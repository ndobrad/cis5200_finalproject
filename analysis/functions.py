import shap
import time
import json

import math, random, os, gc
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Iterable, Optional, Callable

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np, math

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------
# 1) Frozen LM: (MODIFIED to include offset_mapping)
# --------------------------------------------------
@dataclass
class LMConfig:
    model_name: str = "roberta-base"      # Hugging Face model id for both tokenizer and LM
    layer_index: int = 10                 # which hidden layer to read activations from
    max_len: int = 256                    # tokenizer truncation length
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # compute device

class FrozenLM:
    def __init__(self, cfg: LMConfig):
        self.cfg = cfg                    # store config
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)  # load model-specific tokenizer
        self.lm  = AutoModel.from_pretrained(cfg.model_name).to(cfg.device)  # load base LM, move to device
        self.lm.eval()                    # inference mode: disables dropout etc.
        self.lm.requires_grad_(False)     # freeze weights (no gradients)
        # ids to ignore when collecting tokens (special/pad)
        self._ignore = set([
            self.tok.cls_token_id,
            self.tok.sep_token_id,
            self.tok.pad_token_id
        ]) - {None}                       # drop any Nones (some tokenizers may not define all)

    @torch.no_grad()                      # no grad context for everything in this method
    def token_activations(self, texts: List[str]) -> Tuple[
        List[torch.Tensor],               # acts_per_ex: per-example [T_i, d_model] activations
        List[List[int]],                  # ids_per_ex: per-example kept token ids
        List[List[Tuple[int, int]]]       # offs_per_ex: per-example char offsets for kept tokens
    ]:
        """
        Returns:
         acts_per_ex: list of [T_i, d_model] tensors (on CPU)
         ids_per_ex:  list of token id lists (kept tokens only)
         offs_per_ex: list of (start_char, end_char) tuples for kept tokens
        """
        batch = self.tok(                 # tokenize a batch of raw strings
            texts,
            return_tensors="pt",          # return PyTorch tensors
            padding=True,                 # pad to max length within the batch
            truncation=True,              # cut longer sequences
            max_length=self.cfg.max_len,
            return_offsets_mapping=True   # also return per-token (start,end) char offsets
        )
        # print(batch)                      # debug print of the BatchEncoding (safe to remove)

        # Pop offsets before sending to model: they’re metadata, not model inputs
        offset_mapping_list = batch.pop("offset_mapping")

        # Move remaining model inputs (input_ids, attention_mask, etc.) to device
        batch = batch.to(self.cfg.device)

        # Run the LM and collect all hidden states; pick one layer
        hs = self.lm(**batch, output_hidden_states=True).hidden_states
        X = hs[self.cfg.layer_index]      # [B, T, d_model] activations at chosen layer

        acts_per_ex, ids_per_ex, offs_per_ex = [], [], []
        for i in range(X.size(0)):        # iterate over batch dimension
            ids = batch["input_ids"][i].tolist()               # token ids for example i
            offsets = offset_mapping_list[i].tolist()          # char offsets for example i

            # keep indices that are not special/pad
            keep = [j for j, tid in enumerate(ids) if tid not in self._ignore]

            # slice to kept tokens; detach+cpu to return lightweight tensors
            acts_per_ex.append(X[i, keep].detach().cpu())      # [T_i, d_model]
            ids_per_ex.append([ids[j] for j in keep])          # token ids for kept tokens
            offs_per_ex.append([offsets[j] for j in keep])     # (start,end) chars for kept tokens

        return acts_per_ex, ids_per_ex, offs_per_ex

# ------------------------------------------
# 2) Sparse Autoencoder (feature dictionary)
# ------------------------------------------
class SAE(nn.Module):
    """
    Bias-free linear encoder/decoder with ReLU code and column-normalized decoder.
    """
    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.E = nn.Linear(d_model, k, bias=False)
        self.D = nn.Linear(k, d_model, bias=False)
        nn.init.kaiming_uniform_(self.E.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.D.weight)

    def forward(self, X):                         # X: [B, d]
        Z = torch.relu(self.E(X))                 # [B, k]
        X_hat = self.D(Z)                         # [B, d]
        return X_hat, Z

def proj_col_norm(W: torch.Tensor):
    with torch.no_grad():
        W /= (W.norm(dim=0, keepdim=True) + 1e-8)

def train_sae(token_batches: Iterable[torch.Tensor],
              d_model: int,
              k: int = 8*768,          # adjust after you know d_model
              lr: float = 3e-4,
              lam: float = 1e-3,        # L1 on codes; tune for 2-6 active feats/token
              steps: int = 50_000,
              device: str = "cuda" if torch.cuda.is_available() else "cpu",
              log_every: int = 200) -> SAE:
    sae = SAE(d_model=d_model, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    step = 0
    run_recon, run_spars = 0.0, 0.0

    data_iter = iter(token_batches)
    while step < steps:
        try:
            X = next(data_iter).to(device)               # [B, d]
        except StopIteration:
            data_iter = iter(token_batches)
            X = next(data_iter).to(device)

        X_hat, Z = sae(X)
        recon = ((X - X_hat)**2).mean()
        spars = lam * Z.abs().mean()
        loss = recon + spars

        opt.zero_grad()
        loss.backward()
        opt.step()

        proj_col_norm(sae.D.weight)

        run_recon += recon.item()
        run_spars += (spars.item() / lam)
        step += 1
        if step % log_every == 0:
            print(f"[{step}/{steps}] recon={run_recon/log_every:.4f}  active/token={run_spars/log_every:.3f}")
            run_recon, run_spars = 0.0, 0.0
    return sae

class TokenBatcher:
    """
    Streams random token/phrase activations for SAE training.

    New:
      - text_cleaner: pre-cleans HTML/IDs/URLs
      - stopwords: expanded set (lowercased)
      - ngram_sizes: allow multi-word windows (e.g., (1,2,3))
      - min_token_len: drop ultra-short subwords (e.g., 's')
      - min_meaningful_ratio: require proportion of meaningful tokens per window
      - agg: how to collapse within-window activations: 'mean' or 'max'
    """
    def __init__(self,
                 texts: List[str],
                 lm: FrozenLM,
                 batch_tokens: int = 8192,
                 shuffle: bool = True,
                 stopwords: Optional[set] = None,
                 ngram_sizes: Tuple[int, ...] = (1,),
                 min_token_len: int = 3,
                 min_meaningful_ratio: float = 1.0,  # require all tokens meaningful by default
                 agg: str = "mean",
                 text_cleaner: Optional[Callable[[str], str]] = None):
        self.texts = texts
        self.lm = lm
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.stopwords = set(w.lower() for w in (stopwords or set()))
        self.ngram_sizes = tuple(sorted({int(n) for n in ngram_sizes if int(n) >= 1}))
        self.min_token_len = int(min_token_len)
        self.min_meaningful_ratio = float(min_meaningful_ratio)
        assert agg in {"mean", "max"}
        self.agg = agg
        # basic cleaner if none provided
        def _default_clean(x: str) -> str:
            import re
            x = re.sub(r"<br\s*/?>", " ", x)                 # HTML breaks
            x = re.sub(r"\[\[.*?\]\]", " ", x)               # [[VIDEOID:...]]
            x = re.sub(r"https?://\S+|www\.\S+", " ", x)     # URLs
            x = re.sub(r"\s+", " ", x).strip()
            return x
        self.text_cleaner = text_cleaner or _default_clean

    def _norm_tok(self, s: str) -> str:
        import re
        s = s.replace("Ġ", " ").strip()
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        return s.lower()

    def __iter__(self):
        idxs = list(range(0, len(self.texts), 32))
        if self.shuffle:
            random.shuffle(idxs)

        for s in idxs:
            raw = self.texts[s: s+32]
            sub = [self.text_cleaner(t) for t in raw]

            #acts_list, ids_list = self.lm.token_activations(sub)
            acts_list, ids_list, _ = self.lm.token_activations(sub) # <-- FIX: Catch 3rd value
            win_vecs = []

            for X, tok_ids in zip(acts_list, ids_list):
                if X.numel() == 0:
                    continue

                toks = self.lm.tok.convert_ids_to_tokens(tok_ids)
                toks = [self._norm_tok(t) for t in toks]
                T = X.size(0)

                # precompute meaningful mask per token
                is_meaningful = [
                    (w.isalpha() and len(w) >= self.min_token_len and (w not in self.stopwords))
                    for w in toks
                ]

                for n in self.ngram_sizes:
                    if T < n:
                        continue
                    for t in range(0, T - n + 1):
                        frac = sum(is_meaningful[t:t+n]) / n
                        if frac < self.min_meaningful_ratio:
                            continue
                        seg = X[t:t+n]
                        if self.agg == "mean":
                            win_vecs.append(seg.mean(dim=0))
                        else:
                            win_vecs.append(seg.max(dim=0).values)

            if not win_vecs:
                continue
            Xw = torch.stack(win_vecs, dim=0)
            for t in range(0, Xw.size(0), self.batch_tokens):
                yield Xw[t:t+self.batch_tokens]
                
# -------------------------------------------------
# 3) Map SAE features to token spans (exemplars)
# -------------------------------------------------
def decode_tokens(tok: AutoTokenizer, ids: List[int]) -> str:
    # For RoBERTa/BPE-like tokenizers: join then fix spaces
    s = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(ids))
    return s

@torch.no_grad()
def feature_scores_for_text(lm: FrozenLM, sae: SAE, text: str, device,
                            top_p: float = 0.995) -> Dict[int, List[Tuple[str, float]]]:
    # ensure both inputs and SAE live on the same device
    device = torch.device(device)
    sae = sae.to(device)

    acts, ids = lm.token_activations([text])   # CPU tensors by design
    X = acts[0].to(device)                     # [T, d] → move to same device as SAE

    with torch.no_grad():
        Z = torch.relu(sae.E(X))               # [T, k], computed on `device`

    Z_np = Z.detach().cpu().numpy()            # move to CPU before .numpy()

    tok_ids = ids[0]
    k = Z.size(1)
    out = {}
    for j in range(k):
        col = Z_np[:, j]
        if col.max() <= 0:
            continue
        pos = col[col > 0]
        thr = np.quantile(pos, top_p) if pos.size > 0 else np.inf

        runs, cur = [], []
        for t, val in enumerate(col):
            if val > thr:
                cur.append((t, val))
            else:
                if cur:
                    runs.append(cur); cur = []
        if cur: runs.append(cur)

        spans = []
        for run in runs:
            idxs = [t for t, _ in run]
            score = float(np.mean([v for _, v in run]))
            span_text = decode_tokens(lm.tok, [tok_ids[t] for t in idxs])
            spans.append((span_text, score))
        if spans:
            out[j] = sorted(spans, key=lambda x: -x[1])[:5]
    return out

# ---------------------------------------------------------
# 4) Aggregate to review-level features for prediction
# ---------------------------------------------------------
@torch.no_grad()
def review_features(lm: FrozenLM, sae: SAE, texts: List[str],
                    pool: str = "max", topk: int = 3) -> np.ndarray:
    """
    Returns [N, k] matrix of review-level SAE features.
    pool:
      "max"  → max_t z_{t,j}
      "topk" → mean of top-k tokens per feature (length-robust) 
    """
    # Use SAE's actual device as source of truth
    dev = next(sae.parameters()).device
    k = sae.E.weight.size(1)

    feats = []
    for i in tqdm(range(0, len(texts), 32), desc="SAE->review features"):
        sub = texts[i:i+32]
        acts_list, _, _ = lm.token_activations(sub)  # CPU tensors by design
        for X in acts_list:
            # Edge case: no kept tokens (e.g., empty text after stripping specials)
            if X.numel() == 0:
                feats.append(np.zeros(k, dtype=np.float32))
                continue

            X = X.to(dev)                      # <-- move inputs onto SAE's device
            with torch.no_grad():
                Z = torch.relu(sae.E(X))       # [T, k] on `dev`

                if pool == "max":
                    v = Z.max(dim=0).values    # [k]
                else:  # "topk"
                    kk = min(topk, Z.size(0))
                    if kk == 0:
                        v = torch.zeros(k, device=dev)
                    else:
                        vals, _ = torch.topk(Z, k=kk, dim=0)  # [kk, k] This controls how many of the strongest token activations per SAE feature you average to make one review-level value.
                        v = vals.mean(dim=0)                  # [k]

            feats.append(v.detach().cpu().numpy())  # back to CPU before .numpy()

    return np.stack(feats, axis=0)   # [N, k]

def fit_downstream_RF(F_train: np.ndarray, y_train: np.ndarray,
                   F_valid: np.ndarray, y_valid: np.ndarray,
                   n_estimators: int = 800,
                   max_depth: int = 10,
                   min_samples_leaf: int = 5,
                   n_jobs: int = -1,
                   random_state: int = 42):
    """
    RandomForest regressor for log(inter-review-time).
    Returns: (model, {"R2": ..., "RMSE": ...})
    Notes:
      - No scaling needed (RF is scale-invariant).
      - `min_samples_leaf=5` and `max_features="sqrt"` add regularization for wide SAE features.
      - Increase `n_estimators` (e.g., 1200–2000) for more stability if time allows.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        oob_score=False,
        bootstrap=True,
    )
    rf.fit(F_train, y_train)
    pred = rf.predict(F_valid)
    rmse = math.sqrt(mean_squared_error(y_valid, pred))
    r2   = r2_score(y_valid, pred)
    return rf, {"R2": r2, "RMSE": rmse}


def make_sae_feature_df(lm: FrozenLM,
                        sae: SAE,
                        df: pd.DataFrame,
                        text_col: str = "text",
                        id_col: str = "review_id",
                        pool: str = "topk",
                        topk: int = 3,
                        prefix: str = "sae_") -> pd.DataFrame:
    """
    Builds a DataFrame with one row per review and one column per SAE feature:
      [id_col, sae_0, sae_1, ..., sae_{K-1}]
    Uses the existing review_features(...) aggregator from your pipeline.
    """
    # Compute review-level SAE features for all rows (order preserved)
    X = review_features(lm, sae, df[text_col].tolist(), pool=pool, topk=topk)  # shape [N, K]
    # Build column names and assemble DataFrame
    k = X.shape[1]
    cols = [f"{prefix}{j}" for j in range(k)]
    feat_df = pd.DataFrame(X, columns=cols)
    # Attach the identifier to enable a clean merge
    feat_df[id_col] = df[id_col].values
    # Keep id first, then features
    return feat_df[[id_col] + cols]

from typing import List, Dict, Any, Tuple
## Second trial
def feature_triggers_with_offsets_hf(
    lm: FrozenLM,
    sae: SAE,
    text: str,
    j: int,
    *,
    top_p: float = 0.98,                 # lower than 0.995 → longer runs
    min_run_len: int = 2,                # require ≥2-token spans
    stopwords: Optional[set] = None,     # e.g., STOP from your TokenBatcher
    min_token_len: int = 3               # drop tiny subwords like "s"
) -> List[Dict[str, Any]]:
    """
    Return spans where SAE feature j fires strongly.
    Each item: {feature, token_start/end, char_start/end, score_mean, score_max,
                token_text, span_text}
    """
    dev = next(sae.parameters()).device
    stop = set(w.lower() for w in (stopwords or set()))

    # --- helper: normalize tokenizer string token to a plain word form
    def _norm_tok(s: str) -> str:
        import re
        s = s.replace("Ġ", " ").strip()
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        return s.lower()

    # --- 1) offsets for char mapping (non-special tokens only)
    enc = lm.tok(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=lm.cfg.max_len,
        padding=False,
    )
    input_ids = enc["input_ids"][0].tolist()
    offsets   = enc["offset_mapping"][0].tolist()
    ignore = {lm.tok.cls_token_id, lm.tok.sep_token_id, lm.tok.pad_token_id} - {None}
    keep_idx = [i for i, tid in enumerate(input_ids) if tid not in ignore]
    kept_offsets = [offsets[i] for i in keep_idx]

    # --- 2) token activations for same kept tokens
    acts, ids = lm.token_activations([text])   # acts[0]: [T, d], ids[0]: kept token ids
    if not acts or acts[0].numel() == 0:
        return []
    X = acts[0].to(dev)
    tok_ids_kept = ids[0]

    # safety: align lengths if tokenizer bookkeeping differs
    T = min(X.shape[0], len(tok_ids_kept), len(kept_offsets))
    X = X[:T]
    kept_offsets = kept_offsets[:T]
    tok_ids_kept = tok_ids_kept[:T]

    # get per-token codes for feature j
    with torch.no_grad():
        Z = torch.relu(sae.E(X))                      # [T, K] on dev
    col = Z[:, j].detach().cpu().numpy()              # [T]

    # --- 3) mask uninformative tokens (stopwords, digits, short pieces)
    toks = lm.tok.convert_ids_to_tokens(tok_ids_kept)
    words = [_norm_tok(t) for t in toks]
    keep_token = np.array(
        [(w.isalpha() and (len(w) >= min_token_len) and (w not in stop)) for w in words],
        dtype=bool
    )
    col_masked = col.copy()
    col_masked[~keep_token] = 0.0

    pos = col_masked[col_masked > 0]
    if pos.size == 0:
        return []

    thr = np.quantile(pos, top_p)

    # --- 4) collect contiguous runs above threshold
    runs = []
    s = None
    for t, v in enumerate(col_masked):
        if v > thr and s is None:
            s = t
        elif (v <= thr or t == len(col_masked) - 1) and s is not None:
            e = t if v <= thr else t + 1            # [s, e)
            if (e - s) >= min_run_len:
                runs.append((s, e))
            s = None

    if not runs:
        return []

    # --- 5) expand char slice to full word boundaries for readability
    def _expand_to_word_bounds(txt: str, a: int, b: int) -> tuple[int, int]:
        L = a
        while L > 0 and txt[L-1].isalnum():
            L -= 1
        R = b
        while R < len(txt) and txt[R].isalnum():
            R += 1
        return L, R

    out: List[Dict[str, Any]] = []
    for a, b in runs:
        c0 = kept_offsets[a][0]
        c1 = kept_offsets[b-1][1]

        # expand to cover entire word (helps avoid "[[l]]int" cases)
        c0e, c1e = _expand_to_word_bounds(text, c0, c1)

        seg = col[a:b]
        t_max = int(np.argmax(seg) + a)
        token_text = lm.tok.convert_tokens_to_string(
            lm.tok.convert_ids_to_tokens([tok_ids_kept[t_max]])
        )
        span_text = text[c0e:c1e]

        out.append({
            "feature": j,
            "token_start": a, "token_end": b,
            "char_start": int(c0e), "char_end": int(c1e),
            "score_mean": float(seg.mean()), "score_max": float(seg.max()),
            "token_at_max": t_max, "token_text": token_text,
            "span_text": span_text,
        })
    return out
    
def load_sae(path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(path, map_location=device)
    d_model = ckpt["arch"]["d_model"]; k = ckpt["arch"]["k"]
    model = SAE(d_model=d_model, k=k).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # re-normalize decoder columns (safety)
    with torch.no_grad():
        model.D.weight.data /= (model.D.weight.data.norm(dim=0, keepdim=True) + 1e-8)
    return model, ckpt.get("meta", {})

@torch.no_grad()  # Disable autograd for the whole function (no gradients needed for inspection)
def get_feature_spans(
    lm: FrozenLM, 
    sae: SAE, 
    text: str, 
    device,
    top_p: float = 0.995
) -> Dict[int, List[Tuple[str, float, Tuple[int, int]]]]:
    """
    Find activating text spans for *all* SAE features within a single review.

    Returns:
        A dict mapping each feature index j → a (short) list of its strongest spans:
          [
            (span_text, mean_activation_score_over_span, (char_start, char_end)),
            ...
          ]
        where (char_start, char_end) are character offsets into the *original* text.
    """
    # --- Device hygiene: ensure model and inputs live on the same device ---
    device = torch.device(device)   # Normalize string/device-like into a torch.device
    sae = sae.to(device)            # Move SAE parameters to the target device

    # --- Tokenize + run the frozen LM once to get per-token hidden states + offsets ---
    # acts: per-example token activations from the chosen layer; ids: token IDs; offsets: (char_start, char_end)
    # We pass a *single* text, so we take [0] later.
    acts, ids, offsets = lm.token_activations([text]) # <get the token-level embeddings>

    # Edge case: if no tokens were kept (e.g., empty string), bail out early.
    if not acts[0].numel():
        return {}

    # Unpack per-token arrays for this example
    X = acts[0].to(device)     # [T, d] tensor of token hidden states (T tokens, d = hidden size)
    tok_ids = ids[0]           # list[int] of kept token IDs, aligned with rows of X
    tok_offsets = offsets[0]   # list[tuple(int,int)] character spans aligned with rows of X

    # --- Encode tokens with the SAE to get sparse codes Z (one code per token) ---
    # SAE encoder E: X @ E^T → codes; ReLU enforces nonnegativity/sparsity.
    # <Compute per-token SAE>
    with torch.no_grad():                  # Redundant with decorator, kept for clarity near the op
        Z = torch.relu(sae.E(X))           # [T, K] where K is the number of SAE features

    # Move to CPU/NumPy for convenient thresholding and slicing
    Z_np = Z.detach().cpu().numpy()

    k = Z.size(1)  # number of SAE features (columns in Z)
    out: Dict[int, List[Tuple[str, float, Tuple[int, int]]]] = {}

    # --- For each feature j, find contiguous token runs where it fires strongly ---
    for j in range(k):
        # <Get the SAE activation across tokens>
        # <Correct: Take the column z_{:,j} for feature j>
        col = Z_np[:, j]                   # 1>-D array of activations z_{t,j} across tokens t=0..T-1

        # Skip features that never fire in this text
        if col.max() <= 0:
            continue

        # Consider only positive activations; ignore numerical zeros
        pos = col[col > 1e-8]
        if pos.size == 0:
            continue

        # Data-driven threshold: keep tokens whose activation is above the top_p-quantile *within this review*
        # Example: top_p=0.995 ≈ top 0.5% of positive activations for this feature in this text
        # <Keep only the strongest tokens>
        thr = np.quantile(pos, top_p)

        # Group above-threshold tokens into contiguous runs (spans)
        # <For each SAE feature, save the token idx and activation values that is above top 0.5%>
        runs, cur = [], []
        for t, val in enumerate(col):
            if val >= thr:
                cur.append((t, val))       # extend current run
            else:
                if cur:
                    runs.append(cur)       # close current run
                    cur = []
        if cur:
            runs.append(cur)               # close trailing run, if any

        # Convert token runs into readable spans with character offsets
        # <Decode a human-readable span from token IDs>
        spans: List[Tuple[str, float, Tuple[int, int]]] = []
        for run in runs:
            # Token indices covered by this run (into X / tok_ids / tok_offsets)
            idxs = [t for t, _ in run]

            # Span score: mean activation over the run (you could also use max)
            score = float(np.mean([v for _, v in run]))

            # Decode a human-readable span string from token IDs
            # (decode_tokens should merge subwords properly for the tokenizer)
            span_text = decode_tokens(lm.tok, [tok_ids[t] for t in idxs])

            # Map token indices to *character* offsets in the original text:
            # from the first token's start to the last token's end
            char_start = tok_offsets[idxs[0]][0]
            char_end   = tok_offsets[idxs[-1]][1]

            spans.append((span_text, score, (char_start, char_end)))

        # Keep up to 5 strongest spans for this feature, sorted by score (descending)
        if spans:
            out[j] = sorted(spans, key=lambda x: -x[1])[:5]

    return out

def inspect_sae_feature(
    feature_idx: int,
    lm: FrozenLM, 
    sae: SAE, 
    F_va: np.ndarray, 
    texts_va: List[str], 
    order: np.ndarray,
    n_reviews_to_show: int = 5,
):
    """
    Inspects a specific SAE feature by finding and highlighting the top 
    N reviews from the validation set that highly activate this feature.

    Args:
        feature_idx: The index of the SAE feature to inspect.
        lm: The FrozenLM instance.
        sae: The trained SAE model instance.
        F_va: Review-level feature activations array [N_samples, K].
        texts_va: List of raw review texts corresponding to F_va.
        order: Array of feature indices sorted by importance (e.g., SHAP).
        n_reviews_to_show: Number of top examples to display.
    """
    
    if feature_idx not in order and feature_idx < F_va.shape[1]:
        print(f"Warning: Feature {feature_idx} not found among SHAP-ranked features, but proceeding.")
    elif feature_idx >= F_va.shape[1]:
        print(f"Error: Feature index {feature_idx} is out of bounds (max index is {F_va.shape[1]-1}).")
        return

    print(f"\n{'='*60}")
    print(f"Inspecting SAE Feature: {feature_idx}")
    print(f"{'='*60}\n")

    # 1. Find the validation reviews where this feature was *most* active
    activations_for_feature = F_va[:, feature_idx]
    top_review_indices = np.argsort(activations_for_feature)[::-1]

    # 2. Analyze and highlight the top N reviews
    for i in range(min(n_reviews_to_show, len(top_review_indices))):
        review_index = top_review_indices[i]
        text = texts_va[review_index]
        review_level_score = F_va[review_index, feature_idx]

        # Get all feature spans for this *single* review
        feature_data = get_feature_spans(lm, sae, text, device=lm.cfg.device)
        print("feature_data", feature_data)

        # Get the specific spans for *our* feature of interest
        spans_for_feature = feature_data.get(feature_idx, [])
        
        # Collect just the (start, end) character tuples for the highlighter
        char_spans = [span[2] for span in spans_for_feature]

        print(f"--- Top Example {i+1} (Review Index: {review_index}) ---")
        print(f"Review-Level Feature Activation: {review_level_score:.4f}")
        
        print("\nHighlighted Review (Feature Activation Points in RED):")
        print(highlight_text_spans(text, char_spans))
        print("\n")
        
        print("Activating Spans Found:")
        if spans_for_feature:
            for span_text, score, (start, end) in spans_for_feature:
                print(f"  - (Score: {score:.4f}) @ [{start}:{end}]: '{span_text}'")
        else:
            print("  - (No spans found above token-level threshold for this review)")
        print("-" * 60 + "\n")

def highlight_text_spans(
    text: str, 
    char_spans: List[Tuple[int, int]], 
    color_code: str = "\033[91m"
) -> str:
    """
    Highlights a text with ANSI color codes given character spans.
    NOTE: This assumes spans are non-overlapping.
    """
    reset_code = "\033[0m"
    
    # Sort spans by start index to process them in order
    sorted_spans = sorted(char_spans, key=lambda x: x[0])
    
    parts = []
    last_pos = 0
    for start, end in sorted_spans:
        # Add the text *before* the current span
        parts.append(text[last_pos:start])
        # Add the *highlighted* span
        parts.append(f"{color_code}{text[start:end]}{reset_code}")
        last_pos = end
    
    # Add any remaining text after the last span
    parts.append(text[last_pos:])
    
    return "".join(parts)

def r1_regression(X, y, feature_names=None, alpha=0.05, robust_se="HC3"):
    """
    OLS with robust standard errors; returns full table and significant rows.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape [n_samples, n_features]
        SAE embeddings (review-level).
    y : array-like, shape [n_samples]
        Target (y_log_irt).
    feature_names : list[str] or None
        Names for columns in X. If None, uses f"sae_f{i}".
    alpha : float
        Significance level for filtering.
    robust_se : {"HC0","HC1","HC2","HC3", None}
        Heteroskedasticity-robust covariance type. None → classical OLS SE.

    Returns
    -------
    results_df : pd.DataFrame
        Coefficient table (including p-values and FDR q-values).
    sig_df : pd.DataFrame
        Subset of rows significant at `alpha` after BH-FDR.
    model : statsmodels regression results
        Fitted model object.
    """
    # Convert to DataFrame and sanitize
    if isinstance(X, np.ndarray):
        n_features = X.shape[1]
        cols = feature_names or [f"sae_f{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=cols)
    else:
        X_df = X.copy()
        if feature_names is not None:
            X_df.columns = feature_names

    y = pd.Series(np.asarray(y).reshape(-1), name="y")

    # Drop any rows with non-finite values
    mask = np.isfinite(X_df.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
    X_df = X_df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Add intercept
    X_with_const = sm.add_constant(X_df, has_constant="add")

    # Fit OLS
    model = sm.OLS(y, X_with_const)
    fit = model.fit()

    # Build coefficient table (exclude intercept from feature filtering)
    params = fit.params
    bse = fit.bse
    tvals = fit.tvalues
    pvals = fit.pvalues
    conf_int = fit.conf_int()

    results_df = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "se": bse.values,
        "t": tvals.values,
        "p": pvals.values,
        "ci_low": conf_int[0].values,
        "ci_high": conf_int[1].values,
    })

    return results_df