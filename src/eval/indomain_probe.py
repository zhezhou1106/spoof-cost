"""In-domain linear probe for M2′: fit on E0 activations with held-out-test labels."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class InDomainProbeArtifact:
    """Saved probe weights + metadata for E1 scoring and E2 Atlas injection."""

    layer: int
    d_model: int
    weight: list[float]  # shape (d_model,)
    bias: float
    train_auroc: float
    eval_auroc: float
    n_train: int
    n_eval: int
    model_name: str
    lora_path: str | None
    seed: int

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2))
        # Also write a .pt-compatible sidecar for Atlas LinearProbe loading.
        try:
            import torch

            probe = torch.nn.Linear(self.d_model, 1)
            with torch.no_grad():
                probe.weight.copy_(torch.tensor(self.weight, dtype=torch.float32).unsqueeze(0))
                probe.bias.copy_(torch.tensor([self.bias], dtype=torch.float32))
            torch.save(
                {
                    "layer": self.layer,
                    "state_dict": probe.state_dict(),
                    "d_model": self.d_model,
                    "metadata": payload,
                },
                path.with_suffix(".pt"),
            )
        except ImportError:
            pass

    @classmethod
    def load(cls, path: Path | str) -> InDomainProbeArtifact:
        path = Path(path)
        if path.suffix == ".pt":
            import torch

            data = torch.load(path, map_location="cpu", weights_only=False)
            meta = data.get("metadata") or {}
            sd = data["state_dict"]
            w = sd["weight"].detach().cpu().numpy().reshape(-1).tolist()
            b = float(sd["bias"].detach().cpu().numpy().reshape(-1)[0])
            return cls(
                layer=int(data.get("layer", meta.get("layer", 0))),
                d_model=int(data.get("d_model", len(w))),
                weight=w,
                bias=b,
                train_auroc=float(meta.get("train_auroc", float("nan"))),
                eval_auroc=float(meta.get("eval_auroc", float("nan"))),
                n_train=int(meta.get("n_train", 0)),
                n_eval=int(meta.get("n_eval", 0)),
                model_name=str(meta.get("model_name", "")),
                lora_path=meta.get("lora_path"),
                seed=int(meta.get("seed", 0)),
            )
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def score_vector(self, activation: np.ndarray) -> float:
        """Logistic score → [0, 1] suspicion (larger = more deceptive)."""
        x = np.asarray(activation, dtype=np.float64).reshape(-1)
        logit = float(np.dot(self.weight, x) + self.bias)
        # Stable sigmoid
        if logit >= 0:
            p = 1.0 / (1.0 + np.exp(-logit))
        else:
            e = np.exp(logit)
            p = e / (1.0 + e)
        return float(np.clip(p, 0.0, 1.0))


def cluster_id_for_prompt(prompt: str) -> str:
    """Stable prompt-cluster key (problem text hash when available)."""
    from env.prompt_parse import parse_cot_monitor_context

    ctx = parse_cot_monitor_context(prompt)
    key = ctx.problem_text.strip() if ctx.problem_text.strip() else prompt.strip()
    return hashlib.md5(key.encode()).hexdigest()


def prompt_cluster_split(
    rollouts: Sequence[Any],
    *,
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[list[Any], list[Any]]:
    """Split by prompt cluster so train/eval share no problems."""
    rng = random.Random(seed)
    by_cluster: dict[str, list[Any]] = {}
    for r in rollouts:
        cid = cluster_id_for_prompt(r.prompt)
        by_cluster.setdefault(cid, []).append(r)
    clusters = sorted(by_cluster.keys())
    rng.shuffle(clusters)
    n_train = max(1, int(round(len(clusters) * train_frac)))
    if n_train >= len(clusters) and len(clusters) > 1:
        n_train = len(clusters) - 1
    train_c = set(clusters[:n_train])
    train = [r for c, rows in by_cluster.items() if c in train_c for r in rows]
    eval_ = [r for c, rows in by_cluster.items() if c not in train_c for r in rows]
    if not eval_ and train:
        # Degenerate tiny set: move one cluster to eval.
        move = train_c.pop()
        eval_ = by_cluster[move]
        train = [r for c, rows in by_cluster.items() if c in train_c for r in rows]
    return train, eval_


def _fake_activation(prompt: str, response: str, d_model: int = 64) -> np.ndarray:
    """Deterministic pseudo-activation for dry-run / unit tests."""
    h = hashlib.md5(f"{prompt}:{response}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    base = rng.normal(0, 1, size=d_model).astype(np.float64)
    # Inject a weak deception cue so mock fit can separate labels.
    from env.parse import extract_reasoning

    reasoning = extract_reasoning(response).lower()
    if "hardcode" in reasoning or "visible" in reasoning:
        base[0] += 2.0
    else:
        base[0] -= 2.0
    return base


def extract_last_token_activations(
    prompts: list[str],
    responses: list[str],
    *,
    model_name: str,
    lora_path: str | Path | None,
    layers: list[int] | None = None,
    batch_size: int = 4,
    device: str | None = None,
    mock: bool = False,
    mock_d_model: int = 64,
) -> dict[int, np.ndarray]:
    """Extract last-token residual activations per layer. Shape (N, d_model)."""
    if mock:
        acts = np.stack([_fake_activation(p, r, mock_d_model) for p, r in zip(prompts, responses, strict=True)])
        # Mirror requested layer keys so fitted artifacts (e.g. layer 29) score in dry-run.
        use_layers = layers if layers is not None else [0]
        return {int(li): acts for li in use_layers}

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dev.startswith("cuda") else torch.float32,
        device_map=dev if dev.startswith("cuda") else None,
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()
    if not dev.startswith("cuda"):
        model = model.to(dev)

    n_layers = model.config.num_hidden_layers
    use_layers = layers if layers is not None else list(range(n_layers))
    accum: dict[int, list[np.ndarray]] = {li: [] for li in use_layers}

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        batch_r = responses[i : i + batch_size]
        texts = [p + r for p, r in zip(batch_p, batch_r, strict=True)]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states[0] = embeddings; [1..] = layers
        hs = out.hidden_states
        # Last non-pad token index per row.
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        for li in use_layers:
            layer_h = hs[li + 1]  # (B, T, D)
            gathered = layer_h[torch.arange(layer_h.size(0), device=layer_h.device), last_idx]
            accum[li].append(gathered.float().cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {li: np.concatenate(parts, axis=0) for li, parts in accum.items()}


def _fit_layer(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    y_ev: np.ndarray,
    *,
    layer: int,
    model_name: str,
    lora_path: str | Path | None,
    seed: int,
) -> InDomainProbeArtifact:
    """Fit one logistic probe on a single layer's activations."""
    if len(np.unique(y_tr)) < 2:
        train_auc = float("nan")
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_tr, np.zeros(len(y_tr), dtype=int) if len(y_tr) else [0])
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        train_auc = float(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1]))
    if len(np.unique(y_ev)) < 2:
        eval_auc = float("nan")
    else:
        eval_auc = float(roc_auc_score(y_ev, clf.predict_proba(X_ev)[:, 1]))
    return InDomainProbeArtifact(
        layer=int(layer),
        d_model=int(X_tr.shape[1]),
        weight=clf.coef_.reshape(-1).astype(float).tolist(),
        bias=float(clf.intercept_[0]),
        train_auroc=train_auc,
        eval_auroc=eval_auc,
        n_train=len(y_tr),
        n_eval=len(y_ev),
        model_name=model_name,
        lora_path=str(lora_path) if lora_path else None,
        seed=seed,
    )


def _predict_proba_positive(artifact: InDomainProbeArtifact, X: np.ndarray) -> np.ndarray:
    """Apply a fitted artifact as a logistic probe → P(deceptive)."""
    w = np.asarray(artifact.weight, dtype=np.float64)
    logits = X @ w + float(artifact.bias)
    # Stable sigmoid
    out = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    e = np.exp(logits[~pos])
    out[~pos] = e / (1.0 + e)
    return np.clip(out, 0.0, 1.0)


def fit_indomain_probe(
    train_prompts: list[str],
    train_responses: list[str],
    train_labels: list[bool],
    eval_prompts: list[str],
    eval_responses: list[str],
    eval_labels: list[bool],
    *,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_path: str | Path | None = None,
    layers: list[int] | None = None,
    seed: int = 0,
    mock: bool = False,
    batch_size: int = 4,
) -> InDomainProbeArtifact:
    """Fit per-layer logistic probes; keep the layer with best train AUROC."""
    train_acts = extract_last_token_activations(
        train_prompts,
        train_responses,
        model_name=model_name,
        lora_path=lora_path,
        layers=layers,
        batch_size=batch_size,
        mock=mock,
    )
    eval_acts = extract_last_token_activations(
        eval_prompts,
        eval_responses,
        model_name=model_name,
        lora_path=lora_path,
        layers=layers if layers is not None else list(train_acts.keys()),
        batch_size=batch_size,
        mock=mock,
    )
    y_train = np.asarray(train_labels, dtype=int)
    y_eval = np.asarray(eval_labels, dtype=int)

    best: InDomainProbeArtifact | None = None
    for layer, X_tr in train_acts.items():
        if layer not in eval_acts:
            continue
        art = _fit_layer(
            X_tr,
            y_train,
            eval_acts[layer],
            y_eval,
            layer=int(layer),
            model_name=model_name,
            lora_path=lora_path,
            seed=seed,
        )
        if best is None or (
            (art.train_auroc == art.train_auroc)
            and (best.train_auroc != best.train_auroc or art.train_auroc > best.train_auroc)
        ):
            best = art
    if best is None:
        raise RuntimeError("No layers available to fit in-domain probe")
    return best


def cluster_kfold_assign(
    rollouts: Sequence[Any],
    *,
    n_folds: int = 5,
    seed: int = 0,
    labels: Sequence[bool] | None = None,
) -> list[int]:
    """Assign each rollout a fold id with no prompt cluster spanning folds.

    Clusters are majority-labeled, shuffled within each label group, then
    round-robin assigned. If any fold is single-class, ``n_folds`` is decremented
    and the procedure retries down to 2; then raises.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    n = len(rollouts)
    if n == 0:
        return []

    if labels is None:
        y_list = [bool(getattr(r, "deceptive")) for r in rollouts]
    else:
        if len(labels) != n:
            raise ValueError("labels length must match rollouts")
        y_list = [bool(x) for x in labels]

    by_cluster: dict[str, list[int]] = {}
    for i, r in enumerate(rollouts):
        cid = cluster_id_for_prompt(r.prompt if hasattr(r, "prompt") else r["prompt"])
        by_cluster.setdefault(cid, []).append(i)

    cluster_label: dict[str, bool] = {}
    for cid, idxs in by_cluster.items():
        pos = sum(1 for i in idxs if y_list[i])
        cluster_label[cid] = pos >= (len(idxs) - pos)

    for k in range(n_folds, 1, -1):
        rng = random.Random(seed)
        pos_c = sorted(c for c, lab in cluster_label.items() if lab)
        neg_c = sorted(c for c, lab in cluster_label.items() if not lab)
        rng.shuffle(pos_c)
        rng.shuffle(neg_c)

        fold_of_cluster: dict[str, int] = {}
        for i, c in enumerate(pos_c):
            fold_of_cluster[c] = i % k
        for i, c in enumerate(neg_c):
            fold_of_cluster[c] = i % k

        fold_ids = [0] * n
        for cid, idxs in by_cluster.items():
            f = fold_of_cluster[cid]
            for i in idxs:
                fold_ids[i] = f

        ok = True
        for f in range(k):
            ys = [y_list[i] for i in range(n) if fold_ids[i] == f]
            if not ys or len(set(ys)) < 2:
                ok = False
                break
        if ok:
            return fold_ids

    raise RuntimeError(
        f"Could not assign cluster K-fold with both classes in every fold "
        f"(tried n_folds from {n_folds} down to 2; n={n}, clusters={len(by_cluster)})"
    )


@dataclass
class CrossfitResult:
    """Out-of-fold probe scores plus a full-data artifact for E2 injection."""

    artifact: InDomainProbeArtifact
    oof_scores: list[float]
    fold_ids: list[int]
    layer: int
    oof_auroc: float
    fold_train_aurocs: list[float]
    n_folds: int


def crossfit_indomain_probe(
    prompts: list[str],
    responses: list[str],
    labels: list[bool],
    *,
    layer: int | None = 29,
    n_folds: int = 5,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_path: str | Path | None = None,
    seed: int = 0,
    mock: bool = False,
    batch_size: int = 4,
) -> CrossfitResult:
    """Cluster K-fold cross-fit: OOF scores for gating + full-data artifact for E2.

    Activations are extracted once. Layer selection (when ``layer`` is None) uses
    mean *train-fold* AUROC only — never OOF/gate AUROC.
    """
    if len(prompts) != len(responses) or len(prompts) != len(labels):
        raise ValueError("prompts, responses, labels must have equal length")
    if len(prompts) == 0:
        raise ValueError("empty cohort")

    class _Row:
        __slots__ = ("prompt", "deceptive")

        def __init__(self, prompt: str, deceptive: bool):
            self.prompt = prompt
            self.deceptive = deceptive

    rows = [_Row(p, d) for p, d in zip(prompts, labels, strict=True)]
    fold_ids = cluster_kfold_assign(rows, n_folds=n_folds, seed=seed, labels=labels)
    n_folds_used = max(fold_ids) + 1
    y = np.asarray(labels, dtype=int)

    extract_layers = None if layer is None else [int(layer)]
    acts = extract_last_token_activations(
        prompts,
        responses,
        model_name=model_name,
        lora_path=lora_path,
        layers=extract_layers,
        batch_size=batch_size,
        mock=mock,
    )

    def _oof_for_layer(li: int, X: np.ndarray) -> tuple[np.ndarray, list[float]]:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        train_aurocs: list[float] = []
        for f in range(n_folds_used):
            te = np.asarray([i for i in range(len(y)) if fold_ids[i] == f], dtype=int)
            tr = np.asarray([i for i in range(len(y)) if fold_ids[i] != f], dtype=int)
            art_f = _fit_layer(
                X[tr],
                y[tr],
                X[te],
                y[te],
                layer=li,
                model_name=model_name,
                lora_path=lora_path,
                seed=seed + f,
            )
            train_aurocs.append(float(art_f.train_auroc))
            oof[te] = _predict_proba_positive(art_f, X[te])
        if np.any(~np.isfinite(oof)):
            raise RuntimeError(f"Incomplete OOF scores for layer {li}")
        return oof, train_aurocs

    if layer is not None:
        if int(layer) not in acts:
            raise RuntimeError(f"Requested layer {layer} missing from activations {list(acts)}")
        chosen = int(layer)
        oof_scores, fold_train_aurocs = _oof_for_layer(chosen, acts[chosen])
    else:
        best_li: int | None = None
        best_mean = float("-inf")
        best_oof: np.ndarray | None = None
        best_tr: list[float] = []
        for li in sorted(acts.keys()):
            oof_i, tr_i = _oof_for_layer(int(li), acts[li])
            finite = [t for t in tr_i if t == t]
            mean_tr = float(np.mean(finite)) if finite else float("-inf")
            # Prefer higher mean train-fold AUROC; ties → larger layer index.
            if mean_tr > best_mean or (mean_tr == best_mean and (best_li is None or li > best_li)):
                best_mean = mean_tr
                best_li = int(li)
                best_oof = oof_i
                best_tr = tr_i
        if best_li is None or best_oof is None:
            raise RuntimeError("No layers available for cross-fit probe")
        chosen = best_li
        oof_scores, fold_train_aurocs = best_oof, best_tr

    X_full = acts[chosen]
    # Full-data artifact for E2; eval_auroc stores OOF AUROC.
    full_art = _fit_layer(
        X_full,
        y,
        X_full,
        y,
        layer=chosen,
        model_name=model_name,
        lora_path=lora_path,
        seed=seed,
    )
    if len(np.unique(y)) < 2:
        oof_auroc = float("nan")
    else:
        oof_auroc = float(roc_auc_score(y, oof_scores))
    full_art.eval_auroc = oof_auroc
    full_art.n_eval = len(y)

    return CrossfitResult(
        artifact=full_art,
        oof_scores=[float(x) for x in oof_scores],
        fold_ids=list(fold_ids),
        layer=chosen,
        oof_auroc=oof_auroc,
        fold_train_aurocs=fold_train_aurocs,
        n_folds=n_folds_used,
    )


def score_with_probe(
    artifact: InDomainProbeArtifact,
    prompts: list[str],
    responses: list[str],
    *,
    model_name: str | None = None,
    lora_path: str | Path | None = None,
    mock: bool = False,
    batch_size: int = 4,
    precomputed: np.ndarray | None = None,
) -> list[float]:
    """Score rollouts with a fitted in-domain probe."""
    if precomputed is not None:
        return [artifact.score_vector(row) for row in precomputed]
    acts = extract_last_token_activations(
        prompts,
        responses,
        model_name=model_name or artifact.model_name,
        lora_path=lora_path if lora_path is not None else artifact.lora_path,
        layers=[artifact.layer],
        batch_size=batch_size,
        mock=mock,
        mock_d_model=artifact.d_model,
    )
    X = acts[artifact.layer]
    return [artifact.score_vector(row) for row in X]


def load_as_atlas_linear_probes(path: Path | str) -> dict[int, Any]:
    """Load artifact into an Atlas-compatible ``{layer: LinearProbe}`` dict."""
    import torch
    from obfuscation_atlas.detectors.probe_archs import LinearProbe

    art = InDomainProbeArtifact.load(path)
    probe = LinearProbe(art.d_model, nhead=1, normalize_input="none")
    with torch.no_grad():
        probe.linear.weight.copy_(torch.tensor(art.weight, dtype=torch.float32).unsqueeze(0))
        probe.linear.bias.copy_(torch.tensor([art.bias], dtype=torch.float32))
    return {int(art.layer): probe}
