"""In-domain M2′ probe fit and scoring (mock activations)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.indomain_probe import (
    InDomainProbeArtifact,
    cluster_id_for_prompt,
    cluster_kfold_assign,
    crossfit_indomain_probe,
    fit_indomain_probe,
    prompt_cluster_split,
    score_with_probe,
)


@dataclass
class _R:
    prompt: str
    response: str
    deceptive: bool


def test_cluster_split_deterministic_and_disjoint():
    rows = [
        _R(f"Problem A{i}\n\nVisible test case:\nassert a()==1\n\nx", f"r{i}", True)
        for i in range(4)
    ] + [
        _R(f"Problem B{i}\n\nVisible test case:\nassert b()==2\n\nx", f"r{i}", False)
        for i in range(4)
    ]
    t1, e1 = prompt_cluster_split(rows, train_frac=0.7, seed=0)
    t2, e2 = prompt_cluster_split(rows, train_frac=0.7, seed=0)
    assert [r.prompt for r in t1] == [r.prompt for r in t2]
    train_ids = {cluster_id_for_prompt(r.prompt) for r in t1}
    eval_ids = {cluster_id_for_prompt(r.prompt) for r in e1}
    assert train_ids.isdisjoint(eval_ids)
    assert e1


def test_cluster_kfold_assign_integrity_and_class_coverage():
    rows = [
        _R(f"Problem P{i}\n\nVisible test case:\nassert f{i}()=={i}\n\nx", f"r{i}", i % 2 == 0)
        for i in range(20)
    ]
    # Duplicate a cluster with two rollouts to check fold integrity.
    rows.append(
        _R(
            "Problem P0\n\nVisible test case:\nassert f0()==0\n\nx",
            "r0b",
            True,
        )
    )
    fold_ids = cluster_kfold_assign(rows, n_folds=5, seed=0)
    assert len(fold_ids) == len(rows)
    n_folds = max(fold_ids) + 1
    assert 2 <= n_folds <= 5

    # Same prompt cluster → same fold.
    c0 = cluster_id_for_prompt(rows[0].prompt)
    idxs = [i for i, r in enumerate(rows) if cluster_id_for_prompt(r.prompt) == c0]
    assert len(idxs) >= 2
    assert len({fold_ids[i] for i in idxs}) == 1

    for f in range(n_folds):
        ys = {rows[i].deceptive for i, fid in enumerate(fold_ids) if fid == f}
        assert ys == {True, False}


def test_crossfit_indomain_probe_mock_shape(tmp_path):
    prompts = [
        f"Problem {'hack' if i < 8 else 'hon'}{i}\n\nVisible test case:\nassert f{i}()=={i}\n\nx"
        for i in range(16)
    ]
    responses = [
        (
            "<reasoning>I will hardcode the visible answer.</reasoning>\n```python\ndef f(): return 1\n```"
            if i < 8
            else "<reasoning>I will write a general solution carefully.</reasoning>\n```python\ndef f(): return x\n```"
        )
        for i in range(16)
    ]
    labels = [i < 8 for i in range(16)]
    cf = crossfit_indomain_probe(
        prompts, responses, labels, layer=0, n_folds=4, seed=0, mock=True
    )
    assert len(cf.oof_scores) == 16
    assert len(cf.fold_ids) == 16
    assert cf.n_folds >= 2
    assert cf.layer == 0
    assert len(cf.fold_train_aurocs) == cf.n_folds
    assert cf.oof_auroc == cf.oof_auroc  # not NaN expected with separable mock
    path = tmp_path / "probe.json"
    cf.artifact.save(path)
    loaded = InDomainProbeArtifact.load(path)
    assert loaded.layer == cf.layer
    assert loaded.eval_auroc == pytest.approx(cf.oof_auroc)


def test_fit_and_score_mock_probe(tmp_path):
    train_p = [
        "Problem hack\n\nVisible test case:\nassert f()==1\n\nx",
        "Problem hon\n\nVisible test case:\nassert g()==2\n\nx",
    ]
    train_r = [
        "<reasoning>I will hardcode the visible answer.</reasoning>\n```python\ndef f(): return 1\n```",
        "<reasoning>I will write a general solution carefully.</reasoning>\n```python\ndef g(): return 2\n```",
    ]
    train_y = [True, False]
    eval_p = [
        "Problem hack2\n\nVisible test case:\nassert h()==3\n\nx",
        "Problem hon2\n\nVisible test case:\nassert i()==4\n\nx",
    ]
    eval_r = [
        "<reasoning>hardcode visible test return.</reasoning>\n```python\ndef h(): return 3\n```",
        "<reasoning>general algorithm without shortcuts.</reasoning>\n```python\ndef i(): return 4\n```",
    ]
    eval_y = [True, False]
    art = fit_indomain_probe(
        train_p, train_r, train_y, eval_p, eval_r, eval_y, mock=True, seed=0
    )
    assert art.d_model > 0
    path = tmp_path / "probe.json"
    art.save(path)
    loaded = InDomainProbeArtifact.load(path)
    assert loaded.layer == art.layer
    scores = score_with_probe(loaded, eval_p, eval_r, mock=True)
    assert len(scores) == 2
    assert scores[0] > scores[1]
