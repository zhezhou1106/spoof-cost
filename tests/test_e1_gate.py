"""E1 matched-strength gate helpers."""

from __future__ import annotations

from eval.auroc import arm_passes_e1_gate, build_matched_strength_table


def test_arm_passes_e1_gate():
    assert arm_passes_e1_gate(0.70, 0.55) is True
    assert arm_passes_e1_gate(0.70, 0.45) is False
    assert arm_passes_e1_gate(0.60, 0.55) is False
    assert arm_passes_e1_gate(float("nan"), 0.55) is False
    assert arm_passes_e1_gate(0.70, None) is True


def test_matched_strength_table_schema():
    results = {
        "scl": {
            "auroc": 0.74,
            "psi_mean_hacking": 0.9,
            "psi_mean_honest": 0.5,
            "bootstrap": {"ci_low": 0.64, "ci_high": 0.85},
        },
        "judge": {
            "auroc": 0.50,
            "psi_mean_hacking": 0.77,
            "psi_mean_honest": 0.77,
            "bootstrap": {"ci_low": 0.40, "ci_high": 0.60},
        },
    }
    table = build_matched_strength_table(results, gate=0.65, ci_floor=0.50)
    assert len(table) == 2
    by_name = {r["monitor"]: r for r in table}
    assert by_name["scl"]["gate_pass"] is True
    assert by_name["judge"]["gate_pass"] is False
    for row in table:
        assert set(row) >= {
            "monitor",
            "auroc",
            "ci_low",
            "ci_high",
            "psi_mean_hacking",
            "psi_mean_honest",
            "gate_pass",
        }
    passed = [r["monitor"] for r in table if r["gate_pass"]]
    negatives = [r["monitor"] for r in table if not r["gate_pass"]]
    assert passed == ["scl"]
    assert negatives == ["judge"]
