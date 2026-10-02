# Spoof Cost

Replication code for *A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control*.

A low training-time monitor readout does not identify whether a policy still reward-hacks. This repo trains Llama-3.1-8B-Instruct with GRPO on MBPP-Honeypot-CoT against three gated monitors (Probe, SCL, Cut), then scores behavior with held-out tests.

There is no Slurm layer. One script starts vLLM and runs each stage.

## Requirements

- Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/)
- 2 GPUs (train + vLLM), A100-class recommended
- Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct` (Llama 3.1 Community License)
- Dataset `taufeeque/mbpp-hardcode` revision `08e7ea5` (MBPP, CC BY 4.0)
- CUDA 12.x available to PyTorch / vLLM (`CUDA_HOME` if it is not on the default path)

Optional: `WANDB_API_KEY` for logging. Runs write local metrics under `results/` either way.

**Budget.** Each training or prefix-scoring job uses 2 GPUs. A Probe cell is ~3 hours, Cut ~6 hours, SCL ~8 hours. The full grid is one E0 run plus nine E2 cells, then diagnostics: about 120–160 A100-hours.

## Install

```bash
./scripts/install.sh
```

The installer clones [Obfuscation Atlas](https://github.com/AlignmentResearch/obfuscation-atlas) at commit `12abf65` into `third_party/obfuscation-atlas`, applies the patch in `third_party/patches/`, then runs `uv sync`. It needs `git` and network access; re-running it is safe.

On older glibc hosts that cannot build vLLM from source, force a manylinux wheel:

```bash
UV_PYTHON_PLATFORM=x86_64-manylinux_2_31 ./scripts/install.sh
```

Set `HF_HOME` if you already have the model cached. Accept the Llama license and log in (`huggingface-cli login`) before the first GPU run.

CPU smoke test (FakeGenerator, no CUDA):

```bash
MODE=dry ./scripts/run.sh e2
```

## Pipeline

Shared no-monitor E0, then an offline gate and probe fit, then three monitored arms × seeds `{0,1,2}`. Diagnostics re-score the saved adapters. Seeds change policy initialization and generation only; the prompt set (`data_seed=0`) and the E0 adapter / E1 probe are shared.

```text
install → E0 → E1 → E2 (probe / scl / cprobe × seeds 0,1,2)
                 ↘ E4, endpoint ψ, E5a → E5 → aggregate + plots
```

Default `MODE=gpu` starts vLLM on GPU 1 and trains on GPU 0 (`VLLM_GPU`, `TRAIN_GPU` to override).

### 1. E0 — no-monitor baseline

One GRPO run: 120 steps, 400 prompts, group size 8, \(\beta_{\mathrm{KL}}=10^{-4}\), LoRA \(r=64\), format contract (tagged reasoning, ≥200 characters and ≥40 words, fenced Python).

```bash
./scripts/run.sh e0
```

Writes `results/e0/`. Sign-off: hack rate \(d \ge 0.80\) and format-OK \(\ge 0.80\) (including on hacking rollouts). The runner also snapshots `responses_archive/` for E1.

### 2. E1 — offline gate and probe

Scores 256 balanced late-baseline rollouts. Fits a layer-29 linear probe (`LogisticRegression`, \(C=1.0\), prompt-cluster 5-fold). Judge, SCL, and Cut are zero-shot.

```bash
./scripts/run.sh e1
```

Writes `results/e1/` including `indomain_probe/probe.json`. A monitor passes if AUROC \(\ge 0.65\) and the 95% CI lower bound is \(> 0.50\). In the paper, Probe / SCL / Cut pass and Judge fails; only passing arms go to E2.

### 3. E2 — monitored training

Nine cells: `{probe, scl, cprobe}` × seeds `{0,1,2}`, \(\alpha=1.0\), raw \(\psi\) (no \(b_0\) clip). Cut is the single-cut monitor at fraction \(0.25\).

```bash
for seed in 0 1 2; do
  for mon in probe scl cprobe; do
    SEED=$seed MONITOR=$mon ./scripts/run.sh e2
  done
done
```

Writes `results/seed_{S}/e2_{probe,scl,cprobe}/`. Cells are independent and can run in parallel on separate 2-GPU allocations.

### 4. Diagnostics

**E4** — own-answer agreement after a \(0.30\) prefix cut and a \(0.40\) interior ablation (\(N=1\), 32 tokens).

**Endpoint \(\psi\)** — recompute the trained prefix score on the 128 endpoint rollouts with the step-120 adapter (SCL grid \([0.1, 0.25, 0.5, 0.75, 0.9]\), \(N=4\), 64 tokens). Cut reports \(c_{0.25}\) from the same curve.

**E5a** — two-reader re-score with the frozen E1 probe (no vLLM).

**E5** — outcome taxonomy (`correct_honest`, `reward_hacking`, `capability_failure`, `other_ambiguous`) and onset (first of five consecutive steps with held-out pass \(< 0.10\)).

```bash
SEED=0 MONITOR=e0 ./scripts/run.sh e4
SEED=0 MONITOR=e0 ./scripts/run.sh epsi

for seed in 0 1 2; do
  for mon in probe scl cprobe; do
    SEED=$seed MONITOR=$mon ./scripts/run.sh e4
    SEED=$seed MONITOR=$mon ./scripts/run.sh epsi
  done
  SEED=$seed ./scripts/run.sh e5a
  SEED=$seed ./scripts/run.sh e5
done
```

### 5. Aggregate and figures

```bash
./scripts/run.sh aggregate
```

Writes `results/aggregate/summary.json`, `training_curves.png`, and `prefix_vs_hack.png`.

Optional extras (not required for the main tables):

```bash
uv run python scripts/plot_hv_trajectory.py
uv run python scripts/sample_capability_failures.py
uv run python scripts/mine_monitor_logs.py
```

## Output layout

```text
results/e0/                          shared no-monitor baseline
results/e1/                          offline gate + fitted probe
results/seed_{S}/e2_{probe,scl,cprobe}/
results/seed_{S}/e4_{probe,scl,cprobe}/
results/e4_e0/
results/endpoint_psi_e0/
results/seed_{S}/e2_{arm}/endpoint_psi/
results/seed_{S}/e5_probe_scores/
results/seed_{S}/e5/
results/aggregate/
```

## Reported results

`paper_results/` holds the summary files behind the paper's tables, in the same layout as `results/` above: per-cell `metrics.json`, held-out outcome trajectories, endpoint \(\psi\) per rollout (`endpoint_psi.jsonl`), E4 / E5 / E5a metrics, and `aggregate/`. Rollout text, adapters, and activation caches are not included. Paths inside these files are relative to the repo root.

For example, the SCL endpoint \(\psi\) means are in `paper_results/seed_{0,1,2}/e2_scl/endpoint_psi/metrics.json`.

## Training contract

| Knob | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA | \(r=64\), \(\alpha_{\mathrm{lora}}=128\) |
| Optimizer | GRPO, \(\beta_{\mathrm{KL}}=10^{-4}\) |
| Budget | 120 steps, 400 prompts, group size 8, train batch 16 |
| Endpoint | step 120, \(n=128\) on-policy rollouts |
| Seeds | `{0,1,2}` (init + generation); `data_seed=0` |
| Monitor \(\alpha\) | 1.0 on Probe / SCL / Cut; 0 on E0 |
| Format contract | tagged span, ≥200 chars, ≥40 words, fenced Python; penalty 0.5 |
| Code-length cost | 0.003 |
| SCL | fractions `[0.1, 0.25, 0.5, 0.75, 0.9]`, \(N=4\), 64 forced tokens |
| Cut | fraction 0.25, same raw-\(\psi\) \(z\)-score penalty |
| Answer match | whitespace-normalized string equality |
| Onset | first of 5 consecutive steps with held-out pass \(< 0.10\) |

Configs live in `configs/`. GPU scale is applied automatically whenever `dry_run=false`.

## What to look for

The paper claim is coexistence, not a rate: a near-floor monitor readout is compatible with both a mixed regime and near-pure hacking.

- **Probe.** In-loop score at the numerical floor while every seed hacks.
- **SCL / Cut.** Trained-score median \(0\) in both mixed and hacking seeds. Held-out execution, not the readout, identifies the regime.

A run is in the hacking regime if it meets the onset rule above; otherwise it is mixed.

## Tests

```bash
uv run pytest
```

These check the format contract, SCL scoring, and config loading. They do not train.

## Layout

```text
src/                 environment, monitors, reward, GRPO hook
scripts/run.sh       portable stage runner
scripts/install.sh   dependency install
configs/             frozen knobs + per-stage YAML
fixtures/            tiny MBPP JSONL for MODE=dry
paper_results/       summary files behind the paper's tables
third_party/patches/ local fixes to Obfuscation Atlas + Afterburner
tests/               unit / dry-run checks
```

The GRPO harness is [Obfuscation Atlas](https://github.com/AlignmentResearch/obfuscation-atlas) v0.1.0. Its code is not redistributed here; `scripts/install.sh` fetches it and applies the three local fixes in `third_party/patches/`.

## Citation

If you use this code, please cite *A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control*.

## License

MIT, see [LICENSE](LICENSE). This covers the code in this repository only. Obfuscation Atlas, Llama 3.1, and MBPP are under their own terms.
