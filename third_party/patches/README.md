# Local patches to `third_party/obfuscation-atlas`

Obfuscation Atlas is not vendored in this repo. `scripts/install.sh` clones
upstream at `12abf65` into `third_party/obfuscation-atlas` and applies this
patch.

| File | Base | Patched |
|---|---|---|
| `obfuscation-atlas-spoof-cost.patch` | `12abf65` (upstream `main`) | applied by `scripts/install.sh` |

## What the patch changes

1. **`obfuscation_atlas/training/grpo_trainer.py` — reward-info alignment.**
   Prompt-length filtering compacts `prompts` / `trigger_present_map`, but `prompt_indices` kept
   the original row IDs. After compaction those IDs are stale, so `reward_infos[i]` could pair a
   prompt with another row's test cases. Reindexed to `0..N-1`.

2. **`obfuscation_atlas/training/obfuscation_training.py` — exact prompt count.**
   Selects 32 extra rows per label before the length filter, then caps back to
   `max_train + max_val`. Without headroom the CoT template drops a few long MBPP rows
   and the corpus lands at 399 instead of 400.

3. **`third_party/afterburner/.../generation/vllm/server.py` — forward the generation seed.**
   `SamplingParams` never received `seed`, so `grpo.vllm.generation_seed` was inert.
   Adds `seed` to both request models and passes it through.

## Applying by hand

```bash
git clone https://github.com/AlignmentResearch/obfuscation-atlas third_party/obfuscation-atlas
cd third_party/obfuscation-atlas
git checkout 12abf65
git apply ../patches/obfuscation-atlas-spoof-cost.patch
```
