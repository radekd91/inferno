# EMOTE Transformer backend diagnostics

These utilities reproduce and measure a silent EMOTE inference regression seen
after moving to PyTorch 2.1.1. They are diagnostic tools, not the production
compatibility implementation.

## Update (2 August 2026)

The regression is resolved by the model-scoped
`LegacyTransformerEncoderLayer` implementation in
`inferno/models/temporal/TransformerCompatibility.py`. Archived EMOTE and
FLINT models select it by their original training dates. The global monkey
patch below remains only as a reproduction and comparison tool.

## Training-date compatibility policy (29 October 2026)

- `BertPriorDecoder` (EMOTE) and `L2lDecoder` (FLINT, including its subclasses)
  use the legacy path only when their own `cfg.inout.time` is strictly before
  `2023_12_16_09-39-43`. This cutoff is exactly one week after the latest released
  legacy training run (EMOTE_v2: `2023_12_09_09-39-43`). It is a compatibility
  policy boundary, not an inferred PyTorch installation date.
- The timestamp format is `YYYY_MM_DD_HH-MM-SS`, as saved by the training
  scripts. It is compared as recorded, without assuming a timezone.
- EMOTE and its FLINT prior resolve compatibility independently. A new EMOTE
  using an old FLINT uses standard EMOTE layers and legacy FLINT layers.
- Other decoder types retain their existing implementation and do not need a
  training timestamp or compatibility flag. THUNDER configs are unchanged.
- Missing, malformed or impossible dates on affected models without an explicit
  override raise `ValueError`.
  Restore `inout.time` from the original run config; do not substitute a file
  modification date, download date or today's date for an old checkpoint.
- New training initializes `inout.time` before model construction. Resuming or
  forking a checkpoint preserves its original training timestamp even if the
  output directory gets a new date.
- The earlier unconditional fallback has been removed. An optional
  `model.sequence_decoder.force_legacy_transformer_encoder_path: true` forces
  legacy behavior; `false` forces standard PyTorch. Omitted or `null` means
  automatic date selection. Other values raise `ValueError`. This flag is never
  inserted automatically and an EMOTE override is not inherited by its FLINT
  prior: set an override in each model's own config if needed.

The selected path and the date used to select it are printed at construction.
The implementation preserves parameter names and state-dict structure and does
not change global PyTorch backend settings.

Run the regression tests in an environment with PyTorch and OmegaConf:

```bash
python -m unittest discover -s tests -p 'test_transformer_compatibility.py' -v
```

`../run_emote_backend_diagnostic.py` forwards its remaining arguments to
`../demo_eval_talking_head_on_audio.py` and selects one backend experiment with
`--backend-test`:

- `slow_transformer` replaces `torch.nn.TransformerEncoderLayer.forward` for
  the current Python process with PyTorch's ordinary (non-fused) forward path.
- `math_sdp` disables the flash and memory-efficient SDPA backends.
- `slow_transformer_math_sdp` combines those two experiments.
- `cudnn_off` disables cuDNN for the process.

The `slow_transformer` option is intentionally a process-global monkey patch.
It proved that the fused `TransformerEncoderLayer` evaluation path was the
source of the broken EMOTE output. Do not import the launcher as a library or
use its monkey patch as a model-scoped production fix.

Example:

```bash
python inferno_apps/TalkingHead/demos/run_emote_backend_diagnostic.py \
    --backend-test slow_transformer \
    --model_name EMOTE_v2 \
    --path_to_audio <audio.wav> \
    --output_folder <output-directory>
```

`compare_emote_backend_outputs.py` compares the FLAME pickle files produced by
multiple experiment directories. It reports finiteness and numerical distance
from a selected baseline:

```bash
python inferno_apps/TalkingHead/demos/emote_backend_diagnostics/compare_emote_backend_outputs.py \
    <diagnostic-root> --baseline vanilla_numeric
```

Generated checkpoints, meshes, videos, pickle files, and result directories
are not source artifacts and should not be committed.
