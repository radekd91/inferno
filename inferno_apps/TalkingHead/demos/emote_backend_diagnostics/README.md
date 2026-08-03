# EMOTE Transformer backend diagnostics

These utilities reproduce and measure a silent EMOTE inference regression seen
after moving to PyTorch 2.1.1. They are diagnostic tools, not the production
compatibility implementation.

## Update (2 August 2026)

The regression is resolved by the model-scoped
`LegacyTransformerEncoderLayer` implementation in
`inferno/models/temporal/TransformerCompatibility.py`. Archived EMOTE and
FLINT configs default to that compatibility path in memory. Configs that
explicitly set `force_legacy_transformer_encoder_path: false` use PyTorch's
unaltered `TransformerEncoderLayer`. The global monkey patch below remains
only as a reproduction and comparison tool.

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
