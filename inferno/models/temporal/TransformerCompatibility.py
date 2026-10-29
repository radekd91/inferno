"""Compatibility helpers for legacy TransformerEncoderLayer checkpoints.

PyTorch 2.x can bypass TransformerEncoderLayer.forward's ordinary Python path
during evaluation and call a fused implementation. Some legacy EMOTE/FLINT
checkpoints produce substantially different motion through that fused path.
The compatibility layer below preserves the ordinary path without changing
parameters or state-dict keys. Only EMOTE/FLINT decoders trained before the
documented cutoff use it; other models keep PyTorch's implementation.
"""

from datetime import datetime

import torch
import torch.nn.functional as F


# inout.time is an original training-run timestamp, with no timezone suffix.
# One week after the latest legacy release's training timestamp:
# EMOTE_v2, 2023_12_09_09-39-43. Runs at/after this use standard PyTorch.
LEGACY_TRANSFORMER_CUTOFF = datetime(2023, 12, 16, 9, 39, 43)
TRAINING_TIME_FORMAT = "%Y_%m_%d_%H-%M-%S"
AFFECTED_DECODERS = frozenset({"BertPriorDecoder", "L2lDecoder"})
FORCE_LEGACY_TRANSFORMER_ENCODER_PATH = "force_legacy_transformer_encoder_path"


def use_legacy_transformer_encoder_path(decoder_type, training_time, *, override=None):
    """Resolve compatibility from decoder family and original training date.

    Do not guess when an affected model has no usable provenance. In
    particular, today's date, file timestamps and a parent EMOTE model's date
    cannot substitute for a FLINT checkpoint's own inout.time.
    """
    if decoder_type not in AFFECTED_DECODERS:
        return False
    if override is not None:
        if not isinstance(override, bool):
            raise ValueError(
                f"{FORCE_LEGACY_TRANSFORMER_ENCODER_PATH} must be a boolean "
                f"or null (automatic); got {override!r}."
            )
        print(
            f"{decoder_type}: {'legacy compatibility' if override else 'standard PyTorch'} "
            "transformer path (explicit override)."
        )
        return override
    try:
        trained_at = datetime.strptime(training_time, TRAINING_TIME_FORMAT)
        if trained_at.strftime(TRAINING_TIME_FORMAT) != training_time:
            raise ValueError("Non-canonical training timestamp")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{decoder_type} requires its original training date in cfg.inout.time "
            f"(YYYY_MM_DD_HH-MM-SS); got {training_time!r}. Restore the date "
            "from the original run config. New training must initialize inout.time "
            "before constructing the model. Alternatively, explicitly set "
            f"{FORCE_LEGACY_TRANSFORMER_ENCODER_PATH} to true or false in the decoder config."
        ) from error
    legacy = trained_at < LEGACY_TRANSFORMER_CUTOFF
    print(
        f"{decoder_type}: {'legacy compatibility' if legacy else 'standard PyTorch'} "
        f"transformer path (trained {training_time}, "
        f"cutoff {LEGACY_TRANSFORMER_CUTOFF.strftime(TRAINING_TIME_FORMAT)})."
    )
    return legacy


class LegacyTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    """Run the ordinary Transformer encoder path, bypassing fused inference.

    On the original PyTorch 1.12 environment the base implementation already
    uses the required path, so it is called directly. On PyTorch 2.x the body
    mirrors the ordinary branch of TransformerEncoderLayer.forward.
    """

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
    ):
        # PyTorch 1.12, used by the original EMOTE environment, predates the
        # mask canonicalization helpers and does not have the problematic fused
        # evaluation path.
        if not hasattr(F, "_canonical_mask"):
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            x = self.norm2(x + self._ff_block(x))
        return x


def transformer_encoder_layer_class(force_legacy_path):
    """Select the class using the model's resolved date-based compatibility.

    Returning PyTorch's class directly when disabled ensures new models retain
    the exact upstream class and forward implementation.
    """

    if force_legacy_path:
        return LegacyTransformerEncoderLayer
    return torch.nn.TransformerEncoderLayer
