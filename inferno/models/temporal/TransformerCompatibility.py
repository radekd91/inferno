"""Compatibility helpers for legacy TransformerEncoderLayer checkpoints.

PyTorch 2.x can bypass TransformerEncoderLayer.forward's ordinary Python path
during evaluation and call a fused implementation. Some legacy EMOTE/FLINT
checkpoints produce substantially different motion through that fused path.
The compatibility layer below preserves the ordinary path without changing
parameters, state-dict keys, or unflagged models.
"""

import torch
import torch.nn.functional as F


FORCE_LEGACY_TRANSFORMER_ENCODER_PATH = (
    "force_legacy_transformer_encoder_path"
)


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
    """Select the compatibility class only for an explicitly flagged model.

    Returning PyTorch's class directly when disabled ensures new models retain
    the exact upstream class and forward implementation.
    """

    if force_legacy_path:
        return LegacyTransformerEncoderLayer
    return torch.nn.TransformerEncoderLayer
