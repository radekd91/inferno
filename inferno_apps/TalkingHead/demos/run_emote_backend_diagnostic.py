"""Run the existing EMOTE demo with selectable PyTorch backend diagnostics.

This launcher intentionally leaves the existing demo and model code unchanged.
All arguments after ``--backend-test`` are forwarded to
``demo_eval_talking_head_on_audio.py``.
"""

import argparse
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _install_slow_transformer_encoder_path():
    """Bypass torch 2.1's fused TransformerEncoderLayer inference path."""

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
    ):
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
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, src_mask, src_key_padding_mask, is_causal=is_causal
                )
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    torch.nn.TransformerEncoderLayer.forward = forward


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--backend-test",
        required=True,
        choices=[
            "math_sdp",
            "slow_transformer",
            "slow_transformer_math_sdp",
            "cudnn_off",
        ],
    )
    args, demo_args = parser.parse_known_args()

    if "slow_transformer" in args.backend_test:
        _install_slow_transformer_encoder_path()

    if "math_sdp" in args.backend_test:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    if args.backend_test == "cudnn_off":
        torch.backends.cudnn.enabled = False

    print(f"EMOTE backend diagnostic: {args.backend_test}")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(
        "SDPA enabled (flash, memory-efficient, math): "
        f"{torch.backends.cuda.flash_sdp_enabled()}, "
        f"{torch.backends.cuda.mem_efficient_sdp_enabled()}, "
        f"{torch.backends.cuda.math_sdp_enabled()}"
    )
    print(f"cuDNN enabled/version: {torch.backends.cudnn.enabled}/{torch.backends.cudnn.version()}")

    demo_path = Path(__file__).with_name("demo_eval_talking_head_on_audio.py")
    sys.argv = [str(demo_path), *demo_args]
    runpy.run_path(str(demo_path), run_name="__main__")


if __name__ == "__main__":
    main()
