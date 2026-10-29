"""Regression checks for automatic EMOTE/FLINT training-date compatibility."""

import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from inferno.models.temporal.TransformerCompatibility import (
    LegacyTransformerEncoderLayer,
    transformer_encoder_layer_class,
    use_legacy_transformer_encoder_path,
)


class TrainingDateTests(unittest.TestCase):
    def test_released_training_dates(self):
        for decoder, date in (
            ("BertPriorDecoder", "2023_05_18_01-26-32"),
            ("BertPriorDecoder", "2023_12_09_09-39-43"),
            ("L2lDecoder", "2023_04_19_00-20-32"),
            ("L2lDecoder", "2023_11_09_18-00-42"),
        ):
            with self.subTest(decoder=decoder, date=date):
                self.assertTrue(use_legacy_transformer_encoder_path(decoder, date))

    def test_cutoff_is_exclusive(self):
        for decoder in ("BertPriorDecoder", "L2lDecoder"):
            for date, expected in (
                ("2023_12_16_09-39-42", True),
                ("2023_12_16_09-39-43", False),
                ("2027_01_01_00-00-00", False),
            ):
                with self.subTest(decoder=decoder, date=date):
                    self.assertIs(use_legacy_transformer_encoder_path(decoder, date), expected)

    def test_invalid_dates_never_fall_back(self):
        for decoder in ("BertPriorDecoder", "L2lDecoder"):
            for date in (None, "", "todo", False, 2023, "2023_02_30_00-00-00",
                         "2023-05-18", "2023_5_18_01-26-32"):
                with self.subTest(decoder=decoder, date=date):
                    with self.assertRaisesRegex(ValueError, "cfg.inout.time"):
                        use_legacy_transformer_encoder_path(decoder, date)

    def test_unaffected_models_do_not_require_dates(self):
        for decoder in ("FlameBertDecoder", "VertexTransformer", "flame_prior", "CodeTalkerDecoder"):
            self.assertFalse(use_legacy_transformer_encoder_path(decoder, None))
            self.assertFalse(use_legacy_transformer_encoder_path(decoder, "2023_01_01_00-00-00"))

    def test_explicit_override_wins_over_date(self):
        for decoder in ("BertPriorDecoder", "L2lDecoder"):
            for date in (None, "unknown", "2023_05_18_01-26-32", "2027_01_01_00-00-00"):
                for override in (True, False):
                    self.assertIs(
                        use_legacy_transformer_encoder_path(decoder, date, override=override),
                        override,
                    )
            for override in ("false", "true", 0, 1):
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    use_legacy_transformer_encoder_path(decoder, None, override=override)
        self.assertFalse(use_legacy_transformer_encoder_path("flame_prior", None, override=True))


class TransformerIsolationTests(unittest.TestCase):
    def test_legacy_matches_ordinary_path_and_preserves_checkpoints(self):
        torch.manual_seed(9)
        for norm_first in (False, True):
            for activation in ("relu", "gelu"):
                kwargs = dict(d_model=16, nhead=2, dim_feedforward=32,
                              dropout=0, batch_first=True, norm_first=norm_first,
                              activation=activation)
                standard = torch.nn.TransformerEncoderLayer(**kwargs).eval()
                legacy = LegacyTransformerEncoderLayer(**kwargs).eval()
                legacy.load_state_dict(standard.state_dict(), strict=True)
                self.assertEqual(list(standard.state_dict()), list(legacy.state_dict()))
                src = torch.randn(2, 7, 16)
                # Floating attention bias is important for the archived models.
                mask = torch.randn(4, 7, 7) * 0.1
                # Grad-enabled evaluation bypasses PyTorch 2.1's fused layer.
                expected = standard(src, src_mask=mask).detach()
                with torch.no_grad():
                    actual = legacy(src, src_mask=mask)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_legacy_cannot_call_fused_layer_or_patch_other_models(self):
        original_forward = torch.nn.TransformerEncoderLayer.forward
        standard_class = transformer_encoder_layer_class(False)
        self.assertIs(standard_class, torch.nn.TransformerEncoderLayer)
        standard = standard_class(16, 2, dropout=0, batch_first=True).eval()
        src = torch.randn(2, 7, 16)
        with torch.no_grad():
            before = standard(src)
        legacy = transformer_encoder_layer_class(True)(16, 2, dropout=0, batch_first=True).eval()
        with patch.object(torch, "_transformer_encoder_layer_fwd", side_effect=AssertionError("fused")):
            with torch.no_grad():
                legacy(src)
        self.assertIs(torch.nn.TransformerEncoderLayer.forward, original_forward)
        with torch.no_grad():
            after = standard(src)
        torch.testing.assert_close(after, before, rtol=0, atol=0)


class DecoderConstructionTests(unittest.TestCase):
    """Exercise the actual constructors/factory, not only the date predicate."""

    @staticmethod
    def flint_config():
        return OmegaConf.create(dict(
            type="L2lDecoder", feature_dim=16, intermediate_size=32,
            nhead=2, num_layers=1, activation="gelu", dropout=0,
            positional_encoding={"type": "none"},
        ))

    def test_flint_constructor_uses_date_without_mutating_config(self):
        from inferno.models.temporal.motion_prior.L2lMotionPrior import L2lDecoder
        sizes = OmegaConf.create(dict(quant_factor=1, sequence_length=8))
        for date, expected in (
            ("2023_11_09_18-00-42", LegacyTransformerEncoderLayer),
            ("2023_12_16_09-39-43", torch.nn.TransformerEncoderLayer),
        ):
            cfg = self.flint_config()
            before = OmegaConf.to_container(cfg)
            decoder = L2lDecoder(cfg, sizes, 8, training_time=date)
            self.assertIs(type(decoder.decoder_transformer.layers[0]), expected)
            self.assertEqual(OmegaConf.to_container(cfg), before)
        with self.assertRaisesRegex(ValueError, "cfg.inout.time"):
            L2lDecoder(self.flint_config(), sizes, 8)

        for override, expected in ((True, LegacyTransformerEncoderLayer),
                                   (False, torch.nn.TransformerEncoderLayer)):
            cfg = self.flint_config()
            cfg.force_legacy_transformer_encoder_path = override
            decoder = L2lDecoder(cfg, sizes, 8)
            self.assertIs(type(decoder.decoder_transformer.layers[0]), expected)

    def test_emote_factory_uses_own_date_and_does_not_override_prior(self):
        from inferno.models.temporal.BlockFactory import sequence_decoder_from_cfg
        from inferno.models.temporal.motion_prior.L2lMotionPrior import L2lDecoder

        class Prior(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = L2lDecoder(
                    DecoderConstructionTests.flint_config(),
                    OmegaConf.create(dict(quant_factor=1, sequence_length=8)),
                    8, training_time="2023_11_09_18-00-42",
                )

            def latent_frame_size(self): return 2
            def quant_factor(self): return 1
            def bottleneck_dim(self): return 16
            def discard_encoder(self): pass
            def get_flame(self): return torch.nn.Identity()

        for date, expected in (
            ("2023_12_09_09-39-43", LegacyTransformerEncoderLayer),
            ("2023_12_16_09-39-43", torch.nn.TransformerEncoderLayer),
        ):
            cfg = OmegaConf.create(dict(
                inout={"time": date}, data={"train_subjects": ["M003"]},
                model=dict(
                    output={"predict_expcode": True, "predict_jawpose": True},
                    sequence_decoder=dict(
                        type="BertPriorDecoder", feature_dim=16, num_layers=1,
                        nhead=2, activation="gelu", dropout=0,
                        style_embedding="none", positional_encoding={"type": "none"},
                        motion_prior={"path": "test-flint"}, squash_before=False,
                    ),
                ),
            ))
            with patch("inferno.models.talkinghead.FaceFormerDecoder.load_motion_prior_net",
                       side_effect=lambda path, trainable: Prior()) as load_prior:
                decoder = sequence_decoder_from_cfg(cfg)
            load_prior.assert_called_once_with("test-flint", False)
            self.assertIs(type(decoder.bert_decoder.layers[0]), expected)
            self.assertIs(type(decoder.motion_prior.decoder.decoder_transformer.layers[0]),
                          LegacyTransformerEncoderLayer)
            self.assertNotIn("force_legacy_transformer_encoder_path", cfg.model.sequence_decoder)


if __name__ == "__main__":
    unittest.main()
