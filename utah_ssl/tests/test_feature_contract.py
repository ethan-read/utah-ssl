from __future__ import annotations

import unittest

from utah_ssl.feature_contract import (
    SUPPORTED_FEATURE_MODES,
    resolve_feature_contract,
)
from utah_ssl.experiment_contract import SignalSpec


class FeatureContractTests(unittest.TestCase):
    def test_signal_mode_has_no_global_default(self) -> None:
        with self.assertRaises(TypeError):
            SignalSpec()
        self.assertEqual(SignalSpec.sbp_only(sbp_dim=128).mode, "sbp_only")
        with self.assertRaisesRegex(ValueError, "tx_dim must be zero"):
            SignalSpec(mode="sbp_only", tx_dim=128, sbp_dim=128)

    def test_supported_modes_have_expected_layouts(self) -> None:
        self.assertEqual(
            SUPPORTED_FEATURE_MODES,
            ("tx_only", "sbp_only", "tx_sbp"),
        )
        expected = {
            "tx_only": (("tx",), 128, 0),
            "sbp_only": (("sbp",), 128, 0),
            "tx_sbp": (("tx", "sbp"), 256, 128),
        }
        for mode, (modalities, full_dim, sbp_start) in expected.items():
            contract = resolve_feature_contract(mode)
            self.assertEqual(contract.modalities, modalities)
            self.assertEqual(
                contract.full_dim(tx_dim=128, sbp_dim=128),
                full_dim,
            )
            if contract.uses_sbp:
                self.assertEqual(
                    contract.feature_start("sbp", tx_dim=128),
                    sbp_start,
                )

    def test_sbp_only_row_compatibility(self) -> None:
        contract = resolve_feature_contract("sbp_only")
        self.assertTrue(
            contract.row_is_compatible(
                has_tx=False,
                has_sbp=True,
                n_tx_features=0,
                n_sbp_features=128,
                tx_dim=128,
                sbp_dim=128,
            )
        )
        self.assertFalse(
            contract.row_is_compatible(
                has_tx=True,
                has_sbp=True,
                n_tx_features=128,
                n_sbp_features=127,
                tx_dim=128,
                sbp_dim=128,
            )
        )

    def test_signal_spec_enforces_or_explicitly_pads_short_arrays(self) -> None:
        strict = SignalSpec.sbp_only(sbp_dim=128)
        with self.assertRaisesRegex(ValueError, "contains only 64 columns"):
            strict.selected_columns_for_width("sbp", 64)

        padded = SignalSpec.sbp_only(
            sbp_dim=128,
            missing_channel_policy="zero_pad",
        )
        self.assertEqual(padded.selected_columns_for_width("sbp", 64), (0, 64))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_mode must be one of"):
            resolve_feature_contract("unknown")


if __name__ == "__main__":
    unittest.main()
