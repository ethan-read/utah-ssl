from __future__ import annotations

import csv
import unittest
from pathlib import Path

from utah_ssl.canonical_data import DEFAULT_PHONEME_VOCABULARY


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1]
    / "design"
    / "articulatory_feature_taxonomy.csv"
)
NOT_APPLICABLE = "not_applicable"
VOWEL_POSTURE_COLUMNS = (
    "vowel_nucleus_height",
    "vowel_nucleus_backness",
    "vowel_nucleus_rounding",
)
OFFGLIDE_COLUMNS = (
    "vowel_offglide_height",
    "vowel_offglide_backness",
    "vowel_offglide_rounding",
)
ARTICULATORY_COLUMNS = (
    "primary_articulators",
    "constriction_gesture",
    "consonant_place",
    "vowel_dynamic",
    *VOWEL_POSTURE_COLUMNS,
    *OFFGLIDE_COLUMNS,
    "rhotic",
    "voicing",
    "nasal",
)


class ArticulatoryFeatureTaxonomyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TAXONOMY_PATH.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_symbol = {row["symbol"]: row for row in cls.rows}

    def test_rows_match_canonical_vocabulary_exactly(self) -> None:
        expected_symbols = DEFAULT_PHONEME_VOCABULARY["index_to_symbol"]
        self.assertEqual(len(self.rows), 41)
        self.assertEqual(len(self.by_symbol), 41)
        self.assertEqual([int(row["phoneme_id"]) for row in self.rows], list(range(41)))
        self.assertEqual([row["symbol"] for row in self.rows], expected_symbols)

    def test_declared_values_are_closed_enums(self) -> None:
        allowed = {
            "segment_family": {"vowel", "consonant", "silence_boundary", "ctc_blank"},
            "constriction_gesture": {
                "closure",
                "frication",
                "closure_to_frication",
                "approximant",
                "lateral_approximant",
                "vowel",
                NOT_APPLICABLE,
            },
            "consonant_place": {
                "bilabial",
                "labiodental",
                "dental",
                "alveolar",
                "postalveolar",
                "palatal",
                "velar",
                "labial_velar",
                "glottal",
                NOT_APPLICABLE,
            },
            "vowel_dynamic": {"steady", "diphthong", NOT_APPLICABLE},
            "vowel_nucleus_height": {"high", "mid", "low", NOT_APPLICABLE},
            "vowel_nucleus_backness": {"front", "central", "back", NOT_APPLICABLE},
            "vowel_nucleus_rounding": {"rounded", "unrounded", NOT_APPLICABLE},
            "vowel_offglide_height": {"high", "mid", "low", NOT_APPLICABLE},
            "vowel_offglide_backness": {"front", "central", "back", NOT_APPLICABLE},
            "vowel_offglide_rounding": {"rounded", "unrounded", NOT_APPLICABLE},
            "rhotic": {"yes", "no", NOT_APPLICABLE},
            "voicing": {"voiced", "voiceless", NOT_APPLICABLE},
            "nasal": {"yes", "no", NOT_APPLICABLE},
        }
        allowed_articulators = {"lips", "tongue_front", "tongue_body", "larynx"}
        for row in self.rows:
            for column, values in allowed.items():
                self.assertIn(row[column], values, f"{row['symbol']}:{column}")
            if row["primary_articulators"] != NOT_APPLICABLE:
                parts = row["primary_articulators"].split("|")
                self.assertEqual(len(parts), len(set(parts)), row["symbol"])
                self.assertTrue(set(parts).issubset(allowed_articulators), row["symbol"])

    def test_anchor_consonants_match_declared_gestures(self) -> None:
        expected = {
            "P": ("lips", "closure", "bilabial", "voiceless", "no"),
            "B": ("lips", "closure", "bilabial", "voiced", "no"),
            "M": ("lips", "closure", "bilabial", "voiced", "yes"),
            "T": ("tongue_front", "closure", "alveolar", "voiceless", "no"),
            "D": ("tongue_front", "closure", "alveolar", "voiced", "no"),
            "N": ("tongue_front", "closure", "alveolar", "voiced", "yes"),
        }
        for symbol, values in expected.items():
            row = self.by_symbol[symbol]
            observed = tuple(
                row[column]
                for column in (
                    "primary_articulators",
                    "constriction_gesture",
                    "consonant_place",
                    "voicing",
                    "nasal",
                )
            )
            self.assertEqual(observed, values, symbol)

    def test_consonant_category_memberships_match_version_one_rules(self) -> None:
        def symbols_with(column: str, value: str) -> set[str]:
            return {
                row["symbol"]
                for row in self.rows
                if row["segment_family"] == "consonant" and row[column] == value
            }

        expected_places = {
            "bilabial": {"P", "B", "M"},
            "labiodental": {"F", "V"},
            "dental": {"TH", "DH"},
            "alveolar": {"T", "D", "N", "S", "Z", "L"},
            "postalveolar": {"SH", "ZH", "CH", "JH", "R"},
            "palatal": {"Y"},
            "velar": {"K", "G", "NG"},
            "labial_velar": {"W"},
            "glottal": {"HH"},
        }
        for place, expected in expected_places.items():
            self.assertEqual(symbols_with("consonant_place", place), expected, place)

        self.assertEqual(
            symbols_with("constriction_gesture", "closure"),
            {"P", "B", "M", "T", "D", "N", "K", "G", "NG"},
        )
        self.assertEqual(
            symbols_with("constriction_gesture", "frication"),
            {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
        )
        self.assertEqual(
            symbols_with("constriction_gesture", "closure_to_frication"),
            {"CH", "JH"},
        )
        self.assertEqual(
            symbols_with("constriction_gesture", "approximant"),
            {"R", "W", "Y"},
        )
        self.assertEqual(
            symbols_with("constriction_gesture", "lateral_approximant"),
            {"L"},
        )
        self.assertEqual(
            symbols_with("voicing", "voiceless"),
            {"CH", "F", "HH", "K", "P", "S", "SH", "T", "TH"},
        )
        self.assertEqual(symbols_with("nasal", "yes"), {"M", "N", "NG"})

    def test_vowels_have_complete_nuclei_and_five_diphthongs(self) -> None:
        vowels = [row for row in self.rows if row["segment_family"] == "vowel"]
        self.assertEqual(len(vowels), 15)
        for row in vowels:
            self.assertTrue(
                all(row[column] != NOT_APPLICABLE for column in VOWEL_POSTURE_COLUMNS),
                row["symbol"],
            )
            self.assertEqual(row["consonant_place"], NOT_APPLICABLE)
            self.assertEqual(row["constriction_gesture"], "vowel")

        diphthongs = [row for row in vowels if row["vowel_dynamic"] == "diphthong"]
        self.assertEqual({row["symbol"] for row in diphthongs}, {"AW", "AY", "EY", "OW", "OY"})
        for row in diphthongs:
            self.assertTrue(
                all(row[column] != NOT_APPLICABLE for column in OFFGLIDE_COLUMNS),
                row["symbol"],
            )
        for row in vowels:
            if row["vowel_dynamic"] == "steady":
                self.assertTrue(
                    all(row[column] == NOT_APPLICABLE for column in OFFGLIDE_COLUMNS),
                    row["symbol"],
                )

    def test_vowel_postures_match_version_one_rules(self) -> None:
        posture_columns = (
            "vowel_nucleus_height",
            "vowel_nucleus_backness",
            "vowel_nucleus_rounding",
            "vowel_offglide_height",
            "vowel_offglide_backness",
            "vowel_offglide_rounding",
        )
        na = NOT_APPLICABLE
        expected = {
            "AA": ("low", "back", "unrounded", na, na, na),
            "AE": ("low", "front", "unrounded", na, na, na),
            "AH": ("mid", "central", "unrounded", na, na, na),
            "AO": ("mid", "back", "rounded", na, na, na),
            "AW": ("low", "central", "unrounded", "high", "back", "rounded"),
            "AY": ("low", "central", "unrounded", "high", "front", "unrounded"),
            "EH": ("mid", "front", "unrounded", na, na, na),
            "ER": ("mid", "central", "unrounded", na, na, na),
            "EY": ("mid", "front", "unrounded", "high", "front", "unrounded"),
            "IH": ("high", "front", "unrounded", na, na, na),
            "IY": ("high", "front", "unrounded", na, na, na),
            "OW": ("mid", "back", "rounded", "high", "back", "rounded"),
            "OY": ("mid", "back", "rounded", "high", "front", "unrounded"),
            "UH": ("high", "back", "rounded", na, na, na),
            "UW": ("high", "back", "rounded", na, na, na),
        }
        for symbol, values in expected.items():
            observed = tuple(self.by_symbol[symbol][column] for column in posture_columns)
            self.assertEqual(observed, values, symbol)
        self.assertEqual(self.by_symbol["ER"]["rhotic"], "yes")

    def test_consonants_do_not_carry_vowel_posture(self) -> None:
        consonants = [row for row in self.rows if row["segment_family"] == "consonant"]
        self.assertEqual(len(consonants), 24)
        for row in consonants:
            self.assertNotEqual(row["consonant_place"], NOT_APPLICABLE, row["symbol"])
            self.assertEqual(row["vowel_dynamic"], NOT_APPLICABLE, row["symbol"])
            for column in (*VOWEL_POSTURE_COLUMNS, *OFFGLIDE_COLUMNS, "rhotic"):
                self.assertEqual(row[column], NOT_APPLICABLE, f"{row['symbol']}:{column}")

    def test_multi_articulator_targets_are_preserved(self) -> None:
        self.assertEqual(self.by_symbol["W"]["primary_articulators"], "lips|tongue_body")
        self.assertEqual(
            self.by_symbol["R"]["primary_articulators"],
            "tongue_front|tongue_body",
        )

    def test_blank_and_silence_have_no_articulatory_targets(self) -> None:
        self.assertEqual(self.by_symbol["BLANK"]["segment_family"], "ctc_blank")
        self.assertEqual(self.by_symbol["SIL"]["segment_family"], "silence_boundary")
        for symbol in ("BLANK", "SIL"):
            for column in ARTICULATORY_COLUMNS:
                self.assertEqual(
                    self.by_symbol[symbol][column],
                    NOT_APPLICABLE,
                    f"{symbol}:{column}",
                )


if __name__ == "__main__":
    unittest.main()
