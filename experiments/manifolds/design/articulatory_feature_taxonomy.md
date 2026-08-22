# Canonical articulatory feature taxonomy, version 1

## Purpose and authority

This record defines coarse articulatory targets for the 41-token phoneme
vocabulary used by the Utah-array speech decoders. The machine-readable source
of truth is
[`articulatory_feature_taxonomy.csv`](articulatory_feature_taxonomy.csv); the
tables below are its human-readable rendering.

These values describe an intended, broad American-English production target.
They are not measured tongue, lip, larynx, or velum kinematics. A transcript-
derived phoneme fixes the intended discrete target but does not remove
coarticulation, reduction, speaker-specific realization, or uncertainty in the
GRU-assisted timing used by later analyses.

`BLANK` is the CTC alignment state. `SIL` is the dataset's silence or
word-boundary target. Neither is an articulatory phone, and they must remain
distinct.

## Source and adaptation boundary

- **Repository observation:** IDs and symbols follow
  `utah_ssl.canonical_data.DEFAULT_PHONEME_VOCABULARY` exactly.
- **Reported:** the International Phonetic Association organizes pulmonic
  consonants by place and manner and vowels by tongue position and rounding;
  see the [official IPA chart archive](https://www.internationalphoneticassociation.org/content/ipa-chart-archive).
- **Reported:** Mortensen et al. describe representing IPA segments with
  articulatory feature vectors in
  [PanPhon](https://aclanthology.org/C16-1328/) (COLING 2016).
- **Adapted:** this repository uses a deliberately coarser, small-sample target
  system rather than copying the IPA chart or PanPhon feature inventory. The
  active-articulator labels, coarse vowel bins, diphthong endpoints, and
  `not_applicable` policy are local analysis decisions.

No PanPhon code or data table is copied into this record. PanPhon is cited as a
feature-representation precedent; the local rows were assembled for the
repository vocabulary.

## Field definitions

| Field | Meaning |
|---|---|
| `phoneme_id`, `symbol` | Exact decoder token identity and order. |
| `ipa` | Broad IPA rendering for readability, not a narrow realization. |
| `segment_family` | `vowel`, `consonant`, `silence_boundary`, or `ctc_blank`. |
| `primary_articulators` | Coarse articulator set: `lips`, `tongue_front`, `tongue_body`, or `larynx`; simultaneous values use `|`. |
| `constriction_gesture` | `closure`, `frication`, `closure_to_frication`, `approximant`, `lateral_approximant`, or `vowel`. |
| `consonant_place` | Coarse consonant constriction location. |
| `vowel_dynamic` | `steady` or `diphthong`. |
| `vowel_nucleus_*` | Coarse initial or steady tongue height/backness and lip rounding. |
| `vowel_offglide_*` | Coarse diphthong endpoint; `not_applicable` for steady vowels. |
| `rhotic` | Vowel rhoticity; consonantal `R` is represented by its articulator, place, and approximant fields. |
| `voicing` | Canonical laryngeal voicing target. |
| `nasal` | Canonical velum-lowering/nasality target. |

All unavailable or inapplicable categorical values are the literal string
`not_applicable`. Later probes are independent targets rather than a cascading
classifier. Each probe should select rows for which its target is applicable.

## Non-articulatory states

| ID | Symbol | Family | Interpretation |
|---:|---|---|---|
| 0 | `BLANK` | `ctc_blank` | CTC alignment state; not a phoneme or silence. |
| 40 | `SIL` | `silence_boundary` | Silence or word-boundary target; not CTC blank. |

## Consonants

| ID | Phone | IPA | Primary articulator(s) | Constriction | Place | Voice | Nasal |
|---:|---|---|---|---|---|---|---|
| 7 | `B` | b | lips | closure | bilabial | voiced | no |
| 8 | `CH` | tʃ | tongue front | closure → frication | postalveolar | voiceless | no |
| 9 | `D` | d | tongue front | closure | alveolar | voiced | no |
| 10 | `DH` | ð | tongue front | frication | dental | voiced | no |
| 14 | `F` | f | lips | frication | labiodental | voiceless | no |
| 15 | `G` | ɡ | tongue body | closure | velar | voiced | no |
| 16 | `HH` | h | larynx | frication | glottal | voiceless | no |
| 19 | `JH` | dʒ | tongue front | closure → frication | postalveolar | voiced | no |
| 20 | `K` | k | tongue body | closure | velar | voiceless | no |
| 21 | `L` | l | tongue front | lateral approximant | alveolar | voiced | no |
| 22 | `M` | m | lips | closure | bilabial | voiced | yes |
| 23 | `N` | n | tongue front | closure | alveolar | voiced | yes |
| 24 | `NG` | ŋ | tongue body | closure | velar | voiced | yes |
| 27 | `P` | p | lips | closure | bilabial | voiceless | no |
| 28 | `R` | ɹ | tongue front \| tongue body | approximant | postalveolar | voiced | no |
| 29 | `S` | s | tongue front | frication | alveolar | voiceless | no |
| 30 | `SH` | ʃ | tongue front | frication | postalveolar | voiceless | no |
| 31 | `T` | t | tongue front | closure | alveolar | voiceless | no |
| 32 | `TH` | θ | tongue front | frication | dental | voiceless | no |
| 35 | `V` | v | lips | frication | labiodental | voiced | no |
| 36 | `W` | w | lips \| tongue body | approximant | labial-velar | voiced | no |
| 37 | `Y` | j | tongue body | approximant | palatal | voiced | no |
| 38 | `Z` | z | tongue front | frication | alveolar | voiced | no |
| 39 | `ZH` | ʒ | tongue front | frication | postalveolar | voiced | no |

`R` uses both tongue-front and tongue-body labels so the target does not
pretend that a transcript distinguishes retroflex from bunched production.
`HH` receives the canonical glottal label even though its oral posture is
strongly conditioned by neighboring vowels.

## Vowels

Postures are written as `height backness rounding`. Diphthongs show
`nucleus → off-glide`.

| ID | Phone | IPA | Primary articulator(s) | Dynamic | Canonical posture | Rhotic | Voice | Nasal |
|---:|---|---|---|---|---|---|---|---|
| 1 | `AA` | ɑ | tongue body | steady | low back unrounded | no | voiced | no |
| 2 | `AE` | æ | tongue body | steady | low front unrounded | no | voiced | no |
| 3 | `AH` | ʌ~ə | tongue body | steady | mid central unrounded | no | voiced | no |
| 4 | `AO` | ɔ | lips \| tongue body | steady | mid back rounded | no | voiced | no |
| 5 | `AW` | aʊ | lips \| tongue body | diphthong | low central unrounded → high back rounded | no | voiced | no |
| 6 | `AY` | aɪ | tongue body | diphthong | low central unrounded → high front unrounded | no | voiced | no |
| 11 | `EH` | ɛ | tongue body | steady | mid front unrounded | no | voiced | no |
| 12 | `ER` | ɝ~ɚ | tongue body | steady | mid central unrounded | yes | voiced | no |
| 13 | `EY` | eɪ | tongue body | diphthong | mid front unrounded → high front unrounded | no | voiced | no |
| 17 | `IH` | ɪ | tongue body | steady | high front unrounded | no | voiced | no |
| 18 | `IY` | i | tongue body | steady | high front unrounded | no | voiced | no |
| 25 | `OW` | oʊ | lips \| tongue body | diphthong | mid back rounded → high back rounded | no | voiced | no |
| 26 | `OY` | ɔɪ | lips \| tongue body | diphthong | mid back rounded → high front unrounded | no | voiced | no |
| 33 | `UH` | ʊ | lips \| tongue body | steady | high back rounded | no | voiced | no |
| 34 | `UW` | u | lips \| tongue body | steady | high back rounded | no | voiced | no |

The repository vocabulary omits stress markers. Consequently, `AH` covers a
coarse mid-central target spanning common `ʌ`- and `ə`-like labels, and `ER`
uses one coarse rhotic mid-central target for `ɝ`- and `ɚ`-like labels. These
are fixed analysis labels, not claims that the realized postures are identical.

## Intended probe progression

Future linear probes should evaluate these targets independently in the
following descriptive progression:

1. segment family;
2. primary articulator membership as separate binary targets;
3. constriction gesture;
4. consonant place or vowel posture;
5. voicing and nasality.

Reference phoneme sequences should be CTC-aligned to the frozen GRU outputs.
Training probes on attributes copied from the GRU's own native predictions
would largely test whether a second readout can reproduce the existing linear
phoneme head. Even reference-constrained alignment remains model-assisted and
does not create independently measured phoneme timing.

## AI assistance

Codex drafted the version-1 schema, CSV rows, documentation, and validation
tests by combining the repository vocabulary with the cited IPA and PanPhon
feature frameworks and the experiment-specific coarse-target decisions. No
external code or data table was copied. The taxonomy requires human review
before it is treated as a biological ground truth or used for a public claim.
