---
name: trace-paper-derived-idea
description: Trace a published neuroscience or machine-learning idea into this repository with explicit source, inference, adaptation, code, licensing, and AI-assistance provenance. Use when extracting architecture or training details from a paper, implementing or reviewing a BIT-style, POSSM-style, Willett-derived, or other published method, updating paper notes or design notes, checking whether local code is official or independently implemented, or preparing provenance for a public release. Do not use for generic literature recommendations or unsupported summaries from memory.
---

# Trace Paper-Derived Idea

Make the boundary between the publication, released code, local inference, and repository evidence impossible to miss.

## Establish the source

1. Read `AGENTS.md`, the relevant file under `docs/paper_notes/`, and the owning experiment's design and provenance notes.
2. Open the primary paper itself before asserting exact architecture, data, hyperparameter, or result details. Use repository paper notes only as an index and prior summary.
3. Identify the source precisely when available:
   - title, authors, publication or preprint venue, year, and version;
   - DOI, arXiv identifier, publisher URL, or stable local PDF path;
   - exact page, section, figure, table, appendix, or equation supporting each implementation-relevant claim.
4. Search for an author- or institution-released code repository when code provenance matters. Record repository URL, commit or release, relevant files, and license.
5. Prefer primary papers, supplements, official repositories, and author documentation. Do not treat secondary summaries or another model's answer as authoritative evidence.

If a specific paper, repository, license, or current release status is not already available locally, browse for it rather than guessing. Cite the exact source used.

## Label every statement

Use these labels in working notes and provenance reviews:

- **Reported:** explicitly stated or shown in the paper or official supplement.
- **Released-code behavior:** directly observed in identified upstream code.
- **Inferred:** necessary interpretation of incomplete or ambiguous descriptions.
- **Adapted:** intentional local change from the reported or released method.
- **Repository observation:** empirical behavior measured in this repository.
- **Unverified:** plausible but not yet tied to a primary source.

Never collapse these categories. A local implementation can be paper-derived without being official, faithful, independently implemented, or a verified reproduction.

Apply the repository's known boundaries:

- POSSM code has not been released in the currently documented project context. Describe `experiments/possm_style` as a paper-derived implementation built from reported details and explicit inference, not as official code or a reference implementation.
- The Willett-style GRU includes an LLM-assisted Python port/adaptation of likely TensorFlow source with unresolved exact upstream files and licensing. Preserve `experiments/supervised_baselines/PROVENANCE.md`; never call it a manual translation or independent reimplementation without new evidence.

## Build an adaptation map

For each material idea, record:

| Source element | Evidence location | Local implementation | Classification | Deviation or uncertainty | Validation |
|---|---|---|---|---|---|
| Architecture, objective, data rule, or training choice | Paper section or upstream file | Repository path and symbol | Reported, released-code behavior, inferred, or adapted | Exact difference or unknown | Test, reproduction check, or missing validation |

Trace dependencies as well as headline architecture: preprocessing, splits, normalization, tokenization or binning, channel handling, loss, augmentation, optimization, checkpoint selection, and evaluation. These often determine whether a claimed reproduction is meaningful.

Do not silently “fix” an unusual reported detail. Record it as reported, then document any local correction as an adaptation.

## Audit provenance and licensing

When upstream code may have influenced local code:

1. Identify the likely source repository and exact files if possible.
2. Distinguish copied, translated, LLM-ported, adapted, and independently written portions.
3. Record commit, version, copyright owner, license, required notices, and redistribution constraints.
4. Mark unresolved items explicitly.
5. Require provenance and licensing review before public release when origin or redistribution rights remain uncertain.

Do not provide legal clearance. State what was verified and what still requires review. Avoid large verbatim passages or code copies when a citation and concise paraphrase suffice.

## Record substantive AI assistance

When AI materially generated or transformed analytical code, methods, figures, or manuscript content, retain a concise disclosure containing:

- affected content;
- action performed;
- tool and model/version when known;
- purpose;
- human review, tests, and validation performed.

Do not fabricate unavailable model or prompt details, and do not log routine editing or every conversational prompt. If an LLM ported code, say so directly and avoid implying a manual translation.

## Put information in the right place

- Put source-focused summaries in `docs/paper_notes/`.
- Put repository-specific adaptations and design decisions in the owning experiment's `design/` directory.
- Put code-origin and redistribution disclosures beside the affected implementation in `PROVENANCE.md`.
- Put empirical findings in the owning experiment's `results/` directory only after `$evaluate-decoding-result` establishes them.
- Keep citations near the exact claims they support.

When asked only to review provenance, report findings without editing. When asked to update documentation, preserve existing uncertainty, update affected links, and run `git diff --check`.

Return the source identity, adaptation map, unresolved provenance or licensing items, validation status, and exact files changed.
