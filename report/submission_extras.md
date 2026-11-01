# Kept out of the report body (spec §8 asks for a Contribution Statement in the submission)

## Contribution Statement
- **@hwupu (Member A):** project lead (PO); planning, code with an AI coding agent (data loader, metric registry, scoring, evaluation, failure mining, figures), drafted the report, selected the five cases by quantitative criteria.
- **@ALLEN07056 (Member B):** code reviewer for all PRs, cross-platform verification on WSL, verification of results and failure-case explanations, report review.

**Generative AI disclosure (draft).**
- Claude Code (Claude Opus 5.5) was used to write code, run the experiments and draft the text.
- Every AI interaction that produced kept output is logged in `doc/PROMPT_LOG.md` (P-001 to P-010), and every design choice in `doc/DECISIONS.md`.
- Humans chose the case-selection rules, reviewed every PR, and own the final explanations.
