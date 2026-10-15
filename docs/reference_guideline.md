# Reference-summary guideline — internal benchmark (v1.0)

Applies to every row of `references.csv`. The proposal requires this guideline to be fixed
**before** evaluation starts. Changing it means bumping `benchmark.guideline_version` in
`configs/default.yaml`, and the validator then flags references written under the old version.

## 1. Roles
- **Writer**: fluent in Modern Standard Arabic. Reads the full text and writes the reference and key points.
- **Reviewer**: a second fluent Arabic speaker, **not** the writer. Checks the reference against
  the checklist in §6 and sets `review_status`.
- The writer and reviewer must not see any model output for that text beforehand. No machine
  translation, LLM or summarization tool may be used to draft or edit references, except under §1a.

## 1a. Model-drafted references (deviation from proposal §11.1, needs project-lead sign-off)
At the user's request (2026-10-15) the first version of all 240 references was drafted by a
model (`reference_source = llm_draft`, `writer_id = claude-opus-5.5`) following §2–§5.
- Each row stays `review_status = pending` until a fluent Arabic **human** reviewer checks it
  against §6, fixes or rejects it, and sets `reviewer_id` and `review_status`. The validator
  rejects a reviewer equal to the writer, and `--strict` fails on any unapproved row.
- The reviewer reads the source text first, then the draft. Edits by the reviewer are allowed;
  keep `reference_source = llm_draft` and describe substantial changes in `review_notes`.
- Reports, the final benchmark description and any ROUGE claim must call these references
  **model-drafted, human-verified** and give the counts per `reference_source` (the validator
  prints `references_by_source` / `approved_by_source`).
- The drafts file is kept out of git (it contains the reference answers of the test set).

## 2. Length
- Aim for `ref_target_words`, which is about 20% of the input's word count. The guideline range
  is 15–25%.
- Hard limits, enforced by the validator: between `ref_min_words` and `ref_max_words`. That is
  10–30% of the input, and never fewer than 30 or more than 300 words. Both numbers are
  precomputed in each row.
- A *word* is any whitespace-separated token containing a letter or digit. Punctuation alone
  doesn't count.

## 3. Content (coverage)
1. First list the text's **key points** in `key_points`, one per line, at least 3. A key point
   is a fact a reader must know: who, what, when, where, numbers, outcomes, decisions.
2. The reference must cover **at least 80% of the key points**, and the most important ones first.
3. Leave out background, quotes that add no facts, repetition, and the author's opinions
   unless the opinion is the point of the text.

## 4. Accuracy (critical)
- Copy names of people, organizations and places, numbers, percentages, currencies and dates
  **exactly** as in the source. Don't round, convert calendars or translate names.
- Add **nothing** that isn't in the text. No outside knowledge, no inferred causes.
- Keep event attribution exact: who said, did or decided what. Reported speech must stay
  attributed ("قال الوزير إن …").
- Keep hedges the source uses ("قد", "من المتوقع", "بحسب").

## 5. Style
- Modern Standard Arabic, third person, full sentences, a single paragraph. No bullet points
  and no headings.
- Paraphrasing is encouraged. Copying short spans is fine, especially names and figures. Don't
  just paste the first sentences of the text.
- Use the source's digit style (Arabic-Indic or Western). Use standard spelling: hamza,
  ة/ه and ى/ي as in the source. No diacritics except where needed to disambiguate.
- No meta-phrases such as "يتناول هذا النص…" or "يتحدث المقال عن…".

## 6. Reviewer checklist (all must hold before `review_status = approved`)
- [ ] Length is inside the hard limits and close to the target.
- [ ] Every name, number and date in the reference appears identically in the source.
- [ ] Nothing is unsupported by the source. Attribution is correct.
- [ ] At least 80% of the listed key points are covered. The key points themselves are correct and complete.
- [ ] Fluent, coherent, grammatical MSA.

Otherwise set `review_status = needs_revision` and explain in `review_notes`. The writer then
revises, and the reviewer checks again.

## 7. Columns
| Column | Filled by | Notes |
|---|---|---|
| `reference_summary` | writer | the summary text |
| `key_points` | writer | one per line (or separated by `\|`), at least 3 |
| `reference_source` | writer | `human`; `source` if the text came with a credible author-written summary that meets §2–§5; `llm_draft` under §1a |
| `writer_id`, `reviewer_id` | both | short stable initials, e.g. `MD`, `R2` |
| `review_status` | reviewer | `pending` → `approved` / `needs_revision` / `rejected` |
| `review_notes` | reviewer | reason for any status other than approved |
| `guideline_version` | prefilled | don't edit |
