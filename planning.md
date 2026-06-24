# Provenance Guard — Planning Document

This document is the design specification for Provenance Guard, written before any implementation code. It defines the architecture, detection approach, uncertainty model, and UX decisions that subsequent milestones will implement.

---

## Architecture

### Submission flow (POST /submit)

```
  Creator
    │
    │ POST /submit { text, creator_id }
    ▼
┌──────────────────┐
│  Flask endpoint  │── assigns content_id (UUID)
└──────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│          Multi-signal detection pipeline      │
│                                                │
│   ┌──────────────────┐   ┌──────────────────┐ │
│   │ Signal 1:        │   │ Signal 2:        │ │
│   │ Groq LLM         │   │ Stylometric      │ │
│   │ (semantic)       │   │ heuristics       │ │
│   │ score: 0–1       │   │ (structural)     │ │
│   │                  │   │ score: 0–1       │ │
│   └────────┬─────────┘   └────────┬─────────┘ │
│            │ llm_score            │ struct_score │
│            ▼                      ▼             │
│      ┌────────────────────────────────────┐     │
│      │   Confidence scoring combiner      │     │
│      │   weighted avg + human-bias shift  │     │
│      │   output: combined_score 0–1       │     │
│      └─────────────────┬──────────────────┘     │
│                        ▼                         │
│              ┌──────────────────────┐           │
│              │ Label generator       │           │
│              │ picks one of 3 texts  │           │
│              └──────────┬───────────┘           │
└─────────────────────────┼───────────────────────┘
                          ▼
              ┌────────────────────────┐
              │  Structured audit log  │
              │  (content_id, both     │
              │   signal scores,       │
              │   combined score,      │
              │   attribution, label,  │
              │   timestamp, status)   │
              └────────────┬───────────┘
                           ▼
              Response → Creator
              { content_id, attribution,
                confidence, label }
```

### Appeal flow (POST /appeal)

```
  Creator
    │
    │ POST /appeal { content_id, creator_reasoning }
    ▼
┌──────────────────┐
│  Flask endpoint  │
└──────────────────┘
    │
    ▼
┌────────────────────────────────┐
│ Lookup original decision       │
│ by content_id in log           │
└────────────┬───────────────────┘
             ▼
┌────────────────────────────────┐
│ Update status → "under_review" │
└────────────┬───────────────────┘
             ▼
┌────────────────────────────────┐
│ Append appeal entry to audit   │
│ log (linked by content_id)     │
└────────────┬───────────────────┘
             ▼
  Response { content_id,
             status: "under_review",
             appeal_logged: true }
```

### Narrative

A piece of text submitted via `POST /submit` is given a unique `content_id` and run through two independent detectors in parallel: a Groq LLM judge that assesses semantic and stylistic coherence, and a stylometric heuristics module that measures structural properties (sentence length variance, vocabulary diversity, punctuation density). Their individual scores feed a confidence combiner that applies a weighted average with a small constant bias toward the "human" side — this encodes the asymmetry that flagging a human's work as AI is more harmful than missing an AI submission. The combined score maps to one of three label variants. The full decision record — both raw signal scores, the combined confidence, the attribution, the chosen label text, the timestamp, and the status — is written as a structured entry to the audit log before the response returns to the creator. An appeal submitted later via `POST /appeal` looks up the original record by `content_id`, flips its status to `under_review`, and appends an appeal entry to the same audit-log thread so a human reviewer can see the original decision and the creator's reasoning side by side.

---

## Detection signals

### Signal 1 — Groq LLM judge (semantic)

**Property measured.** Holistic semantic and stylistic coherence: hedging patterns (`it is important to note`, `furthermore`), generic structure (intro → balanced points → conclusion), uniform register, and the kind of "average-of-the-internet" phrasing that LLM training tends to converge on. Llama-3.3-70b-versatile is asked via a structured prompt to return a probability that the text is AI-generated.

**Output format.** A single float in `[0.0, 1.0]` where `0.0` = confidently human-written and `1.0` = confidently AI-generated. Parsed from a constrained JSON response from the model.

**Exact prompt (used at call time).**
```
You are an AI-vs-human writing classifier. Read the text and return
ONLY a JSON object of the form {"ai_probability": <float between 0 and 1>}
with no commentary, where 0.0 means certainly human-written and 1.0
means certainly AI-generated. Consider hedging phrases, generic
structure, uniform register, and whether the text looks like a polished
LLM draft.

Text:
"""
{text}
"""
```
The function will parse the model's JSON, clamp the float to `[0, 1]`, and return it. If parsing fails, the function returns `0.5` (the maximum-uncertainty value) so the rest of the pipeline degrades gracefully without crashing.

**Why this property differs.** LLMs are trained to produce predictable, low-perplexity continuations; they overuse certain transitional phrases and avoid the idiosyncratic quirks (typos, abrupt topic shifts, dialect, dead metaphors) that humans leave behind.

**Blind spots.**
- Lightly edited AI text: a human rewriting 10–20% of an LLM draft can pass the judge's surface checks.
- Non-native English formal writing: heavy hedging and structured argumentation by a fluent non-native writer can look statistically similar to LLM prose.
- Short fragments under ~80 words: too little signal for the model to find a confident pattern.

### Signal 2 — Stylometric heuristics (structural)

**Properties measured.** Three computable statistics computed in pure Python:

1. **Sentence length variance (slv).** Standard deviation of the length (in words) of each sentence in the text. To normalize to `[0, 1]`, divide by a reference value of `12.0` and clamp: `slv_score = clamp(slv / 12.0, 0, 1)`. Higher variance → lower AI-likelihood → smaller score.

2. **Type-token ratio (ttr).** `unique_words / total_words`, computed case-insensitively after stripping punctuation. Reference for "AI-like" vocabulary diversity: around `0.45`. We invert so that low TTR (more repetitive) scores closer to 1: `ttr_score = clamp(1.0 - (ttr - 0.30) / 0.30, 0, 1)`. Practically: TTR of `0.30` or below → `1.0` (max AI-likelihood); TTR of `0.60` or above → `0.0`.

3. **Punctuation density (pd).** Punctuation marks (`. , ; : ! ? - —`) per 100 words. AI text sits around `4` per 100 words; human text around `7` per 100 words. `pd_score = clamp(1.0 - (pd - 3.0) / 5.0, 0, 1)`.

These three are averaged uniformly into a single structural score:
`struct_score = (slv_score + ttr_score + pd_score) / 3.0`.

**Edge handling.** If the text has fewer than 3 sentences or fewer than 30 total words, the heuristics return `0.5` (no opinion) — short text is unreliable for structural statistics, and an uncertain value is safer than a guessed one.

**Output format.** A single float in `[0.0, 1.0]` (same scale as Signal 1).

**Why these properties differ.** AI text is produced by sampling one likely next token at a time, which produces locally smooth output without the punctuation flair and length variation a human editor leaves in.

**Blind spots.**
- Very short text (a poem, a haiku): too few sentences and tokens for variance/diversity to be statistically meaningful.
- Polished human essays deliberately written for uniformity (e.g., op-eds) will score AI-like on the heuristics.
- Conversational/casual human text that happens to be uniform (formulaic recipes, listicles) will be flagged as AI.
- The metrics have no semantic content at all — a string of vocabulary words that passes TTR but is nonsense would look "human" to it.

These two signals are deliberately independent: one is semantic, the other is structural. Where they agree the system can be confident; where they disagree the system should land in the "uncertain" band rather than guessing.

---

## Uncertainty representation

### What does a confidence score mean?

The combined score is the system's calibrated estimate of `P(text is AI-generated | signals)`. Interpreted for users:

- **`< 0.30`** → Likely human-written. Strong human-side evidence from both signals.
- **`0.30 – 0.70`** → Uncertain. Signals disagree, or both are weak.
- **`> 0.70`** → Likely AI-generated. Both signals point the same direction.

These thresholds are not symmetric around 0.5. The "Likely AI" band starts at 0.70, not 0.50, because a false positive (labeling a human as AI) is more harmful on a writing platform than a false negative. A creator who is wrongly flagged faces an accusation of dishonesty; a piece of AI text that slips through is a lesser harm the platform can address through audience transparency rather than creator blame.

### Combining the two signals

```python
def combine(llm_score: float, struct_score: float) -> dict:
    # weights from spec: LLM judge gets more weight than heuristics
    raw = 0.6 * llm_score + 0.4 * struct_score

    # disagreement penalty: when signals fight, lean toward uncertainty
    if abs(llm_score - struct_score) > 0.3:
        raw = raw * 0.85 + 0.5 * 0.15   # pull 15% toward 0.5

    confidence = max(0.0, min(1.0, raw))

    if confidence > 0.70:
        attribution = "likely_ai"
    elif confidence < 0.30:
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    return {"confidence": confidence, "attribution": attribution}
```

Weights: `w_llm = 0.6`, `w_struct = 0.4`. The LLM judge is given more weight because semantic evidence carries more information than three structural metrics, but the structural signal is not negligible — it can catch cases the LLM misses (edited AI drafts) and can contradict the LLM when the LLM is biased (non-native English).

---

## Transparency label design

Three variants. Each one is the literal text shown to a reader on the platform, paired with a short headline.

### Variant A — high-confidence AI (combined score > 0.70)

**Headline:** "Likely AI-generated"
**Body:**
> Independent analysis of this submission suggests it was very likely produced by an AI writing assistant. Signals used: stylometric analysis and a language-model assessment. If you are the creator and believe this is incorrect, you may submit an appeal from your dashboard.

### Variant B — high-confidence human (combined score < 0.30)

**Headline:** "Likely human-written"
**Body:**
> Both the language-model assessment and stylometric analysis suggest this text was written by a human. This is a probabilistic judgment, not a guarantee — AI-assisted writing is not always detectable.

### Variant C — uncertain (0.30 ≤ score ≤ 0.70)

**Headline:** "Uncertain — verification recommended"
**Body:**
> The two analysis signals used by Provenance Guard disagree on this submission, or both are weak. We cannot make a confident attribution in either direction. Readers should treat the authorship of this piece as unverified.

These three texts must change with the score — they are not the same string regardless of band. The full label object — both the `headline` and `body` strings — is what the API returns in the `label` field of the `/submit` response (see the API surface section below for the exact response shape).

**Selection logic (used by M5):**
```python
if confidence > 0.70:
    return LABELS["likely_ai"]
elif confidence < 0.30:
    return LABELS["likely_human"]
else:
    return LABELS["uncertain"]
```
The category assigned here uses the *same* thresholds as `combine()` so a content record's `attribution` field matches its `label` variant.

---

## Appeals workflow

### Who can submit

The creator identified by the `creator_id` associated with the original `content_id`. Authentication of creator identity is out of scope for this project (the spec doesn't require it); in production this would be tied to an authenticated session.

### Information provided

```json
{
  "content_id": "uuid-from-original-submit",
  "creator_reasoning": "free-text explanation from the creator"
}
```

### What the system does

1. Look up the original decision by `content_id` in the audit log.
2. If found, append a new audit-log entry linked to that `content_id` with:
   - `event`: `"appeal"`
   - `creator_reasoning`: the supplied text
   - `previous_status`: the original status
   - `new_status`: `"under_review"`
   - `timestamp`: ISO 8601
3. Update the original record's `status` field to `"under_review"` so subsequent log reads reflect the change.
4. Return a confirmation JSON to the creator.

No automated re-classification is performed. The decision waits for human review.

### What a human reviewer sees in the appeal queue

A simple view: `GET /log?status=under_review` returns all log entries whose current status is `under_review`. Each entry exposes:

- `content_id` — links the appeal to the original submission.
- Original `text` from the submission event.
- `llm_score` and `stylometric_score` (raw, so the reviewer can see what the system saw).
- `confidence` (the combined score).
- `attribution` and `label` from the original decision.
- `creator_reasoning` — the free-text appeal argument.
- `appeal_timestamp` — when the appeal was filed.

A real product would render these in a dashboard; for grading visibility we surface them through `/log` with an optional status filter. The minimum record fields are sufficient for a reviewer to make a manual override decision.

---

## Anticipated edge cases

1. **Poetry with strong repetition and simple vocabulary.** A poem like "the rose / the rose / the garden / the rose" has very low TTR (high repetition) and short sentences, both of which the stylometric signal may score as AI-like. Mitigation: the LLM judge is weighted higher than the heuristics and poetry is often short enough that the LLM judge will not converge strongly on either side, pushing the combined score into the uncertain band — which is the honest outcome.

2. **Polished op-ed writing.** A human journalist writing a tightly-edited opinion piece will have uniform sentence length, low punctuation density, and cautious vocabulary. The stylometric signal will read it as AI. Mitigation: the LLM judge is more reliable on formal semantic content; it will likely disagree, and the disagreement penalty will keep the combined score out of the "Likely AI" band.

3. **Non-native English formal writing.** A fluent non-native speaker may deploy formal hedging ("it is important to note that…") at higher rates than native casual writers. Both signals may tag this as AI-like. Mitigation: the asymmetric threshold (>0.70 to flag as "Likely AI") and the appeal path catch false positives. The cost is some false negatives on actual AI from non-native writers — an acceptable trade given that false positives are the worse failure mode.

4. **Text under ~80 words.** Too little for either signal to be confident. The system should naturally fall into the uncertain band on short input, which is the safest behavior.

5. **Lightly edited AI output.** A writer uses an LLM to draft and rewrites ~15% by hand. The heuristics may catch something the LLM judge misses (or vice versa). Result will usually be "uncertain" — exactly the outcome that justifies the appeals workflow.

---

## AI Tool Plan

This section pre-specifies what spec sections will be fed to an AI assistant in each implementation milestone, what will be asked for, and how the output will be verified before being committed.

### Milestone 3 — submission endpoint + first signal
- **Sections provided to AI:** Detection signals (Signal 1 Groq), Architecture diagram, the Submission endpoint shape from the API surface outline below.
- **What to generate:** Flask app skeleton, `POST /submit` route stub, the Groq Signal 1 function (returns float in [0,1]), initial SQLite/JSON audit-log writer, `GET /log` endpoint.
- **Verification:** Run the Flask app standalone, POST a hardcoded response, inspect `/log` shows one entry, call the Signal 1 function directly with a known AI-looking and known human-looking input and confirm the scores are in plausible ranges before wiring it into the endpoint.

### Milestone 4 — second signal + confidence scoring
- **Sections provided:** Detection signals (both), Uncertainty representation, Architecture diagram.
- **What to generate:** Signal 2 stylometric function (returns float in [0,1]), the confidence combiner with the asymmetric thresholds and disagreement penalty, wiring both signals into `/submit`.
- **Verification:** Run the four specified test inputs from Milestone 4 (clearly AI, clearly human, formal human, lightly edited AI) and confirm the combined scores produce meaningfully different values across them. Both individual signal scores must also be visible in the audit log.

### Milestone 5 — production layer
- **Sections provided:** Transparency label design, Appeals workflow, Architecture diagram, the API surface shape.
- **What to generate:** Label generator function that maps combined score to one of the three exact label texts, `POST /appeal` endpoint with status update and audit-log append, Flask-Limiter setup with rate limits as defined below.
- **Verification:** Submit three inputs that hit each of the three label bands and confirm the returned `label` text exactly matches the Verification variants above (reading the response back, not just trusting the function). POST an appeal and confirm the audit log contains both the original entry and the appeal entry tied by `content_id` with `status: "under_review"`. Run the rapid-fire curl loop to confirm 429 responses after the per-minute limit.

---

## API surface (contract for all later code)

### `POST /submit`
**Request:**
```json
{
  "text": "string (required)",
  "creator_id": "string (required)"
}
```
**Response (200):**
```json
{
  "content_id": "uuid",
  "creator_id": "string",
  "attribution": "likely_ai | likely_human | uncertain",
  "confidence": 0.0,
  "label": {
    "headline": "string",
    "body": "string"
  },
  "llm_score": 0.0,
  "stylometric_score": 0.0
}
```

### `POST /appeal`
**Request:**
```json
{
  "content_id": "uuid (required)",
  "creator_reasoning": "string (required)"
}
```
**Response (200):**
```json
{
  "content_id": "uuid",
  "status": "under_review",
  "appeal_logged": true
}
```
**Response (404):** if `content_id` not found.

### `GET /log`
**Response (200):**
```json
{
  "entries": [ /* most-recent log entries, structured JSON */ ]
}
```

### Rate limiting
- `POST /submit`: 10 per minute, 100 per day per IP address.
- Rationale: a creator submitting their own work would realistically submit a handful of pieces per day, never ten per minute. The per-minute limit blocks naive flood scripts; the daily limit blocks slower sustained abuse. Limits chosen specifically to be defensible, not arbitrary.

---

## Stretch features (planned)

To be re-scoped and updated **before** starting any stretch work, per the Milestone 5 instruction.

- *Ensemble detection (3+ signals)* — possible candidates: perplexity proxy via token-frequency rarity, repetition-of-phrase counter, n-gram repetition of function-word trigrams. Not started.
- *Provenance certificate ("verified human" credential)* — design idea: an additional verification step (e.g., a timed writing sample prompt) the creator completes, after which their `creator_id` is flagged `verified_human = true` and downstream labels downgrade AI-likelihood for that creator's content. Not started.
- *Analytics dashboard* — minimum entries: aggregate detection patterns by attribution band, appeal rate, false-positive rate derived from appeal outcomes. Plus one additional metric TBD before implementation.
- *Multi-modal support* — second content type: image descriptions using EXIF metadata plus OCR (stub library such as `pytesseract`). Not started.

## Spec sign-off (Milestone 2)

- All five required questions answered with specific, implementation-ready answers above.
- Three transparency label variants written out verbatim and mapped to thresholds.
- Confidence scoring produces three different labels at different score ranges (not a binary flip at 0.5); thresholds are asymmetric (0.30 / 0.70) per the false-positive asymmetry.
- `## Architecture` section contains the diagram and the 2–3-sentence flow narrative.
- `## AI Tool Plan` covers Milestones 3, 4, and 5 with specific spec sections provided to the AI tool, what is asked for, and verification steps.

No code has been written yet; implementation in Milestones 3–5 will be driven from this document.
