# Provenance Guard

A backend for creative-sharing platforms that classifies submitted text content as AI-generated or human-written, surfaces a transparency label to readers, and lets creators appeal classifications. See `planning.md` for the full design specification.

## Quick start

```bash
python -m venv .venv
# activate (Windows) .venv\Scripts\activate
#         (mac/linux) source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add your GROQ_API_KEY
python -m provenance_guard   # serves on 127.0.0.1:5000 by default; PORT env overrides
```

## Endpoints

| Method | Path     | Purpose                                                                |
|--------|----------|------------------------------------------------------------------------|
| GET    | /health  | Liveness check                                                         |
| POST   | /submit  | Classify a submission (`{text, creator_id}`) — returns label + scores  |
| POST   | /appeal  | Contest a classification (`{content_id, creator_reasoning}`)           |
| GET    | /log     | Read recent audit-log entries (optional `?status=under_review` filter) |

## Project layout

```
provenance_guard/
  __init__.py
  __main__.py        # `python -m provenance_guard`
  app.py             # Flask routes
  signals.py         # imports both signal functions
  signal_groq.py     # Signal 1 — semantic (Groq llama-3.3-70b)
  signal_stylometry.py  # Signal 2 — stylometric heuristics (M4)
  scoring.py         # confidence combiner (M3 placeholder, full impl in M4)
  audit.py           # SQLite-backed structured audit log
```

The full implementation has been built milestone by milestone against `planning.md`; this README is finished as part of Milestone 6.
