"""Flask application for Provenance Guard.

Defines the API surface from planning.md:
    POST /submit  — classify a submission
    POST /appeal  — contest a classification
    GET  /log     — read the structured audit log
"""

from __future__ import annotations

import logging
import os
import uuid

from flask import Flask, jsonify, request

from dotenv import load_dotenv

from .audit import init_log, get_log
from .scoring import combine_signals
from .signals import groq_signal, stylometric_signal

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("provenance_guard")


# ----------------------------------------------------------------------
# Placeholder label set used until Milestone 5 wires the full variants.
# Variants will be returned by a label_generator() function then.
# ----------------------------------------------------------------------
_PLACEHOLDER_LABELS = {
    "likely_ai": {
        "headline": "Likely AI-generated",
        "body": "Signals suggest AI authorship. See /appeal to contest.",
    },
    "likely_human": {
        "headline": "Likely human-written",
        "body": "Signals suggest human authorship. Probabilistic, not a guarantee.",
    },
    "uncertain": {
        "headline": "Uncertain",
        "body": "Signals disagree or are weak. Authorship is unverified.",
    },
}


def create_app(db_path: str | None = None) -> Flask:
    """Application factory.

    `db_path` defaults to ./provenance_guard.db so tests can override it.
    """
    app = Flask(__name__)
    init_log(db_path or os.environ.get(
        "PROVENANCE_DB", "provenance_guard.db"
    ))
    log.info("Audit log initialised at %s", db_path or "provenance_guard.db")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/submit", methods=["POST"])
    def submit():
        body = request.get_json(silent=True) or {}
        text = body.get("text", "")
        creator_id = body.get("creator_id", "")
        if not text or not creator_id:
            return (
                jsonify({"error": "text and creator_id are required"}),
                400,
            )

        content_id = str(uuid.uuid4())

        llm_score = groq_signal(text)
        struct_score = stylometric_signal(text)
        combined = combine_signals(llm_score, struct_score)

        label = _PLACEHOLDER_LABELS[combined["attribution"]]

        log_entry = get_log().append(
            content_id=content_id,
            event="submission",
            payload={
                "creator_id": creator_id,
                "text": text,
                "attribution": combined["attribution"],
                "confidence": round(combined["confidence"], 4),
                "llm_score": round(llm_score, 4),
                "stylometric_score": round(struct_score, 4),
                "label": label,
                "status": "classified",
            },
            status="classified",
        )
        log.info("submission %s by %s -> %s", content_id, creator_id, combined["attribution"])

        return (
            jsonify(
                {
                    "content_id": content_id,
                    "creator_id": creator_id,
                    "attribution": log_entry["attribution"],
                    "confidence": log_entry["confidence"],
                    "label": log_entry["label"],
                    "llm_score": log_entry["llm_score"],
                    "stylometric_score": log_entry["stylometric_score"],
                }
            ),
            200,
        )

    @app.route("/log", methods=["GET"])
    def read_log():
        limit = int(request.args.get("limit", "50"))
        status = request.args.get("status")
        entries = get_log().get_entries(limit=limit, status=status)
        return jsonify({"entries": entries}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
