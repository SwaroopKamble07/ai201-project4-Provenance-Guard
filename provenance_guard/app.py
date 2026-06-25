"""Flask application for Provenance Guard.

Routes (per planning.md):
    POST /submit   — classify a submission
    POST /appeal   — contest a classification
    GET  /log      — read the structured audit log
    GET  /health   — liveness

Production layer (Milestone 5):
    - Real three-variant transparency label
    - POST /appeal wired with status flip + audit-log append
    - Flask-Limiter on /submit (10/minute, 100/day per IP)
"""

from __future__ import annotations

import logging
import os
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from dotenv import load_dotenv

from .audit import init_log, get_log
from .labels import label_for
from .scoring import combine_signals
from .signals import groq_signal, stylometric_signal

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("provenance_guard")


def create_app(db_path: str | None = None) -> Flask:
    """Application factory.

    `db_path` defaults so tests can override it.
    """
    app = Flask(__name__)
    init_log(db_path or os.environ.get("PROVENANCE_DB", "provenance_guard.db"))
    log.info(
        "Audit log initialised at %s",
        db_path or os.environ.get("PROVENANCE_DB", "provenance_guard.db"),
    )

    # Flask-Limiter ≥3.x requires an explicit storage_uri.
    # In-memory storage is fine for development & tests.
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
    )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/submit", methods=["POST"])
    @limiter.limit("10 per minute;100 per day")
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
        label = label_for(combined["confidence"])

        log_entry = get_log().append(
            content_id=content_id,
            event="submission",
            status="classified",
            payload={
                "creator_id": creator_id,
                "text": text,
                "attribution": combined["attribution"],
                "confidence": round(combined["confidence"], 4),
                "llm_score": round(llm_score, 4),
                "stylometric_score": round(struct_score, 4),
                "label": label,
                "appealed": False,
            },
        )
        log.info(
            "submission %s by %s -> %s (conf=%.3f)",
            content_id,
            creator_id,
            combined["attribution"],
            combined["confidence"],
        )

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

    @app.route("/appeal", methods=["POST"])
    def appeal():
        body = request.get_json(silent=True) or {}
        content_id = (body.get("content_id") or "").strip()
        creator_reasoning = (body.get("creator_reasoning") or "").strip()
        if not content_id or not creator_reasoning:
            return (
                jsonify(
                    {"error": "content_id and creator_reasoning are required"}
                ),
                400,
            )

        existing = get_log().get_by_content_id(content_id)
        if not existing:
            return (
                jsonify({"error": "content_id not found"}),
                404,
            )

        original = existing[0]
        previous_status = original.get("status", "classified")
        appeal_entry = get_log().append(
            content_id=content_id,
            event="appeal",
            status="under_review",
            payload={
                "creator_id": original.get("creator_id", ""),
                "creator_reasoning": creator_reasoning,
                "previous_status": previous_status,
                "appealed": True,
            },
        )
        # Flip the original record's status (and any sibling rows) so a
        # reviewer reading /log later sees the current state.
        get_log().update_status(content_id, "under_review")

        log.info(
            "appeal %s by %s (prev=%s) -> under_review",
            content_id,
            original.get("creator_id", ""),
            previous_status,
        )

        return (
            jsonify(
                {
                    "content_id": content_id,
                    "status": "under_review",
                    "appeal_logged": True,
                    "appeal_event_id": appeal_entry["timestamp"],
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
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
