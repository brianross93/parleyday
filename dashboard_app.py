import os
import traceback
from datetime import datetime

from flask import Flask, render_template, request

from data_pipeline import DEFAULT_DB_PATH
from env_config import load_local_env

load_local_env()

from llm_judge import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    generate_chat_reply,
    generate_slate_judgment,
)
from refresh_slate import refresh_slate
from quantum_parlay_oracle import run_oracle


app = Flask(__name__)
LATEST_RESULT_RAW: dict | None = None
LATEST_RESULT_VIEW: dict | None = None
LATEST_FORM: dict | None = None
CHAT_HISTORY: list[dict] = []


def default_form() -> dict:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sport": "mlb",
        "slate_mode": "auto",
        "score_source": "implied",
        "refresh_first": True,
        "kalshi_pages": 25,
        "fallback": False,
        "samples_per_beta": 250,
        "warmup": 50,
        "thin": 3,
        "bytes": 65536,
        "llm_judge": False,
        "llm_model": DEFAULT_OPENAI_MODEL,
        "llm_reasoning_effort": DEFAULT_REASONING_EFFORT,
    }


def _normalize_leg_for_template(leg: dict | None) -> dict | None:
    if leg is None:
        return None
    normalized = dict(leg)
    normalized.setdefault("implied_prob", 0.0)
    normalized.setdefault("score_delta", 0.0)
    normalized.setdefault("trust_score", 0.0)
    normalized.setdefault("pricing_label", "Unknown")
    normalized.setdefault("pricing_source", "")
    normalized.setdefault("pricing_reason", "")
    normalized.setdefault("notes", "")
    normalized.setdefault("activation", 0.0)
    normalized.setdefault("category", "")
    normalized.setdefault("game", "")
    normalized.setdefault("label", "")
    normalized.setdefault("sport", "")
    return normalized


def _normalize_judgment_for_template(judgment: dict | None) -> dict | None:
    if judgment is None:
        return None
    normalized = dict(judgment)
    normalized.setdefault("status", "unavailable")
    normalized.setdefault("model", "")
    normalized.setdefault("reasoning_effort", "")
    normalized.setdefault("message", "")
    normalized.setdefault("text", "")
    normalized.setdefault("generated_at", None)
    return normalized


def _normalize_chat_history_for_template(history: list[dict] | None) -> list[dict]:
    normalized_history = []
    for entry in history or []:
        normalized_history.append(
            {
                "role": str(entry.get("role") or "assistant"),
                "content": str(entry.get("content") or ""),
                "generated_at": entry.get("generated_at"),
            }
        )
    return normalized_history


def _normalize_result_for_template(result: dict | None) -> dict | None:
    if result is None:
        return None
    normalized = dict(result)
    normalized["top_legs"] = [_normalize_leg_for_template(leg) for leg in result.get("top_legs", [])]
    normalized["fades"] = [_normalize_leg_for_template(leg) for leg in result.get("fades", [])]
    normalized["moonshot"] = _normalize_leg_for_template(result.get("moonshot"))
    normalized["tier_parlays"] = [
        {
            **parlay,
            "legs": [_normalize_leg_for_template(leg) for leg in parlay.get("legs", [])],
        }
        for parlay in result.get("tier_parlays", [])
    ]
    normalized["llm_judgment"] = _normalize_judgment_for_template(result.get("llm_judgment"))
    normalized["llm_chat"] = _normalize_chat_history_for_template(result.get("llm_chat"))
    return normalized


@app.route("/", methods=["GET", "POST"])
def index():
    global LATEST_FORM, LATEST_RESULT_RAW, LATEST_RESULT_VIEW, CHAT_HISTORY

    form = dict(LATEST_FORM or default_form())
    result = LATEST_RESULT_VIEW
    error = None
    refresh_result = None

    if request.method == "POST":
        action = request.form.get("action", "run")
        incoming_form = {
            "date": request.form.get("date", form["date"]),
            "sport": request.form.get("sport", form["sport"]),
            "slate_mode": request.form.get("slate_mode", form["slate_mode"]),
            "score_source": request.form.get("score_source", form["score_source"]),
            "refresh_first": request.form.get("refresh_first") == "on",
            "kalshi_pages": int(request.form.get("kalshi_pages", form["kalshi_pages"])),
            "fallback": request.form.get("fallback") == "on",
            "samples_per_beta": int(request.form.get("samples_per_beta", form["samples_per_beta"])),
            "warmup": int(request.form.get("warmup", form["warmup"])),
            "thin": int(request.form.get("thin", form["thin"])),
            "bytes": int(request.form.get("bytes", form["bytes"])),
            "llm_judge": request.form.get("llm_judge") == "on",
            "llm_model": request.form.get("llm_model", form["llm_model"]).strip() or form["llm_model"],
            "llm_reasoning_effort": request.form.get(
                "llm_reasoning_effort",
                form["llm_reasoning_effort"],
            ).strip()
            or form["llm_reasoning_effort"],
        }

        form = incoming_form

        try:
            if action == "chat":
                if LATEST_RESULT_RAW is None:
                    error = "Run a slate first so the chatbox has context."
                else:
                    chat_message = request.form.get("chat_message", "").strip()
                    chat_result = generate_chat_reply(
                        LATEST_RESULT_RAW,
                        chat_message,
                        history=CHAT_HISTORY,
                        model=form["llm_model"],
                        reasoning_effort=form["llm_reasoning_effort"],
                    )
                    CHAT_HISTORY.append({"role": "user", "content": chat_message, "generated_at": None})
                    if chat_result["status"] == "ok":
                        CHAT_HISTORY.append(
                            {
                                "role": "assistant",
                                "content": chat_result["text"],
                                "generated_at": chat_result["generated_at"],
                            }
                        )
                    else:
                        CHAT_HISTORY.append(
                            {
                                "role": "assistant",
                                "content": chat_result["message"],
                                "generated_at": chat_result["generated_at"],
                            }
                        )
                    LATEST_RESULT_RAW["llm_chat"] = list(CHAT_HISTORY)
                    LATEST_RESULT_VIEW = _normalize_result_for_template(LATEST_RESULT_RAW)
                    result = LATEST_RESULT_VIEW
            else:
                CHAT_HISTORY = []
                if form["refresh_first"]:
                    refresh_result = refresh_slate(
                        date_str=form["date"],
                        sport=form["sport"],
                        db_path=os.getenv("PARLEYDAY_DB_PATH", DEFAULT_DB_PATH),
                        kalshi_pages=form["kalshi_pages"],
                    )
                result = run_oracle(
                    date_str=form["date"],
                    sport=form["sport"],
                    slate_mode=form["slate_mode"],
                    score_source=form["score_source"],
                    kalshi_pages=form["kalshi_pages"],
                    fallback=form["fallback"],
                    n_bytes=form["bytes"],
                    samples_per_beta=form["samples_per_beta"],
                    warmup=form["warmup"],
                    thin=form["thin"],
                )
                if refresh_result is not None:
                    result["refresh"] = refresh_result
                result["llm_chat"] = []
                if form["llm_judge"]:
                    result["llm_judgment"] = generate_slate_judgment(
                        result,
                        model=form["llm_model"],
                        reasoning_effort=form["llm_reasoning_effort"],
                    )
                LATEST_RESULT_RAW = result
                LATEST_RESULT_VIEW = _normalize_result_for_template(result)
                result = LATEST_RESULT_VIEW
        except Exception as exc:
            traceback.print_exc()
            error = str(exc)

        LATEST_FORM = dict(form)

    return render_template("index.html", form=form, result=result, error=error)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
