import os
import tempfile
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request

from basketball_viewer import build_possession_view_payload, default_viewer_form
from data_pipeline import DEFAULT_DB_PATH
from dfs_results import save_dfs_build
from dfs_strategy import build_mlb_contest_lineups, build_nba_contest_lineups
from env_config import load_local_env

load_local_env()

from llm_judge import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_WEB_SEARCH,
    generate_intuition_theses,
    generate_chat_reply,
    generate_thesis_judgment,
    generate_verified_theses,
    generate_slate_judgment,
)
from refresh_slate import refresh_slate
from quantum_parlay_oracle import run_oracle


app = Flask(__name__)
LATEST_RESULT_RAW: dict | None = None
LATEST_RESULT_VIEW: dict | None = None
LATEST_FORM: dict | None = None
CHAT_HISTORY: list[dict] = []
LATEST_DFS_RESULT: dict | None = None
LATEST_DFS_FORM: dict | None = None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def default_form() -> dict:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sport": "mlb",
        "slate_mode": "auto",
        "score_source": "implied",
        "props_only": False,
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
        "llm_web_search": DEFAULT_WEB_SEARCH,
    }


def default_dfs_form() -> dict:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sport": "nba",
        "salary_csv_path": r"C:\Users\brssn\Downloads\DKSalaries.csv",
        "contest_type": "head_to_head",
        "uploaded_csv_name": "",
    }


def _normalize_leg_for_template(leg: dict | None) -> dict | None:
    if leg is None:
        return None
    normalized = dict(leg)
    normalized.setdefault("implied_prob", 0.0)
    normalized.setdefault("score_delta", 0.0)
    normalized.setdefault("trust_score", 0.0)
    normalized.setdefault("entry_price", normalized.get("implied_prob", 0.0))
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
    normalized.setdefault("web_search_enabled", False)
    normalized.setdefault("sources", [])
    return normalized


def _normalize_thesis_judgment_for_template(judgment: dict | None) -> dict | None:
    normalized = _normalize_judgment_for_template(judgment)
    if normalized is None:
        return None
    normalized["ranked_theses"] = []
    for item in judgment.get("ranked_theses", []) if judgment else []:
        merged = dict(item)
        merged["quote_comparison"] = dict(item.get("quote_comparison") or {})
        merged["best_candidate"] = (
            {
                **item.get("best_candidate", {}),
                "legs": [_normalize_leg_for_template(leg) for leg in (item.get("best_candidate", {}) or {}).get("legs", [])],
            }
            if item.get("best_candidate")
            else None
        )
        if merged["verdict"] == "pass" and not merged["best_candidate"]:
            continue
        normalized["ranked_theses"].append(merged)
    normalized.setdefault("call", "")
    normalized.setdefault("portfolio_note", "")
    return normalized


def _build_quote_comparison(candidate: dict | None, quoted_payout: float) -> dict | None:
    if not candidate or quoted_payout <= 0:
        return None
    synthetic_payout = _safe_float(candidate.get("payout_estimate"))
    model_joint_prob = _safe_float(candidate.get("model_joint_prob"))
    market_joint_prob = _safe_float(candidate.get("market_joint_prob"))
    model_fair_payout = (1.0 / model_joint_prob) if model_joint_prob > 0 else None
    synthetic_gap_ratio = (quoted_payout / synthetic_payout) if synthetic_payout > 0 else None
    model_gap_ratio = (quoted_payout / model_fair_payout) if model_fair_payout and model_fair_payout > 0 else None
    market_implied_gap_ratio = (quoted_payout * market_joint_prob) if market_joint_prob > 0 else None
    return {
        "quoted_payout": quoted_payout,
        "synthetic_payout": synthetic_payout if synthetic_payout > 0 else None,
        "model_fair_payout": model_fair_payout,
        "market_joint_prob": market_joint_prob if market_joint_prob > 0 else None,
        "model_joint_prob": model_joint_prob if model_joint_prob > 0 else None,
        "quote_vs_synthetic_ratio": synthetic_gap_ratio,
        "quote_vs_model_fair_ratio": model_gap_ratio,
        "quote_vs_market_joint_multiple": market_implied_gap_ratio,
    }


def _apply_quote_comparison(result: dict, thesis_id: str, quoted_payout: float) -> bool:
    thesis_judgment = result.get("thesis_judgment")
    if not thesis_judgment:
        return False
    updated = False
    for item in thesis_judgment.get("ranked_theses", []) or []:
        if str(item.get("thesis_id")) != thesis_id:
            continue
        comparison = _build_quote_comparison(item.get("best_candidate"), quoted_payout)
        if comparison is None:
            return False
        item["quote_comparison"] = comparison
        updated = True
        break
    return updated


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
    normalized["thesis_judgment"] = _normalize_thesis_judgment_for_template(result.get("thesis_judgment"))
    normalized["llm_chat"] = _normalize_chat_history_for_template(result.get("llm_chat"))
    normalized["theses"] = result.get("theses", []) or []
    normalized["thesis_candidates"] = result.get("thesis_candidates", []) or []
    normalized["intuition_theses"] = result.get("intuition_theses", []) or []
    normalized["intuition_thesis_meta"] = _normalize_judgment_for_template(result.get("intuition_thesis_meta"))
    normalized["thesis_verification_meta"] = _normalize_judgment_for_template(result.get("thesis_verification_meta"))
    return normalized


def _normalize_dfs_result_for_template(result) -> dict | None:
    if result is None:
        return None
    if is_dataclass(result):
        result = asdict(result)
    normalized = dict(result)
    normalized["focus_players"] = list(normalized.get("focus_players") or [])
    normalized["fade_players"] = list(normalized.get("fade_players") or [])
    normalized["build_reasons"] = list(normalized.get("build_reasons") or [])
    normalized["lineups"] = list(normalized.get("lineups") or [])
    normalized["lineup_cards"] = list(normalized.get("lineup_cards") or [])
    normalized["lineup_families"] = list(normalized.get("lineup_families") or [])
    normalized["game_boosts"] = dict(normalized.get("game_boosts") or {})
    return normalized


def _apply_thesis_verification(result: dict, verification_result: dict | None) -> None:
    if not verification_result:
        return
    updates = {
        item.get("thesis_id"): item
        for item in (verification_result.get("verified_theses") or [])
        if item.get("thesis_id")
    }
    for bucket_name in ("theses", "intuition_theses"):
        updated_bucket = []
        for thesis in result.get(bucket_name, []) or []:
            merged = dict(thesis)
            update = updates.get(thesis.get("thesis_id"))
            if update:
                merged["verification_status"] = update.get("verification_status", "unverified")
                merged["verified_confidence"] = update.get("updated_confidence", thesis.get("confidence"))
                merged["verification_notes"] = update.get("verification_notes", [])
                merged["verification_sources"] = update.get("sources", [])
            else:
                merged.setdefault("verification_status", "unverified")
                merged.setdefault("verified_confidence", thesis.get("confidence"))
                merged.setdefault("verification_notes", [])
                merged.setdefault("verification_sources", [])
            updated_bucket.append(merged)
        result[bucket_name] = updated_bucket


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
            "props_only": request.form.get("props_only") == "on",
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
            "llm_web_search": request.form.get("llm_web_search") == "on",
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
                        web_search=form["llm_web_search"],
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
            elif action == "compare_quote":
                if LATEST_RESULT_RAW is None:
                    error = "Run a slate first so the quote comparison has something to attach to."
                else:
                    thesis_id = request.form.get("thesis_id", "").strip()
                    quoted_payout = _safe_float(request.form.get("quoted_payout"), 0.0)
                    if not thesis_id:
                        error = "Missing thesis id for quote comparison."
                    elif quoted_payout <= 0:
                        error = "Enter the observed Kalshi combo payout as a positive number, like 333."
                    elif not _apply_quote_comparison(LATEST_RESULT_RAW, thesis_id, quoted_payout):
                        error = "Could not attach that quote to the current thesis board entry."
                    else:
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
                    props_only=form["props_only"],
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
                    intuition_result = generate_intuition_theses(
                        result,
                        model=form["llm_model"],
                        reasoning_effort=form["llm_reasoning_effort"],
                        web_search=form["llm_web_search"],
                    )
                    result["intuition_theses"] = intuition_result.get("theses", [])
                    result["intuition_thesis_meta"] = intuition_result
                    verification_result = generate_verified_theses(
                        result,
                        model=form["llm_model"],
                        reasoning_effort=form["llm_reasoning_effort"],
                        web_search=form["llm_web_search"],
                    )
                    result["thesis_verification_meta"] = verification_result
                    _apply_thesis_verification(result, verification_result)
                    result["thesis_judgment"] = generate_thesis_judgment(
                        result,
                        model=form["llm_model"],
                        reasoning_effort=form["llm_reasoning_effort"],
                        web_search=form["llm_web_search"],
                    )
                    result["llm_judgment"] = generate_slate_judgment(
                        result,
                        model=form["llm_model"],
                        reasoning_effort=form["llm_reasoning_effort"],
                        web_search=form["llm_web_search"],
                    )
                else:
                    result["intuition_theses"] = []
                    result["intuition_thesis_meta"] = None
                    result["thesis_verification_meta"] = None
                    result["thesis_judgment"] = None
                LATEST_RESULT_RAW = result
                LATEST_RESULT_VIEW = _normalize_result_for_template(result)
                result = LATEST_RESULT_VIEW
        except Exception as exc:
            traceback.print_exc()
            error = str(exc)

        LATEST_FORM = dict(form)

    return render_template("index.html", form=form, result=result, error=error)


@app.route("/basketball-viewer", methods=["GET", "POST"])
@app.route("/basketball-match-view", methods=["GET", "POST"])
def basketball_viewer():
    form = default_viewer_form()
    error = None
    payload = None

    if request.method == "POST":
        form = {
            "view_mode": request.form.get("view_mode", form["view_mode"]),
            "data_mode": request.form.get("data_mode", form["data_mode"]),
            "date": request.form.get("date", form["date"]),
            "matchup": request.form.get("matchup", form["matchup"]),
            "csv_path": request.form.get("csv_path", form["csv_path"]),
            "seed": int(request.form.get("seed", form["seed"])),
            "play_family": request.form.get("play_family", form["play_family"]),
            "coverage": request.form.get("coverage", form["coverage"]),
            "entry_type": request.form.get("entry_type", form["entry_type"]),
            "entry_source": request.form.get("entry_source", form["entry_source"]),
            "offense_team": request.form.get("offense_team", form["offense_team"]),
        }

    try:
        payload = build_possession_view_payload(
            view_mode=str(form["view_mode"]),
            data_mode=str(form["data_mode"]),
            date=str(form["date"]),
            matchup=str(form["matchup"]),
            csv_path=str(form["csv_path"]),
            seed=int(form["seed"]),
            play_family=str(form["play_family"]),
            coverage=str(form["coverage"]),
            entry_type=str(form["entry_type"]),
            entry_source=str(form["entry_source"]),
            offense_team=str(form["offense_team"]),
        )
    except Exception as exc:
        traceback.print_exc()
        error = str(exc)

    return render_template("basketball_viewer.html", form=form, payload=payload, error=error)


@app.route("/dfs", methods=["GET", "POST"])
def dfs():
    global LATEST_DFS_FORM, LATEST_DFS_RESULT

    form = dict(LATEST_DFS_FORM or default_dfs_form())
    result = LATEST_DFS_RESULT
    error = None

    if request.method == "POST":
        form = {
            "date": request.form.get("date", form["date"]),
            "sport": request.form.get("sport", form["sport"]).strip() or form["sport"],
            "salary_csv_path": request.form.get("salary_csv_path", form["salary_csv_path"]).strip() or form["salary_csv_path"],
            "contest_type": request.form.get("contest_type", form["contest_type"]).strip() or form["contest_type"],
            "uploaded_csv_name": "",
        }
        try:
            salary_csv_path = form["salary_csv_path"]
            uploaded_file = request.files.get("salary_csv_upload")
            if uploaded_file and uploaded_file.filename:
                suffix = Path(uploaded_file.filename).suffix or ".csv"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    uploaded_file.save(handle.name)
                    salary_csv_path = handle.name
                form["uploaded_csv_name"] = uploaded_file.filename
            builder = build_nba_contest_lineups if form["sport"] == "nba" else build_mlb_contest_lineups
            dfs_result = builder(date_str=form["date"], salary_csv_path=salary_csv_path, contest_type=form["contest_type"])
            saved_build_id = save_dfs_build(
                slate_date=form["date"],
                result=dfs_result,
                salary_csv_path=salary_csv_path,
                input_label=form["uploaded_csv_name"] or os.path.basename(salary_csv_path),
            )
            result = _normalize_dfs_result_for_template(dfs_result)
            result["saved_build_id"] = saved_build_id
            result["input_csv_path"] = salary_csv_path
            result["uploaded_csv_name"] = form["uploaded_csv_name"]
            LATEST_DFS_RESULT = result
        except Exception as exc:
            traceback.print_exc()
            error = str(exc)
        LATEST_DFS_FORM = dict(form)

    return render_template("dfs.html", form=form, result=result, error=error)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
