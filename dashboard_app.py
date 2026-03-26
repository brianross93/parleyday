import os
from datetime import datetime

from flask import Flask, render_template, request

from quantum_parlay_oracle import run_oracle


app = Flask(__name__)


def default_form() -> dict:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sport": "mlb",
        "slate_mode": "auto",
        "score_source": "implied",
        "kalshi_pages": 25,
        "fallback": False,
        "samples_per_beta": 250,
        "warmup": 50,
        "thin": 3,
        "bytes": 65536,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    form = default_form()
    result = None
    error = None

    if request.method == "POST":
        form = {
            "date": request.form.get("date", form["date"]),
            "sport": request.form.get("sport", form["sport"]),
            "slate_mode": request.form.get("slate_mode", form["slate_mode"]),
            "score_source": request.form.get("score_source", form["score_source"]),
            "kalshi_pages": int(request.form.get("kalshi_pages", form["kalshi_pages"])),
            "fallback": request.form.get("fallback") == "on",
            "samples_per_beta": int(request.form.get("samples_per_beta", form["samples_per_beta"])),
            "warmup": int(request.form.get("warmup", form["warmup"])),
            "thin": int(request.form.get("thin", form["thin"])),
            "bytes": int(request.form.get("bytes", form["bytes"])),
        }

        try:
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
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", form=form, result=result, error=error)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
