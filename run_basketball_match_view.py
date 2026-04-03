from __future__ import annotations

import argparse
import os
import threading
import time
import webbrowser

from dashboard_app import app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the local basketball match view on top of the basketball simulator."
    )
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local server without opening the browser automatically.",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/basketball-match-view"
    print(f"Starting basketball match view at {url}")
    print("Main dashboard remains available at " f"http://{args.host}:{args.port}/")

    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, host=args.host, port=args.port)


def _open_browser(url: str) -> None:
    time.sleep(0.8)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
