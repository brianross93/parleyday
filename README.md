# parleyday

Local Flask dashboard for live MLB/NBA parlay state search using Kalshi market data, MLB Stats API inputs, and a backtest framework for residual-model experiments.

## Run locally

```powershell
pip install -r requirements.txt
python dashboard_app.py
```

Then open `http://127.0.0.1:5000`.

## Deploy

This repo includes:

- `wsgi.py`
- `Procfile`
- `render.yaml`

so it can be deployed on a simple Python host like Render or exposed from a local machine with a tunnel.
