# LSCO Blood Logistics Simulation

Spatiotemporal simulation of Role II blood logistics across 5-node FRSD network.  
McWhirter DODTR 2024 casualty registry · Bellamy 1984 sepsis model · 30-day LSCO horizon.

## Local Development

```bash
pip install -r requirements.txt
python server.py
# Open http://localhost:5000
```

## Deploy to Railway

1. Push this repo to GitHub
2. Create new project on [railway.app](https://railway.app) → "Deploy from GitHub repo"
3. Set environment variable in Railway dashboard:
   - `ACCESS_TOKEN` = any strong random string (e.g. output of `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
4. Railway auto-deploys on every push to main

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PORT` | Auto-set by Railway | Server port |
| `ACCESS_TOKEN` | Recommended | Token for `/run` and `/health` endpoints |

## Architecture

```
GitHub repo
    └── Railway (nixpacks build)
            ├── server.py          — Flask API + static file serving
            ├── simulation_tiered.py — Deterministic sim engine
            └── static/
                    └── index.html — Single-file React UI
```

## References

- McWhirter DL et al. DODTR 2024 (n=15,581) — casualty tier distribution
- Bellamy RF. Mil Med. 1984;149(2):55-62 — sepsis/MOF mortality curve
- Seamons et al. — original 5-node LSCO network geometry
