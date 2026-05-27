---
title: Tennis Value Bet
emoji: 🎾
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.57.0
app_file: app/dashboard.py
pinned: false
license: cc-by-nc-sa-4.0
---

Personal, non-commercial tennis value-betting dashboard.

## Data attribution

Match data: **Jeff Sackmann's tennis_atp / tennis_wta datasets**
(https://github.com/JeffSackmann), licensed CC BY-NC-SA 4.0. This repository
redistributes a derived SQLite of the same data under the same license.

Live odds: The Odds API (the-odds-api.com), personal/non-commercial tier.

## Deployment secrets

Set as Streamlit Community Cloud secrets (Manage app → Settings → Secrets,
TOML format):

```toml
ODDS_API_KEY = "your-odds-api-key"
APP_PASSWORD = "your-app-password"
```

When `APP_PASSWORD` is unset the dashboard runs unprotected (local use).
