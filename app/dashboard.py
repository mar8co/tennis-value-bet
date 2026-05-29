"""Streamlit dashboard: today's tennis matches + value-bet analysis.

    streamlit run app/dashboard.py

The serve/return model is wired to live matches from The Odds API; serve
parameters come from Sackmann data via name matching; recalibrations
validated on the historical backtests are applied by default. Password
gate active behind `APP_PASSWORD`.
"""
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _LOCAL_TZ = ZoneInfo("Europe/Rome")
except Exception:
    _LOCAL_TZ = timezone.utc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from tvb import config
from tvb.calibration import apply_logistic, apply_temperature
from tvb.elo import compute_elo, win_probability
from tvb.ingest import read_matches
from tvb.odds_api import fetch_tennis_odds, match_context
from tvb.pipeline import evaluate_match, rank_value_bets
from tvb.ratings import find_player_id, matchup_probs, player_names
from tvb.serve_return import player_serve_return
from tvb.simulator import monte_carlo
try:
    from tvb.bet_tracker import (accuracy_by_player, equity_curve, get_bets_df,
                                  get_pending_bets, log_bet, performance_stats,
                                  update_results)
    _TRACKER_OK = True
    _TRACKER_ERR = ""
except Exception as _e:
    _TRACKER_OK = False
    _TRACKER_ERR = str(_e)

    def _noop(*a, **kw):  # stub so the rest of the file doesn't NameError
        return False
    log_bet = _noop

    def accuracy_by_player(tour=""): import pandas as pd; return pd.DataFrame()
    def equity_curve(tour=""): import pandas as pd; return pd.DataFrame()
    def get_bets_df(tour=""): import pandas as pd; return pd.DataFrame()
    def get_pending_bets(tour=""): import pandas as pd; return pd.DataFrame()
    def performance_stats(tour=""): return {"n_pending":0,"n_resolved":0,"n_won":0,"n_lost":0,"win_rate":0.0,"total_staked":0.0,"total_profit":0.0,"roi":0.0}
    def update_results(key): return 0

# These were added later; import separately so a stale cache on the above
# functions doesn't break the entire tracker.
try:
    from tvb.bet_tracker import (manual_resolve_bet, resolve_match_manual,
                                  scores_debug, get_pending_matches)
except Exception:
    def manual_resolve_bet(bet_id, result): return False  # type: ignore[misc]
    def resolve_match_manual(match_id, winner, total_games=None): return 0  # type: ignore[misc]
    def scores_debug(api_key): return []  # type: ignore[misc]
    def get_pending_matches(tour=""): import pandas as pd; return pd.DataFrame()  # type: ignore[misc]

try:
    from tvb.bet_tracker import update_from_sackmann, clear_sackmann_cache
except Exception:
    def update_from_sackmann(): return 0  # type: ignore[misc]
    def clear_sackmann_cache(): pass  # type: ignore[misc]

try:
    from tvb.bet_tracker import update_from_rapidapi
except Exception:
    def update_from_rapidapi(k): return 0  # type: ignore[misc]

try:
    from tvb.bet_tracker import update_from_espn
except Exception:
    def update_from_espn(): return 0  # type: ignore[misc]

try:
    from tvb.bet_tracker import void_unresolvable_bets
except Exception:
    def void_unresolvable_bets(): return 0  # type: ignore[misc]


try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False
    def _st_autorefresh(*a, **kw): return 0  # type: ignore[misc]

st.set_page_config(page_title="Tennis Value Bet", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html { font-size: 17px; }
    code, pre { font-family: ui-monospace, 'JetBrains Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _gate():
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("_auth_ok"):
        return
    st.title("🎾 Tennis Value Bet")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Entra")
    if ok:
        if pw == expected:
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Password errata.")
    st.stop()


_gate()


# ------------------------------------------- cached data + simulation
@st.cache_data(show_spinner="Carico dati Sackmann + Elo...")
def load_base(tour: str):
    matches = read_matches(tour)
    return matches, player_names(matches), compute_elo(matches, by_surface=True)


@st.cache_data(show_spinner="Carico statistiche servizio/risposta...")
def load_serve_return(tour: str, surface: str):
    matches, _, _ = load_base(tour)
    return player_serve_return(matches[matches["surface"] == surface])


@st.cache_data(show_spinner="Simulazione in corso...")
def _simulate(p0: float, p1: float, best_of: int, n_sims: int):
    return monte_carlo(p0, p1, best_of=best_of, n_sims=n_sims)


# -------------------------------------------- live-match resolution
def _resolve_live_params(tour: str, surface: str, name0: str, name1: str):
    fallback = (0.63, 0.63, None)
    try:
        _, names, elo = load_base(tour)
        sr = load_serve_return(tour, surface)
    except Exception:
        return (*fallback, f"⚠ Dati {tour.upper()} non disponibili.")
    id0 = find_player_id(names, name0)
    id1 = find_player_id(names, name1)
    missing = [n for n, i in ((name0, id0), (name1, id1)) if i is None]
    if missing:
        return (*fallback, "⚠ Non trovati nei dati Sackmann: "
                + ", ".join(missing))
    try:
        p0, p1 = matchup_probs(sr, id0, id1, tour)
    except KeyError:
        return (*fallback,
                f"⚠ Statistiche servizio su {surface} mancanti.")

    def _elo(pid):
        e = elo[(elo["player_id"] == pid) & (elo["surface"] == surface)]
        return float(e["elo"].iloc[0]) if not e.empty else config.ELO_BASE

    return (p0, p1, win_probability(_elo(id0), _elo(id1)),
            f"✓ Parametri dai dati Sackmann ({surface})")


# ------------------------------------------- API key + live fetch
def _parse_dt(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _match_label(m) -> str:
    dt = _parse_dt(m.commence_time)
    if dt is None:
        return f"{m.player1} vs {m.player2} - — ⏳"
    if dt <= datetime.now(timezone.utc):
        return f"{m.player1} vs {m.player2} - LIVE 🟥"
    local = dt.astimezone(_LOCAL_TZ).strftime("%H:%M")
    return f"{m.player1} vs {m.player2} - {local} 🏁"


def _live_score_url(name0: str, name1: str) -> str:
    q = urllib.parse.quote_plus(f"{name0} vs {name1} live score")
    return f"https://www.google.com/search?q={q}"


def _api_key():
    return (st.session_state.get("_api_key_override")
            or os.environ.get("ODDS_API_KEY", ""))


def _apply_live_odds():
    m = (st.session_state.get("live_matches") or {}).get(
        st.session_state.get("live_sel"))
    if not m:
        return

    def c(x):
        return round(min(max(x, 1.01), 50.0), 2)

    st.session_state["mw0"] = c(m.odds1)
    st.session_state["mw1"] = c(m.odds2)
    _, _, bo = match_context(m.sport_key)
    st.session_state["best_of"] = bo
    if m.total_line is not None:
        st.session_state["ltg"] = round(min(max(m.total_line, 10.0), 40.0), 1)
        st.session_state["ov"] = c(m.over_odds)
        st.session_state["un"] = c(m.under_odds)
    if m.hcap_line is not None:
        st.session_state["lh"] = round(min(max(m.hcap_line, -12.0), 12.0), 1)
        st.session_state["h0"] = c(m.hcap_odds1)
        st.session_state["h1"] = c(m.hcap_odds2)


def _auto_log_all_matches() -> int:
    """Scan every live match and log positive-EV bets to the tracker.

    Uses calibrated thresholds (config.MIN_EDGE) without a min_prob filter
    so Over/Under and Handicap bets are captured for accuracy tracking.
    Each match is processed once per session; INSERT OR IGNORE in log_bet
    handles deduplication across sessions.
    """
    matches = st.session_state.get("live_matches") or {}
    if not matches:
        return 0
    scanned = st.session_state.setdefault("_scanned_match_ids", set())
    new_ids = set(matches.keys()) - scanned
    if not new_ids:
        return 0
    total = 0
    for mid in new_ids:
        m = matches[mid]
        tour, surface, bo_match = match_context(m.sport_key)
        name0, name1 = m.player1, m.player2
        p0, p1, _, status = _resolve_live_params(tour, surface, name0, name1)
        if not status.startswith("✓"):
            scanned.add(mid)   # skip: no Sackmann data, fallback probs are unreliable
            continue
        book = _simulate(float(p0), float(p1), bo_match, config.N_SIMS)
        mkts: dict = {"Match winner": {"selections": [
            {"label": name0, "odds": m.odds1,
             "model": lambda b: float(apply_temperature(
                 b.p_match_winner(0), config.SR_TEMPERATURE))},
            {"label": name1, "odds": m.odds2,
             "model": lambda b: float(apply_temperature(
                 b.p_match_winner(1), config.SR_TEMPERATURE))},
        ]}}
        if m.total_line is not None and m.over_odds and m.under_odds:
            tg = m.total_line
            mkts["Total games"] = {"selections": [
                {"label": f"Over {tg}", "odds": m.over_odds,
                 "model": lambda b: b.p_total_over(tg + config.SR_TOTAL_SHIFT)},
                {"label": f"Under {tg}", "odds": m.under_odds,
                 "model": lambda b: 1.0 - b.p_total_over(tg + config.SR_TOTAL_SHIFT)},
            ]}
        if m.hcap_line is not None and m.hcap_odds1 and m.hcap_odds2:
            hl = m.hcap_line
            mkts["Handicap games"] = {"selections": [
                {"label": f"{name0} {hl:+}", "odds": m.hcap_odds1,
                 "model": lambda b: float(apply_temperature(
                     b.p_handicap(0, hl), config.SR_HANDICAP_TEMPERATURE))},
                {"label": f"{name1} {-hl:+}", "odds": m.hcap_odds2,
                 "model": lambda b: float(apply_temperature(
                     b.p_handicap(1, -hl), config.SR_HANDICAP_TEMPERATURE))},
            ]}
        evl = evaluate_match(f"{name0} vs {name1}", book, mkts)
        for vb in rank_value_bets(evl, min_edge=config.MIN_EDGE, min_prob=0.0):
            if log_bet(player1=name0, player2=name1,
                       commence_time=m.commence_time, sport_key=m.sport_key,
                       market=vb.market, selection=vb.selection,
                       odds=vb.odds, model_prob=vb.model_prob,
                       edge=vb.edge, ev=vb.ev, kelly=vb.kelly,
                       stake=10.0):
                total += 1
        scanned.add(mid)
    return total


def _fetch_matches():
    key = _api_key()
    if not key:
        st.session_state["_fetch_error"] = (
            "Chiave API non configurata. Impostala come secret "
            "`ODDS_API_KEY` o inseriscila in Impostazioni → Chiave API.")
        st.session_state["live_matches"] = {}
        return
    try:
        found = fetch_tennis_odds(key)
        new_matches = {
            f"{m.player1}|{m.player2}|{m.commence_time}": m for m in found}
        prev_sel = st.session_state.get("live_sel")
        st.session_state["live_matches"] = new_matches
        st.session_state.pop("_fetch_error", None)
        if found:
            if prev_sel not in new_matches:
                st.session_state["live_sel"] = next(iter(new_matches))
            _apply_live_odds()
            _n = _auto_log_all_matches()
            if _n:
                st.toast(f"✅ {_n} nuove bet registrate da {len(new_matches)} match.")
    except Exception as exc:
        st.session_state.pop("live_matches", None)
        st.session_state["_fetch_error"] = f"Errore API: {exc}"


# ----------------------------------------- session-state defaults
for _k, _v in (("mw0", 1.55), ("mw1", 2.45), ("ltg", 22.5), ("ov", 1.92),
               ("un", 1.88), ("lh", -3.5), ("h0", 1.90), ("h1", 1.90),
               ("s10", 1.62), ("s11", 2.30), ("tby", 2.40), ("tbn", 1.55),
               ("best_of", 3)):
    st.session_state.setdefault(_k, _v)

if "live_matches" not in st.session_state and _api_key():
    _fetch_matches()
elif "live_matches" not in st.session_state:
    st.session_state["live_matches"] = {}


# ------------------------------------------------------------------ title
st.title("🎾 Tennis Value Bet")
st.caption("Analisi value-bet sui match di oggi · modello di simulazione "
           "punto-su-punto vs quote bookmaker · progetto personale.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Impostazioni")

    best_of = st.radio("Formato match", [3, 5], horizontal=True,
                       key="best_of",
                       help="Si aggiorna automaticamente al torneo del match "
                            "live selezionato.")
    n_sims = st.select_slider("Simulazioni Monte Carlo",
                              options=[2000, 5000, 10000, 20000, 50000],
                              value=config.N_SIMS)
    min_edge = st.slider("Edge minimo per value bet", 0.0, 0.15,
                         config.MIN_EDGE, 0.01)
    min_prob = st.slider("Probabilità minima vittoria", 0.40, 0.80, 0.55,
                         0.05,
                         help="Filtra le giocate sotto questa probabilità "
                              "stimata. Valori alti = picks più sicuri, "
                              "meno volatili.")
    recalibrate = st.checkbox(
        "Ricalibrazione validata", value=True,
        help=f"Correzioni dai backtest ATP — Match winner T="
             f"{config.SR_TEMPERATURE}, Total games δ={config.SR_TOTAL_SHIFT}, "
             f"Handicap T={config.SR_HANDICAP_TEMPERATURE}, Tie-break "
             f"logistica.")

    st.divider()
    with st.expander("Chiave API personale (avanzato)"):
        st.caption("L'app normalmente usa la chiave configurata come secret. "
                   "Inseriscine una qui solo se vuoi sovrascriverla.")
        ov = st.text_input("Chiave alternativa", type="password",
                           key="_api_key_override_input",
                           value=st.session_state.get(
                               "_api_key_override") or "")
        if st.button("Applica chiave e ricarica"):
            st.session_state["_api_key_override"] = ov.strip() or None
            st.session_state.pop("live_matches", None)
            st.rerun()


# ------------------------------------------------------------------ tabs
tab_analysis, tab_perf = st.tabs(["🎾 Analisi match", "📊 Performance tracker"])


# ================================================================ TAB ANALISI
with tab_analysis:

    # -------------------------------------------------------- match picker
    hl, hr = st.columns([5, 1])
    hl.subheader("📅 Partite di oggi")
    if hr.button("🔄 Aggiorna", use_container_width=True):
        with st.spinner("Aggiorno le quote..."):
            _fetch_matches()

    if st.session_state.get("_fetch_error"):
        st.error(st.session_state["_fetch_error"])

    live = st.session_state.get("live_matches") or {}
    if not live:
        st.info("Nessun match disponibile al momento. Apri le **Impostazioni** "
                "(barra a sinistra) per controllare la chiave API, oppure premi "
                "**Aggiorna** più tardi.")
    else:
        st.selectbox("Match", options=list(live.keys()), key="live_sel",
                     format_func=lambda k: _match_label(live[k]),
                     on_change=_apply_live_odds, label_visibility="collapsed")

        m = live[st.session_state["live_sel"]]
        tour, surface, bo_match = match_context(m.sport_key)
        name0, name1 = m.player1, m.player2
        p0, p1, elo_xcheck, status = _resolve_live_params(
            tour, surface, name0, name1)

        # ----------------------------------------------- match card
        with st.container(border=True):
            head, info = st.columns([3, 2])
            with head:
                st.markdown(f"### {name0}  vs  {name1}")
                dt_match = _parse_dt(m.commence_time)
                if dt_match is not None and dt_match <= datetime.now(timezone.utc):
                    when_html = ("<span style='color:#8A3E1A;font-weight:700'>"
                                 "🟥 LIVE</span>")
                elif dt_match is not None:
                    when_html = (
                        f"🏁 {dt_match.astimezone(_LOCAL_TZ).strftime('%H:%M')}")
                else:
                    when_html = "—"
                st.markdown(
                    f"<div style='opacity:.7; font-size:.95em'>{tour.upper()} · "
                    f"{surface} · best-of-{bo_match} · {when_html}</div>",
                    unsafe_allow_html=True)
            with info:
                (st.caption if status.startswith("✓") else st.warning)(status)
                a, b = st.columns(2)
                a.metric(f"{name0[:14]} — p", f"{p0 * 100:.1f}%")
                b.metric(f"{name1[:14]} — p", f"{p1 * 100:.1f}%")

        _score_url = _live_score_url(name0, name1)
        st.markdown(
            f'<a href="{_score_url}" target="_blank" rel="noopener" '
            f'style="color:#c1440e; font-weight:600; text-decoration:none; '
            f'font-size:.95em">Vedi punteggio live ↗</a>',
            unsafe_allow_html=True)

        if tour == "wta" and recalibrate:
            st.warning(
                "**WTA match**: le correzioni di ricalibrazione sono state "
                "validate su dati ATP. Su WTA sono un'approssimazione — "
                "considera di disattivarle (Impostazioni → Ricalibrazione "
                "validata).")

        # -------------------------------------------------------- quote section
        st.subheader("Quote bookmaker")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Match winner**")
            st.number_input(name0, 1.01, 50.0, key="mw0")
            st.number_input(name1, 1.01, 50.0, key="mw1")
        with c2:
            st.markdown("**Total games**")
            line_tg = st.number_input("Linea", 10.0, 40.0, step=0.5, key="ltg")
            st.number_input("Over", 1.01, 50.0, key="ov")
            st.number_input("Under", 1.01, 50.0, key="un")
        with c3:
            st.markdown("**Handicap games**")
            line_h = st.number_input(f"Linea {name0}", -12.0, 12.0, step=0.5,
                                     key="lh")
            st.number_input(f"{name0} ({line_h:+})", 1.01, 50.0, key="h0")
            st.number_input(f"{name1} ({-line_h:+})", 1.01, 50.0, key="h1")

        with st.expander(
                "Mercati a inserimento manuale (non disponibili via API)"):
            st.caption(
                "The Odds API non fornisce quote per **vincente 1° set** e "
                "**tie-break** sul tennis. Questi due mercati restano a "
                "inserimento manuale e i valori non cambiano automaticamente "
                "al cambio del match.")
            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown("**Vincente 1° set**")
                st.number_input(name0, 1.01, 50.0, key="s10")
                st.number_input(name1, 1.01, 50.0, key="s11")
            with cm2:
                st.markdown("**Tie-break**")
                st.number_input("Sì", 1.01, 50.0, key="tby")
                st.number_input("No", 1.01, 50.0, key="tbn")

        # ------------------------------------------- auto-analyze
        temp = config.SR_TEMPERATURE if recalibrate else 1.0
        tg_delta = config.SR_TOTAL_SHIFT if recalibrate else 0.0
        hcap_temp = config.SR_HANDICAP_TEMPERATURE if recalibrate else 1.0
        tb_a, tb_b = config.SR_TIEBREAK_LOGISTIC if recalibrate else (1.0, 0.0)

        book = _simulate(float(p0), float(p1), int(best_of), int(n_sims))

        s = st.session_state
        markets = {
            "Match winner": {"selections": [
                {"label": name0, "odds": s["mw0"],
                 "model": lambda b: float(apply_temperature(
                     b.p_match_winner(0), temp))},
                {"label": name1, "odds": s["mw1"],
                 "model": lambda b: float(apply_temperature(
                     b.p_match_winner(1), temp))},
            ]},
            "Total games": {"selections": [
                {"label": f"Over {line_tg}", "odds": s["ov"],
                 "model": lambda b: b.p_total_over(line_tg + tg_delta)},
                {"label": f"Under {line_tg}", "odds": s["un"],
                 "model": lambda b: 1.0 - b.p_total_over(line_tg + tg_delta)},
            ]},
            "Handicap games": {"selections": [
                {"label": f"{name0} {line_h:+}", "odds": s["h0"],
                 "model": lambda b: float(apply_temperature(
                     b.p_handicap(0, line_h), hcap_temp))},
                {"label": f"{name1} {-line_h:+}", "odds": s["h1"],
                 "model": lambda b: float(apply_temperature(
                     b.p_handicap(1, -line_h), hcap_temp))},
            ]},
            "Vincente 1° set": {"selections": [
                {"label": name0, "odds": s["s10"],
                 "model": lambda b: b.p_set1_winner(0)},
                {"label": name1, "odds": s["s11"],
                 "model": lambda b: b.p_set1_winner(1)},
            ]},
            "Tie-break": {"selections": [
                {"label": "Sì", "odds": s["tby"],
                 "model": lambda b: float(apply_logistic(
                     b.p_tiebreak_yes(), tb_a, tb_b))},
                {"label": "No", "odds": s["tbn"],
                 "model": lambda b: 1.0 - float(apply_logistic(
                     b.p_tiebreak_yes(), tb_a, tb_b))},
            ]},
        }

        match_name = f"{name0} vs {name1}"
        bets = evaluate_match(match_name, book, markets)
        ranked = rank_value_bets(bets, min_edge=min_edge, min_prob=min_prob)

        # ----------------------- auto-log value bets to tracker
        for vb in ranked:
            log_bet(
                player1=name0, player2=name1,
                commence_time=m.commence_time,
                sport_key=m.sport_key,
                market=vb.market, selection=vb.selection,
                odds=vb.odds, model_prob=vb.model_prob,
                edge=vb.edge, ev=vb.ev, kelly=vb.kelly,
                stake=10.0)

        # --------------------------------------------------------------- top picks
        st.subheader(f"💎 Migliori value bet — {len(ranked)} sopra soglia")
        if not ranked:
            st.info("Nessuna value bet sopra la soglia di edge impostata "
                    "(prova ad abbassarla in Impostazioni).")
        else:
            st.caption("Ordinate per Kelly (combina edge e probabilità di vittoria) "
                       "— in alto le giocate più fattibili. Le prime 3 in evidenza, "
                       "le altre sotto.")
            BADGES = {
                "alta":  ("🟢", "ALTA",  "#5A8C50"),
                "media": ("🔵", "MEDIA", "#2563eb"),
                "bassa": ("⚪", "BASSA", "#6b7280"),
            }
            top_n = min(3, len(ranked))
            for i, vb in enumerate(ranked[:top_n], 1):
                with st.container(border=True):
                    head, ee, ev, cf = st.columns([4, 1, 1, 1])
                    head.markdown(
                        f"#### #{i} · {vb.market} — {vb.selection}  "
                        f"@{vb.odds:.2f}"
                        f"  \nModello **{vb.model_prob * 100:.1f}%**  ·  "
                        f"mercato {vb.market_prob * 100:.1f}%")
                    ee.metric("Edge", f"{vb.edge * 100:+.1f}%")
                    ev.metric("EV", f"{vb.ev * 100:+.1f}%")
                    emoji, lbl, color = BADGES.get(vb.confidence,
                                                   ("⚪", "—", "#6b7280"))
                    cf.markdown(
                        f"<div style='font-size:1.8em; text-align:center; "
                        f"line-height:1'>{emoji}</div>"
                        f"<div style='color:{color}; font-weight:700; "
                        f"text-align:center; margin-top:.2em'>{lbl}</div>"
                        f"<div style='font-size:.85em; text-align:center; "
                        f"opacity:.8; margin-top:.3em'>Kelly "
                        f"{vb.kelly * 100:.1f}%</div>",
                        unsafe_allow_html=True)

            if len(ranked) > top_n:
                st.markdown("**Altre value bet**")
                for i, vb in enumerate(ranked[top_n:], top_n + 1):
                    em = BADGES.get(vb.confidence, ("⚪",))[0]
                    st.markdown(
                        f"{em} **{i}. {vb.market} — {vb.selection}**  "
                        f"@{vb.odds:.2f}  ·  EV {vb.ev * 100:+.1f}%  ·  "
                        f"edge {vb.edge * 100:+.1f}%  ·  Kelly "
                        f"{vb.kelly * 100:.1f}%")

        # --------------------------------------------- diagnostics + full table
        with st.expander("Diagnostica modello"):
            p_mw0_raw = book.p_match_winner(0)
            p_mw0 = float(apply_temperature(p_mw0_raw, temp))
            games_raw = book.total_games.mean()
            p_tb_raw = book.p_tiebreak_yes()
            p_tb = float(apply_logistic(p_tb_raw, tb_a, tb_b))
            n_cols = 4 if elo_xcheck is not None else 3
            cols = st.columns(n_cols)
            cols[0].metric(f"Vittoria {name0[:14]} (modello)",
                           f"{p_mw0 * 100:.1f}%",
                           help=f"Grezza: {p_mw0_raw * 100:.1f}%")
            cols[1].metric("Media game totali",
                           f"{games_raw - tg_delta:.1f}",
                           help=f"Grezza: {games_raw:.1f}")
            cols[2].metric("Tie-break (almeno 1)",
                           f"{p_tb * 100:.1f}%",
                           help=f"Grezza: {p_tb_raw * 100:.1f}%")
            if elo_xcheck is not None:
                cols[3].metric(f"Vittoria {name0[:14]} (Elo)",
                               f"{elo_xcheck * 100:.1f}%",
                               help="Controprova indipendente dal modello a punti.")

        with st.expander("Tutti i mercati analizzati"):
            st.dataframe([vars(b) for b in bets], width="stretch")


# ============================================================ TAB PERFORMANCE
with tab_perf:
    st.subheader("📊 Performance tracker")
    st.caption(
        "**Tutti** i match disponibili vengono analizzati automaticamente ad "
        "ogni aggiornamento delle quote: ogni value bet con edge ≥ 3% e EV "
        "positivo viene registrata con stake simulato di **€10** (Match winner, "
        "Total games e Handicap). I risultati vengono aggiornati via "
        "The Odds API Scores.")

    if not _TRACKER_OK:
        st.error(f"⚠️ Tracker non disponibile — errore di importazione:\n\n`{_TRACKER_ERR}`")
        st.stop()

    if not os.environ.get("DATABASE_URL"):
        st.info(
            "💾 **Storico locale** · Su Streamlit Cloud i dati si perdono ad ogni "
            "redeploy. Per uno storico permanente aggiungi `DATABASE_URL` nei "
            "*Secrets* dell'app (es. Supabase o Neon — entrambi gratuiti).")

    _now = time.time()
    _key = _api_key()

    # Ogni ora: Sackmann GitHub (gratuito, nessuna quota consumata)
    _last_sack = st.session_state.get("_last_sackmann_check", 0)
    if (_now - _last_sack) > 3600:
        _sack_n = update_from_sackmann()
        st.session_state["_last_sackmann_check"] = _now
        if _sack_n:
            st.toast(f"✅ {_sack_n} risultat{'o' if _sack_n == 1 else 'i'} "
                     f"aggiornati automaticamente.")

    # ---- manual refresh button
    if st.button("🔄 Aggiorna risultati ora"):
        _total = 0
        _msgs = []
        with st.spinner("Verifico risultati..."):
            try:
                _n_espn = update_from_espn()
                _msgs.append(f"ESPN: {_n_espn} risolti")
                _total += _n_espn
            except Exception as _e:
                _msgs.append(f"ESPN errore: {_e}")
            if _key:
                try:
                    _n_odds = update_results(_key, days_from=3)
                    _msgs.append(f"Odds API: {_n_odds} risolti")
                    _total += _n_odds
                except Exception as _e:
                    _msgs.append(f"Odds API errore: {_e}")
            try:
                _n_void = void_unresolvable_bets()
                if _n_void:
                    _msgs.append(f"Annullati: {_n_void} (mercati non risolvibili)")
            except Exception as _e:
                _msgs.append(f"Void errore: {_e}")
            st.session_state["_last_results_check"] = time.time()
        st.session_state["_update_msg"] = " | ".join(_msgs)
        st.session_state["_update_total"] = _total
        st.rerun()

    if "_update_msg" in st.session_state:
        st.info(st.session_state["_update_msg"])
        if st.session_state.get("_update_total", 0):
            st.success(f"✅ Risolti {st.session_state['_update_total']} nuovi risultati.")

    _last_upd = max(st.session_state.get("_last_sackmann_check", 0),
                    st.session_state.get("_last_results_check", 0))
    if _last_upd:
        st.caption(f"Ultimo aggiornamento: "
                   f"{datetime.fromtimestamp(_last_upd, tz=_LOCAL_TZ).strftime('%H:%M:%S')}")

    # ---- audit duplicati
    with st.expander("🔎 Audit dati — cerca duplicati", expanded=False):
        st.caption("Verifica se la stessa partita è stata loggata più volte con match_id diversi.")
        if st.button("Analizza duplicati nel DB"):
            from tvb.bet_tracker import _read as _bt_read, _engine, _is_pg
            # substr(commence_time,1,10) funziona sia su SQLite che PostgreSQL
            _dup = _bt_read("""
                SELECT player1, player2,
                       substr(commence_time, 1, 10) as match_date,
                       market, selection,
                       COUNT(DISTINCT match_id) as n_match_ids,
                       COUNT(*) as n_bets,
                       SUM(CASE WHEN result='won' THEN 1 ELSE 0 END) as n_won
                FROM bets
                GROUP BY player1, player2, substr(commence_time, 1, 10), market, selection
                HAVING COUNT(DISTINCT match_id) > 1
                ORDER BY n_match_ids DESC
            """)
            if _dup.empty:
                st.success("✅ Nessun duplicato trovato.")
            else:
                st.error(f"⚠️ Trovati {len(_dup)} gruppi con duplicati!")
                st.dataframe(_dup, use_container_width=True)
                st.session_state["_has_duplicates"] = True

        if st.session_state.get("_has_duplicates"):
            if st.button("🗑️ Elimina duplicati (mantieni solo il primo per ogni match)"):
                from tvb.bet_tracker import _engine as _eng
                from sqlalchemy import text as _sqlt
                with _eng().begin() as _conn:
                    _conn.execute(_sqlt("""
                        DELETE FROM bets WHERE id NOT IN (
                            SELECT MIN(id) FROM bets
                            GROUP BY player1, player2,
                                     substr(commence_time, 1, 10),
                                     market, selection
                        )
                    """))
                st.success("Duplicati eliminati.")
                st.session_state.pop("_has_duplicates", None)
                st.rerun()

    # ---- diagnostica API
    with st.expander("🔍 Diagnostica API", expanded=False):
        _rapi_key = os.environ.get("RAPIDAPI_KEY", "")
        st.write(f"ODDS_API_KEY: {'✅' if _key else '❌'}  |  RAPIDAPI_KEY: {'✅' if _rapi_key else '❌'}")

        if st.button("⚡ Risolvi da ESPN direttamente"):
            import requests as _req
            from tvb.bet_tracker import (_norm_name as _nn, _match_one_name as _mon,
                                          _resolve_match as _res, _read as _rd)
            _pend3 = _rd("SELECT DISTINCT player1, player2, match_id FROM bets WHERE result='pending'")
            # Fetch ESPN all dates from pending bets
            _espn_pairs: dict = {}
            _dates3 = set(str(ct)[:10] for ct in get_pending_matches()["commence_time"])
            _dates3.add(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            for _d3 in sorted(_dates3):
                for _tour3 in ("atp", "wta"):
                    _r3 = _req.get(
                        f"https://site.api.espn.com/apis/site/v2/sports/tennis/{_tour3}/scoreboard",
                        params={"dates": _d3.replace("-","")},
                        headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
                    if _r3.status_code != 200:
                        continue
                    for _ev3 in _r3.json().get("events",[]):
                        for _g3 in _ev3.get("groupings",[]):
                            for _c3 in _g3.get("competitions",[]):
                                _cx3 = _c3.get("competitors",[])
                                _w3 = next((c for c in _cx3 if c.get("winner")), None)
                                _l3 = next((c for c in _cx3 if not c.get("winner")), None)
                                if not _w3 or not _l3:
                                    continue
                                _wn3 = (_w3.get("athlete",{}) or {}).get("displayName","") or _w3.get("displayName","")
                                _ln3 = (_l3.get("athlete",{}) or {}).get("displayName","") or _l3.get("displayName","")
                                if _wn3 and _ln3:
                                    _espn_pairs[(_nn(_wn3),_nn(_ln3))] = _wn3
                                    _espn_pairs[(_nn(_ln3),_nn(_wn3))] = _wn3

            st.write(f"ESPN pairs trovati: {len(_espn_pairs)//2}")
            _resolved3 = 0
            _log3 = []
            _seen3: set = set()
            for _prow3 in _pend3.itertuples(index=False):
                _pk3 = (_nn(_prow3.player1), _nn(_prow3.player2))
                if _pk3 in _seen3:
                    continue
                _winner3 = None
                for (_wk,_lk), _wv in _espn_pairs.items():
                    if (_mon(_wk, _prow3.player1) and _mon(_lk, _prow3.player2)) or \
                       (_mon(_wk, _prow3.player2) and _mon(_lk, _prow3.player1)):
                        _winner3 = _wv
                        break
                if not _winner3:
                    continue
                _seen3.add(_pk3)
                _db_w3 = _prow3.player1 if _mon(_winner3, _prow3.player1) else _prow3.player2
                # Resolve ALL match_ids for this pair
                _all_ids3 = _rd("SELECT DISTINCT match_id FROM bets WHERE result='pending' AND player1=:p1 AND player2=:p2",
                                  {"p1": _prow3.player1, "p2": _prow3.player2})
                for _mid3 in _all_ids3["match_id"]:
                    try:
                        _n3 = _res(_mid3, _db_w3)
                        _resolved3 += _n3
                        if _n3 == 0:
                            # debug: see what's in the DB for this match_id
                            _bets3 = _rd("SELECT id, market, selection, result FROM bets WHERE match_id=:m", {"m": _mid3})
                            _log3.append(f"  ⚠️ {_mid3[:40]} → _res=0, bets={_bets3[['market','selection','result']].to_dict('records')[:2]}")
                    except Exception as _ex3:
                        _log3.append(f"  ❌ errore: {_ex3}")
                if _all_ids3.shape[0] > 0:
                    _log3.append(f"✅ {_prow3.player1} vs {_prow3.player2} → vincitore: {_db_w3} ({_all_ids3.shape[0]} match_id, risolti: {_n3})")
            st.write(f"**Risolti: {_resolved3}**")
            if _log3:
                st.code("\n".join(_log3))
            else:
                st.warning("Nessuna partita trovata in ESPN da risolvere.")

        if st.button("🔬 Test ESPN vs pending (debug)"):
            import requests as _req
            from tvb.bet_tracker import _norm_name as _nn, _match_one_name as _mon
            _pend2 = get_pending_matches()

            # Fetch ESPN for yesterday+today
            _all_espn = []
            for _d in [
                (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d"),
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ]:
                _r2 = _req.get(
                    "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard",
                    params={"dates": _d.replace("-", "")},
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if _r2.status_code == 200:
                    for _ev in _r2.json().get("events", []):
                        for _grp in _ev.get("groupings", []):
                            for _comp in _grp.get("competitions", []):
                                _cx = _comp.get("competitors", [])
                                _w = next((c for c in _cx if c.get("winner")), None)
                                _l = next((c for c in _cx if not c.get("winner")), None)
                                if _w and _l:
                                    _wn = _w.get("athlete",{}).get("displayName","") or _w.get("displayName","")
                                    _ln = _l.get("athlete",{}).get("displayName","") or _l.get("displayName","")
                                    if _wn and _ln:
                                        _all_espn.append((_wn, _ln))

            st.write(f"Totale risultati ESPN: {len(_all_espn)}")

            # Check each pending match
            _found, _notfound = [], []
            for _prow in _pend2.itertuples(index=False):
                _match = None
                for _we, _le in _all_espn:
                    if (_mon(_we, _prow.player1) and _mon(_le, _prow.player2)) or \
                       (_mon(_we, _prow.player2) and _mon(_le, _prow.player1)):
                        _match = f"{_we} def. {_le}"
                        break
                if _match:
                    _found.append(f"✅ {_prow.player1} vs {_prow.player2} → {_match}")
                else:
                    _notfound.append(f"❌ {_prow.player1} vs {_prow.player2}")

            st.write(f"**Trovati:** {len(_found)} | **Non trovati:** {len(_notfound)}")
            if _found:
                st.code("\n".join(_found[:20]))
            if _notfound:
                st.code("\n".join(_notfound[:20]))

        _pend = get_pending_matches()
        if _pend.empty:
            st.info("Nessuna partita pending nel database.")
        else:
            st.write(f"**Partite pending:** {len(_pend)}")
            st.dataframe(_pend[["player1", "player2", "commence_time", "sport_key"]],
                         use_container_width=True)

        if _rapi_key and st.button("Test RapidAPI risultati ieri"):
            import requests as _req
            from datetime import timedelta
            _yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            _host = "tennis-api-atp-wta-itf.p.rapidapi.com"
            _hdrs = {"X-RapidAPI-Key": _rapi_key, "X-RapidAPI-Host": _host}
            # prova endpoint results
            for _url in [
                f"https://{_host}/tennis/v2/atp/results/{_yesterday}",
                f"https://{_host}/tennis/v2/atp/fixtures/{_yesterday}",
            ]:
                _r = _req.get(_url, headers=_hdrs, timeout=15)
                st.write(f"`{_url}` → **{_r.status_code}**")
                if _r.status_code == 200:
                    _data = _r.json()
                    _items = _data.get("data", _data) if isinstance(_data, dict) else _data
                    if isinstance(_items, list) and _items:
                        # mostra solo primi 2 con timeGame valorizzato
                        _with_score = [m for m in _items if m.get("timeGame")]
                        st.write(f"Totale: {len(_items)} | Con score: {len(_with_score)}")
                        st.json((_with_score or _items)[:2])
                    break

        if _key and st.button("Test Odds API scores (debug)"):
            _dbg = scores_debug(_key)
            if not _dbg:
                st.error("Nessun dato dall'Odds API.")
            for _d in _dbg:
                st.markdown(f"**`{_d['sport_key']}`** — totali: {_d['total_events']} | "
                            f"completati: {_d['completed']} | matched: {_d['matched_pending']}")
                if _d["sample_completed"]:
                    st.json(_d["sample_completed"])

    # ---- tour filter
    st.divider()
    _tour_sel = st.radio("Circuito", ["Tutti", "ATP", "WTA"],
                         horizontal=True, index=0,
                         help="Filtra tutte le statistiche per circuito")
    _tf = "" if _tour_sel == "Tutti" else _tour_sel.lower()

    # ---- today's proposals ordered by Kelly
    _pending = get_pending_bets(tour=_tf)
    n_pending_display = len(_pending)
    st.markdown(f"#### 📋 Proposte in attesa — {n_pending_display} giocate")
    if _pending.empty:
        st.info(
            "Nessuna proposta in attesa. Le value bet vengono registrate "
            "automaticamente al caricamento del palinsesto (solo per i "
            "giocatori presenti nel database Sackmann).")
    else:
        _disp = _pending.copy()
        _disp["Orario"] = _disp["commence_time"].apply(
            lambda s: (_parse_dt(s) or "").astimezone(_LOCAL_TZ).strftime(
                "%d/%m %H:%M") if _parse_dt(s) else s)
        _disp["Match"] = _disp["player1"] + " vs " + _disp["player2"]
        _disp["ATP/WTA"] = _disp["tour"].str.upper()
        _disp["model_prob"] = (_disp["model_prob"] * 100).round(1)
        _disp["edge"] = (_disp["edge"] * 100).round(1)
        _disp["kelly"] = (_disp["kelly"] * 100).round(1)
        st.dataframe(
            _disp[["Orario", "Match", "ATP/WTA", "market", "selection",
                   "odds", "model_prob", "edge", "kelly"]].rename(columns={
                "market": "Mercato", "selection": "Selezione",
                "odds": "Quota", "model_prob": "P(mod)%",
                "edge": "Edge%", "kelly": "Kelly%"}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kelly%": st.column_config.ProgressColumn(
                    "Kelly%", min_value=0, max_value=15, format="%.1f%%"),
                "Edge%": st.column_config.NumberColumn(format="+%.1f%%"),
                "P(mod)%": st.column_config.NumberColumn(format="%.1f%%"),
                "Quota": st.column_config.NumberColumn(format="%.2f"),
            })

    # ---- bulk result entry
    _pending_matches = get_pending_matches(tour=_tf)
    if not _pending_matches.empty:
        with st.expander(
                f"📝 Inserisci risultati in blocco  —  "
                f"{len(_pending_matches)} partite in attesa", expanded=False):
            st.caption(
                "Imposta il **Vincitore** per le partite finite, "
                "inserisci i **Game totali** se vuoi risolvere anche le bet Total games "
                "(es. 6-3 6-4 = 19), poi clicca *Salva*. "
                "Lascia '⏳' per le partite non ancora finite.")
            _winners_bulk: dict = {}
            _games_bulk: dict = {}
            with st.form("_bulk_results_form"):
                _hc = st.columns([3, 1, 2, 1])
                _hc[0].markdown("**Partita**")
                _hc[1].markdown("**Ora**")
                _hc[2].markdown("**Vincitore**")
                _hc[3].markdown("**Game**")
                for _bi, _br in enumerate(
                        _pending_matches.itertuples(index=False)):
                    _rc = st.columns([3, 1, 2, 1])
                    _dt = _parse_dt(_br.commence_time)
                    _rc[0].markdown(
                        f"**{_br.player1}** vs {_br.player2}  \n"
                        f"<small style='opacity:.6'>{_br.n_bets} bet · "
                        f"{_br.tour.upper()}</small>",
                        unsafe_allow_html=True)
                    _rc[1].caption(
                        _dt.astimezone(_LOCAL_TZ).strftime("%d/%m %H:%M")
                        if _dt else "—")
                    _winners_bulk[_br.match_id] = _rc[2].selectbox(
                        "W", ["⏳", _br.player1, _br.player2],
                        key=f"_bw_{_bi}", label_visibility="collapsed")
                    _games_bulk[_br.match_id] = _rc[3].number_input(
                        "G", 0, 200, 0, step=1,
                        key=f"_bg_{_bi}", label_visibility="collapsed")
                _bulk_ok = st.form_submit_button(
                    "✅ Salva risultati selezionati",
                    type="primary", use_container_width=True)
            if _bulk_ok:
                _tot_r = 0
                for _mid, _w in _winners_bulk.items():
                    if _w == "⏳":
                        continue
                    _tg = int(_games_bulk[_mid]) if _games_bulk.get(_mid, 0) > 0 else None
                    _tot_r += resolve_match_manual(_mid, _w, total_games=_tg)
                if _tot_r:
                    st.success(f"✅ {_tot_r} giocate aggiornate.")
                    st.rerun()
                else:
                    st.warning("Nessun vincitore selezionato.")

    stats = performance_stats(tour=_tf)
    n_resolved = stats["n_resolved"]
    n_scanned = len(st.session_state.get("_scanned_match_ids", set()))

    # ---- summary metrics
    st.divider()
    m0, m1, m2, m3, m4, m5 = st.columns(6)
    m0.metric("Partite analizzate", n_scanned or "—",
              help="Match scansionati automaticamente in questa sessione")
    m1.metric("Value bet trovate", n_resolved + stats["n_pending"],
              help="Bets con edge ≥ 3% e EV > 0 su tutti i match · risolte + in attesa")
    m2.metric("Risolte", n_resolved)
    m3.metric("% Vittorie",
              f"{stats['win_rate'] * 100:.1f}%" if n_resolved else "—",
              help=f"{stats['n_won']} vinte · {stats['n_lost']} perse")
    profit = stats["total_profit"]
    m4.metric("Profitto netto",
              f"€ {profit:+.2f}" if n_resolved else "—",
              delta=f"{stats['roi'] * 100:+.1f}% ROI" if n_resolved else None,
              delta_color="normal")
    m5.metric("Stake totale",
              f"€ {stats['total_staked']:.0f}" if n_resolved else "—")

    # ---- budget calculator
    st.divider()
    _b_left, _b_right = st.columns([1, 3])
    _budget = _b_left.number_input(
        "💰 Budget iniziale (€)",
        min_value=10.0, max_value=1_000_000.0,
        value=float(st.session_state.get("_user_budget", 200.0)),
        step=10.0, format="%.0f",
        help="Il tuo bankroll di partenza. Usato per calcolare il bankroll attuale "
             "e quante giocate puoi ancora fare.",
        key="_user_budget")
    _bankroll = _budget + profit
    _growth_pct = (profit / _budget * 100) if _budget else 0.0
    _staked_pct = (stats["total_staked"] / _budget * 100) if _budget else 0.0
    _bets_avail = max(0, int(_bankroll / 10))
    _bm1, _bm2, _bm3, _bm4 = _b_right.columns(4)
    _bm1.metric(
        "Bankroll attuale",
        f"€ {_bankroll:.2f}",
        delta=f"{_growth_pct:+.1f}%" if n_resolved else None,
        delta_color="normal")
    _bm2.metric(
        "P&L netto",
        f"€ {profit:+.2f}" if n_resolved else "€ 0.00",
        delta=f"{_growth_pct:+.1f}% sul budget" if n_resolved else None,
        delta_color="normal")
    _bm3.metric(
        "Stake impegnato",
        f"€ {stats['total_staked']:.0f}" if stats["total_staked"] else "€ 0",
        delta=f"{_staked_pct:.1f}% del budget" if stats["total_staked"] else None,
        delta_color="off")
    _bm4.metric(
        "Giocate disponibili",
        str(_bets_avail),
        help="Quante bet da €10 puoi ancora piazzare col bankroll attuale")

    if n_resolved == 0:
        if n_pending_display > 0:
            st.info(
                f"✅ **{n_pending_display} giocate salvate nel database** — "
                "nessun match ancora terminato. I risultati verranno aggiornati "
                "automaticamente appena le partite si concludono. "
                "Le giocate già registrate non vengono mai modificate: "
                "vengono solo aggiunte nuove bet.")
        else:
            st.info(
                "Nessuna proposta ancora registrata. Le value bet vengono salvate "
                "automaticamente al primo aggiornamento del palinsesto "
                "(solo per i giocatori presenti nel database Sackmann).")
    else:
        # ---- equity curve
        st.divider()
        st.markdown("#### Curva di equity (€ cumulati)")
        eq = equity_curve(tour=_tf)
        if not eq.empty:
            st.line_chart(eq.set_index("Data")["Profitto cumulato (€)"],
                          use_container_width=True)

        # ---- win/loss progress bars
        st.divider()
        st.markdown("#### Vittorie vs Sconfitte")
        wr = stats["win_rate"]
        col_w, col_l = st.columns(2)
        col_w.metric("Vinte", stats["n_won"])
        col_w.progress(wr, text=f"{wr * 100:.1f}%")
        col_l.metric("Perse", stats["n_lost"])
        col_l.progress(1.0 - wr, text=f"{(1 - wr) * 100:.1f}%")

        # ---- accuracy by player
        st.divider()
        st.markdown("#### Accuratezza per giocatore / esito")
        by_player = accuracy_by_player(tour=_tf)
        if not by_player.empty:
            st.dataframe(
                by_player,
                use_container_width=True,
                column_config={
                    "% Accuratezza": st.column_config.ProgressColumn(
                        "% Accuratezza", min_value=0, max_value=100,
                        format="%.1f%%"),
                    "Profitto netto (€)": st.column_config.NumberColumn(
                        "Profitto netto (€)", format="€ %.2f"),
                })

    # ---- scores API debug (button-gated — never auto-fires)
    st.divider()
    with st.expander("🔍 Debug scores API"):
        st.caption(
            "Controlla cosa restituisce The Odds API Scores. "
            "Ogni click consuma crediti API — usalo solo per diagnosticare.")
        if _key:
            if st.button("Controlla scores API", key="_dbg_scores_btn"):
                with st.spinner("Interrogo scores API..."):
                    _dbg = scores_debug(_key)
                if not _dbg:
                    st.info("Nessuna bet pending con sport_key valido.")
                for _d in _dbg:
                    _fn = st.success if _d["matched_pending"] > 0 else st.warning
                    _fn(f"**`{_d['sport_key']}`** — "
                        f"{_d['total_events']} eventi · "
                        f"{_d['completed']} completati · "
                        f"**{_d['matched_pending']} match con bet nel DB**")
                    for _s in _d["sample_completed"]:
                        st.markdown(
                            f"- `{_s['p1']}` vs `{_s['p2']}` · "
                            f"scores={_s['scores']} · winner=`{_s['winner']}`")
        else:
            st.warning("Chiave API non configurata.")

    # ---- full history
    st.divider()
    with st.expander("Storico completo delle bet"):
        df_all = get_bets_df(tour=_tf)
        if df_all.empty:
            st.info("Nessuna bet registrata.")
        else:
            _result_colors = {"won": "🟢", "lost": "🔴", "pending": "🟡"}
            df_display = df_all.copy()
            df_display["result"] = df_display["result"].map(
                lambda r: f"{_result_colors.get(r, '')} {r}")
            st.dataframe(
                df_display[[
                    "logged_at", "tour", "player1", "player2", "market",
                    "selection", "odds", "model_prob", "edge",
                    "stake", "result", "profit"
                ]].rename(columns={
                    "logged_at": "Data",
                    "tour": "Circuito",
                    "player1": "Giocatore 1",
                    "player2": "Giocatore 2",
                    "market": "Mercato",
                    "selection": "Selezione",
                    "odds": "Quota",
                    "model_prob": "P(modello)",
                    "edge": "Edge",
                    "stake": "Stake (€)",
                    "result": "Risultato",
                    "profit": "P&L (€)",
                }),
                use_container_width=True,
                column_config={
                    "Edge": st.column_config.NumberColumn(format="%.1f%%"),
                    "P(modello)": st.column_config.NumberColumn(format="%.1%"),
                    "P&L (€)": st.column_config.NumberColumn(format="€ %.2f"),
                })
