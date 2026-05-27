"""Streamlit dashboard: today's tennis matches + value-bet analysis.

    streamlit run app/dashboard.py

The serve/return model is wired to live matches from The Odds API; serve
parameters come from Sackmann data via name matching; recalibrations
validated on the historical backtests are applied by default. Password
gate active behind `APP_PASSWORD`.
"""
import os
import sys
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

st.set_page_config(page_title="Tennis Value Bet", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------- font + size override
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    /* Just bump the base size — the font itself is applied by Streamlit's
       theme (font = "Plus Jakarta Sans" in .streamlit/config.toml). Don't
       touch font-family on generic elements: Streamlit's Material Symbols
       icons live in <span> elements and any broad span-level override
       hijacks their icon font and renders ligature names as text. */
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
    """Cached Monte Carlo — same (p0, p1, best_of, n_sims) returns same result."""
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
    """Dropdown label: 'P1 vs P2 - HH:MM 🏁' for upcoming, '... - LIVE 🟥' for in-play."""
    dt = _parse_dt(m.commence_time)
    if dt is None:
        return f"{m.player1} vs {m.player2} - — ⏳"
    if dt <= datetime.now(timezone.utc):
        return f"{m.player1} vs {m.player2} - LIVE 🟥"
    local = dt.astimezone(_LOCAL_TZ).strftime("%H:%M")
    return f"{m.player1} vs {m.player2} - {local} 🏁"


def _live_score_url(name0: str, name1: str) -> str:
    """Google search URL for 'P1 vs P2' — first results usually include
    live-score sites (SofaScore, Flashscore, ATP/WTA)."""
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
        # Stable IDs as dict keys (don't change when the time becomes LIVE);
        # display labels are computed by format_func on every render.
        new_matches = {
            f"{m.player1}|{m.player2}|{m.commence_time}": m for m in found}
        prev_sel = st.session_state.get("live_sel")
        st.session_state["live_matches"] = new_matches
        st.session_state.pop("_fetch_error", None)
        if found:
            # Keep the user's current match if it still exists; fall back to
            # the first match only when the previous one is gone.
            if prev_sel not in new_matches:
                st.session_state["live_sel"] = next(iter(new_matches))
            _apply_live_odds()
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


# ------------------------------------------------------------ match picker
hl, hr = st.columns([5, 1])
hl.subheader("📅 Partite di oggi")
if hr.button("🔄 Aggiorna", use_container_width=True):
    with st.spinner("Aggiorno le quote..."):
        _fetch_matches()
    # No explicit st.rerun(): the rest of the script renders below this
    # button and reads the freshly updated session_state on the same run.

if st.session_state.get("_fetch_error"):
    st.error(st.session_state["_fetch_error"])

live = st.session_state.get("live_matches") or {}
if not live:
    st.info("Nessun match disponibile al momento. Apri le **Impostazioni** "
            "(barra a sinistra) per controllare la chiave API, oppure premi "
            "**Aggiorna** più tardi.")
    st.stop()

st.selectbox("Match", options=list(live.keys()), key="live_sel",
             format_func=lambda k: _match_label(live[k]),
             on_change=_apply_live_odds, label_visibility="collapsed")

m = live[st.session_state["live_sel"]]
tour, surface, bo_match = match_context(m.sport_key)
name0, name1 = m.player1, m.player2
p0, p1, elo_xcheck, status = _resolve_live_params(tour, surface, name0, name1)

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
            when_html = (f"🏁 {dt_match.astimezone(_LOCAL_TZ).strftime('%H:%M')}")
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

# Live score link — opens a Google search for "P1 vs P2" in a new tab
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
        "considera di disattivarle (Impostazioni → Ricalibrazione validata).")

# ------------------------------------------------------------ quote section
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

with st.expander("Mercati a inserimento manuale (non disponibili via API)"):
    st.caption(
        "The Odds API non fornisce quote per **vincente 1° set** e "
        "**tie-break** sul tennis. Questi due mercati restano a inserimento "
        "manuale e i valori non cambiano automaticamente al cambio del match.")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("**Vincente 1° set**")
        st.number_input(name0, 1.01, 50.0, key="s10")
        st.number_input(name1, 1.01, 50.0, key="s11")
    with cm2:
        st.markdown("**Tie-break**")
        st.number_input("Sì", 1.01, 50.0, key="tby")
        st.number_input("No", 1.01, 50.0, key="tbn")


# ------------------------------------------- auto-analyze (no button)
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
                f"#### #{i} · {vb.market} — {vb.selection}  @{vb.odds:.2f}"
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
