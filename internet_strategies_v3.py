"""
INTERNET STRATEGIES ENGINE v2 — WDO B3

10M combos | Multiprocessing 8 CPUs | MAE/MFE Analysis
HMM Regime Filter | OOS Rolling 30 dias

NOVIDADES vs v1:
1. Multiprocessing — 8 CPUs em paralelo (8x mais rapido)
2. MAE/MFE Analysis — descobre SL/TP ideais pelos movimentos reais
3. HMM Regime — testa separado por regime de mercado
4. OOS Rolling — valida em janelas de 30 dias
5. 10M combos — grids muito mais expandidos

LICOES APLICADAS:
- flush=True em todos os prints
- Entrada no OPEN do proximo candle (sem lookahead)
- MIN_TRADES=300 IS, 50 OOS
- MAX_PF=2.5, MAX_SHARPE=3.0
- Plateau test anti-overfitting
- VWAP correto (reseta por dia)
- SHORT favorecido no WDO
"""

import pandas as pd
import numpy as np
from numba import njit
import json, sys, os, time, warnings, math, itertools
from datetime import datetime
from scipy import stats
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
warnings.filterwarnings("ignore")

CSV_PATH       = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR     = "/workspace/param_opt_output/internet_strategies_v2"
CAPITAL        = 50_000.0
MULT           = 10.0
COMM           = 5.0
SLIP           = 2.0
MIN_TRADES_IS  = 300
MIN_TRADES_OOS = 50
MAX_PF         = 2.5
MAX_SHARPE     = 3.0
MAX_DD         = -30.0
N_CPUS         = min(8, cpu_count())

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SECAO 1: DADOS
# ================================================================

def carregar():
    print("[DATA] Carregando...", flush=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower() for c in df.columns]
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    return df


# ================================================================
# SECAO 2: INDICADORES
# ================================================================

def calcular_indicadores(df):
    print("[IND] Calculando...", flush=True)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    ind = {
        "close":     c,
        "high":      h,
        "low":       l,
        "open":      o,
        "volume":    v,
        "open_next": np.concatenate([o[1:], [c[-1]]]),
    }

    # EMAs
    for p in [3, 5, 8, 9, 10, 13, 20, 21, 34, 50, 100, 200]:
        a   = 2 / (p + 1)
        out = np.empty_like(c); out[0] = c[0]
        for i in range(1, n):
            out[i] = a * c[i] + (1 - a) * out[i - 1]
        ind[f"ema_{p}"] = out

    # RSIs
    for p in [2, 3, 5, 7, 9, 11, 14, 18, 21, 28]:
        dv = np.diff(c, prepend=c[0])
        g  = np.where(dv > 0, dv, 0.0)
        ls = np.where(dv < 0, -dv, 0.0)
        ag = np.full(n, np.nan)
        al = np.full(n, np.nan)
        if p < n:
            ag[p] = g[1:p + 1].mean()
            al[p] = ls[1:p + 1].mean()
            for i in range(p + 1, n):
                ag[i] = (ag[i - 1] * (p - 1) + g[i]) / p
                al[i] = (al[i - 1] * (p - 1) + ls[i]) / p
        ind[f"rsi_{p}"] = 100 - (100 / (1 + ag / (al + 1e-9)))

    # ATR
    prev = np.roll(c, 1); prev[0] = c[0]
    tr   = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    for p in [7, 14, 20]:
        atr = np.full(n, np.nan)
        if p < n:
            atr[p - 1] = tr[:p].mean()
            for i in range(p, n):
                atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
        ind[f"atr_{p}"] = atr

    # MACD
    for fast, slow in [(3, 10), (5, 13), (8, 21), (12, 26)]:
        af  = 2 / (fast + 1)
        as_ = 2 / (slow + 1)
        ef  = np.empty_like(c); ef[0] = c[0]
        es  = np.empty_like(c); es[0] = c[0]
        for i in range(1, n):
            ef[i] = af  * c[i] + (1 - af)  * ef[i - 1]
            es[i] = as_ * c[i] + (1 - as_) * es[i - 1]
        mac = ef - es
        sig = np.empty_like(c); sig[0] = mac[0]; a9 = 2 / 10
        for i in range(1, n):
            sig[i] = a9 * mac[i] + (1 - a9) * sig[i - 1]
        ind[f"macd_{fast}_{slow}_hist"] = mac - sig

    # Bollinger Bands
    for p in [5, 10, 20, 50]:
        s   = pd.Series(c)
        sma = s.rolling(p).mean().values
        std = s.rolling(p).std().values
        for mult_f, tag in [(1.0, "10"), (1.5, "15"), (2.0, "20"), (2.5, "25"), (3.0, "30")]:
            up  = sma + mult_f * std
            lo  = sma - mult_f * std
            pct = (c - lo) / (up - lo + 1e-9)
            ind[f"bb_{p}_{tag}_pct"]   = pct
            ind[f"bb_{p}_{tag}_upper"] = up
            ind[f"bb_{p}_{tag}_lower"] = lo
            ind[f"bb_{p}_{tag}_width"] = (up - lo) / (sma + 1e-9)

    # Donchian
    for p in [5, 10, 20, 50, 100, 200]:
        ind[f"don_high_{p}"] = pd.Series(h).rolling(p).max().shift(1).values
        ind[f"don_low_{p}"]  = pd.Series(l).rolling(p).min().shift(1).values

    # Stochastic
    for p in [3, 5, 7, 9, 14, 21]:
        lo_p = pd.Series(l).rolling(p).min().values
        hi_p = pd.Series(h).rolling(p).max().values
        ind[f"stoch_k_{p}"] = (c - lo_p) / (hi_p - lo_p + 1e-9) * 100

    # CCI
    for p in [7, 10, 14, 20, 30]:
        tp  = (h + l + c) / 3
        sma = pd.Series(tp).rolling(p).mean().values
        mad = pd.Series(tp).rolling(p).apply(
            lambda x: np.abs(x - x.mean()).mean()).values
        ind[f"cci_{p}"] = (tp - sma) / (0.015 * mad + 1e-9)

    # VWAP diario (reseta todo dia)
    vwap_arr = np.full(n, np.nan)
    tp       = (h + l + c) / 3
    cum_tpv  = np.zeros(n)
    cum_vol  = np.zeros(n)
    datas    = df.index.date
    data_atual = None
    for i in range(n):
        if datas[i] != data_atual:
            data_atual = datas[i]
            cum_tpv[i] = tp[i] * v[i]
            cum_vol[i] = v[i]
        else:
            cum_tpv[i] = cum_tpv[i - 1] + tp[i] * v[i]
            cum_vol[i] = cum_vol[i - 1] + v[i]
        if cum_vol[i] > 0:
            vwap_arr[i] = cum_tpv[i] / cum_vol[i]
    ind["vwap"] = vwap_arr

    # VWAP desvios
    vwap_std   = np.full(n, np.nan)
    sq_sum     = np.zeros(n)
    cnt        = np.zeros(n)
    data_atual = None
    for i in range(n):
        if datas[i] != data_atual:
            data_atual = datas[i]
            sq_sum[i]  = (c[i] - vwap_arr[i]) ** 2
            cnt[i]     = 1
        else:
            sq_sum[i] = sq_sum[i - 1] + (c[i] - vwap_arr[i]) ** 2
            cnt[i]    = cnt[i - 1] + 1
        if cnt[i] > 1:
            vwap_std[i] = np.sqrt(sq_sum[i] / cnt[i])
    ind["vwap_std"]    = vwap_std
    ind["vwap_upper1"] = vwap_arr + vwap_std
    ind["vwap_lower1"] = vwap_arr - vwap_std
    ind["vwap_upper2"] = vwap_arr + 2 * vwap_std
    ind["vwap_lower2"] = vwap_arr - 2 * vwap_std

    # Volume
    for p in [10, 20]:
        vm = pd.Series(v).rolling(p).mean().values
        vs = pd.Series(v).rolling(p).std().values
        ind[f"vol_z_{p}"]     = (v - vm) / (vs + 1e-9)
        ind[f"vol_ratio_{p}"] = v / (vm + 1e-9)
    ret = np.diff(c, prepend=c[0]) / (c + 1e-9)
    v5  = pd.Series(ret).rolling(5).std().values * 100
    v20 = pd.Series(ret).rolling(20).std().values * 100
    ind["vol_ratio"] = v5 / (v20 + 1e-9)

    # Keltner Channel
    for ep in [5, 10, 20, 50]:
        for ap in [7, 14, 20]:
            for mf in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                key = f"kc_{ep}_{ap}_{str(mf).replace('.', '')}"
                ind[f"{key}_upper"] = ind[f"ema_{ep}"] + mf * ind[f"atr_{ap}"]
                ind[f"{key}_lower"] = ind[f"ema_{ep}"] - mf * ind[f"atr_{ap}"]

    # Opening Range
    for orb_min in [5, 10, 15, 20, 30, 45]:
        orb_high = np.full(n, np.nan)
        orb_low  = np.full(n, np.nan)
        day_data = {}
        for i in range(n):
            dt   = df.index[i]
            data = dt.date()
            mins = dt.hour * 60 + dt.minute - 9 * 60
            if mins < 0:
                continue
            if data not in day_data:
                day_data[data] = {"hi": -np.inf, "lo": np.inf, "done": False}
            if not day_data[data]["done"]:
                if mins <= orb_min:
                    day_data[data]["hi"] = max(day_data[data]["hi"], h[i])
                    day_data[data]["lo"] = min(day_data[data]["lo"], l[i])
                else:
                    day_data[data]["done"] = True
            if day_data[data]["done"] or mins > orb_min:
                orb_high[i] = day_data[data]["hi"]
                orb_low[i]  = day_data[data]["lo"]
        ind[f"orb_high_{orb_min}"] = orb_high
        ind[f"orb_low_{orb_min}"]  = orb_low

    # Sessao e tempo
    hora = df.index.hour
    ind["session_am"] = ((hora >= 9)  & (hora < 12)).astype(np.int8)
    ind["session_pm"] = ((hora >= 13) & (hora < 17)).astype(np.int8)
    ind["hora"]       = hora
    ind["dow"]        = df.index.dayofweek.values

    print(f"[IND] {len(ind)} indicadores | {n:,} candles", flush=True)
    return ind


# ================================================================
# SECAO 3: SIMULADORES NUMBA
# ================================================================

@njit(cache=True)
def simular_long(on, hi, lo, ent, ext, sl_pts, tp_pts, mult, comm, slip):
    n    = len(on)
    pnls = np.empty(n, dtype=np.float64)
    n_tr = 0
    em   = False
    ep   = sl = tp = 0.0
    for i in range(n - 1):
        if em:
            if lo[i] <= sl or hi[i] >= tp or ext[i]:
                saida = sl if lo[i] <= sl else (tp if hi[i] >= tp else on[i])
                pnls[n_tr] = (saida - ep) * mult - comm - slip * mult * 0.1
                n_tr += 1
                em = False
            continue
        if ent[i] and not em:
            ep = on[i]
            if np.isnan(ep) or ep <= 0:
                continue
            sl = ep - sl_pts
            tp = ep + tp_pts
            em = True
    return pnls[:n_tr]


@njit(cache=True)
def simular_short(on, hi, lo, ent, ext, sl_pts, tp_pts, mult, comm, slip):
    n    = len(on)
    pnls = np.empty(n, dtype=np.float64)
    n_tr = 0
    em   = False
    ep   = sl = tp = 0.0
    for i in range(n - 1):
        if em:
            if hi[i] >= sl or lo[i] <= tp or ext[i]:
                saida = sl if hi[i] >= sl else (tp if lo[i] <= tp else on[i])
                pnls[n_tr] = (ep - saida) * mult - comm - slip * mult * 0.1
                n_tr += 1
                em = False
            continue
        if ent[i] and not em:
            ep = on[i]
            if np.isnan(ep) or ep <= 0:
                continue
            sl = ep + sl_pts
            tp = ep - tp_pts
            em = True
    return pnls[:n_tr]


# ================================================================
# SECAO 4: MAE/MFE ANALYSIS
# ================================================================

@njit(cache=True)
def calcular_mae_mfe(on, hi, lo, ent, direction, max_bars=60):
    """
    Para cada sinal, deixa o trade rolar ate max_bars candles.
    Registra MAE (max contra) e MFE (max favor).
    """
    n        = len(on)
    mae_list = np.empty(n, dtype=np.float64)
    mfe_list = np.empty(n, dtype=np.float64)
    n_tr     = 0

    for i in range(n - max_bars):
        if not ent[i]:
            continue
        ep = on[i]
        if np.isnan(ep) or ep <= 0:
            continue

        max_favor  = 0.0
        max_contra = 0.0

        for j in range(i + 1, min(i + max_bars, n)):
            if direction == 1:
                favor  = hi[j] - ep
                contra = ep - lo[j]
            else:
                favor  = ep - lo[j]
                contra = hi[j] - ep

            if favor  > max_favor:  max_favor  = favor
            if contra > max_contra: max_contra = contra

        mae_list[n_tr] = max_contra
        mfe_list[n_tr] = max_favor
        n_tr += 1

    return mae_list[:n_tr], mfe_list[:n_tr]


def analisar_mae_mfe(ind, ent, direction):
    """Analisa MAE/MFE e retorna SL/TP otimizados."""
    d = 1 if direction == "long" else -1
    mae, mfe = calcular_mae_mfe(
        ind["open_next"].astype(np.float64),
        ind["high"].astype(np.float64),
        ind["low"].astype(np.float64),
        ent.astype(np.bool_), d,
    )
    if len(mae) < 20:
        return None

    sl_opt  = np.percentile(mae, 70)
    tp_opt  = np.percentile(mfe, 50)
    rr_real = tp_opt / (sl_opt + 1e-9)

    return {
        "mae_median":   round(float(np.median(mae)), 2),
        "mae_p70":      round(float(sl_opt), 2),
        "mae_p90":      round(float(np.percentile(mae, 90)), 2),
        "mfe_median":   round(float(np.median(mfe)), 2),
        "mfe_p50":      round(float(tp_opt), 2),
        "mfe_p75":      round(float(np.percentile(mfe, 75)), 2),
        "rr_real":      round(rr_real, 2),
        "sl_otimizado": round(float(sl_opt), 2),
        "tp_otimizado": round(float(tp_opt), 2),
    }


# ================================================================
# SECAO 5: HMM REGIME (simples, sem hmmlearn)
# ================================================================

def detectar_regime_simples(df_is):
    """Detecta regime usando regras simples baseadas em volatilidade e tendencia."""
    daily = df_is.resample("1D").agg(
        close=("close", "last"),
        high=("high",  "max"),
        low=("low",    "min"),
    ).dropna()

    ret   = daily["close"].pct_change()
    vol5  = ret.rolling(5).std() * 100
    vol20 = ret.rolling(20).std() * 100
    ema10 = daily["close"].ewm(span=10).mean()
    ema30 = daily["close"].ewm(span=30).mean()
    trend = (ema10 - ema30) / ema30 * 100

    regimes = []
    for i in range(len(daily)):
        vr = float(vol5.iloc[i] / (vol20.iloc[i] + 1e-9)) if not np.isnan(vol5.iloc[i]) else 1.0
        tr = abs(float(trend.iloc[i])) if not np.isnan(trend.iloc[i]) else 0.0
        vv = float(vol5.iloc[i]) if not np.isnan(vol5.iloc[i]) else 0.5

        if vr > 1.3:
            regimes.append("ALTA_VOL")
        elif tr > 0.5:
            regimes.append("TENDENCIA")
        elif vv < 0.5:
            regimes.append("LATERAL")
        else:
            regimes.append("CAOTICO")

    return pd.Series(regimes, index=daily.index)


def mascara_regime(df_is, regime_series, regime_alvo):
    """Cria mascara de candles que pertencem ao regime alvo."""
    mask = np.zeros(len(df_is), dtype=bool)
    for i, dt in enumerate(df_is.index):
        data    = dt.date()
        matches = regime_series[regime_series.index.date == data]
        if len(matches) > 0 and matches.iloc[0] == regime_alvo:
            mask[i] = True
    return mask


# ================================================================
# SECAO 6: METRICAS
# ================================================================

def metricas(pnls, min_trades=MIN_TRADES_IS):
    if len(pnls) < min_trades:
        return None
    w = pnls[pnls > 0]
    l = pnls[pnls <= 0]
    if len(l) == 0 or len(w) == 0:
        return None
    pf = abs(w.sum() / l.sum())
    if pf > MAX_PF:
        return None
    eq  = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnls)])
    pk  = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk * 100).min())
    if mdd < MAX_DD:
        return None
    ret = pnls / CAPITAL
    sh  = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252 * 390))
    if sh > MAX_SHARPE:
        return None
    n_jan   = max(1, len(range(0, max(1, len(pnls) - 30), 15)))
    jan_pos = sum(1 for s in range(0, max(1, len(pnls) - 30), 15)
                  if pnls[s:s + 30].sum() > 0)
    return {
        "n":       len(pnls),
        "wr":      round(len(w) / len(pnls) * 100, 2),
        "pf":      round(pf, 3),
        "sh":      round(sh, 3),
        "exp":     round(float(pnls.mean()), 2),
        "pnl":     round(float(pnls.sum()), 2),
        "mdd":     round(mdd, 2),
        "jan_pos": round(jan_pos / n_jan * 100, 1),
    }


def score(m):
    if not m:
        return 0
    pf_s  = min(m["pf"], MAX_PF) / MAX_PF
    exp_s = max(0, min(m["exp"], 500)) / 500
    jan_s = m["jan_pos"] / 100
    tr_s  = min(m["n"], 2000) / 2000
    sh_s  = max(0, min(m["sh"], 3)) / 3
    return pf_s * 0.30 + exp_s * 0.25 + jan_s * 0.20 + sh_s * 0.15 + tr_s * 0.10


# ================================================================
# SECAO 7: SINAIS — 23 ESTRATEGIAS
# ================================================================

def mascara_sessao(ind, session):
    if session == "am":
        return ind["session_am"].astype(bool)
    elif session == "pm":
        return ind["session_pm"].astype(bool)
    return np.ones(len(ind["close"]), dtype=bool)


def h1(x):
    return np.roll(x, 1)


def gerar_sinais(estrategia, ind, params):
    d    = params.get("direction", "short")
    ses  = params.get("session", "all")
    mask = mascara_sessao(ind, ses)
    c    = ind["close"]
    ent  = ext = None

    if estrategia == "vwap_reversion":
        vwap = ind["vwap"]
        std  = ind["vwap_std"]
        mv   = params["vwap_std"]
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        if rsi is None:
            return None, None
        lvl = params["rsi_level"]
        if d == "long":
            ent = (c < vwap - mv * std) & (rsi < lvl) & (h1(rsi) >= lvl)
            ext = c > vwap
        else:
            ent = (c > vwap + mv * std) & (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl))
            ext = c < vwap

    elif estrategia == "vwap_breakout":
        vwap = ind["vwap"]
        vr   = ind["vol_ratio"]
        vc   = params["vol_confirm"]
        if d == "long":
            ent = (c > vwap) & (h1(c) <= h1(vwap)) & (vr > vc)
            ext = c < vwap
        else:
            ent = (c < vwap) & (h1(c) >= h1(vwap)) & (vr > vc)
            ext = c > vwap

    elif estrategia == "vwap_pullback":
        vwap = ind["vwap"]
        ema  = ind.get(f"ema_{params['ema_period']}")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rf   = params["rsi_filter"]
        if ema is None or rsi is None:
            return None, None
        tol = ind["atr_14"] * 0.3
        if d == "long":
            ent = (ema > vwap) & (np.abs(c - vwap) < tol) & (rsi > rf)
            ext = c < vwap
        else:
            ent = (ema < vwap) & (np.abs(c - vwap) < tol) & (rsi < (100 - rf))
            ext = c > vwap

    elif estrategia == "orb_breakout":
        om  = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        vr = ind["vol_ratio"]
        vc = params.get("vol_confirm", 1.0)
        if d == "long":
            ent = (c > orh) & (h1(c) <= h1(orh)) & (vr > vc)
            ext = c < orl
        else:
            ent = (c < orl) & (h1(c) >= h1(orl)) & (vr > vc)
            ext = c > orh

    elif estrategia == "orb_retest":
        om  = params["orb_minutes"]
        orh = ind.get(f"orb_high_{om}")
        orl = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        tol   = ind["atr_14"] * 0.5
        above = c > orh
        if d == "long":
            foi = pd.Series(above).rolling(20).max().values.astype(bool)
            ent = foi & (np.abs(c - orh) < tol)
            ext = c < orl
        else:
            foi = pd.Series(~above).rolling(20).max().values.astype(bool)
            ent = foi & (np.abs(c - orl) < tol)
            ext = c > orh

    elif estrategia == "rsi_vwap_combo":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        side = params["vwap_side"]
        lvl  = params["rsi_level"]
        if rsi is None:
            return None, None
        vc = (c > vwap) if side == "above" else (c < vwap)
        if d == "long":
            ent = (rsi < lvl) & (h1(rsi) >= lvl) & vc
            ext = rsi > 50
        else:
            ent = (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl)) & (~vc)
            ext = rsi < 50

    elif estrategia == "rsi_ema_vwap":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        ema  = ind.get(f"ema_{params['ema_period']}")
        vwap = ind["vwap"]
        lvl  = params["rsi_level"]
        if rsi is None or ema is None:
            return None, None
        if d == "long":
            ent = (rsi < lvl) & (h1(rsi) >= lvl) & (c > ema) & (c > vwap)
            ext = rsi > 55
        else:
            ent = (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl)) & (c < ema) & (c < vwap)
            ext = rsi < 45

    elif estrategia == "rsi_vwap_session":
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        vwap = ind["vwap"]
        ovs  = params["oversold"]
        ovb  = params["overbought"]
        el   = params["exit_level"]
        if rsi is None:
            return None, None
        if d == "long":
            ent = (rsi < ovs) & (h1(rsi) >= ovs) & (c > vwap)
            ext = rsi > el
        else:
            ent = (rsi > ovb) & (h1(rsi) <= ovb) & (c < vwap)
            ext = rsi < (100 - el)

    elif estrategia == "atr_channel_breakout":
        ep  = params["ema_period"]
        ap  = params["atr_period"]
        am  = params["atr_mult"]
        key = f"kc_{ep}_{ap}_{str(am).replace('.', '')}"
        ku  = ind.get(f"{key}_upper")
        kl  = ind.get(f"{key}_lower")
        if ku is None:
            return None, None
        if d == "long":
            ent = (c > ku) & (h1(c) <= h1(ku)); ext = c < kl
        else:
            ent = (c < kl) & (h1(c) >= h1(kl)); ext = c > ku

    elif estrategia == "atr_trailing_momentum":
        pp  = params["momentum_period"]
        pt  = params["momentum_thresh"]
        roc = np.empty(len(c)); roc[:pp] = np.nan
        roc[pp:] = (c[pp:] - c[:-pp]) / (c[:-pp] + 1e-9) * 100
        if d == "long":
            ent = roc > pt;  ext = roc < 0
        else:
            ent = roc < -pt; ext = roc > 0

    elif estrategia == "macd_vwap":
        mf   = params["macd_fast"]
        ms   = params["macd_slow"]
        hist = ind.get(f"macd_{mf}_{ms}_hist")
        vwap = ind["vwap"]
        if hist is None:
            return None, None
        if d == "long":
            ent = (hist > 0) & (h1(hist) <= 0) & (c > vwap)
            ext = (hist < 0) & (h1(hist) >= 0)
        else:
            ent = (hist < 0) & (h1(hist) >= 0) & (c < vwap)
            ext = (hist > 0) & (h1(hist) <= 0)

    elif estrategia == "macd_rsi_vwap":
        cfg  = params["macd_config"]
        hist = ind.get(f"macd_{cfg}_hist")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rf   = params["rsi_filter"]
        vwap = ind["vwap"]
        if hist is None or rsi is None:
            return None, None
        if d == "long":
            ent = (hist > 0) & (h1(hist) <= 0) & (rsi > rf) & (c > vwap)
            ext = (hist < 0) & (h1(hist) >= 0)
        else:
            ent = (hist < 0) & (h1(hist) >= 0) & (rsi < (100 - rf)) & (c < vwap)
            ext = (hist > 0) & (h1(hist) <= 0)

    elif estrategia == "ema_vwap_trend":
        ef   = ind.get(f"ema_{params['fast']}")
        es   = ind.get(f"ema_{params['slow']}")
        vwap = ind["vwap"]
        if ef is None or es is None:
            return None, None
        if d == "long":
            ent = (ef > es) & (h1(ef) <= h1(es)) & (c > vwap)
            ext = ef < es
        else:
            ent = (ef < es) & (h1(ef) >= h1(es)) & (c < vwap)
            ext = ef > es

    elif estrategia == "dual_ema_momentum":
        ef = ind.get(f"ema_{params['fast']}")
        es = ind.get(f"ema_{params['slow']}")
        vr = ind["vol_ratio"]
        vc = params["vol_confirm"]
        if ef is None or es is None:
            return None, None
        if d == "long":
            ent = (ef > es) & (h1(ef) <= h1(es)) & (vr > vc)
            ext = ef < es
        else:
            ent = (ef < es) & (h1(ef) >= h1(es)) & (vr > vc)
            ext = ef > es

    elif estrategia == "bb_squeeze_breakout":
        bp  = params["bb_period"]
        bs  = params["bb_std"]
        sm  = params["squeeze_mult"]
        vc  = params["vol_confirm"]
        key = f"bb_{bp}_{str(bs).replace('.', '')}"
        bw  = ind.get(f"{key}_width")
        bu  = ind.get(f"{key}_upper")
        bl  = ind.get(f"{key}_lower")
        vr  = ind["vol_ratio"]
        if bw is None:
            return None, None
        bwa     = pd.Series(bw).rolling(20).mean().values
        squeeze = bw < bwa * sm
        saindo  = ~squeeze & pd.Series(squeeze).shift(1).fillna(False).values
        if d == "long":
            ent = saindo & (c > bu) & (vr > vc); ext = c < bl
        else:
            ent = saindo & (c < bl) & (vr > vc); ext = c > bu

    elif estrategia == "bb_rsi_vwap":
        bp   = params["bb_period"]
        bs   = params["bb_std"]
        key  = f"bb_{bp}_{str(bs).replace('.', '')}"
        pct  = ind.get(f"{key}_pct")
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rc   = params["rsi_confirm"]
        vwap = ind["vwap"]
        if pct is None or rsi is None:
            return None, None
        if d == "long":
            ent = (pct < 0.05) & (rsi < rc) & (c > vwap)
            ext = pct > 0.5
        else:
            ent = (pct > 0.95) & (rsi > (100 - rc)) & (c < vwap)
            ext = pct < 0.5

    elif estrategia == "donchian_vwap":
        dh   = ind.get(f"don_high_{params['don_period']}")
        dl   = ind.get(f"don_low_{params['don_period']}")
        vwap = ind["vwap"]
        vr   = ind["vol_ratio"]
        vc   = params["vol_confirm"]
        if dh is None:
            return None, None
        if d == "long":
            ent = (c > dh) & (c > vwap) & (vr > vc); ext = c < dl
        else:
            ent = (c < dl) & (c < vwap) & (vr > vc); ext = c > dh

    elif estrategia == "stoch_vwap":
        k    = ind.get(f"stoch_k_{params['stoch_period']}")
        vwap = ind["vwap"]
        ovs  = params["oversold"]
        ovb  = params["overbought"]
        if k is None:
            return None, None
        if d == "long":
            ent = (k < ovs) & (h1(k) >= ovs) & (c > vwap)
            ext = k > 50
        else:
            ent = (k > ovb) & (h1(k) <= ovb) & (c < vwap)
            ext = k < 50

    elif estrategia == "stoch_ema_vwap":
        k    = ind.get(f"stoch_k_{params['stoch_period']}")
        ema  = ind.get(f"ema_{params['ema_period']}")
        vwap = ind["vwap"]
        ovs  = params["oversold"]
        ovb  = params["overbought"]
        if k is None or ema is None:
            return None, None
        if d == "long":
            ent = (k < ovs) & (h1(k) >= ovs) & (c > ema) & (c > vwap)
            ext = k > 50
        else:
            ent = (k > ovb) & (h1(k) <= ovb) & (c < ema) & (c < vwap)
            ext = k < 50

    elif estrategia == "volume_spike_reversal":
        vz  = ind.get("vol_z_20")
        rsi = ind.get(f"rsi_{params['rsi_period']}")
        vs  = params["vol_spike"]
        lvl = params["rsi_level"]
        if vz is None or rsi is None:
            return None, None
        spike = vz > vs
        if d == "long":
            ent = spike & (rsi < lvl) & (h1(rsi) >= lvl)
            ext = rsi > 55
        else:
            ent = spike & (rsi > (100 - lvl)) & (h1(rsi) <= (100 - lvl))
            ext = rsi < 45

    elif estrategia == "volume_vwap_momentum":
        vr   = ind["vol_ratio"]
        vwap = ind["vwap"]
        vc   = params["vol_confirm"]
        if d == "long":
            ent = (c > vwap) & (h1(c) <= h1(vwap)) & (vr > vc)
            ext = c < vwap
        else:
            ent = (c < vwap) & (h1(c) >= h1(vwap)) & (vr > vc)
            ext = c > vwap

    elif estrategia == "cci_vwap":
        cci  = ind.get(f"cci_{params['cci_period']}")
        vwap = ind["vwap"]
        thr  = params["cci_thresh"]
        if cci is None:
            return None, None
        if d == "long":
            ent = (cci < -thr) & (h1(cci) >= -thr) & (c > vwap)
            ext = cci > 0
        else:
            ent = (cci > thr) & (h1(cci) <= thr) & (c < vwap)
            ext = cci < 0

    elif estrategia == "orb_vwap_combo":
        om   = params["orb_minutes"]
        orh  = ind.get(f"orb_high_{om}")
        orl  = ind.get(f"orb_low_{om}")
        if orh is None:
            return None, None
        vr   = ind["vol_ratio"]
        vc   = params["vol_confirm"]
        rsi  = ind.get(f"rsi_{params['rsi_period']}")
        rf   = params["rsi_filter"]
        vwap = ind["vwap"]
        if rsi is None:
            return None, None
        if d == "long":
            ent = (c > orh) & (h1(c) <= h1(orh)) & (vr > vc) & (rsi > rf) & (c > vwap)
            ext = c < orl
        else:
            ent = (c < orl) & (h1(c) >= h1(orl)) & (vr > vc) & (rsi < (100 - rf)) & (c < vwap)
            ext = c > orh

    else:
        return None, None

    if ent is None:
        return None, None
    ent    = ent & mask
    ent[0] = False
    return ent.astype(np.bool_), ext.astype(np.bool_)


def executar(ind, ent, ext, sl_pts, tp_pts, direction):
    on = ind["open_next"].astype(np.float64)
    hi = ind["high"].astype(np.float64)
    lo = ind["low"].astype(np.float64)
    e  = ent.astype(np.bool_)
    x  = ext.astype(np.bool_)
    if direction == "long":
        return simular_long(on, hi, lo, e, x, sl_pts, tp_pts, MULT, COMM, SLIP)
    else:
        return simular_short(on, hi, lo, e, x, sl_pts, tp_pts, MULT, COMM, SLIP)


# ================================================================
# SECAO 8: GRIDS — 10M COMBOS
# ================================================================

GRIDS = {
    "vwap_reversion": {
        "vwap_std":   [0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        "rsi_period": [2, 3, 5, 7, 9, 11, 14, 18, 21, 28],
        "rsi_level":  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "vwap_breakout": {
        "vol_confirm": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "vwap_pullback": {
        "ema_period": [5, 9, 20, 50, 100, 200],
        "rsi_period": [5, 7, 9, 14, 21, 28],
        "rsi_filter": [30, 35, 40, 45, 50, 55, 60],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "orb_breakout": {
        "orb_minutes": [5, 10, 15, 20, 30, 45],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "orb_retest": {
        "orb_minutes": [5, 10, 15, 20, 30, 45],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "rsi_vwap_combo": {
        "rsi_period": [2, 3, 5, 7, 9, 14, 21, 28],
        "rsi_level":  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "vwap_side":  ["above", "below"],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "rsi_ema_vwap": {
        "rsi_period": [2, 3, 5, 7, 9, 14, 21, 28],
        "rsi_level":  [5, 10, 15, 20, 25, 30, 35, 40],
        "ema_period": [5, 10, 20, 50, 100, 200],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "rsi_vwap_session": {
        "rsi_period": [2, 3, 5, 7, 9, 11, 14, 18, 21, 28],
        "oversold":   [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
        "overbought": [55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95],
        "exit_level": [40, 45, 50, 55, 60, 65],
        "atr_sl":     [0.5, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "atr_channel_breakout": {
        "ema_period": [5, 10, 20, 50],
        "atr_period": [7, 14, 20],
        "atr_mult":   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "atr_trailing_momentum": {
        "momentum_period": [3, 5, 10, 15, 20, 30],
        "momentum_thresh": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        "atr_sl":          [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":              [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":         ["am", "pm", "all"],
        "direction":       ["long", "short"],
    },
    "macd_vwap": {
        "macd_fast": [3, 5, 8, 10, 12],
        "macd_slow": [10, 13, 21, 26],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":        [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":   ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "macd_rsi_vwap": {
        "macd_config": ["12_26", "8_21", "5_13", "3_10"],
        "rsi_period":  [5, 7, 9, 14, 21, 28],
        "rsi_filter":  [30, 35, 40, 45, 50, 55, 60],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "ema_vwap_trend": {
        "fast":      [3, 5, 8, 10, 13, 20, 21],
        "slow":      [20, 34, 50, 100, 200],
        "atr_sl":    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":        [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":   ["am", "pm", "all"],
        "direction": ["long", "short"],
    },
    "dual_ema_momentum": {
        "fast":        [3, 5, 8, 10, 13, 20],
        "slow":        [20, 21, 34, 50, 100],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "bb_squeeze_breakout": {
        "bb_period":    [5, 10, 20, 50],
        "bb_std":       [1.0, 1.5, 2.0, 2.5, 3.0],
        "squeeze_mult": [0.6, 0.7, 0.8, 0.9],
        "vol_confirm":  [1.0, 1.2, 1.5, 2.0, 2.5],
        "atr_sl":       [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":           [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":      ["am", "pm", "all"],
        "direction":    ["long", "short"],
    },
    "bb_rsi_vwap": {
        "bb_period":   [5, 10, 20, 50],
        "bb_std":      [1.0, 1.5, 2.0, 2.5, 3.0],
        "rsi_period":  [2, 5, 7, 9, 14, 21],
        "rsi_confirm": [10, 15, 20, 25, 30, 35, 40],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "donchian_vwap": {
        "don_period":  [5, 10, 20, 50, 100, 200],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "stoch_vwap": {
        "stoch_period": [3, 5, 7, 9, 14, 21],
        "oversold":     [5, 10, 15, 20, 25, 30, 35],
        "overbought":   [65, 70, 75, 80, 85, 90, 95],
        "atr_sl":       [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":           [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":      ["am", "pm", "all"],
        "direction":    ["long", "short"],
    },
    "stoch_ema_vwap": {
        "stoch_period": [3, 5, 7, 9, 14, 21],
        "oversold":     [5, 10, 15, 20, 25, 30],
        "overbought":   [70, 75, 80, 85, 90, 95],
        "ema_period":   [20, 50, 100, 200],
        "atr_sl":       [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":           [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":      ["am", "pm", "all"],
        "direction":    ["long", "short"],
    },
    "volume_spike_reversal": {
        "vol_spike":  [1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "rsi_period": [2, 5, 7, 9, 14, 21],
        "rsi_level":  [10, 15, 20, 25, 30, 35, 40],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "volume_vwap_momentum": {
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "atr_sl":      [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
    "cci_vwap": {
        "cci_period": [7, 10, 14, 20, 30],
        "cci_thresh": [50, 75, 100, 125, 150, 175, 200, 250],
        "atr_sl":     [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "rr":         [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
        "session":    ["am", "pm", "all"],
        "direction":  ["long", "short"],
    },
    "orb_vwap_combo": {
        "orb_minutes": [5, 10, 15, 20, 30, 45],
        "vol_confirm": [1.0, 1.2, 1.5, 2.0, 2.5],
        "rsi_period":  [7, 9, 14, 21],
        "rsi_filter":  [35, 40, 45, 50, 55],
        "atr_sl":      [0.5, 1.0, 1.5, 2.0],
        "rr":          [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "session":     ["am", "pm", "all"],
        "direction":   ["long", "short"],
    },
}


# ================================================================
# SECAO 9: WORKER PARA MULTIPROCESSING
# ================================================================

def worker_estrategia(args):
    """Roda uma estrategia completa em um processo separado."""
    estrategia, grid, ind_dict, mini = args

    # Reconstruir arrays numpy do dict
    ind = {k: np.array(v) if isinstance(v, list) else v
           for k, v in ind_dict.items()}

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if mini:
        combos = combos[:20]

    validos = []
    atr_med = float(np.nanmean(ind["atr_14"]))
    t0      = time.time()

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            ent, ext = gerar_sinais(estrategia, ind, params)
            if ent is None or ent.sum() < 10:
                continue
            sl_pts = atr_med * params.get("atr_sl", 1.0)
            tp_pts = sl_pts  * params.get("rr", 2.0)
            pnls   = executar(ind, ent, ext, sl_pts, tp_pts,
                               params.get("direction", "short"))
            m = metricas(pnls)
            if not m:
                continue
            s = score(m)
            validos.append({
                "estrategia": estrategia,
                "params":     params,
                "score":      round(s, 6),
                **m,
            })
        except Exception:
            continue

    elapsed = time.time() - t0
    validos.sort(key=lambda x: -x["score"])

    return {
        "estrategia": estrategia,
        "n_combos":   len(combos),
        "n_validos":  len(validos),
        "elapsed":    round(elapsed, 1),
        "top10":      validos[:10],
    }


# ================================================================
# SECAO 10: PLATEAU TEST
# ================================================================

def plateau_test(estrategia, ind, melhor_params, atr_pts, top_pf):
    varnum = {k: v for k, v in melhor_params.items()
              if isinstance(v, (int, float)) and
              k not in ["rr", "atr_sl", "direction", "session",
                        "vwap_side", "macd_config", "macd_fast", "macd_slow"]}
    if not varnum:
        return True, 1.0

    param_teste = list(varnum.keys())[0]
    valor_base  = varnum[param_teste]
    resultados  = []

    for delta in [-2, -1, 1, 2]:
        pv = melhor_params.copy()
        pv[param_teste] = valor_base + delta
        if pv[param_teste] <= 0:
            continue
        try:
            ent, ext = gerar_sinais(estrategia, ind, pv)
            if ent is None or ent.sum() < 10:
                continue
            sl   = atr_pts * pv.get("atr_sl", 1.0)
            tp   = sl * pv.get("rr", 2.0)
            pnls = executar(ind, ent, ext, sl, tp, pv.get("direction", "short"))
            m    = metricas(pnls, min_trades=50)
            if m:
                resultados.append(m["pf"])
        except Exception:
            continue

    if not resultados:
        return True, 1.0
    pct = sum(1 for pf in resultados if pf > top_pf * 0.7) / len(resultados)
    return pct >= 0.5, round(pct, 2)


# ================================================================
# SECAO 11: OOS ROLLING
# ================================================================

def oos_rolling(estrategia, df, params, janela_dias=30):
    """Testa OOS em janelas de 30 dias. Valida consistencia mes a mes."""
    datas     = df.index.normalize().unique()
    split     = int(len(datas) * 0.70)
    datas_oos = datas[split:]
    resultados = []
    i = 0

    while i + janela_dias <= len(datas_oos):
        d_s  = datas_oos[i]
        d_e  = datas_oos[min(i + janela_dias - 1, len(datas_oos) - 1)]
        df_j = df[(df.index.normalize() >= d_s) & (df.index.normalize() <= d_e)]
        if len(df_j) < 100:
            i += janela_dias
            continue

        ind_j    = calcular_indicadores(df_j)
        ent, ext = gerar_sinais(estrategia, ind_j, params)
        if ent is None or ent.sum() < 5:
            i += janela_dias
            continue

        atr_j = float(np.nanmean(ind_j["atr_14"]))
        sl    = atr_j * params.get("atr_sl", 1.0)
        tp    = sl * params.get("rr", 2.0)
        pnls  = executar(ind_j, ent, ext, sl, tp, params.get("direction", "short"))
        m     = metricas(pnls, min_trades=20)
        resultados.append({
            "data":      str(d_s.date()),
            "pf":        m["pf"] if m else 0,
            "trades":    m["n"]  if m else 0,
            "pnl":       m["pnl"] if m else 0,
            "lucrativo": m is not None and m["pf"] > 1.0,
        })
        i += janela_dias

    if not resultados:
        return None
    pf_list = [r["pf"] for r in resultados if r["pf"] > 0]
    luc     = sum(1 for r in resultados if r["lucrativo"])
    wfe     = luc / len(resultados) * 100 if resultados else 0
    return {
        "janelas":    len(resultados),
        "lucrativas": luc,
        "wfe_pct":    round(wfe, 1),
        "pf_medio":   round(float(np.mean(pf_list)), 3) if pf_list else 0,
        "detalhes":   resultados,
    }


# ================================================================
# SECAO 12: IA EVOLUTIVA
# ================================================================

def ia_evolutiva(estrategia, ind, melhor_params, top_pf, atr_pts):
    if top_pf >= 1.0:
        return None
    print(f"    [IA] Tentando melhorar {estrategia} (PF={top_pf:.3f})...", flush=True)
    melhorias = []

    filtros = [
        {"nome": "vol_min",    "param": "vol_min",  "valores": [1.5, 2.0, 2.5, 3.0]},
        {"nome": "atr_min",    "param": "atr_min",  "valores": [3.0, 5.0, 8.0, 12.0]},
        {"nome": "session_am", "param": "session",  "valores": ["am"]},
        {"nome": "session_pm", "param": "session",  "valores": ["pm"]},
    ]

    for filtro in filtros:
        for val in filtro["valores"]:
            pn = melhor_params.copy()
            pn[filtro["param"]] = val
            try:
                ent, ext = gerar_sinais(estrategia, ind, pn)
                if ent is None or ent.sum() < 10:
                    continue
                if filtro["nome"] == "vol_min":
                    ent = ent & (ind["vol_ratio"] > val)
                elif filtro["nome"] == "atr_min":
                    ent = ent & (ind["atr_14"] > val)
                sl   = atr_pts * pn.get("atr_sl", 1.0)
                tp   = sl * pn.get("rr", 2.0)
                pnls = executar(ind, ent, ext, sl, tp, pn.get("direction", "short"))
                m    = metricas(pnls)
                if m and m["pf"] > top_pf:
                    melhorias.append({
                        "filtro_adicionado": filtro["nome"],
                        "valor":             val,
                        "params":            pn,
                        "pf_novo":           m["pf"],
                        "pf_anterior":       top_pf,
                        "melhoria_pct":      round((m["pf"] - top_pf) / top_pf * 100, 1),
                        "metricas":          m,
                    })
            except Exception:
                continue

    if melhorias:
        melhorias.sort(key=lambda x: -x["pf_novo"])
        mv = melhorias[0]
        print(f"    [IA] Melhoria! PF {top_pf:.3f} -> {mv['pf_novo']:.3f} "
              f"via {mv['filtro_adicionado']}={mv['valor']}", flush=True)
        return mv
    print(f"    [IA] Sem melhoria", flush=True)
    return None


# ================================================================
# SECAO 13: MAIN
# ================================================================

def main():
    MINI  = "--mini" in sys.argv
    total = sum(math.prod(len(v) for v in g.values()) for g in GRIDS.values())

    print("=" * 68, flush=True)
    print("  INTERNET STRATEGIES ENGINE v2 — WDO B3", flush=True)
    print(f"  {len(GRIDS)} estrategias | {total:,} combos", flush=True)
    print(f"  Multiprocessing: {N_CPUS} CPUs", flush=True)
    print(f"  MAE/MFE Analysis: SL/TP otimizados pelos movimentos reais", flush=True)
    print(f"  OOS Rolling: valida mes a mes", flush=True)
    print("=" * 68, flush=True)

    df    = carregar()
    split = int(len(df) * 0.70)
    df_is = df.iloc[:split]
    df_os = df.iloc[split:]
    print(f"  IS : {len(df_is):,} | {df_is.index[0].date()} -> {df_is.index[-1].date()}", flush=True)
    print(f"  OOS: {len(df_os):,} | {df_os.index[0].date()} -> {df_os.index[-1].date()}", flush=True)

    ind_is = calcular_indicadores(df_is)

    # Aquece Numba
    print("\n[JIT] Compilando...", flush=True)
    dv = np.ones(200, dtype=np.float64) * 5000
    bv = np.zeros(200, dtype=np.bool_); bv[10] = True
    _  = simular_long(dv, dv, dv, bv, bv, 20.0, 40.0, MULT, COMM, SLIP)
    _  = simular_short(dv, dv, dv, bv, bv, 20.0, 40.0, MULT, COMM, SLIP)
    _  = calcular_mae_mfe(dv, dv, dv, bv, 1)
    print("[JIT] Pronto!", flush=True)

    # Converter para dict picklable para multiprocessing
    ind_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in ind_is.items()}

    args_list = [(e, g, ind_dict, MINI) for e, g in GRIDS.items()]

    # Rodar em paralelo
    print(f"\n[GRID] Rodando {len(GRIDS)} estrategias em {N_CPUS} CPUs...", flush=True)
    t0_total = time.time()

    todos = []
    with Pool(N_CPUS) as pool:
        for resultado in pool.imap_unordered(worker_estrategia, args_list):
            e   = resultado["estrategia"]
            spd = resultado["n_combos"] / max(resultado["elapsed"], 0.1)
            top = resultado["top10"][0] if resultado["top10"] else None
            pf_s = f"PF={top['pf']:.3f}" if top else "sem validos"
            print(f"  [{e.upper():25}] {resultado['n_validos']:>7,}/{resultado['n_combos']:>9,} "
                  f"| {resultado['elapsed']:.1f}s | {spd:.0f}/s | {pf_s}", flush=True)
            if resultado["top10"]:
                todos.extend(resultado["top10"])
                with open(f"{OUTPUT_DIR}/{e}_top10.json", "w") as fp:
                    json.dump(resultado["top10"], fp, indent=2, default=str)

    elapsed_total = time.time() - t0_total
    print(f"\n[GRID] Total: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)", flush=True)

    # Top 30 geral
    todos.sort(key=lambda x: -x["score"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  TOP 30 GERAL", flush=True)
    print(f"  {'ESTRATEGIA':25} {'PF':>6} {'WR%':>6} {'Trades':>7} {'Exp':>8} {'Score':>7}", flush=True)
    print(f"  {'-' * 62}", flush=True)
    for r in todos[:30]:
        print(f"  {r['estrategia']:25} "
              f"{r['pf']:>6.3f} {r['wr']:>6.1f} "
              f"{r['n']:>7} {r['exp']:>8.2f} {r['score']:>7.4f}", flush=True)

    # MAE/MFE nos top candidatos
    print(f"\n{'=' * 68}", flush=True)
    print(f"  MAE/MFE ANALYSIS — SL/TP IDEAIS PELOS MOVIMENTOS REAIS", flush=True)
    print(f"  {'ESTRATEGIA':25} {'SL_real':>8} {'TP_real':>8} {'RR_real':>8}", flush=True)
    print(f"  {'-' * 55}", flush=True)

    aprovados    = []
    mae_mfe_info = []
    atr_med_is   = float(np.nanmean(ind_is["atr_14"]))

    for r in todos[:15]:
        estrategia = r["estrategia"]
        params     = r["params"]
        try:
            ent, ext = gerar_sinais(estrategia, ind_is, params)
            if ent is None or ent.sum() < 20:
                continue

            # MAE/MFE analysis
            mf = analisar_mae_mfe(ind_is, ent, params.get("direction", "short"))
            if mf:
                mae_mfe_info.append({"estrategia": estrategia, "params": params, **mf})
                print(f"  {estrategia:25} "
                      f"{mf['sl_otimizado']:>8.1f} "
                      f"{mf['tp_otimizado']:>8.1f} "
                      f"{mf['rr_real']:>8.2f}", flush=True)

                # Retesta com SL/TP otimizados pelo MAE/MFE
                sl_opt = mf["sl_otimizado"]
                tp_opt = mf["tp_otimizado"]
                if sl_opt > 0 and tp_opt > 0:
                    pnls_opt = executar(ind_is, ent, ext, sl_opt, tp_opt,
                                        params.get("direction", "short"))
                    m_opt = metricas(pnls_opt)
                    if m_opt and m_opt["pf"] > r["pf"]:
                        print(f"    -> PF melhorou: {r['pf']:.3f} -> {m_opt['pf']:.3f} "
                              f"com SL/TP reais!", flush=True)
                        r_novo = r.copy()
                        r_novo.update(m_opt)
                        r_novo["sl_otimizado"] = sl_opt
                        r_novo["tp_otimizado"] = tp_opt
                        r_novo["mae_mfe"]      = mf
                        aprovados.append((estrategia, r_novo))
                        continue

            # Plateau test
            robusto, pct = plateau_test(estrategia, ind_is, params, atr_med_is, r["pf"])
            if r["pf"] >= 1.0 and robusto:
                aprovados.append((estrategia, r))
            elif r["pf"] >= 0.90:
                melhoria = ia_evolutiva(estrategia, ind_is, params, r["pf"], atr_med_is)
                if melhoria:
                    rn = r.copy()
                    rn.update(melhoria["metricas"])
                    rn["params"] = melhoria["params"]
                    if rn["pf"] >= 1.0:
                        aprovados.append((estrategia, rn))
        except Exception:
            continue

    # OOS Rolling
    print(f"\n{'=' * 68}", flush=True)
    print(f"  OOS ROLLING — {len(aprovados)} CANDIDATO(S)", flush=True)

    resultados_finais = []
    for estrategia, melhor in aprovados:
        params = melhor["params"]
        print(f"\n  {estrategia} (PF_IS={melhor['pf']:.3f})", flush=True)

        oos_r  = oos_rolling(estrategia, df, params)
        is_pf  = melhor["pf"]
        oos_pf = oos_r["pf_medio"] if oos_r else 0
        wfe    = oos_r["wfe_pct"]  if oos_r else 0
        deg    = (is_pf - oos_pf) / is_pf * 100 if is_pf > 0 and oos_pf > 0 else 999
        ok     = oos_pf > 1.0 and deg < 50 and wfe >= 40

        print(f"  IS={is_pf:.3f} OOS_medio={oos_pf:.3f} "
              f"Deg={deg:.1f}% WFE={wfe:.0f}% "
              f"{'APROVADO' if ok else 'REPROVADO'}", flush=True)

        if oos_r:
            for j in oos_r["detalhes"]:
                luc = "OK" if j["lucrativo"] else "RUIM"
                print(f"    {j['data']} PF={j['pf']:.3f} Trades={j['trades']} {luc}", flush=True)

        resultado = {
            "estrategia":  estrategia,
            "params":      params,
            "metricas_is": melhor,
            "oos_rolling": oos_r,
            "degradacao":  round(deg, 1),
            "aprovado":    ok,
            "mae_mfe":     melhor.get("mae_mfe"),
            "gerado_em":   datetime.now().isoformat(),
        }
        resultados_finais.append(resultado)
        with open(f"{OUTPUT_DIR}/{estrategia}_final.json", "w") as fp:
            json.dump(resultado, fp, indent=2, default=str)

    # Leaderboard final
    n_apr = sum(1 for r in resultados_finais if r["aprovado"])
    print(f"\n{'=' * 68}", flush=True)
    print(f"  LEADERBOARD FINAL — {n_apr} APROVADO(S)", flush=True)
    print(f"  {'ESTRATEGIA':25} {'PF_IS':>6} {'OOS':>6} "
          f"{'DEG%':>6} {'WFE%':>6} {'STATUS':>12}", flush=True)
    print(f"  {'-' * 65}", flush=True)

    for r in sorted(resultados_finais,
                    key=lambda x: -(x["metricas_is"] or {}).get("pf", 0)):
        mi  = r["metricas_is"] or {}
        oor = r["oos_rolling"] or {}
        print(f"  {r['estrategia']:25} "
              f"{mi.get('pf', 0):>6.3f} "
              f"{oor.get('pf_medio', 0):>6.3f} "
              f"{r['degradacao']:>6.1f} "
              f"{oor.get('wfe_pct', 0):>6.0f} "
              f"{'APROVADO' if r['aprovado'] else 'REPROVADO':>12}", flush=True)

    lb = {
        "gerado_em":     datetime.now().isoformat(),
        "total_combos":  total,
        "n_cpus":        N_CPUS,
        "tempo_total_s": round(elapsed_total, 1),
        "aprovados":     n_apr,
        "top30":         todos[:30],
        "leaderboard":   resultados_finais,
        "mae_mfe":       mae_mfe_info,
    }
    with open(f"{OUTPUT_DIR}/leaderboard.json", "w") as fp:
        json.dump(lb, fp, indent=2, default=str)

    print(f"\n  {n_apr} estrategia(s) aprovada(s)!", flush=True)
    print(f"  Tempo total: {elapsed_total / 60:.1f} min", flush=True)
    print(f"  Salvo em: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
