"""
FlowTrace Technical Indicator Library

Computes 38+ technical indicators from OHLCV DataFrames.
No external dependencies beyond numpy/pandas (already installed).

Categories:
  - Trend Indicators (13): SMA, EMA, WMA, MACD, PSAR, ADX, Ichimoku, Aroon,
    SuperTrend, VWAP, HMA, DEMA, TEMA
  - Oscillators (15): RSI, Stochastic, StochRSI, CCI, Williams %R, ROC,
    Momentum, RSI Divergence, Ultimate Oscillator, DPO, Elder, CMO, KST, TSI, PPO
  - Volume Indicators (4): OBV, MFI, CMF, A/D
  - Bill Williams (6): Alligator, Fractals, AO, AC, Gator, BW-MFI
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3


# ============================================================================
# TREND INDICATORS (13)
# ============================================================================

def compute_trend_indicators(df: pd.DataFrame, indicators: list = None) -> dict:
    """Compute requested trend indicators from OHLCV DataFrame."""
    if indicators is None:
        indicators = [
            "SMA", "EMA", "WMA", "MACD", "PSAR", "ADX", "ICHIMOKU",
            "AROON", "SUPERTREND", "VWAP", "HMA", "DEMA", "TEMA",
        ]

    results = {}
    close = df["Close"]

    if "SMA" in indicators:
        results["SMA_20"] = round(float(_sma(close, 20).iloc[-1]), 4) if len(close) >= 20 else None
        results["SMA_50"] = round(float(_sma(close, 50).iloc[-1]), 4) if len(close) >= 50 else None
        results["SMA_200"] = round(float(_sma(close, 200).iloc[-1]), 4) if len(close) >= 200 else None
        results["price_vs_SMA50"] = "above" if results.get("SMA_50") and close.iloc[-1] > results["SMA_50"] else "below"
        results["price_vs_SMA200"] = "above" if results.get("SMA_200") and close.iloc[-1] > results["SMA_200"] else "below"

    if "EMA" in indicators:
        results["EMA_5"] = round(float(_ema(close, 5).iloc[-1]), 4) if len(close) >= 5 else None
        results["EMA_9"] = round(float(_ema(close, 9).iloc[-1]), 4) if len(close) >= 9 else None
        results["EMA_21"] = round(float(_ema(close, 21).iloc[-1]), 4) if len(close) >= 21 else None
        results["EMA_50"] = round(float(_ema(close, 50).iloc[-1]), 4) if len(close) >= 50 else None

    if "WMA" in indicators:
        results["WMA_20"] = round(float(_wma(close, 20).iloc[-1]), 4) if len(close) >= 20 else None

    if "MACD" in indicators and len(close) >= 26:
        macd_line = _ema(close, 12) - _ema(close, 26)
        signal_line = _ema(macd_line, 9)
        histogram = macd_line - signal_line
        results["MACD_line"] = round(float(macd_line.iloc[-1]), 4)
        results["MACD_signal"] = round(float(signal_line.iloc[-1]), 4)
        results["MACD_histogram"] = round(float(histogram.iloc[-1]), 4)
        # Crossover detection
        if len(histogram) >= 2:
            prev_hist = histogram.iloc[-2]
            curr_hist = histogram.iloc[-1]
            if prev_hist < 0 and curr_hist >= 0:
                results["MACD_crossover"] = "bullish_cross"
            elif prev_hist > 0 and curr_hist <= 0:
                results["MACD_crossover"] = "bearish_cross"
            else:
                results["MACD_crossover"] = "none"

    if "PSAR" in indicators and len(df) >= 5:
        results.update(_parabolic_sar(df))

    if "ADX" in indicators and len(df) >= 28:
        results.update(_adx(df))

    if "ICHIMOKU" in indicators and len(df) >= 52:
        results.update(_ichimoku(df))

    if "AROON" in indicators and len(df) >= 25:
        results.update(_aroon(df))

    if "SUPERTREND" in indicators and len(df) >= 14:
        results.update(_supertrend(df))

    if "VWAP" in indicators and "Volume" in df.columns:
        tp = _typical_price(df)
        cumulative_tp_vol = (tp * df["Volume"]).cumsum()
        cumulative_vol = df["Volume"].cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
        results["VWAP"] = round(float(vwap.iloc[-1]), 4)

    if "HMA" in indicators and len(close) >= 20:
        half_period = 10
        full_wma = _wma(close, 20)
        half_wma = _wma(close, half_period)
        diff = 2 * half_wma - full_wma
        sqrt_period = int(np.sqrt(20))
        hma = _wma(diff.dropna(), sqrt_period)
        results["HMA_20"] = round(float(hma.iloc[-1]), 4) if len(hma) > 0 else None

    if "DEMA" in indicators and len(close) >= 20:
        ema1 = _ema(close, 20)
        ema2 = _ema(ema1, 20)
        results["DEMA_20"] = round(float((2 * ema1 - ema2).iloc[-1]), 4)

    if "TEMA" in indicators and len(close) >= 20:
        ema1 = _ema(close, 20)
        ema2 = _ema(ema1, 20)
        ema3 = _ema(ema2, 20)
        results["TEMA_20"] = round(float((3 * ema1 - 3 * ema2 + ema3).iloc[-1]), 4)

    return results


def _parabolic_sar(df: pd.DataFrame, af_start=0.02, af_max=0.2) -> dict:
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)

    psar = np.zeros(n)
    af = af_start
    bull = True
    ep = low[0]
    psar[0] = high[0]

    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[max(0, i-2)])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = af_start
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[max(0, i-2)])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = af_start
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)

    return {
        "PSAR": round(float(psar[-1]), 4),
        "PSAR_trend": "bullish" if bull else "bearish",
    }


def _adx(df: pd.DataFrame, period: int = 14) -> dict:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = _true_range(df)
    atr = _ema(tr, period)

    plus_di = 100 * _ema(plus_dm, period) / atr
    minus_di = 100 * _ema(minus_dm, period) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _ema(dx, period)

    return {
        "ADX": round(float(adx.iloc[-1]), 2),
        "plus_DI": round(float(plus_di.iloc[-1]), 2),
        "minus_DI": round(float(minus_di.iloc[-1]), 2),
        "ADX_trend_strength": "strong" if adx.iloc[-1] > 25 else "weak",
    }


def _ichimoku(df: pd.DataFrame) -> dict:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    return {
        "Ichimoku_tenkan": round(float(tenkan.iloc[-1]), 4),
        "Ichimoku_kijun": round(float(kijun.iloc[-1]), 4),
        "Ichimoku_senkou_a": round(float(senkou_a.iloc[-1]), 4) if pd.notna(senkou_a.iloc[-1]) else None,
        "Ichimoku_senkou_b": round(float(senkou_b.iloc[-1]), 4) if pd.notna(senkou_b.iloc[-1]) else None,
        "Ichimoku_cloud": "bullish" if close.iloc[-1] > max(
            senkou_a.iloc[-1] or 0, senkou_b.iloc[-1] or 0
        ) else "bearish",
    }


def _aroon(df: pd.DataFrame, period: int = 25) -> dict:
    high = df["High"]
    low = df["Low"]

    aroon_up = 100 * high.rolling(period + 1).apply(lambda x: x.argmax(), raw=True) / period
    aroon_down = 100 * low.rolling(period + 1).apply(lambda x: x.argmin(), raw=True) / period

    return {
        "Aroon_up": round(float(aroon_up.iloc[-1]), 2),
        "Aroon_down": round(float(aroon_down.iloc[-1]), 2),
        "Aroon_oscillator": round(float(aroon_up.iloc[-1] - aroon_down.iloc[-1]), 2),
    }


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    tr = _true_range(df)
    atr = _sma(tr, period)
    hl2 = (df["High"] + df["Low"]) / 2

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    close = df["Close"]
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    supertrend.iloc[period] = upper.iloc[period]
    direction.iloc[period] = -1

    for i in range(period + 1, len(df)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = max(lower.iloc[i], supertrend.iloc[i-1]) if direction.iloc[i-1] == 1 else lower.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(upper.iloc[i], supertrend.iloc[i-1]) if direction.iloc[i-1] == -1 else upper.iloc[i]
            direction.iloc[i] = -1

    return {
        "SuperTrend": round(float(supertrend.iloc[-1]), 4),
        "SuperTrend_direction": "bullish" if direction.iloc[-1] == 1 else "bearish",
    }


# ============================================================================
# OSCILLATORS (15)
# ============================================================================

def compute_oscillators(df: pd.DataFrame, indicators: list = None) -> dict:
    """Compute requested oscillator indicators."""
    if indicators is None:
        indicators = [
            "RSI", "STOCH", "STOCH_RSI", "CCI", "WILLR", "ROC", "MOM",
            "RSI_DIV", "UO", "DPO", "ELDER", "CMO", "KST", "TSI", "PPO",
        ]

    results = {}
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    if "RSI" in indicators and len(close) >= 14:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = _ema(gain, 14)
        avg_loss = _ema(loss, 14)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])
        results["RSI_14"] = round(rsi_val, 2)
        if rsi_val > 70:
            results["RSI_signal"] = "overbought"
        elif rsi_val < 30:
            results["RSI_signal"] = "oversold"
        else:
            results["RSI_signal"] = "neutral"

    if "STOCH" in indicators and len(df) >= 14:
        l14 = low.rolling(14).min()
        h14 = high.rolling(14).max()
        denom = (h14 - l14).replace(0, np.nan)
        k = 100 * (close - l14) / denom
        d = _sma(k, 3)
        results["Stoch_K"] = round(float(k.iloc[-1]), 2)
        results["Stoch_D"] = round(float(d.iloc[-1]), 2)
        results["Stoch_signal"] = "overbought" if k.iloc[-1] > 80 else "oversold" if k.iloc[-1] < 20 else "neutral"

    if "STOCH_RSI" in indicators and len(close) >= 28:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = _ema(gain, 14)
        avg_loss = _ema(loss, 14)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        results["StochRSI"] = round(float(stoch_rsi.iloc[-1]), 4) if pd.notna(stoch_rsi.iloc[-1]) else None

    if "CCI" in indicators and len(df) >= 20:
        tp = _typical_price(df)
        tp_sma = _sma(tp, 20)
        mean_dev = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))
        results["CCI_20"] = round(float(cci.iloc[-1]), 2)
        results["CCI_signal"] = "overbought" if cci.iloc[-1] > 100 else "oversold" if cci.iloc[-1] < -100 else "neutral"

    if "WILLR" in indicators and len(df) >= 14:
        h14 = high.rolling(14).max()
        l14 = low.rolling(14).min()
        willr = -100 * (h14 - close) / (h14 - l14)
        results["WilliamsR_14"] = round(float(willr.iloc[-1]), 2)
        results["WilliamsR_signal"] = "overbought" if willr.iloc[-1] > -20 else "oversold" if willr.iloc[-1] < -80 else "neutral"

    if "ROC" in indicators and len(close) >= 12:
        roc = ((close - close.shift(12)) / close.shift(12)) * 100
        results["ROC_12"] = round(float(roc.iloc[-1]), 2)

    if "MOM" in indicators and len(close) >= 10:
        mom = close - close.shift(10)
        results["Momentum_10"] = round(float(mom.iloc[-1]), 4)

    if "RSI_DIV" in indicators and len(close) >= 28:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = _ema(gain, 14)
        avg_loss = _ema(loss, 14)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Simple divergence: price making new high but RSI not, or vice versa
        price_up = close.iloc[-1] > close.iloc[-14]
        rsi_up = rsi.iloc[-1] > rsi.iloc[-14]
        if price_up and not rsi_up:
            results["RSI_divergence"] = "bearish_divergence"
        elif not price_up and rsi_up:
            results["RSI_divergence"] = "bullish_divergence"
        else:
            results["RSI_divergence"] = "none"

    if "UO" in indicators and len(df) >= 28:
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = _true_range(df)
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        results["UltimateOsc"] = round(float(uo.iloc[-1]), 2)

    if "DPO" in indicators and len(close) >= 20:
        shift = 20 // 2 + 1
        sma20 = _sma(close, 20)
        dpo = close.shift(shift) - sma20
        if pd.notna(dpo.iloc[-1]):
            results["DPO_20"] = round(float(dpo.iloc[-1]), 4)

    if "ELDER" in indicators and len(df) >= 13:
        ema13 = _ema(close, 13)
        results["Elder_bull_power"] = round(float(high.iloc[-1] - ema13.iloc[-1]), 4)
        results["Elder_bear_power"] = round(float(low.iloc[-1] - ema13.iloc[-1]), 4)

    if "CMO" in indicators and len(close) >= 14:
        delta = close.diff()
        sum_up = delta.where(delta > 0, 0).rolling(14).sum()
        sum_down = (-delta).where(delta < 0, 0).rolling(14).sum()
        cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down)
        results["CMO_14"] = round(float(cmo.iloc[-1]), 2) if pd.notna(cmo.iloc[-1]) else None

    if "KST" in indicators and len(close) >= 30:
        roc1 = ((close - close.shift(10)) / close.shift(10)) * 100
        roc2 = ((close - close.shift(15)) / close.shift(15)) * 100
        roc3 = ((close - close.shift(20)) / close.shift(20)) * 100
        roc4 = ((close - close.shift(30)) / close.shift(30)) * 100
        kst = _sma(roc1, 10) * 1 + _sma(roc2, 10) * 2 + _sma(roc3, 10) * 3 + _sma(roc4, 15) * 4
        kst_signal = _sma(kst, 9)
        results["KST"] = round(float(kst.iloc[-1]), 2) if pd.notna(kst.iloc[-1]) else None
        results["KST_signal_line"] = round(float(kst_signal.iloc[-1]), 2) if pd.notna(kst_signal.iloc[-1]) else None

    if "TSI" in indicators and len(close) >= 25:
        delta = close.diff()
        double_smoothed = _ema(_ema(delta, 25), 13)
        double_smoothed_abs = _ema(_ema(delta.abs(), 25), 13)
        tsi = 100 * double_smoothed / double_smoothed_abs
        results["TSI"] = round(float(tsi.iloc[-1]), 2) if pd.notna(tsi.iloc[-1]) else None

    if "PPO" in indicators and len(close) >= 26:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        ppo = 100 * (ema12 - ema26) / ema26
        results["PPO"] = round(float(ppo.iloc[-1]), 2)
        results["PPO_signal"] = round(float(_ema(ppo, 9).iloc[-1]), 2)

    return results


# ============================================================================
# VOLUME INDICATORS (4)
# ============================================================================

def compute_volume_indicators(df: pd.DataFrame, indicators: list = None) -> dict:
    """Compute requested volume indicators."""
    if indicators is None:
        indicators = ["OBV", "MFI", "CMF", "AD"]

    results = {}
    close = df["Close"]
    volume = df.get("Volume", pd.Series(dtype=float))

    if volume.empty or volume.sum() == 0:
        return results

    if "OBV" in indicators:
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        results["OBV"] = int(obv.iloc[-1])
        # Trend: compare current OBV to 20-period SMA of OBV
        if len(obv) >= 20:
            obv_sma = _sma(obv, 20).iloc[-1]
            results["OBV_trend"] = "rising" if obv.iloc[-1] > obv_sma else "falling"

    if "MFI" in indicators and len(df) >= 14:
        tp = _typical_price(df)
        mf = tp * volume
        delta_tp = tp.diff()
        pos_mf = mf.where(delta_tp > 0, 0).rolling(14).sum()
        neg_mf = mf.where(delta_tp <= 0, 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
        mfi_val = float(mfi.iloc[-1])
        results["MFI_14"] = round(mfi_val, 2)
        results["MFI_signal"] = "overbought" if mfi_val > 80 else "oversold" if mfi_val < 20 else "neutral"

    if "CMF" in indicators and len(df) >= 20:
        clv = ((close - df["Low"]) - (df["High"] - close)) / (df["High"] - df["Low"]).replace(0, np.nan)
        cmf = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
        results["CMF_20"] = round(float(cmf.iloc[-1]), 4) if pd.notna(cmf.iloc[-1]) else None
        if results.get("CMF_20") is not None:
            results["CMF_signal"] = "accumulation" if results["CMF_20"] > 0 else "distribution"

    if "AD" in indicators:
        clv = ((close - df["Low"]) - (df["High"] - close)) / (df["High"] - df["Low"]).replace(0, np.nan)
        ad = (clv.fillna(0) * volume).cumsum()
        results["AD"] = round(float(ad.iloc[-1]), 0)

    return results


# ============================================================================
# BILL WILLIAMS INDICATORS (6)
# ============================================================================

def compute_bill_williams(df: pd.DataFrame, indicators: list = None) -> dict:
    """Compute requested Bill Williams indicators."""
    if indicators is None:
        indicators = ["ALLIGATOR", "FRACTALS", "AO", "AC", "GATOR", "BW_MFI"]

    results = {}
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    midprice = (high + low) / 2

    if "ALLIGATOR" in indicators and len(df) >= 21:
        # Jaw (blue): 13-period SMMA offset 8
        # Teeth (red): 8-period SMMA offset 5
        # Lips (green): 5-period SMMA offset 3
        jaw = _sma(midprice, 13).shift(8)
        teeth = _sma(midprice, 8).shift(5)
        lips = _sma(midprice, 5).shift(3)
        results["Alligator_jaw"] = round(float(jaw.iloc[-1]), 4) if pd.notna(jaw.iloc[-1]) else None
        results["Alligator_teeth"] = round(float(teeth.iloc[-1]), 4) if pd.notna(teeth.iloc[-1]) else None
        results["Alligator_lips"] = round(float(lips.iloc[-1]), 4) if pd.notna(lips.iloc[-1]) else None

        # Determine state
        if results["Alligator_jaw"] and results["Alligator_teeth"] and results["Alligator_lips"]:
            j, t, l = results["Alligator_jaw"], results["Alligator_teeth"], results["Alligator_lips"]
            if l > t > j:
                results["Alligator_state"] = "uptrend"
            elif l < t < j:
                results["Alligator_state"] = "downtrend"
            else:
                results["Alligator_state"] = "sleeping"

    if "FRACTALS" in indicators and len(df) >= 5:
        # Bullish fractal: middle bar is lowest of 5
        # Bearish fractal: middle bar is highest of 5
        up_fractals = []
        down_fractals = []
        for i in range(2, len(df) - 2):
            if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and
                    high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]):
                up_fractals.append((i, float(high.iloc[i])))
            if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and
                    low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]):
                down_fractals.append((i, float(low.iloc[i])))

        results["Fractal_up_latest"] = round(up_fractals[-1][1], 4) if up_fractals else None
        results["Fractal_down_latest"] = round(down_fractals[-1][1], 4) if down_fractals else None

    if "AO" in indicators and len(df) >= 34:
        ao = _sma(midprice, 5) - _sma(midprice, 34)
        results["AO"] = round(float(ao.iloc[-1]), 4)
        if len(ao.dropna()) >= 2:
            results["AO_direction"] = "rising" if ao.iloc[-1] > ao.iloc[-2] else "falling"

    if "AC" in indicators and len(df) >= 39:
        ao = _sma(midprice, 5) - _sma(midprice, 34)
        ac = ao - _sma(ao, 5)
        results["AC"] = round(float(ac.iloc[-1]), 4) if pd.notna(ac.iloc[-1]) else None

    if "GATOR" in indicators and len(df) >= 21:
        jaw = _sma(midprice, 13).shift(8)
        teeth = _sma(midprice, 8).shift(5)
        lips = _sma(midprice, 5).shift(3)
        upper_hist = (jaw - teeth).abs()
        lower_hist = -(teeth - lips).abs()
        results["Gator_upper"] = round(float(upper_hist.iloc[-1]), 4) if pd.notna(upper_hist.iloc[-1]) else None
        results["Gator_lower"] = round(float(lower_hist.iloc[-1]), 4) if pd.notna(lower_hist.iloc[-1]) else None

    if "BW_MFI" in indicators and "Volume" in df.columns:
        bw_mfi = (high - low) / df["Volume"].replace(0, np.nan)
        results["BW_MFI"] = round(float(bw_mfi.iloc[-1]), 8) if pd.notna(bw_mfi.iloc[-1]) else None

    return results


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def compute_all(df: pd.DataFrame, indicators: list = None) -> dict:
    """
    Compute all requested indicators (or all if None).

    If indicators is a list, only compute those specific ones.
    Each indicator name is matched against all four categories.
    """
    results = {}
    results.update(compute_trend_indicators(df, indicators))
    results.update(compute_oscillators(df, indicators))
    results.update(compute_volume_indicators(df, indicators))
    results.update(compute_bill_williams(df, indicators))

    # Add current price for reference
    results["current_price"] = round(float(df["Close"].iloc[-1]), 4)
    results["current_volume"] = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else None

    # Clean NaN/inf values recursively — replace with None for safe JSON serialization
    def _clean_nan(val):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        elif isinstance(val, dict):
            return {k: _clean_nan(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_clean_nan(v) for v in val]
        return val

    return _clean_nan(results)


def get_indicators_for_timeframe(style: str) -> list:
    """
    Return recommended indicator set based on trading style.

    Args:
        style: "day_trader", "swing_single_week", "swing_multi_week", "value_investor"
    """
    presets = {
        "day_trader": [
            "RSI", "STOCH", "MACD", "EMA", "VWAP", "OBV",
            "MFI", "SUPERTREND", "PSAR", "CCI", "WILLR",
            "ELDER", "FRACTALS",
        ],
        "swing_single_week": [
            "RSI", "MACD", "STOCH", "STOCH_RSI", "EMA", "SMA",
            "VWAP", "OBV", "MFI", "CCI", "SUPERTREND",
        ],
        "swing_multi_week": [
            "RSI", "MACD", "STOCH", "SMA", "EMA", "ADX",
            "OBV", "CMF", "ALLIGATOR", "AO",
        ],
        "value_investor": [
            "SMA", "MACD", "RSI", "OBV",
        ],
    }
    return presets.get(style, presets["swing_multi_week"])


def interpret_signals(results: dict) -> list:
    """
    Produce human-readable signal interpretations from computed indicators.
    Returns list of signal strings like "RSI oversold (28.3)", "MACD bullish crossover".
    """
    signals = []

    # RSI
    rsi_sig = results.get("RSI_signal")
    if rsi_sig == "overbought":
        signals.append(f"RSI overbought ({results.get('RSI_14', '?')})")
    elif rsi_sig == "oversold":
        signals.append(f"RSI oversold ({results.get('RSI_14', '?')})")

    # MACD crossover
    macd_cross = results.get("MACD_crossover")
    if macd_cross == "bullish_cross":
        signals.append("MACD bullish crossover")
    elif macd_cross == "bearish_cross":
        signals.append("MACD bearish crossover")

    # Stochastic
    stoch_sig = results.get("Stoch_signal")
    if stoch_sig == "overbought":
        signals.append(f"Stochastic overbought ({results.get('Stoch_K', '?')})")
    elif stoch_sig == "oversold":
        signals.append(f"Stochastic oversold ({results.get('Stoch_K', '?')})")

    # ADX
    adx_strength = results.get("ADX_trend_strength")
    if adx_strength == "strong":
        signals.append(f"Strong trend (ADX={results.get('ADX', '?')})")

    # Price vs SMAs
    if results.get("price_vs_SMA50") == "above" and results.get("price_vs_SMA200") == "above":
        signals.append("Price above 50 & 200 SMA (bullish structure)")
    elif results.get("price_vs_SMA50") == "below" and results.get("price_vs_SMA200") == "below":
        signals.append("Price below 50 & 200 SMA (bearish structure)")

    # SuperTrend
    st_dir = results.get("SuperTrend_direction")
    if st_dir:
        signals.append(f"SuperTrend {st_dir}")

    # PSAR
    psar_trend = results.get("PSAR_trend")
    if psar_trend:
        signals.append(f"Parabolic SAR {psar_trend}")

    # Ichimoku
    cloud = results.get("Ichimoku_cloud")
    if cloud:
        signals.append(f"Ichimoku cloud {cloud}")

    # Volume
    obv_trend = results.get("OBV_trend")
    if obv_trend:
        signals.append(f"OBV {obv_trend}")

    mfi_sig = results.get("MFI_signal")
    if mfi_sig in ("overbought", "oversold"):
        signals.append(f"MFI {mfi_sig} ({results.get('MFI_14', '?')})")

    cmf_sig = results.get("CMF_signal")
    if cmf_sig:
        signals.append(f"Chaikin Money Flow: {cmf_sig}")

    # Bill Williams
    alligator = results.get("Alligator_state")
    if alligator:
        signals.append(f"Alligator: {alligator}")

    ao_dir = results.get("AO_direction")
    if ao_dir:
        signals.append(f"Awesome Oscillator {ao_dir}")

    # RSI divergence
    div = results.get("RSI_divergence")
    if div and div != "none":
        signals.append(f"RSI {div.replace('_', ' ')}")

    # CCI
    cci_sig = results.get("CCI_signal")
    if cci_sig in ("overbought", "oversold"):
        signals.append(f"CCI {cci_sig} ({results.get('CCI_20', '?')})")

    # Williams %R
    willr_sig = results.get("WilliamsR_signal")
    if willr_sig in ("overbought", "oversold"):
        signals.append(f"Williams %R {willr_sig}")

    return signals
