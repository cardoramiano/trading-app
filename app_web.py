import datetime
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Quant Trading Terminal", layout="wide", initial_sidebar_state="expanded")

def calcola_pattern_e_indicatori(df):
    open_p, high, low, close = df['Open'], df['High'], df['Low'], df['Close']
    body = (close - open_p).abs()
    candle_range = high - low
    
    df['Pattern_Doji'] = np.where(body <= (candle_range * 0.1), 1, 0)
    lower_shadow = np.minimum(open_p, close) - low
    upper_shadow = high - np.maximum(open_p, close)
    df['Pattern_Hammer'] = np.where((lower_shadow >= 2 * body) & (upper_shadow <= body * 0.5) & (body > 0), 1, 0)
    df['Pattern_Shooting_Star'] = np.where((upper_shadow >= 2 * body) & (lower_shadow <= body * 0.5) & (body > 0), -1, 0)

    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_50'] = close.rolling(50).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['Bollinger_Upper'] = sma_20 + (std_20 * 2)
    df['Bollinger_Lower'] = sma_20 - (std_20 * 2)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    df['Return_1d'] = close.pct_change(1)
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std()

    return df

# --- SIDEBAR CONTROLLI ---
st.sidebar.title("QUANT TERMINAL PRO")

query_search = st.sidebar.text_input("Cerca Azienda o Ticker:", "Apple")

ticker_selezionato = "AAPL"
if len(query_search) >= 2:
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query_search}&quotesCount=5"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=3).json()
        options = [f"{q['symbol']} - {q.get('shortname', 'N/A')}" for q in res.get('quotes', []) if 'symbol' in q]
        if options:
            scelta = st.sidebar.selectbox("Seleziona Azienda:", options)
            ticker_selezionato = scelta.split(" - ")[0]
    except Exception:
        ticker_selezionato = query_search.upper()

giorni_futuri = st.sidebar.slider("Orizzonte Previsione (Giorni):", 1, 15, 7)
investimento = st.sidebar.number_input("Capitale da Investire ($):", min_value=100.0, value=2000.0, step=100.0)

btn_run = st.sidebar.button("ELABORA ANALISI PRO", type="primary")

# --- MAIN DASHBOARD ---
st.title("Financial Analytics & Predictive Engine")

if btn_run or ticker_selezionato:
    with st.spinner("Elaborazione dati e addestramento modello AI..."):
        try:
            df = yf.download(ticker_selezionato, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 60:
                st.error("Dati non disponibili per il ticker selezionato.")
                st.stop()

            df = calcola_pattern_e_indicatori(df)
            df.dropna(inplace=True)

            features = [
                'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal_Line',
                'Bollinger_Upper', 'Bollinger_Lower', 'Volatility_20d', 'ATR_14',
                'Pattern_Doji', 'Pattern_Hammer', 'Pattern_Shooting_Star'
            ]

            previsioni_future = []
            ultimi_dati = df[features].iloc[-1:]

            for step in range(1, giorni_futuri + 1):
                df_step = df.copy()
                df_step['Target_Step'] = df_step['Close'].shift(-step)
                df_step_clean = df_step.dropna(subset=['Target_Step'])

                X_step = df_step_clean[features]
                y_step = df_step_clean['Target_Step']

                model_step = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)
                model_step.fit(X_step, y_step)
                pred_step = model_step.predict(ultimi_dati)[0]
                previsioni_future.append(pred_step)

            ultima_data = df.index[-1]
            date_future = []
            curr_date = ultima_data
            while len(date_future) < giorni_futuri:
                curr_date += datetime.timedelta(days=1)
                if curr_date.weekday() < 5:
                    date_future.append(curr_date)

            prezzo_attuale = float(df['Close'].iloc[-1])
            prezzo_finale = float(previsioni_future[-1])
            var_perc = ((prezzo_finale - prezzo_attuale) / prezzo_attuale) * 100
            atr_corrente = float(df['ATR_14'].iloc[-1])

            distanza_sl = 1.5 * atr_corrente
            num_azioni = investimento / prezzo_attuale

            if var_perc >= 0:
                stop_loss = prezzo_attuale - distanza_sl
                take_profit_1 = prezzo_attuale + (1.5 * distanza_sl)
                take_profit_2 = prezzo_attuale + (2.5 * distanza_sl)
                gain_tp1 = num_azioni * (take_profit_1 - prezzo_attuale)
                gain_tp2 = num_azioni * (take_profit_2 - prezzo_attuale)
                loss_sl = num_azioni * (prezzo_attuale - stop_loss)
            else:
                stop_loss = prezzo_attuale + distanza_sl
                take_profit_1 = prezzo_attuale - (1.5 * distanza_sl)
                take_profit_2 = prezzo_attuale - (2.5 * distanza_sl)
                gain_tp1 = num_azioni * (prezzo_attuale - take_profit_1)
                gain_tp2 = num_azioni * (prezzo_attuale - take_profit_2)
                loss_sl = num_azioni * (stop_loss - prezzo_attuale)

            # METRICHE PRINCIPALI IN ALTO
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Ticker", ticker_selezionato)
            kpi2.metric("Prezzo Attuale", f"${prezzo_attuale:.2f}")
            kpi3.metric("Target Stimato", f"${prezzo_finale:.2f}", f"{var_perc:+.2f}%")
            kpi4.metric("Stop Loss", f"${stop_loss:.2f}")

            # GRAFICO INTERATTIVO CON PLOTLY
            dati_rec = df.iloc[-50:]
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=dati_rec.index,
                open=dati_rec['Open'], high=dati_rec['High'],
                low=dati_rec['Low'], close=dati_rec['Close'],
                name='Storico'
            ))

            # Proiezione Futura
            x_fut = [dati_rec.index[-1]] + list(date_future)
            y_fut = [prezzo_attuale] + list(previsioni_future)
            fig.add_trace(go.Scatter(x=x_fut, y=y_fut, mode='lines+markers', name='Proiezione AI', line=dict(color='#FF007F', width=3, dash='dash')))

            # Livelli TP e SL
            fig.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop Loss")
            fig.add_hline(y=take_profit_1, line_dash="dot", line_color="green", annotation_text="Take Profit 1")

            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # CARD RISULTATI
            c1, c2, c3 = st.columns(3)
            c1.success(f"**Profitto TP1:** +${gain_tp1:.2f}")
            c2.success(f"**Profitto TP2:** +${gain_tp2:.2f}")
            c3.error(f"**Perdita Max SL:** -${loss_sl:.2f}")

        except Exception as e:
            st.error(f"Errore durante l'elaborazione: {str(e)}")
