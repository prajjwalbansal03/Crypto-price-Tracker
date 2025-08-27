# crypto_tracker.py
import time
import math
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------- Config --------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
SIMPLE_PRICE_BATCH = 250
MAX_WORKERS = 6  # limit parallelism to avoid triggering rate limits
REQUEST_TIMEOUT = 15

TIMEFRAMES = {"7D": 7, "30D": 30, "90D": 90, "1Y": 365}
PORTFOLIO_TIMEFRAMES = {"30D": 30, "90D": 90, "1Y": 365}


# -------------------- Requests session with retry --------------------
@st.cache_resource
def get_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# -------------------- Cached API helpers --------------------
@st.cache_data(ttl=300)
def cg_ping() -> Dict[str, Any]:
    s = get_session()
    r = s.get(f"{COINGECKO_BASE}/ping", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def get_global_stats() -> Dict[str, Any]:
    s = get_session()
    r = s.get(f"{COINGECKO_BASE}/global", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def get_market_data(vs_currency: str, per_page: int = 250, page: int = 1) -> List[Dict[str, Any]]:
    s = get_session()
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": False,
        "price_change_percentage": "24h",
    }
    r = s.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=90)
def get_simple_price(coin_id: str, vs_currency: str) -> Dict[str, Any]:
    s = get_session()
    params = {"ids": coin_id, "vs_currencies": vs_currency}
    r = s.get(f"{COINGECKO_BASE}/simple/price", params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=90)
def get_simple_prices_bulk(coin_ids: List[str], vs_currency: str) -> Dict[str, Dict[str, float]]:
    """Batch coin ids into chunks acceptable by CoinGecko and aggregate results."""
    if not coin_ids:
        return {}
    s = get_session()
    results: Dict[str, Dict[str, float]] = {}
    # chunk
    for i in range(0, len(coin_ids), SIMPLE_PRICE_BATCH):
        chunk = coin_ids[i : i + SIMPLE_PRICE_BATCH]
        params = {"ids": ",".join(chunk), "vs_currencies": vs_currency}
        r = s.get(f"{COINGECKO_BASE}/simple/price", params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        results.update(r.json() or {})
    return results


@st.cache_data(ttl=600)
def search_coins(query: str) -> List[Dict[str, Any]]:
    s = get_session()
    r = s.get(f"{COINGECKO_BASE}/search", params={"query": query}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("coins", []) or []
    return [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "symbol": (c.get("symbol") or "").upper(),
            "market_cap_rank": c.get("market_cap_rank"),
        }
        for c in data
    ]


@st.cache_data(ttl=600)
def get_market_chart(coin_id: str, vs_currency: str, days: int) -> pd.DataFrame:
    """Return df with columns: date, price (daily)."""
    s = get_session()
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    r = s.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    prices = data.get("prices", []) or []
    if not prices:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df[["date", "price"]].sort_values("date").reset_index(drop=True)
    return df


# -------------------- Chart helpers --------------------
def line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines", name=y_col))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def comparison_chart(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str):
    if df_a.empty or df_b.empty:
        return go.Figure(), pd.DataFrame()
    merged = pd.merge(df_a, df_b, on="date", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        return go.Figure(), pd.DataFrame()

    base_a = merged["price_a"].iloc[0]
    base_b = merged["price_b"].iloc[0]
    merged["idx_a"] = (merged["price_a"] / base_a) * 100.0
    merged["idx_b"] = (merged["price_b"] / base_b) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["idx_a"], mode="lines", name=label_a))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["idx_b"], mode="lines", name=label_b))
    fig.update_layout(
        title="Performance (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Index (100 = start)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )

    # Build metrics efficiently as a list of rows
    rows = []
    for label, col in [(label_a, "price_a"), (label_b, "price_b")]:
        start = merged[col].iloc[0]
        end = merged[col].iloc[-1]
        ret = (end / start - 1.0) * 100.0
        dr = merged[col].pct_change().dropna()
        vol_pct = dr.std() * 100.0 if not dr.empty else 0.0
        rows.append({"Metric": "Return over period (%)", "Asset": label, "Value": f"{ret:.2f}"})
        rows.append({"Metric": "Daily volatility (%)", "Asset": label, "Value": f"{vol_pct:.2f}"})

    metrics = pd.DataFrame(rows)
    # pivot for nicer presentation: each metric row grouped
    metrics_pivot = metrics.pivot(index="Metric", columns="Asset", values="Value").reset_index()
    return fig, metrics_pivot


def pie_chart(labels, values, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# -------------------- App UI --------------------
st.set_page_config(page_title="Crypto Tracker", page_icon="🪙", layout="wide")
st.title("🪙 Crypto Tracker (Optimized)")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("📊 Market Dashboard")
    currency = st.selectbox("Display currency", ["USD", "INR", "EUR", "GBP"], index=0, key="currency")
    vs = currency.lower()

    # Global stats
    try:
        global_stats = get_global_stats()
        mkt = global_stats.get("data", {}) or {}
        st.subheader("🌍 Global Market")
        st.metric("Active Cryptos", mkt.get("active_cryptocurrencies", "N/A"))
        st.metric("Markets", mkt.get("markets", "N/A"))
        btc_dom = mkt.get("market_cap_percentage", {}).get("btc", 0.0)
        st.metric("BTC Dominance", f"{btc_dom:.2f}%")
    except Exception as e:
        st.error(f"Global stats fetch failed: {e}")

    # Gainers / Losers
    try:
        st.subheader("🚀 Top Gainers (24h)")
        coins = get_market_data(vs, per_page=100)
        sorted_coins = sorted(coins, key=lambda x: x.get("price_change_percentage_24h") or 0, reverse=True)
        for c in sorted_coins[:5]:
            pct = c.get("price_change_percentage_24h") or 0
            st.markdown(
                f"**{c.get('name')} ({(c.get('symbol') or '').upper()})** — "
                f"<span style='color:green'>{pct:.2f}%</span>",
                unsafe_allow_html=True,
            )

        st.subheader("📉 Top Losers (24h)")
        sorted_coins = sorted(coins, key=lambda x: x.get("price_change_percentage_24h") or 0)
        for c in sorted_coins[:5]:
            pct = c.get("price_change_percentage_24h") or 0
            st.markdown(
                f"**{c.get('name')} ({(c.get('symbol') or '').upper()})** — "
                f"<span style='color:red'>{pct:.2f}%</span>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Gainers/losers fetch failed: {e}")

    st.divider()
    st.subheader("API Status")
    try:
        pong = cg_ping()
        st.success(f"CoinGecko status: {pong.get('gecko_says', 'OK')}")
    except Exception as e:
        st.error(f"CoinGecko ping failed: {e}")

# -------------------- Tabs --------------------
tab_search, tab_portfolio = st.tabs(["🔎 Crypto Search", "💼 Portfolio"])

# -------------------- Crypto Search --------------------
with tab_search:
    st.subheader("Search a Coin")
    q = st.text_input("Search by name or symbol", value="bitcoin", placeholder="e.g., bitcoin, eth, solana")

    results = search_coins(q.strip()) if q.strip() else []
    if results:
        display_options = [
            f"{c['name']} ({c['symbol']})" + (f" • Rank {c['market_cap_rank']}" if c.get('market_cap_rank') else "")
            for c in results
        ]
        idx = st.selectbox("Select coin (Coin A)", list(range(len(display_options))),
                           format_func=lambda i: display_options[i])
        coin_a = results[idx]
        coin_a_id = coin_a["id"]
        coin_a_label = f"{coin_a['name']} ({coin_a['symbol']})"
    else:
        coin_a_id, coin_a_label = None, None
        st.info("Start typing to search coins…")

    if coin_a_id:
        try:
            data = get_simple_price(coin_a_id, vs)
            price = (data.get(coin_a_id, {}) or {}).get(vs, None)
            if price is not None:
                st.metric(f"Current Price — {coin_a_label}", f"{price} {currency}")
        except Exception as e:
            st.error(f"Price fetch failed: {e}")

        st.markdown("### Historical Price")
        timeframe = st.radio("Timeframe", options=list(TIMEFRAMES.keys()), horizontal=True, index=3)
        days = TIMEFRAMES[timeframe]
        try:
            df_a = get_market_chart(coin_a_id, vs, days)
            fig_a = line_chart(df_a, "date", "price", f"{coin_a_label} — {timeframe}", f"Price ({currency})")
            st.plotly_chart(fig_a, use_container_width=True)
        except Exception as e:
            st.error(f"Historical chart failed: {e}")

        st.markdown("### Compare with another coin")
        compare_on = st.checkbox("Enable comparison (Coin B)", value=False)
        if compare_on:
            q2 = st.text_input("Search Coin B (name or symbol)", value="ethereum", placeholder="e.g., ethereum, bnb")
            results_b = search_coins(q2.strip()) if q2.strip() else []
            if results_b:
                display_b = [
                    f"{c['name']} ({c['symbol']})" + (f" • Rank {c['market_cap_rank']}" if c.get('market_cap_rank') else "")
                    for c in results_b
                ]
                idx_b = st.selectbox("Select coin (Coin B)", list(range(len(display_b))),
                                     format_func=lambda i: display_b[i])
                coin_b = results_b[idx_b]
                coin_b_id = coin_b["id"]
                coin_b_label = f"{coin_b['name']} ({coin_b['symbol']})"
                try:
                    df_b = get_market_chart(coin_b_id, vs, days)
                    fig_cmp, metrics = comparison_chart(df_a, df_b, coin_a_label, coin_b_label)
                    st.plotly_chart(fig_cmp, use_container_width=True)
                    if metrics is not None and not metrics.empty:
                        st.markdown("**Comparison Metrics**")
                        st.dataframe(metrics, use_container_width=True)
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
            else:
                st.info("Start typing to search Coin B…")

# -------------------- Portfolio --------------------
# -------------------- Portfolio --------------------
with tab_portfolio:
    st.subheader("Your Portfolio")

    # Init session state storage
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []  # list of dicts: {id, name, symbol, amount, alert_dir, alert_price}

    # --- Add coin form ---
    with st.expander("➕ Add a coin"):
        q_add = st.text_input("Search coin (name or symbol)", placeholder="e.g., bitcoin, eth, solana", key="q_add")
        results_add = search_coins(q_add.strip()) if q_add.strip() else []
        coin_to_add = None
        if results_add:
            display_add = [
                f"{c['name']} ({c['symbol']})" + (f" • Rank {c['market_cap_rank']}" if c.get('market_cap_rank') else "")
                for c in results_add
            ]
            idx_add = st.selectbox("Select coin", list(range(len(display_add))),format_func=lambda i: display_add[i], key="idx_add")
            coin_to_add = results_add[idx_add]
        amt = st.number_input("Amount you hold", min_value=0.0, step=0.0001, format="%.8f", key="amt_add")
        col_ad1, col_ad2 = st.columns([1, 1])
        with col_ad1:
            alert_dir = st.selectbox("Alert direction", options=["(none)", ">=", "<="], index=0, key="alert_dir_add")
        with col_ad2:
            alert_price = st.number_input(f"Alert price ({currency})", min_value=0.0, step=0.0001, format="%.8f",key="alert_price_add")

        if st.button("Add to portfolio", type="primary", use_container_width=True, key="add_btn"):
            if coin_to_add and amt > 0:
                exists = next((i for i, x in enumerate(st.session_state.portfolio) if x["id"] == coin_to_add["id"]), None)
                entry = {
                    "id": coin_to_add["id"],
                    "name": coin_to_add["name"],
                    "symbol": coin_to_add["symbol"],
                    "amount": float(amt),
                    "alert_dir": alert_dir,
                    "alert_price": float(alert_price) if alert_dir != "(none)" and alert_price > 0 else None,
                }
                if exists is None:
                    st.session_state.portfolio.append(entry)
                else:
                    # merge: increase amount; update alert if provided
                    st.session_state.portfolio[exists]["amount"] += entry["amount"]
                    if entry["alert_dir"] != "(none)":
                        st.session_state.portfolio[exists]["alert_dir"] = entry["alert_dir"]
                        st.session_state.portfolio[exists]["alert_price"] = entry["alert_price"]
                st.success(f"Added/updated {entry['name']}")
            else:
                st.warning("Pick a coin and enter an amount > 0.")

    # Show & edit holdings table
    if st.session_state.portfolio:
        st.markdown("### Your Holdings")
        for i, row in enumerate(st.session_state.portfolio):
            cols = st.columns([3, 2, 2, 2, 1])
            with cols[0]:
                st.write(f"**{row['name']} ({row['symbol']})**")
            with cols[1]:
                row['amount'] = st.number_input("Amount", value=row["amount"], step=0.0001, format="%.8f", key=f"amt_{i}")
            with cols[2]:
                row['alert_dir'] = st.selectbox("Alert dir", ["(none)", ">=", "<="], index=["(none)", ">=", "<="].index(row["alert_dir"]), key=f"adir_{i}")
            with cols[3]:
                row['alert_price'] = st.number_input("Alert price", value=row["alert_price"] or 0.0, step=0.0001, format="%.8f", key=f"aprice_{i}")
            with cols[4]:
                if st.button("🗑", key=f"del_{i}"):
                    st.session_state.portfolio.pop(i)
                    st.rerun()

        # Download + clear options
        rm_cols = st.columns([1, 1, 6])
        with rm_cols[0]:
            if st.button("🗑 Clear portfolio", use_container_width=True, key="clear_pf"):
                st.session_state.portfolio = []
                st.rerun()
        with rm_cols[1]:
            df_port = pd.DataFrame(st.session_state.portfolio)
            st.download_button(
                "⬇️ Download holdings (CSV)",
                data=df_port.to_csv(index=False).encode(),
                file_name="holdings.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # (rest of your portfolio value, pie chart, alerts, history stays same)
    # else:
        # st.info("Your portfolio is empty. Use **➕ Add a coin** to get started.")

        # ---- Current value & distribution ----
        coin_ids = [row["id"] for row in st.session_state.portfolio]
        amounts = {row["id"]: float(row.get("amount", 0.0) or 0.0) for row in st.session_state.portfolio}

        try:
            prices = get_simple_prices_bulk(coin_ids, vs)  # {id: {vs: price}}
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                st.error("⚠️ Rate limit exceeded. Please wait a few seconds and try again.")
                prices = {}
            else:
                raise

        # compute current values
        values = {}
        total_value = 0.0
        for cid in coin_ids:
            p = (prices.get(cid, {}) or {}).get(vs)
            val = (p or 0.0) * amounts.get(cid, 0.0)
            values[cid] = val
            total_value += val

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Current Portfolio Value", f"{total_value:,.2f} {currency}")
        with c2:
            st.caption("Prices via CoinGecko — cached to respect rate limits.")

        # pie
        if total_value > 0 and any(v > 0 for v in values.values()):
            labels = [next(x["name"] for x in st.session_state.portfolio if x["id"] == cid) for cid in coin_ids]
            vals = [values[cid] for cid in coin_ids]
            st.plotly_chart(pie_chart(labels, vals, "Portfolio Distribution"), use_container_width=True)
        else:
            st.info("Your portfolio total is 0. Add amounts or wait for prices.")

        # ---- Alerts check ----
        st.markdown("### Alerts")
        alerts_rows = []
        for row in st.session_state.portfolio:
            if not row.get("alert_price") or row.get("alert_dir") == "(none)":
                alerts_rows.append({"Asset": f"{row['name']} ({row['symbol']})", "Status": "No alert set"})
                continue
            cid = row["id"]
            curr_price = (prices.get(cid, {}) or {}).get(vs)
            if curr_price is None:
                alerts_rows.append({"Asset": f"{row['name']} ({row['symbol']})", "Status": "Price unavailable"})
                continue
            cond = False
            if row["alert_dir"] == ">=":
                cond = curr_price >= row["alert_price"]
            elif row["alert_dir"] == "<=":
                cond = curr_price <= row["alert_price"]

            label = f"price {curr_price:.6g} {currency} | alert {row['alert_dir']} {row['alert_price']}"
            if cond:
                alerts_rows.append({"Asset": f"{row['name']} ({row['symbol']})", "Status": "Triggered", "Details": label})
            else:
                alerts_rows.append({"Asset": f"{row['name']} ({row['symbol']})", "Status": "Not triggered", "Details": label})

        alerts_df = pd.DataFrame(alerts_rows)
        st.dataframe(alerts_df, use_container_width=True)

        # ---- Historical portfolio value ----
        st.markdown("### Portfolio Value Over Time")
        tf = st.radio("Timeframe", options=list(PORTFOLIO_TIMEFRAMES.keys()), horizontal=True, index=2, key="port_tf")
        days = PORTFOLIO_TIMEFRAMES[tf]

        # Build combined portfolio timeseries (parallel fetch)
        combined = None
        # limit coins to those with positive amounts
        positive_coin_ids = [cid for cid in coin_ids if amounts.get(cid, 0.0) > 0]
        if positive_coin_ids:
            futures = {}
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(positive_coin_ids))) as exe:
                for cid in positive_coin_ids:
                    futures[exe.submit(get_market_chart, cid, vs, days)] = cid

                dfs = {}
                for fut in as_completed(futures):
                    cid = futures[fut]
                    try:
                        df_c = fut.result()
                        if df_c.empty:
                            continue
                        df_c = df_c.rename(columns={"price": f"value_{cid}"})
                        # multiply price by coin amount = position value
                        amt = amounts.get(cid, 0.0)
                        df_c[f"value_{cid}"] = df_c[f"value_{cid}"] * amt
                        dfs[cid] = df_c[["date", f"value_{cid}"]]
                    except requests.exceptions.HTTPError as e:
                        if e.response is not None and e.response.status_code == 429:
                            st.error("⚠️ Rate limit while building history. Please wait a few seconds and retry.")
                        else:
                            st.error(f"Failed fetching history for {cid}: {e}")
                    except Exception as e:
                        st.error(f"Failed fetching history for {cid}: {e}")

            # merge all dfs on 'date' (outer) and sum values
            if dfs:
                merged = None
                for i, (cid, dfi) in enumerate(dfs.items()):
                    if merged is None:
                        merged = dfi.copy()
                    else:
                        merged = pd.merge(merged, dfi, on="date", how="outer")
                    # be polite when merging many coins
                    time.sleep(0.05)
                if merged is not None and not merged.empty:
                    value_cols = [c for c in merged.columns if c.startswith("value_")]
                    merged[value_cols] = merged[value_cols].fillna(0.0)
                    merged["portfolio_value"] = merged[value_cols].sum(axis=1)
                    merged = merged.sort_values("date")
                    combined = merged[["date", "portfolio_value"]]

        if combined is not None and not combined.empty:
            fig_port = line_chart(combined, "date", "portfolio_value", f"Portfolio Value — {tf}", f"Value ({currency})")
            st.plotly_chart(fig_port, use_container_width=True)
        else:
            st.info("Add coins with amounts to see portfolio history.")
    else:
        st.info("Your portfolio is empty. Use **➕ Add a coin** to get started.")

st.caption("Data source: CoinGecko API — optimized for caching & rate-limits")
