"""
BIST 100 Portfolio Diversification via Minimum Spanning Tree (MST)
===================================================================
MIS Network Optimization Project

This script:
1. Downloads real price data from Yahoo Finance (yfinance)
2. Computes pairwise Pearson correlations
3. Transforms correlations into distances: d = sqrt(2 * (1 - rho))
4. Builds a complete weighted graph
5. Extracts the Minimum Spanning Tree (Kruskal's algorithm via NetworkX)
6. Visualizes the MST with sector color-coding
7. Prints managerial interpretation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. STOCK UNIVERSE
# ─────────────────────────────────────────────
STOCKS = {
    # Banks
    "GARAN": "Banks",
    "AKBNK": "Banks",
    "YKBNK": "Banks",
    "ISCTR": "Banks",
    "SKBNK": "Banks",
    # Aviation
    "THYAO": "Aviation",
    "PGSUS": "Aviation",
    "TAVHL": "Aviation",
    # Energy
    "TUPRS": "Energy",
    "ASTOR": "Energy",
    "ENJSA": "Energy",
    "AKSEN": "Energy",
    "PETKM": "Energy",
    "ZOREN": "Energy",
    # Technology / Telecom
    "TCELL": "Telecom",
    "ASELS": "Technology",
    "TTKOM": "Telecom",
    "KAREL": "Technology",
    # Retail / Food and Beverage
    "BIMAS": "Retail",
    "CCOLA": "Food & Beverage",
    "MAVI":  "Retail",
    "ULKER": "Food & Beverage",
    "AEFES": "Food & Beverage",
    "TABGD": "Food & Beverage",
    # Automotive
    "DOAS":  "Automotive",
    "FROTO": "Automotive",
    "TOASO": "Automotive",
    "OTKAR": "Automotive",
    # Real Estate
    "EKGYO": "Real Estate",
    "ENKAI": "Real Estate",
    # Holdings
    "KCHOL": "Holdings",
    "SAHOL": "Holdings",
    "TKFEN": "Holdings",
    "DOHOL": "Holdings",
    "AGHOL": "Holdings",
    # Steel
    "EREGL": "Steel",
    "KRDMD": "Steel",
    # Cement
    "CIMSA": "Cement",          # FIX: was "Ciment" in original
    "OYAKC": "Cement",          # FIX: was "Ciment" in original
    # Consumer Appliances
    "VESTL": "Consumer Appliances",
    "ARCLK": "Consumer Appliances",
}

TICKERS = [f"{s}.IS" for s in STOCKS.keys()]

# FIX: All sector keys now exactly match the values used in STOCKS dict above
SECTOR_COLORS = {
    "Banks":                "#2196F3",   # blue
    "Aviation":             "#FF9800",   # orange
    "Energy":               "#4CAF50",   # green
    "Technology":           "#F44336",   # red
    "Telecom":              "#9C27B0",   # purple
    "Retail":               "#795548",   # brown
    "Food & Beverage":      "#E91E63",   # pink
    "Automotive":           "#9E9E9E",   # gray
    "Real Estate":          "#827717",   # olive
    "Holdings":             "#00BCD4",   # cyan
    "Steel":                "#212121",   # near-black
    "Cement":               "#A0522D",   # sienna
    "Consumer Appliances":  "#B0BEC5",   # silver
}


# ─────────────────────────────────────────────
# 2. PRICE DATA
# ─────────────────────────────────────────────
def download_prices(tickers: list = TICKERS, period: str = "1y") -> pd.DataFrame:
    """
    Try to pull live data from Yahoo Finance.
    If the network is unavailable, fall back to simulated data with
    realistic intra-sector / cross-sector correlation structure.
    """
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]

        # Clean up column names
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].replace(".IS", "") for c in raw.columns]
        else:
            raw.columns = [c.replace(".IS", "") for c in raw.columns]

        raw.dropna(axis=1, how="all", inplace=True)
        raw.ffill(inplace=True)
        raw.dropna(inplace=True)

        available = [s for s in STOCKS if s in raw.columns]
        if len(available) >= 8:
            print(f"📥  Live data: {len(available)} stocks loaded from Yahoo Finance\n")
            return raw[available]
    except Exception as e:
        print(f"⚠️  Live data error: {e}")

    # ── Simulated fallback ──────────────────────────────────────────
    print("📥  Network restricted — using simulated BIST price data")
    print("    (Correlation structure mirrors realistic sector dynamics)\n")

    np.random.seed(42)
    n_days = 252        # 1 trading year
    stocks  = list(STOCKS.keys())
    sectors = list(STOCKS.values())
    n = len(stocks)

    # Block-structured correlation matrix
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = (np.random.uniform(0.55, 0.82)
                 if sectors[i] == sectors[j]
                 else np.random.uniform(0.05, 0.35))
            rho[i, j] = rho[j, i] = r

    # Nearest positive-definite fix
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-8)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L   = np.linalg.cholesky(rho)

    z         = np.random.randn(n_days, n)
    corr_z    = z @ L.T
    daily_ret = 0.0005 + corr_z * 0.018
    prices    = 100 * np.exp(np.cumsum(daily_ret, axis=0))

    df = pd.DataFrame(
        prices,
        index=pd.bdate_range(end="2024-12-31", periods=n_days),
        columns=stocks
    )
    print(f"   ✓ {n} stocks simulated over {n_days} trading days\n")
    return df


# ─────────────────────────────────────────────
# 3. CORRELATION → DISTANCE MATRIX
# ─────────────────────────────────────────────
def build_distance_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Mantegna (1999): d = sqrt(2 * (1 - rho))"""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    corr = log_returns.corr()
    return np.sqrt(2 * (1 - corr))


# ─────────────────────────────────────────────
# 4. BUILD COMPLETE GRAPH + MST
# ─────────────────────────────────────────────
def build_graph_and_mst(dist: pd.DataFrame):
    stocks = dist.columns.tolist()
    G = nx.Graph()

    for s in stocks:
        G.add_node(s, sector=STOCKS[s])

    for i, s1 in enumerate(stocks):
        for j, s2 in enumerate(stocks):
            if j > i:
                G.add_edge(s1, s2, weight=round(dist.loc[s1, s2], 4))

    MST = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

    print(f"📊  Complete graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"🌲  MST            : {MST.number_of_nodes()} nodes, {MST.number_of_edges()} edges\n")
    return G, MST


# ─────────────────────────────────────────────
# 5. VISUALIZE MST
# ─────────────────────────────────────────────
def visualize_mst(MST: nx.Graph, output_path: str = "mst_visualization.png"):
    fig, ax = plt.subplots(figsize=(20, 14))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    pos = nx.kamada_kawai_layout(MST, weight="weight")

    node_colors = [SECTOR_COLORS.get(STOCKS.get(n, ""), "#AAAAAA") for n in MST.nodes()]

    weights    = [MST[u][v]["weight"] for u, v in MST.edges()]
    max_w, min_w = max(weights), min(weights)
    edge_widths  = [3.5 - 2.5 * (w - min_w) / (max_w - min_w + 1e-9) for w in weights]

    nx.draw_networkx_edges(
        MST, pos, width=edge_widths,
        edge_color="#888888", alpha=0.7, ax=ax
    )
    nx.draw_networkx_nodes(
        MST, pos, node_color=node_colors,
        node_size=900, alpha=0.95, ax=ax
    )
    nx.draw_networkx_labels(
        MST, pos, font_size=7,
        font_color="white", font_weight="bold", ax=ax
    )

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in MST.edges(data=True)}
    nx.draw_networkx_edge_labels(
        MST, pos, edge_labels=edge_labels,
        font_size=6, font_color="#CCCCCC",
        bbox=dict(boxstyle="round,pad=0.15", fc="#1A1A2E", alpha=0.6), ax=ax
    )

    # Only show sectors that actually appear in the current MST
    active_sectors = set(STOCKS[n] for n in MST.nodes())
    legend_handles = [
        mpatches.Patch(color=SECTOR_COLORS[s], label=s)
        for s in SECTOR_COLORS if s in active_sectors
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        framealpha=0.3, facecolor="#1A1A2E",
        edgecolor="white", labelcolor="white", fontsize=9
    )

    ax.set_title(
        "BIST Selected Stocks — Minimum Spanning Tree of Stock Correlations\n"
        "Edge weight = Mantegna distance  d = √(2·(1−ρ))  |  Thicker edge = Higher correlation",
        color="white", fontsize=13, pad=14
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"🖼️   Visualization saved → {output_path}\n")


# ─────────────────────────────────────────────
# 6. MANAGERIAL REPORT
# ─────────────────────────────────────────────
def managerial_report(MST: nx.Graph, dist: pd.DataFrame):
    print("=" * 60)
    print("         MANAGERIAL INTERPRETATION REPORT")
    print("=" * 60)

    # Hub nodes (most connected)
    degrees = dict(MST.degree())
    hubs    = sorted(degrees, key=lambda x: degrees[x], reverse=True)[:3]
    print("\n🔗  Most Connected Stocks (Hub Nodes):")
    for h in hubs:
        print(f"   {h:8s} | Sector: {STOCKS[h]:22s} | Degree: {degrees[h]}")

    # Most correlated pairs in MST (shortest edges → highest correlation)
    mst_edges = sorted(MST.edges(data=True), key=lambda x: x[2]["weight"])
    print("\n⚡  Highest Correlation in MST (Avoid holding both):")
    for u, v, d in mst_edges[:3]:
        rho = round(1 - (d["weight"] ** 2) / 2, 4)
        print(f"   {u} ↔ {v:8s}  distance={d['weight']:.4f}  ρ≈{rho:.4f}")

    # FIX: Best diversification pairs from the FULL distance matrix (not just MST)
    stocks = dist.columns.tolist()
    all_pairs = [
        (u, v, dist.loc[u, v])
        for i, u in enumerate(stocks)
        for v in stocks[i + 1:]
    ]
    far_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)[:5]
    print("\n✅  Best Diversification Pairs (Lowest correlation):")
    for u, v, d in far_pairs:
        rho = round(1 - (d ** 2) / 2, 4)
        print(f"   {u} ↔ {v:8s}  distance={d:.4f}  ρ≈{rho:.4f}")

    # Leaf nodes (best diversifiers structurally)
    leaves = [n for n in MST.nodes() if MST.degree(n) == 1]
    print(f"\n🍃  Leaf Nodes ({len(leaves)} stocks — structurally most independent):")
    for leaf in leaves:
        print(f"   {leaf:8s} | Sector: {STOCKS[leaf]}")

    # Sector clustering summary
    print("\n📁  Intra-Sector Edges (Concentrated Risk Clusters):")
    intra_found = False
    for sector in sorted(set(STOCKS.values())):
        intra = [
            (u, v, MST[u][v]["weight"])
            for u, v in MST.edges()
            if STOCKS[u] == sector and STOCKS[v] == sector
        ]
        if intra:
            intra_found = True
            pairs_str = ", ".join(f"{u}↔{v}" for u, v, _ in intra)
            print(f"   {sector:22s}: {pairs_str}")
    if not intra_found:
        print("   None detected.")

    # ── Concrete portfolio recommendation ───────────────────────────
    # Algorithm: from each sector, pick the leaf node with the highest
    # average distance to all other selected stocks (greedy max-spread).
    print("\n💼  Recommended Diversified Portfolio:")
    print("   (One stock per sector — selected from leaf nodes by max avg distance)\n")

    leaves_set = set(n for n in MST.nodes() if MST.degree(n) == 1)
    sectors_list = sorted(set(STOCKS.values()))
    portfolio  = []

    for sector in sectors_list:
        candidates = [n for n in MST.nodes() if STOCKS[n] == sector and n in leaves_set]
        if not candidates:
            candidates = [n for n in MST.nodes() if STOCKS[n] == sector]
        if not candidates:
            continue

        if not portfolio:
            pick = max(candidates, key=lambda c: dist.loc[c].mean())
        else:
            pick = max(candidates,
                       key=lambda c: np.mean([dist.loc[c, p] for p in portfolio]))

        portfolio.append(pick)
        avg_d = np.mean([dist.loc[pick, p] for p in portfolio if p != pick]) if len(portfolio) > 1 else None
        avg_str = f"  avg dist to portfolio={avg_d:.4f}" if avg_d else ""
        print(f"   ✅  {pick:8s} | {sector}{avg_str}")

    overall_avg = np.mean([dist.loc[a, b]
                           for i, a in enumerate(portfolio)
                           for b in portfolio[i+1:]])
    print(f"\n   Portfolio size   : {len(portfolio)} stocks")
    print(f"   Avg pairwise ρ   : {round(1 - overall_avg**2 / 2, 4)}"
          "  (closer to 0 = better diversification)")
    print(f"   Stocks selected  : {', '.join(portfolio)}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys, pathlib

    # Save image to Desktop so it's easy to find
    desktop = pathlib.Path.home() / "Desktop"
    desktop.mkdir(exist_ok=True)
    img_path = str(desktop / "mst_visualization.png")

    prices   = download_prices()
    dist_mat = build_distance_matrix(prices)
    G, MST   = build_graph_and_mst(dist_mat)
    visualize_mst(MST, output_path=img_path)
    managerial_report(MST, dist_mat)

    # Auto-open the image
    try:
        if sys.platform == "win32":
            os.startfile(img_path)
        elif sys.platform == "darwin":
            os.system(f'open "{img_path}"')
        else:
            os.system(f'xdg-open "{img_path}"')
    except Exception:
        pass

    print(f"\n✅  Analysis complete.")
    print(f"📂  Image saved to: {img_path}")
