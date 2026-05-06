# BIST Portfolio Diversification via Minimum Spanning Tree (MST)

> **MIS Network Optimization Project** — Applying graph theory to real-world financial decision-making on Borsa Istanbul.

---

## Project Title

**Stock Correlation Network Analysis for Portfolio Risk Management using MST**

---

## Problem Definition

### What is the problem?

A portfolio manager investing in Borsa Istanbul faces a critical challenge: **which stocks should be held together to minimize correlated risk?**

Holding multiple stocks from the same sector (e.g., four bank stocks) provides an illusion of diversification. In reality, when the banking sector drops, all four stocks fall simultaneously — offering no protection at all.

The raw data — daily price movements of dozens of stocks — creates a **complex, high-dimensional web of correlations** that is impossible to interpret by eye. A portfolio manager needs a systematic, algorithmic method to uncover the hidden structure of this correlation network.

### Why is it a MIS problem?

This is a classic **Management Information Systems decision-support problem**:

- **Input data**: Time-series price data from Yahoo Finance (1 year, daily adjusted close)
- **Processing**: Network optimization algorithm (Minimum Spanning Tree)
- **Output**: An actionable recommendation that directly supports a managerial decision — *which assets to include in a diversified portfolio*

It maps directly to the MIS framework: raw data → information → knowledge → decision.

---

## Network Structure

### Nodes

Each **node** represents a publicly traded company on Borsa Istanbul. This project includes **41 stocks across 13 sectors**:

| Sector | Stocks |
|--------|--------|
| Banks | GARAN, AKBNK, YKBNK, ISCTR, SKBNK |
| Aviation | THYAO, PGSUS, TAVHL |
| Energy | TUPRS, ASTOR, ENJSA, AKSEN, PETKM, ZOREN |
| Technology | ASELS, KAREL |
| Telecom | TCELL, TTKOM |
| Retail | BIMAS, MAVI |
| Food & Beverage | CCOLA, ULKER, AEFES, TABGD |
| Automotive | DOAS, FROTO, TOASO, OTKAR |
| Real Estate | EKGYO, ENKAI |
| Holdings | KCHOL, SAHOL, TKFEN, DOHOL, AGHOL |
| Steel | EREGL, KRDMD |
| Cement | CIMSA, OYAKC |
| Consumer Appliances | VESTL, ARCLK |

**Total nodes: 41** ✓ (requirement: min. 6)

### Edges

An **edge** between two stocks indicates a measurable statistical relationship between their price movements. The complete graph contains **820 edges** (all possible pairs). After applying MST, this is reduced to the **40 most structurally important edges**.

**Total edges (MST): 40** ✓ (requirement: min. 8)

### Weights — Mantegna Distance

Edge weights are computed using the **Mantegna (1999) distance metric**, which converts Pearson correlation (ρ) into a proper Euclidean distance:

$$d(i,j) = \sqrt{2 \cdot (1 - \rho_{ij})}$$

| Correlation (ρ) | Distance (d) | Meaning |
|---|---|---|
| +1.0 | 0.000 | Perfectly identical movement |
| +0.5 | 1.000 | Moderate correlation |
| 0.0 | 1.414 | No relationship |
| -1.0 | 2.000 | Perfect inverse movement |

A **short edge = high correlation = similar behavior = risky to hold together**.
A **long edge = low correlation = independent behavior = good for diversification**.

---

## Algorithm Selection: Why Minimum Spanning Tree?

### What is MST?

A **Minimum Spanning Tree** is a subgraph that:
- Connects **all nodes** in the network
- Uses the **minimum total edge weight** possible
- Contains **no cycles** (it is a tree)

### Why MST for this problem?

| Option | Reason for / against |
|--------|----------------------|
| **MST** ✓ | Reveals the backbone of the correlation structure; filters noise; highlights natural clusters |
| Shortest Path | Would only answer "how similar are stock A and B?" — doesn't give the full market picture |
| Maximum Flow | Designed for capacity/throughput problems, not correlation analysis |

MST is the standard approach in **financial econophysics**, used in academic literature since Mantegna (1999). The algorithm used is **Kruskal's algorithm** (via `networkx.minimum_spanning_tree`):

1. Sort all 820 edges by weight (ascending)
2. Greedily add the shortest edge that doesn't create a cycle
3. Stop when all 41 nodes are connected → 40 edges remain

**Time complexity: O(E log E)**

---

## Python Implementation

### Requirements

```
networkx>=3.0
matplotlib>=3.7
pandas>=2.0
numpy>=1.24
yfinance>=0.2
scipy>=1.10
```

Install with:
```bash
pip install networkx matplotlib pandas numpy yfinance scipy
```

### How to Run

```bash
git clone https://github.com/YOUR_USERNAME/bist-mst-portfolio
cd bist-mst-portfolio
pip install -r requirements.txt
python main.py
```

The script will:
1. Download 1 year of live price data from Yahoo Finance
2. Compute the correlation and distance matrix
3. Build the MST
4. Save `mst_visualization.png` to your Desktop and open it automatically
5. Print the full managerial report in the terminal

### Code Structure

```
main.py
│
├── STOCKS dict              → 41 stocks and their 13 sectors
├── SECTOR_COLORS dict       → color mapping for visualization
│
├── download_prices()        → pulls 1-year adjusted close prices from Yahoo Finance
│                              (falls back to simulated data if offline)
│
├── build_distance_matrix()  → log returns → Pearson correlation →
│                              Mantegna distance transformation
│
├── build_graph_and_mst()    → builds complete graph (820 edges)
│                              applies Kruskal's MST → 40 edges remain
│
├── visualize_mst()          → draws color-coded MST with edge weights,
│                              saves to Desktop and auto-opens
│
└── managerial_report()      → hub nodes, high-risk pairs, best diversification
                               pairs, leaf nodes, sector clusters,
                               and concrete portfolio recommendation
```

### Key Code Snippets

**Distance transformation:**
```python
log_returns = np.log(prices / prices.shift(1)).dropna()
corr = log_returns.corr()
dist = np.sqrt(2 * (1 - corr))   # Mantegna distance
```

**MST extraction:**
```python
G   = nx.Graph()
# ... add 41 nodes and 820 edges ...
MST = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
# Result: 41 nodes, 40 edges
```

---

## Results

### MST Visualization

The output graph (`mst_visualization.png`) shows all 41 stocks color-coded by sector. Key observations:

- **Tight bank cluster** — GARAN, AKBNK, YKBNK, ISCTR are densely connected with short edges (ρ ≈ 0.88–0.89)
- **Holdings as hubs** — KCHOL and SAHOL connect across multiple sectors due to their diversified conglomerate structure
- **Energy sub-cluster** — ASTOR acts as a hub (degree 5) connecting TUPRS, ENJSA, AKSEN, ZOREN, PETKM

### Hub Nodes (Most Connected)

| Stock | Sector | Degree | Interpretation |
|-------|--------|--------|----------------|
| KCHOL | Holdings | 10 | Bridges banking, energy, automotive |
| SAHOL | Holdings | 9 | Bridges banking, energy, retail |
| AGHOL | Holdings | 4 | Connects within holdings cluster |

### Most Correlated Pairs (Avoid holding both)

| Pair | Distance | Correlation ρ | Risk |
|------|----------|---------------|------|
| GARAN ↔ YKBNK | 0.4598 | 0.8943 | Very High |
| YKBNK ↔ ISCTR | 0.4836 | 0.8831 | Very High |
| AKBNK ↔ YKBNK | 0.4950 | 0.8775 | Very High |

### Best Diversification Pairs

| Pair | Distance | Correlation ρ |
|------|----------|---------------|
| AKSEN ↔ ASELS | 1.4050 | 0.0130 |
| ASELS ↔ DOAS | 1.4047 | 0.0134 |
| TAVHL ↔ TUPRS | 1.3964 | 0.0250 |

---

## Managerial Interpretation

### Recommended Diversified Portfolio

The algorithm selects one stock per sector, prioritizing **leaf nodes** (structurally most independent) and maximizing average distance to already-selected stocks:

| Stock | Sector |
|-------|--------|
| TOASO | Automotive |
| THYAO | Aviation |
| SKBNK | Banks |
| OYAKC | Cement |
| ARCLK | Consumer Appliances |
| ENJSA | Energy |
| ULKER | Food & Beverage |
| AGHOL | Holdings |
| ENKAI | Real Estate |
| BIMAS | Retail |
| KRDMD | Steel |
| KAREL | Technology |
| TCELL | Telecom |

**Portfolio size: 13 stocks | Average pairwise ρ ≈ 0.19**

A value close to 0 indicates genuine diversification. A fully correlated portfolio would have ρ = 1.0.

### Key Managerial Decisions Supported

| Question | MST Answer |
|----------|------------|
| Should I hold GARAN and YKBNK together? | ❌ No — ρ = 0.89, nearly identical behavior |
| Can I pair TCELL with THYAO? | ✅ Yes — near-zero correlation |
| Which bank gives best diversification? | SKBNK — leaf node, most distant from other sectors |
| What is the riskiest concentration? | Any 2+ bank stocks held simultaneously |
| Which stocks act as market proxies? | KCHOL, SAHOL — hub nodes with 9–10 connections |

### Intra-Sector Risk Clusters

The MST reveals the following high-risk clusters where stocks should not be combined:

- **Banks**: GARAN↔YKBNK, AKBNK↔YKBNK, YKBNK↔ISCTR
- **Holdings**: KCHOL↔SAHOL, KCHOL↔DOHOL, KCHOL↔TKFEN
- **Energy**: TUPRS↔ASTOR, ASTOR↔ENJSA, ASTOR↔AKSEN
- **Aviation**: THYAO↔PGSUS, THYAO↔TAVHL
- **Automotive**: DOAS↔FROTO, DOAS↔OTKAR, TOASO↔OTKAR

---

## Repository Structure

```
bist-mst-portfolio/
│
├── main.py                  # Main analysis script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── network_data.csv         # Node and edge data from MST analysis
├── mst_visualization.png    # Output network graph
├── solution_overview.docx   # Full project report
└── references.docx          # Academic and software references
```

---

## References

- Mantegna, R. N. (1999). *Hierarchical structure in financial markets*. European Physical Journal B, 11(1), 193–197.
- Onnela, J. P. et al. (2003). *Asset trees and asset graphs in financial markets*. Physica Scripta, T106, 48–54.
- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
- NetworkX Documentation: https://networkx.org

---

## License

MIT License — free to use and modify for academic purposes.
