# Statistical Correlation Analysis

Framework for analyzing relationships between cryptocurrency sentiment and price movements using Pearson correlation, R² analysis, and event detection algorithms.

## Implementation

### Core Statistical Engine

**mcp_server/server.py** (589 lines, 13 functions)

Implements:
- Pearson correlation coefficient calculation
- R² (coefficient of determination) computation
- Event detection and impact analysis
- Cross-asset correlation comparison
- Time-series data aggregation

### Mathematical Methods

**Pearson Correlation:**
```
r = Σ((x_i - x̄)(y_i - ȳ)) / √(Σ(x_i - x̄)² · Σ(y_i - ȳ)²)
```

**Coefficient of Determination:**
```
R² = r²
```

**Directional Analysis:**
```
Same direction % = count(sign(Δsentiment) = sign(Δprice)) / total
```

## Analytical Tools

### 1. get_price_history
Historical cryptocurrency price data with statistical summaries.

### 2. get_sentiment_price_correlation
Pearson correlation analysis between sentiment and price.

### 3. find_sentiment_price_events
Sentiment event detection with price impact measurement.

### 4. compare_crypto_sentiment_correlation
Cross-asset correlation comparison.

### 5. get_hourly_sentiment_price
Time-series data export for custom analysis.

## Research Findings

Analysis of June-July 2025 data (4,441 hourly records):

| Asset | Correlation (r) | R² | Interpretation |
|-------|----------------|-----|----------------|
| BTC | 0.158 | 2.5% | Very weak |
| ETH | 0.218 | 4.7% | Weak |
| SOL | 0.220 | 4.8% | Weak |

**Key Insight:** Twitter sentiment explains less than 5% of price variance.
