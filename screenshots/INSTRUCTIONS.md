# Screenshot Instructions

This directory will contain screenshots demonstrating the analysis results.

## Screenshots to Take

### 1. Analysis Demo Output (PRIORITY)

**File name:** `analysis_demo.png`

**How to get it:**
```bash
cd /Users/ewang7/mcpdev-public/sample-data
python3 -c "
import pandas as pd
import numpy as np

print('='*70)
print('CRYPTOCURRENCY SENTIMENT ANALYSIS DEMO')
print('='*70)

# Load data
tweets = pd.read_csv('sample_tweets.csv', parse_dates=['timestamp'])
prices = pd.read_csv('sample_prices.csv', parse_dates=['timestamp'])

print(f'\n1. LOADED DATA')
print(f'   Tweets: {len(tweets)}')
print(f'   Prices: {len(prices)}')

# Aggregate by hour
tweets['hour'] = tweets['timestamp'].dt.floor('H')
prices['hour'] = prices['timestamp'].dt.floor('H')

hourly_sentiment = tweets.groupby('hour').agg({
    'avg_score': 'mean',
    'handle': 'count'
}).rename(columns={'handle': 'tweet_count', 'avg_score': 'hourly_sentiment'})

merged = pd.merge(
    hourly_sentiment,
    prices[['hour', 'price', 'volume_24h']],
    on='hour',
    how='inner'
)

print(f'   Merged hourly records: {len(merged)}')

# Calculate Pearson correlation manually
def calculate_pearson(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sum_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    sum_x2 = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_y2 = sum((y[i] - mean_y) ** 2 for i in range(n))
    denominator = (sum_x2 * sum_y2) ** 0.5
    return sum_xy / denominator if denominator != 0 else 0

sentiment_array = merged['hourly_sentiment'].values
price_array = merged['price'].values

r = calculate_pearson(sentiment_array, price_array)
r_squared = r ** 2

print(f'\n2. PEARSON CORRELATION CALCULATION')
print(f'   ┌─────────────────────────────────────────────────┐')
print(f'   │  Pearson Correlation (r):  {r:>20.4f}  │')
print(f'   │  R² (Variance Explained):  {r_squared:>20.4f}  │')
print(f'   │  Percentage:               {r_squared*100:>19.2f}%  │')
print(f'   └─────────────────────────────────────────────────┘')

# Verify
pandas_r = merged['hourly_sentiment'].corr(merged['price'])
print(f'\n   Verification (pandas corr): {pandas_r:.4f}')
print(f'   Difference: {abs(r - pandas_r):.10f} ✓')

# Interpretation
def interpret(r):
    abs_r = abs(r)
    if abs_r < 0.3: return 'Very weak'
    elif abs_r < 0.5: return 'Weak'
    elif abs_r < 0.7: return 'Moderate'
    elif abs_r < 0.9: return 'Strong'
    else: return 'Very strong'

classification = interpret(r)
direction = 'positive' if r > 0 else 'negative'

print(f'\n3. STATISTICAL INTERPRETATION')
print(f'   Classification: {classification} {direction}')
print(f'   Sentiment explains {r_squared*100:.2f}% of price variance')

# Directional analysis
merged_sorted = merged.sort_values('hour').reset_index(drop=True)
merged_sorted['sentiment_change'] = merged_sorted['hourly_sentiment'].diff()
merged_sorted['price_change'] = merged_sorted['price'].diff()
changes = merged_sorted.dropna()

same_direction = (
    ((changes['sentiment_change'] > 0) & (changes['price_change'] > 0)) |
    ((changes['sentiment_change'] < 0) & (changes['price_change'] < 0))
)
same_direction_pct = (same_direction.sum() / len(changes)) * 100

print(f'\n4. DIRECTIONAL MOVEMENT ANALYSIS')
print(f'   Same direction: {same_direction.sum()}/{len(changes)} ({same_direction_pct:.1f}%)')
print(f'   Opposite: {(~same_direction).sum()}/{len(changes)} ({100-same_direction_pct:.1f}%)')
print(f'   Random chance: 50.0%')

print(f'\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'Twitter sentiment shows a {classification.lower()} correlation with Bitcoin price.')
print(f'Sentiment explains only {r_squared*100:.1f}% of price variance, suggesting')
print(f'limited predictive power for short-term price movements.')
print('='*70)
"
```

**To screenshot:**
1. Run the command above in your terminal
2. Take a screenshot of the full output
3. Save as `analysis_demo.png` in this directory

---

### 2. MCP Tests Passing (PRIORITY)

**File name:** `mcp_tests_passing.png`

**How to get it:**
```bash
cd /Users/ewang7/mcpdev-public/correlation-analysis/mcp_server
python3 test_tools.py
```

**To screenshot:**
1. Run the command above
2. Scroll to show the "TEST SUMMARY" section with "🎉 All tests passed!"
3. Save as `mcp_tests_passing.png`

---

### 3. Sample Data Preview (Optional)

**File name:** `sample_data.png`

**How to get it:**
```bash
cd /Users/ewang7/mcpdev-public/sample-data
head -20 sample_tweets.csv
echo ""
echo "---"
echo ""
head -20 sample_prices.csv
```

**To screenshot:**
1. Run command to show both CSV files
2. Save as `sample_data.png`

---

### 4. GitHub README Preview (Optional)

**File name:** `readme_preview.png`

**How to get it:**
1. Go to https://github.com/EricWang2000/mcpdev-public
2. Scroll to the "Research Findings" section showing the correlation table
3. Take screenshot of the table and summary
4. Save as `readme_preview.png`

---

## After Taking Screenshots

Once you have the screenshots, update the main README.md to reference them:

```markdown
## Demo

See the [analysis demo notebook](sample-data/analysis_demo.ipynb) for a step-by-step walkthrough.

![Analysis Results](screenshots/analysis_demo.png)
*Pearson correlation analysis showing weak relationship between sentiment and price*

### Verification

All analytical tools are tested and verified:

![MCP Tests Passing](screenshots/mcp_tests_passing.png)
*Automated test suite confirms all 5 statistical tools pass*
```
