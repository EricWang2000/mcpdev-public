# Sample Data

Sample datasets for demonstrating the cryptocurrency sentiment analysis and correlation framework without requiring a full PostgreSQL database setup.

## Files

### sample_tweets.csv
100 sample tweets from June 15, 2025 with sentiment scores.

**Columns:**
- `timestamp` - When the tweet was posted (with timezone)
- `handle` - Twitter account handle
- `text` - Tweet content
- `avg_score` - Average sentiment score across multiple NLP models (-1 to +1)

**Statistics:**
- Records: 100 tweets
- Date: June 15, 2025
- Sentiment range: -1.0 (very negative) to +1.0 (very positive)

### sample_prices.csv
24 hours of Bitcoin price data at hourly intervals.

**Columns:**
- `symbol` - Cryptocurrency symbol (BTC)
- `timestamp` - Price timestamp (with timezone)
- `price` - Price in USD
- `volume_24h` - 24-hour trading volume in USD

**Statistics:**
- Records: 24 hourly price points
- Date: June 15, 2025
- Asset: Bitcoin (BTC)
- Price range: $105,497 - $106,032

## Usage

These sample datasets allow you to:

1. **Test correlation analysis** - Run Pearson correlation between sentiment and price
2. **Explore event detection** - Find sentiment spikes and measure price impact
3. **Verify statistical methods** - Validate R² calculations and directional analysis
4. **Run without database** - Analyze data using pandas/CSV instead of PostgreSQL

## Example Analysis

```python
import pandas as pd

# Load data
tweets = pd.read_csv('sample_tweets.csv', parse_dates=['timestamp'])
prices = pd.read_csv('sample_prices.csv', parse_dates=['timestamp'])

# Merge on nearest timestamp
merged = pd.merge_asof(
    tweets.sort_values('timestamp'),
    prices.sort_values('timestamp'),
    on='timestamp',
    direction='nearest'
)

# Calculate correlation
correlation = merged['avg_score'].corr(merged['price'])
print(f"Sentiment-Price Correlation: {correlation:.3f}")
```

## Data Source

This sample data is extracted from the full dataset used in the main analysis:

- **Full dataset:** 19,684 tweets from 88 Twitter accounts
- **Price data:** 4,441 hourly records across 5 cryptocurrencies
- **Time period:** June 1 - July 8, 2025

For research findings from the complete dataset, see the main [README](../README.md).
