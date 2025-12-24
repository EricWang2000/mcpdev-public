# Sentiment Analysis Engine

NLP pipeline for extracting sentiment from cryptocurrency-related social media content with exponential weighted moving averages for temporal trend detection.

## Core Implementation

**Sentiment.py** (425 lines)

Clean, focused sentiment analysis engine implementing:
- Multi-model NLP sentiment scoring (OpenAI, FinBERT, VADER)
- Exponential weighted moving average (EWMA) calculations
- PostgreSQL time-series storage
- Hourly sentiment aggregation for correlation analysis

## Features

### Natural Language Processing
- Three complementary sentiment models:
  - **OpenAI**: Contextual understanding via language models
  - **FinBERT**: Financial text specialization
  - **VADER**: Social media and informal language
- Average score calculation across models for robustness
- Temporal sentiment tracking

### Statistical Methods
**Exponential Weighted Moving Average:**
```
EWMA(t) = α · sentiment(t) + (1 - α) · EWMA(t-1)
```
Where α controls the decay rate of historical sentiment.

### Database Integration
- PostgreSQL storage with time-series optimization
- Efficient aggregation queries for sentiment trends
- Feed-level sentiment tracking (88 Twitter accounts)
- Historical sentiment data retrieval

## Database Schema

```sql
CREATE TABLE tweets (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE,
    feed VARCHAR(100),
    content TEXT,
    avg_score DECIMAL(10, 4),
    sentiment VARCHAR(20)
);

CREATE TABLE analysis_runs (
    run_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE,
    overall_sentiment DECIMAL(10, 4),
    tweet_count INTEGER
);
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
psql -d crypto_analysis -f schema.sql

# Run sentiment analysis
python3 Sentiment.py
```

## Configuration

Database connection settings in Sentiment.py:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_analysis",
    "user": "postgres",
    "password": "your-password"
}
```

## Performance

- Processing rate: ~22 tweets per hour
- Database size: 19,684 tweets analyzed (June-July 2025)
- Query latency: <100ms for aggregated sentiment
- EWMA calculation: Real-time with configurable decay

## Dependencies

- Python 3.8+
- PostgreSQL 12+
- psycopg2 (database connector)
- NLP libraries (see requirements.txt)
