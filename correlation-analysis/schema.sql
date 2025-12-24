-- Simple DatabaseManagement Schema
-- Stores crypto price data in PostgreSQL (no Airtable needed)

-- Price history table
CREATE TABLE IF NOT EXISTS crypto_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    volume_24h DECIMAL(20, 2),
    market_cap DECIMAL(20, 2),
    price_change_24h DECIMAL(10, 4),
    price_change_pct_24h DECIMAL(10, 4),
    source VARCHAR(50) DEFAULT 'coingecko',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timestamp, source)
);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_crypto_prices_symbol_timestamp
    ON crypto_prices(symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_crypto_prices_timestamp
    ON crypto_prices(timestamp DESC);

-- Hourly aggregated data for faster queries
CREATE TABLE IF NOT EXISTS crypto_prices_hourly (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    hour TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(18, 8) NOT NULL,
    high_price DECIMAL(18, 8) NOT NULL,
    low_price DECIMAL(18, 8) NOT NULL,
    close_price DECIMAL(18, 8) NOT NULL,
    avg_price DECIMAL(18, 8) NOT NULL,
    volume_24h DECIMAL(20, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, hour)
);

CREATE INDEX IF NOT EXISTS idx_crypto_prices_hourly_symbol_hour
    ON crypto_prices_hourly(symbol, hour DESC);

-- Sentiment-Price correlation tracking
CREATE TABLE IF NOT EXISTS sentiment_price_correlation (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    btc_price DECIMAL(18, 8),
    eth_price DECIMAL(18, 8),
    overall_sentiment DECIMAL(10, 4),
    negative_feed_pct DECIMAL(10, 4),
    correlation_score DECIMAL(10, 4),
    analysis_run_id INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_price_correlation_timestamp
    ON sentiment_price_correlation(timestamp DESC);

-- Comments for documentation
COMMENT ON TABLE crypto_prices IS 'Raw cryptocurrency price data from free APIs (CoinGecko)';
COMMENT ON TABLE crypto_prices_hourly IS 'Hourly OHLC aggregates for faster queries';
COMMENT ON TABLE sentiment_price_correlation IS 'Correlation between Black Swan sentiment and crypto prices';
