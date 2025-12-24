-- Black Swan Detection Database Schema
-- PostgreSQL Database: blackswan_db

-- Create tables
CREATE TABLE IF NOT EXISTS tweets (
    id SERIAL PRIMARY KEY,
    tweet_id VARCHAR(255) UNIQUE,
    handle VARCHAR(255) NOT NULL,
    text TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tweet_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- Sentiment scores
    openai_score FLOAT,
    openai_label VARCHAR(20),
    openai_confidence FLOAT,
    
    finbert_score FLOAT,
    finbert_label VARCHAR(20),
    finbert_confidence FLOAT,
    
    cryptobert_score FLOAT,
    cryptobert_label VARCHAR(20),
    cryptobert_confidence FLOAT,
    
    cryptonewsbert_score FLOAT,
    cryptonewsbert_label VARCHAR(20),
    cryptonewsbert_confidence FLOAT,
    
    avg_score FLOAT,
    avg_confidence FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    
    -- Sentiment metrics
    overall_score FLOAT,
    negative_percentage FLOAT,
    negative_feed_ratio FLOAT,
    is_black_swan BOOLEAN DEFAULT FALSE,
    
    -- Market data
    btc_price FLOAT,
    btc_change FLOAT,
    btc_volume_spike FLOAT,
    eth_price FLOAT,
    eth_change FLOAT,
    eth_volume_spike FLOAT,
    spy_volatility FLOAT,
    market_stress BOOLEAN,
    
    -- Email alert tracking
    email_sent BOOLEAN DEFAULT FALSE,
    email_sent_at TIMESTAMP WITH TIME ZONE,
    
    -- Run metadata
    total_tweets_analyzed INTEGER,
    new_tweets_added INTEGER,
    feeds_analyzed INTEGER,
    status VARCHAR(50) DEFAULT 'running',
    
    -- Weighted scores
    weighted_overall_score FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feed_scores (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    handle VARCHAR(255) NOT NULL,
    
    -- Scores
    weighted_score FLOAT,
    simple_avg_score FLOAT,
    negative_percentage FLOAT,
    tweet_count INTEGER,
    influence_score FLOAT DEFAULT 0,
    
    -- Metadata
    last_update TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    correct_predictions INTEGER DEFAULT 0,
    total_predictions INTEGER DEFAULT 0,
    accuracy FLOAT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tweets_handle ON tweets(handle);
CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets(timestamp);
CREATE INDEX IF NOT EXISTS idx_tweets_hash ON tweets(tweet_hash);
CREATE INDEX IF NOT EXISTS idx_feed_scores_run_id ON feed_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_feed_scores_handle ON feed_scores(handle);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_timestamp ON analysis_runs(start_time);

-- Create views for common queries
CREATE OR REPLACE VIEW recent_tweets AS
SELECT * FROM tweets 
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW feed_sentiment_summary AS
SELECT 
    handle,
    COUNT(*) as tweet_count,
    AVG(avg_score) as avg_sentiment,
    SUM(CASE WHEN avg_score < 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 as negative_pct
FROM tweets
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY handle;

-- Initialize model performance tracking
INSERT INTO model_performance (model_name) VALUES 
    ('OpenAI'),
    ('FinBERT'),
    ('CryptoBERT'),
    ('CryptoNewsBERT')
ON CONFLICT DO NOTHING;