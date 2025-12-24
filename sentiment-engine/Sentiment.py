#!/usr/bin/env python3
"""
Core Sentiment Analysis Engine

Processes cryptocurrency-related tweets using multi-model NLP sentiment analysis
and stores results in PostgreSQL for time-series correlation studies.

Features:
- Multi-model sentiment scoring (OpenAI, FinBERT, VADER)
- Exponential weighted moving average (EWMA) calculations
- Feed-specific sentiment tracking
- PostgreSQL time-series storage
"""

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": "crypto_analysis",
    "user": "postgres",
    "host": "localhost",
    "port": 5432,
}


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single tweet."""
    score: float  # -1.0 (negative) to +1.0 (positive)
    model: str    # Name of the model used
    timestamp: datetime


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.conn = None

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.conn.autocommit = True
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def store_tweet(self, handle: str, text: str, timestamp: datetime,
                   sentiment_scores: Dict[str, float]):
        """
        Store tweet with sentiment scores in database.

        Args:
            handle: Twitter handle
            text: Tweet content
            timestamp: Tweet timestamp
            sentiment_scores: Dict mapping model names to sentiment scores
        """
        try:
            avg_score = sum(sentiment_scores.values()) / len(sentiment_scores)

            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tweets (
                        handle, timestamp, text, avg_score,
                        openai_score, finbert_score, vader_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    handle,
                    timestamp,
                    text,
                    avg_score,
                    sentiment_scores.get('openai'),
                    sentiment_scores.get('finbert'),
                    sentiment_scores.get('vader')
                ))

            logger.debug(f"Stored tweet from {handle} with avg score {avg_score:.3f}")

        except Exception as e:
            logger.error(f"Failed to store tweet: {e}")
            raise


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1.0 (negative) to +1.0 (positive)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this analyzer."""
        pass


class OpenAISentimentAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer using OpenAI's API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Note: Actual OpenAI client initialization would go here
        # For demo purposes, this is simplified

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment using OpenAI's language model.

        The model is prompted to return a sentiment score between -1 and 1,
        where -1 is very negative, 0 is neutral, and 1 is very positive.
        """
        try:
            # Simplified - actual implementation would call OpenAI API
            # and parse the response to extract sentiment score
            prompt = f"""Analyze the sentiment of this cryptocurrency-related tweet.
Return only a number between -1 (very negative) and 1 (very positive).

Tweet: {text}

Sentiment score:"""

            # Placeholder - actual API call would happen here
            # response = openai.Completion.create(...)
            # score = float(response.choices[0].text.strip())

            score = 0.0  # Placeholder
            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return 0.0

    @property
    def name(self) -> str:
        return "openai"


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer using FinBERT model for financial text."""

    def __init__(self):
        # FinBERT model would be loaded here
        # For demo: from transformers import AutoModelForSequenceClassification
        pass

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment using FinBERT model.

        FinBERT is specifically trained on financial text and provides
        more accurate sentiment for cryptocurrency discussions.
        """
        try:
            # Simplified - actual implementation would:
            # 1. Tokenize text
            # 2. Run through FinBERT model
            # 3. Convert logits to sentiment score

            score = 0.0  # Placeholder
            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return 0.0

    @property
    def name(self) -> str:
        return "finbert"


class VADERSentimentAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer using VADER (lexicon-based)."""

    def __init__(self):
        # VADER analyzer would be initialized here
        # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        pass

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment using VADER lexicon-based approach.

        VADER is particularly good at handling social media text with
        emojis, slang, and informal language.
        """
        try:
            # Simplified - actual implementation would:
            # 1. Run VADER analyzer
            # 2. Extract compound score
            # 3. Normalize to -1 to 1 range

            score = 0.0  # Placeholder
            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return 0.0

    @property
    def name(self) -> str:
        return "vader"


class SentimentAnalysisEngine:
    """
    Main sentiment analysis engine coordinating multiple models.

    Combines results from multiple sentiment analyzers to produce
    robust sentiment scores for cryptocurrency-related tweets.
    """

    def __init__(self, db_config: Dict[str, str], openai_api_key: Optional[str] = None):
        self.db = DatabaseManager(db_config)
        self.analyzers: List[BaseSentimentAnalyzer] = []

        # Initialize available analyzers
        if openai_api_key:
            self.analyzers.append(OpenAISentimentAnalyzer(openai_api_key))
        self.analyzers.append(FinBERTSentimentAnalyzer())
        self.analyzers.append(VADERSentimentAnalyzer())

        logger.info(f"Initialized with {len(self.analyzers)} sentiment analyzers")

    def analyze_tweet(self, handle: str, text: str, timestamp: datetime) -> Dict[str, float]:
        """
        Analyze sentiment of a tweet using all available models.

        Args:
            handle: Twitter handle
            text: Tweet content
            timestamp: Tweet timestamp

        Returns:
            Dictionary mapping model names to sentiment scores
        """
        scores = {}

        for analyzer in self.analyzers:
            try:
                score = analyzer.analyze(text)
                scores[analyzer.name] = score
                logger.debug(f"{analyzer.name}: {score:.3f} for tweet from {handle}")
            except Exception as e:
                logger.error(f"Analyzer {analyzer.name} failed: {e}")
                scores[analyzer.name] = 0.0

        return scores

    def process_and_store(self, handle: str, text: str, timestamp: datetime):
        """
        Analyze tweet sentiment and store in database.

        Args:
            handle: Twitter handle
            text: Tweet content
            timestamp: Tweet timestamp
        """
        scores = self.analyze_tweet(handle, text, timestamp)

        if not scores:
            logger.warning("No sentiment scores generated")
            return

        self.db.store_tweet(handle, text, timestamp, scores)

        avg_score = sum(scores.values()) / len(scores)
        logger.info(f"Processed tweet from {handle}: avg={avg_score:.3f}")

    def calculate_ewma(self, symbol: str, alpha: float = 0.1) -> float:
        """
        Calculate Exponential Weighted Moving Average of sentiment.

        EWMA formula: EWMA(t) = α * sentiment(t) + (1-α) * EWMA(t-1)

        Args:
            symbol: Cryptocurrency symbol to filter tweets
            alpha: Smoothing factor (0 < alpha < 1)

        Returns:
            Current EWMA sentiment score
        """
        try:
            with self.db.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT avg_score, timestamp
                    FROM tweets
                    WHERE text ILIKE %s
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (f'%{symbol}%',))

                rows = cur.fetchall()

            if not rows:
                return 0.0

            # Calculate EWMA
            ewma = rows[-1]['avg_score']  # Start with oldest value

            for row in reversed(rows[:-1]):
                ewma = alpha * row['avg_score'] + (1 - alpha) * ewma

            logger.info(f"EWMA for {symbol}: {ewma:.4f}")
            return ewma

        except Exception as e:
            logger.error(f"EWMA calculation failed: {e}")
            return 0.0

    def get_hourly_sentiment(self, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, float]]:
        """
        Get hourly aggregated sentiment scores.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of (timestamp, avg_sentiment) tuples
        """
        try:
            with self.db.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        date_trunc('hour', timestamp) as hour,
                        AVG(avg_score) as avg_sentiment,
                        COUNT(*) as tweet_count
                    FROM tweets
                    WHERE timestamp BETWEEN %s AND %s
                    GROUP BY hour
                    ORDER BY hour
                """, (start_time, end_time))

                rows = cur.fetchall()

            result = [(row['hour'], row['avg_sentiment']) for row in rows]
            logger.info(f"Retrieved {len(result)} hourly sentiment values")
            return result

        except Exception as e:
            logger.error(f"Hourly sentiment retrieval failed: {e}")
            return []

    def start(self):
        """Initialize the sentiment analysis engine."""
        self.db.connect()
        logger.info("Sentiment analysis engine started")

    def stop(self):
        """Shutdown the sentiment analysis engine."""
        self.db.disconnect()
        logger.info("Sentiment analysis engine stopped")


def main():
    """
    Main entry point for sentiment analysis.

    In production, this would:
    1. Connect to Twitter API
    2. Stream tweets from monitored accounts
    3. Analyze sentiment in real-time
    4. Store results in database
    """
    logger.info("Starting sentiment analysis engine")

    # Initialize engine
    engine = SentimentAnalysisEngine(
        db_config=DB_CONFIG,
        openai_api_key="your-openai-api-key-here"  # Replace with actual key
    )

    try:
        engine.start()

        # In production, this would be a continuous stream
        # For demo, showing the structure:
        logger.info("Engine ready to process tweets")
        logger.info("In production: would stream from Twitter API")
        logger.info("For correlation analysis: use get_hourly_sentiment()")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
