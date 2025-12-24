#!/usr/bin/env python3
"""
Historical Price Backfill Script

Fetches historical hourly crypto prices from CoinGecko to match your tweet timeline.
Uses free API - no authentication required.

This backfills June-July 2025 to correlate with your Black Swan tweets.
"""

import os
import sys
import time
import logging
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": "blackswan_db",
    "user": "postgres",
    "host": "localhost",
    "port": 5432
}

# Crypto symbols to track
SYMBOLS = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']

# CoinGecko symbol mapping
COINGECKO_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'ADA': 'cardano',
    'DOT': 'polkadot'
}

class HistoricalPriceFetcher:
    """Fetch historical crypto prices from CoinGecko"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoTracker/1.0)'
        })

    def fetch_historical_range(self, symbol, start_date, end_date):
        """
        Fetch historical price data for a date range

        CoinGecko free tier:
        - < 90 days: hourly data
        - > 90 days: daily data

        Args:
            symbol: Crypto symbol (BTC, ETH, etc)
            start_date: datetime object for start
            end_date: datetime object for end

        Returns:
            List of price records
        """
        if symbol not in COINGECKO_IDS:
            logger.warning(f"Unknown symbol: {symbol}")
            return []

        coin_id = COINGECKO_IDS[symbol]

        # Convert to Unix timestamps
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_ts,
            'to': end_ts
        }

        try:
            logger.info(f"Fetching {symbol} from {start_date.date()} to {end_date.date()}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse response
            # CoinGecko returns: {"prices": [[timestamp_ms, price], ...],
            #                     "market_caps": [[timestamp_ms, mcap], ...],
            #                     "total_volumes": [[timestamp_ms, volume], ...]}

            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            volumes = data.get('total_volumes', [])

            # Convert to our format
            price_records = []
            for i, (ts_ms, price) in enumerate(prices):
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

                # Get corresponding market cap and volume
                market_cap = market_caps[i][1] if i < len(market_caps) else None
                volume = volumes[i][1] if i < len(volumes) else None

                price_records.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'price': price,
                    'volume_24h': volume,
                    'market_cap': market_cap,
                    'source': 'coingecko_historical'
                })

            logger.info(f"Fetched {len(price_records)} historical records for {symbol}")
            return price_records

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return []

class PriceDatabase:
    """Store historical prices in PostgreSQL"""

    def __init__(self):
        self.conn = None

    def connect(self):
        """Connect to PostgreSQL"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    def save_prices(self, prices):
        """Save prices to database"""
        if not prices:
            logger.warning("No prices to save")
            return 0

        conn = self.connect()
        with conn.cursor() as cur:
            # Prepare data for bulk insert
            values = [
                (
                    p['symbol'],
                    p['timestamp'],
                    p['price'],
                    p.get('volume_24h'),
                    p.get('market_cap'),
                    None,  # price_change_24h (not available in historical)
                    None,  # price_change_pct_24h (not available in historical)
                    p['source']
                )
                for p in prices
            ]

            # Insert with ON CONFLICT DO NOTHING to avoid duplicates
            query = """
                INSERT INTO crypto_prices
                (symbol, timestamp, price, volume_24h, market_cap,
                 price_change_24h, price_change_pct_24h, source)
                VALUES %s
                ON CONFLICT (symbol, timestamp, source) DO NOTHING
            """

            execute_values(cur, query, values)
            conn.commit()

            inserted = cur.rowcount
            logger.info(f"Saved {inserted} new historical price records")
            return inserted

    def get_tweet_date_range(self):
        """Get the date range of tweets in the database"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest,
                    COUNT(*) as total_tweets
                FROM tweets
            """)
            result = cur.fetchone()
            return {
                'earliest': result[0],
                'latest': result[1],
                'total_tweets': result[2]
            }

    def aggregate_hourly(self):
        """Create hourly OHLC aggregates from historical data"""
        conn = self.connect()
        with conn.cursor() as cur:
            # Get the last hour we've aggregated
            cur.execute("""
                SELECT COALESCE(MAX(hour), '1970-01-01'::timestamp)
                FROM crypto_prices_hourly
            """)
            last_hour = cur.fetchone()[0]

            # Aggregate all hours since then
            query = """
                INSERT INTO crypto_prices_hourly
                (symbol, hour, open_price, high_price, low_price, close_price, avg_price, volume_24h)
                SELECT
                    symbol,
                    DATE_TRUNC('hour', timestamp) as hour,
                    (array_agg(price ORDER BY timestamp))[1] as open_price,
                    MAX(price) as high_price,
                    MIN(price) as low_price,
                    (array_agg(price ORDER BY timestamp DESC))[1] as close_price,
                    AVG(price) as avg_price,
                    AVG(volume_24h) as volume_24h
                FROM crypto_prices
                WHERE timestamp > %s
                GROUP BY symbol, DATE_TRUNC('hour', timestamp)
                ON CONFLICT (symbol, hour) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    avg_price = EXCLUDED.avg_price,
                    volume_24h = EXCLUDED.volume_24h
            """

            cur.execute(query, (last_hour,))
            conn.commit()

            aggregated = cur.rowcount
            logger.info(f"Aggregated {aggregated} hourly records")
            return aggregated

    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Backfill historical crypto prices')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD), defaults to tweet start date')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), defaults to tweet end date')
    parser.add_argument('--symbols', type=str, nargs='+', default=SYMBOLS,
                       help='Crypto symbols to fetch (default: BTC ETH SOL ADA DOT)')

    args = parser.parse_args()

    logger.info("Starting historical price backfill...")

    db = PriceDatabase()
    fetcher = HistoricalPriceFetcher()

    try:
        # Get tweet date range if not specified
        if not args.start or not args.end:
            logger.info("Getting tweet date range from database...")
            tweet_range = db.get_tweet_date_range()
            logger.info(f"Tweets: {tweet_range['earliest']} to {tweet_range['latest']} ({tweet_range['total_tweets']} total)")

            start_date = tweet_range['earliest'] if not args.start else datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_date = tweet_range['latest'] if not args.end else datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        else:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

        logger.info(f"Backfilling prices from {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {', '.join(args.symbols)}")

        total_inserted = 0

        # Fetch for each symbol
        for symbol in args.symbols:
            prices = fetcher.fetch_historical_range(symbol, start_date, end_date)

            if prices:
                inserted = db.save_prices(prices)
                total_inserted += inserted

                # Rate limiting - be nice to free API
                logger.info(f"Waiting 1 second before next request...")
                time.sleep(1)
            else:
                logger.warning(f"No prices fetched for {symbol}")

        logger.info(f"Backfill complete! Inserted {total_inserted} total price records")

        # Aggregate into hourly data
        logger.info("Creating hourly aggregates...")
        aggregated = db.aggregate_hourly()
        logger.info(f"Created {aggregated} hourly aggregate records")

        # Show summary
        conn = db.connect()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    symbol,
                    COUNT(*) as records,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM crypto_prices
                WHERE source = 'coingecko_historical'
                GROUP BY symbol
                ORDER BY symbol
            """)

            logger.info("\n=== Backfill Summary ===")
            for row in cur.fetchall():
                logger.info(f"{row[0]}: {row[1]} records from {row[2]} to {row[3]}")

    except Exception as e:
        logger.error(f"Error in backfill: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()
