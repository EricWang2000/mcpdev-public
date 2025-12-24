#!/usr/bin/env python3
"""
Simple Crypto Price Fetcher

Fetches cryptocurrency prices from free APIs and stores in PostgreSQL.
No API keys required for basic usage.

Supported APIs:
- CoinGecko (free tier: 30-50 calls/minute)
- CryptoCompare (free tier: 100,000 calls/month)
"""

import os
import sys
import time
import logging
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import pytz

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

class PriceFetcher:
    """Fetch crypto prices from free APIs"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoTracker/1.0)'
        })

    def fetch_coingecko_prices(self, symbols=None):
        """
        Fetch current prices from CoinGecko (free tier)
        No API key required for basic usage
        """
        if symbols is None:
            symbols = SYMBOLS

        # CoinGecko allows fetching multiple coins in one request
        coin_ids = [COINGECKO_IDS[sym] for sym in symbols if sym in COINGECKO_IDS]
        ids_param = ','.join(coin_ids)

        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': ids_param,
            'vs_currencies': 'usd',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_market_cap': 'true'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            prices = []
            timestamp = datetime.now(pytz.UTC)

            for symbol in symbols:
                if symbol not in COINGECKO_IDS:
                    continue

                coin_id = COINGECKO_IDS[symbol]
                if coin_id in data:
                    coin_data = data[coin_id]
                    prices.append({
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'price': coin_data.get('usd', 0),
                        'volume_24h': coin_data.get('usd_24h_vol'),
                        'market_cap': coin_data.get('usd_market_cap'),
                        'price_change_24h': coin_data.get('usd_24h_change'),
                        'price_change_pct_24h': coin_data.get('usd_24h_change'),
                        'source': 'coingecko'
                    })

            logger.info(f"Fetched {len(prices)} prices from CoinGecko")
            return prices

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            return []

    def fetch_cryptocompare_prices(self, symbols=None):
        """
        Fetch prices from CryptoCompare (free tier backup)
        No API key required for basic usage
        """
        if symbols is None:
            symbols = SYMBOLS

        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        params = {
            'fsyms': ','.join(symbols),
            'tsyms': 'USD'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Error':
                logger.error(f"CryptoCompare error: {data.get('Message')}")
                return []

            prices = []
            timestamp = datetime.now(pytz.UTC)

            raw_data = data.get('RAW', {})
            for symbol in symbols:
                if symbol in raw_data and 'USD' in raw_data[symbol]:
                    coin_data = raw_data[symbol]['USD']
                    prices.append({
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'price': coin_data.get('PRICE', 0),
                        'volume_24h': coin_data.get('VOLUME24HOUR'),
                        'market_cap': coin_data.get('MKTCAP'),
                        'price_change_24h': coin_data.get('CHANGE24HOUR'),
                        'price_change_pct_24h': coin_data.get('CHANGEPCT24HOUR'),
                        'source': 'cryptocompare'
                    })

            logger.info(f"Fetched {len(prices)} prices from CryptoCompare")
            return prices

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from CryptoCompare: {e}")
            return []

class PriceDatabase:
    """Store and manage crypto prices in PostgreSQL"""

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
                    p.get('price_change_24h'),
                    p.get('price_change_pct_24h'),
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
            logger.info(f"Saved {inserted} new price records")
            return inserted

    def aggregate_hourly(self):
        """Create hourly OHLC aggregates"""
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

    def update_sentiment_correlation(self):
        """
        Update sentiment-price correlation table
        Links Black Swan sentiment with crypto prices
        """
        conn = self.connect()
        with conn.cursor() as cur:
            # Get the latest analysis run
            cur.execute("""
                SELECT run_id, end_time, weighted_overall_score
                FROM analysis_runs
                WHERE status = 'completed'
                ORDER BY end_time DESC
                LIMIT 1
            """)
            latest_run = cur.fetchone()

            if not latest_run:
                logger.warning("No completed analysis runs found")
                return 0

            run_id, end_time, sentiment_score = latest_run

            # Get negative feed percentage
            cur.execute("""
                SELECT
                    COUNT(CASE WHEN weighted_score < -0.1 THEN 1 END)::float /
                    COUNT(*)::float * 100 as negative_pct
                FROM feed_weighted_scores
                WHERE run_id = %s
            """, (str(run_id),))
            negative_pct = cur.fetchone()[0]

            # Get prices closest to sentiment analysis time
            cur.execute("""
                SELECT symbol, price
                FROM crypto_prices
                WHERE timestamp <= %s
                  AND symbol IN ('BTC', 'ETH')
                  AND timestamp > %s - INTERVAL '1 hour'
                ORDER BY symbol, ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
            """, (end_time, end_time, end_time))

            prices = {row[0]: row[1] for row in cur.fetchall()}

            # Insert correlation record
            cur.execute("""
                INSERT INTO sentiment_price_correlation
                (timestamp, btc_price, eth_price, overall_sentiment,
                 negative_feed_pct, analysis_run_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                end_time,
                prices.get('BTC'),
                prices.get('ETH'),
                sentiment_score,
                negative_pct,
                run_id
            ))

            conn.commit()
            logger.info(f"Updated sentiment correlation for run {run_id}")
            return 1

    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

def main():
    """Main execution"""
    logger.info("Starting crypto price fetcher...")

    fetcher = PriceFetcher()
    db = PriceDatabase()

    try:
        # Try CoinGecko first (free, no API key needed)
        prices = fetcher.fetch_coingecko_prices()

        # Fallback to CryptoCompare if CoinGecko fails
        if not prices:
            logger.info("Trying CryptoCompare as fallback...")
            prices = fetcher.fetch_cryptocompare_prices()

        if prices:
            # Save to database
            db.save_prices(prices)

            # Aggregate hourly data
            db.aggregate_hourly()

            # Update sentiment correlation
            db.update_sentiment_correlation()

            logger.info("Price fetch completed successfully")
        else:
            logger.error("Failed to fetch prices from all sources")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()
