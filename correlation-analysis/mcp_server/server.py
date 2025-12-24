#!/usr/bin/env python3
"""
Price-Sentiment Correlation MCP Server

Provides tools for analyzing correlation between crypto prices and Twitter sentiment.
Combines data from crypto_prices and tweets tables for statistical analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Sequence
import json

import psycopg2
from psycopg2.extras import RealDictCursor
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("price-sentiment-mcp")

# Database configuration
DB_CONFIG = {
    "dbname": "blackswan_db",
    "user": "postgres",
    "host": "localhost",
    "port": 5432
}

class PriceSentimentMCPServer:
    """MCP Server for Price-Sentiment Correlation Analysis"""

    def __init__(self):
        self.server = Server("price-sentiment-correlation")
        self.db_conn = None
        self._setup_handlers()

    def _get_db_connection(self):
        """Get or create database connection"""
        if self.db_conn is None or self.db_conn.closed:
            self.db_conn = psycopg2.connect(**DB_CONFIG)
            self.db_conn.autocommit = True  # Enable autocommit to avoid transaction issues
        return self.db_conn

    def _reset_connection(self):
        """Reset database connection on error"""
        try:
            if self.db_conn and not self.db_conn.closed:
                self.db_conn.close()
        except:
            pass
        self.db_conn = None

    def _setup_handlers(self):
        """Set up MCP request handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_price_history",
                    description="Get historical crypto prices for a specific symbol and date range. Returns hourly price data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Crypto symbol (BTC, ETH, SOL, ADA, DOT)",
                                "enum": ["BTC", "ETH", "SOL", "ADA", "DOT"]
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD format)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD format)"
                            }
                        },
                        "required": ["symbol", "start_date", "end_date"]
                    }
                ),
                Tool(
                    name="get_sentiment_price_correlation",
                    description="Calculate correlation between sentiment and price for a given date range. Returns correlation coefficient and detailed analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Crypto symbol to analyze",
                                "enum": ["BTC", "ETH", "SOL", "ADA", "DOT"]
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)"
                            }
                        },
                        "required": ["symbol", "start_date", "end_date"]
                    }
                ),
                Tool(
                    name="find_sentiment_price_events",
                    description="Find periods where sentiment changed significantly and analyze corresponding price movements. Helps identify if sentiment predicts price.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Crypto symbol",
                                "enum": ["BTC", "ETH", "SOL", "ADA", "DOT"]
                            },
                            "sentiment_threshold": {
                                "type": "number",
                                "description": "Sentiment threshold (e.g., -0.3 for very negative)",
                                "default": -0.3
                            },
                            "hours_after": {
                                "type": "integer",
                                "description": "Hours after sentiment change to check price (default: 12)",
                                "default": 12
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="compare_crypto_sentiment_correlation",
                    description="Compare how different cryptocurrencies correlate with the same Twitter sentiment. Shows which crypto is most/least affected by sentiment.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["BTC", "ETH", "SOL", "ADA", "DOT"]
                                },
                                "description": "List of crypto symbols to compare"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)"
                            }
                        },
                        "required": ["symbols", "start_date", "end_date"]
                    }
                ),
                Tool(
                    name="get_hourly_sentiment_price",
                    description="Get hourly sentiment and price data side-by-side for detailed analysis. Returns time series data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Crypto symbol",
                                "enum": ["BTC", "ETH", "SOL", "ADA", "DOT"]
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max hours to return (default: 168 = 1 week)",
                                "default": 168
                            }
                        },
                        "required": ["symbol", "start_date", "end_date"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
            """Handle tool calls"""
            try:
                if name == "get_price_history":
                    return await self.get_price_history(
                        symbol=arguments.get("symbol"),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date")
                    )
                elif name == "get_sentiment_price_correlation":
                    return await self.get_sentiment_price_correlation(
                        symbol=arguments.get("symbol"),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date")
                    )
                elif name == "find_sentiment_price_events":
                    return await self.find_sentiment_price_events(
                        symbol=arguments.get("symbol"),
                        sentiment_threshold=arguments.get("sentiment_threshold", -0.3),
                        hours_after=arguments.get("hours_after", 12)
                    )
                elif name == "compare_crypto_sentiment_correlation":
                    return await self.compare_crypto_sentiment_correlation(
                        symbols=arguments.get("symbols", []),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date")
                    )
                elif name == "get_hourly_sentiment_price":
                    return await self.get_hourly_sentiment_price(
                        symbol=arguments.get("symbol"),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date"),
                        limit=arguments.get("limit", 168)
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in {name}: {e}", exc_info=True)
                # Reset connection on error to avoid transaction state issues
                self._reset_connection()
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def get_price_history(self, symbol: str, start_date: str, end_date: str) -> Sequence[TextContent]:
        """Get price history for a crypto"""
        conn = self._get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    timestamp,
                    price,
                    volume_24h,
                    market_cap
                FROM crypto_prices
                WHERE symbol = %s
                  AND timestamp >= %s::timestamp
                  AND timestamp < %s::timestamp + INTERVAL '1 day'
                ORDER BY timestamp
            """, (symbol, start_date, end_date))

            prices = cur.fetchall()

            if not prices:
                return [TextContent(type="text", text=f"No price data found for {symbol} between {start_date} and {end_date}")]

            # Calculate statistics
            avg_price = sum(float(p['price']) for p in prices) / len(prices)
            min_price = min(float(p['price']) for p in prices)
            max_price = max(float(p['price']) for p in prices)
            price_range = ((max_price - min_price) / min_price * 100)

            response = f"""# {symbol} Price History: {start_date} to {end_date}

## Summary
- **Data Points**: {len(prices)}
- **Average Price**: ${avg_price:,.2f}
- **Price Range**: ${min_price:,.2f} - ${max_price:,.2f}
- **Volatility**: {price_range:.2f}%

## Sample Data (First 10 hours)
| Timestamp | Price | Volume 24h |
|-----------|-------|------------|
"""
            for p in prices[:10]:
                vol = f"${float(p['volume_24h'])/1e9:.2f}B" if p['volume_24h'] else "N/A"
                response += f"| {p['timestamp'].strftime('%Y-%m-%d %H:%M')} | ${float(p['price']):,.2f} | {vol} |\n"

            if len(prices) > 10:
                response += f"\n... and {len(prices) - 10} more records"

            return [TextContent(type="text", text=response)]

    async def get_sentiment_price_correlation(self, symbol: str, start_date: str, end_date: str) -> Sequence[TextContent]:
        """Calculate correlation between sentiment and price"""
        conn = self._get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get hourly sentiment and price data
            cur.execute("""
                WITH hourly_data AS (
                    SELECT
                        DATE_TRUNC('hour', t.timestamp) as hour,
                        AVG(t.avg_score) as avg_sentiment,
                        COUNT(t.*) as tweet_count,
                        AVG(p.price) as avg_price
                    FROM tweets t
                    LEFT JOIN crypto_prices p ON
                        DATE_TRUNC('hour', p.timestamp) = DATE_TRUNC('hour', t.timestamp)
                        AND p.symbol = %s
                    WHERE t.timestamp >= %s::timestamp
                      AND t.timestamp < %s::timestamp + INTERVAL '1 day'
                      AND p.price IS NOT NULL
                    GROUP BY DATE_TRUNC('hour', t.timestamp)
                    HAVING COUNT(t.*) >= 3
                )
                SELECT
                    hour,
                    avg_sentiment,
                    tweet_count,
                    avg_price,
                    avg_price - LAG(avg_price) OVER (ORDER BY hour) as price_change,
                    avg_sentiment - LAG(avg_sentiment) OVER (ORDER BY hour) as sentiment_change
                FROM hourly_data
                ORDER BY hour
            """, (symbol, start_date, end_date))

            data = cur.fetchall()

            if len(data) < 5:
                return [TextContent(type="text", text=f"Insufficient data for correlation analysis. Need at least 5 hourly data points with both sentiment and price.")]

            # Calculate correlation coefficient
            valid_data = [(float(d['avg_sentiment']), float(d['avg_price'])) for d in data if d['avg_sentiment'] is not None and d['avg_price'] is not None]

            if len(valid_data) < 5:
                return [TextContent(type="text", text="Insufficient valid data points for correlation.")]

            # Simple Pearson correlation
            n = len(valid_data)
            sum_x = sum(x for x, y in valid_data)
            sum_y = sum(y for x, y in valid_data)
            sum_xy = sum(x * y for x, y in valid_data)
            sum_x2 = sum(x * x for x, y in valid_data)
            sum_y2 = sum(y * y for x, y in valid_data)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

            correlation = numerator / denominator if denominator != 0 else 0

            # Analyze direction of changes
            changes = [(float(d['sentiment_change']), float(d['price_change'])) for d in data if d['sentiment_change'] is not None and d['price_change'] is not None]
            same_direction = sum(1 for s, p in changes if (s > 0 and p > 0) or (s < 0 and p < 0))
            opposite_direction = sum(1 for s, p in changes if (s > 0 and p < 0) or (s < 0 and p > 0))

            response = f"""# Sentiment-Price Correlation Analysis: {symbol}

## Period: {start_date} to {end_date}

### Correlation Coefficient
**{correlation:.4f}** {"(Moderate Negative)" if correlation < -0.3 else "(Weak Negative)" if correlation < 0 else "(Weak Positive)" if correlation < 0.3 else "(Moderate Positive)"}

- **-1.0**: Perfect inverse correlation (sentiment down = price up)
- **0.0**: No correlation
- **+1.0**: Perfect positive correlation (sentiment up = price up)

### Change Direction Analysis
- **Same Direction**: {same_direction} hours ({same_direction/len(changes)*100:.1f}%)
  - Sentiment ↑ and Price ↑, or Sentiment ↓ and Price ↓
- **Opposite Direction**: {opposite_direction} hours ({opposite_direction/len(changes)*100:.1f}%)
  - Sentiment ↑ but Price ↓, or Sentiment ↓ but Price ↑

### Data Quality
- **Hours Analyzed**: {len(data)}
- **Valid Data Points**: {len(valid_data)}
- **Average Tweets/Hour**: {sum(d['tweet_count'] for d in data) / len(data):.1f}

### Interpretation
"""
            if abs(correlation) < 0.2:
                response += "⚠️ **Very Weak Correlation**: Sentiment and price show minimal relationship. Other factors dominate price movements.\n"
            elif abs(correlation) < 0.4:
                response += "📊 **Weak Correlation**: Some relationship exists but is not strong enough for reliable predictions. Use as one of many indicators.\n"
            elif abs(correlation) < 0.6:
                response += "📈 **Moderate Correlation**: Noticeable relationship, but still significant noise. May be useful as a supplementary signal.\n"
            else:
                response += "🎯 **Strong Correlation**: Significant relationship detected. However, correlation ≠ causation. Further analysis needed.\n"

            if same_direction > opposite_direction:
                response += "\n✅ Changes mostly move together, suggesting sentiment may reflect or influence price.\n"
            else:
                response += "\n❌ Changes often move in opposite directions, suggesting inverse or weak relationship.\n"

            return [TextContent(type="text", text=response)]

    async def find_sentiment_price_events(self, symbol: str, sentiment_threshold: float, hours_after: int) -> Sequence[TextContent]:
        """Find sentiment events and their price impact"""
        conn = self._get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                WITH hourly_sentiment AS (
                    SELECT
                        DATE_TRUNC('hour', timestamp) as hour,
                        AVG(avg_score) as avg_sentiment,
                        COUNT(*) as tweet_count
                    FROM tweets
                    WHERE timestamp >= '2025-06-01'
                    GROUP BY DATE_TRUNC('hour', timestamp)
                    HAVING AVG(avg_score) < %s
                       AND COUNT(*) >= 5
                )
                SELECT
                    s.hour as event_time,
                    s.avg_sentiment,
                    s.tweet_count,
                    p1.price as price_at_event,
                    p2.price as price_after,
                    ((p2.price - p1.price) / p1.price * 100) as price_change_pct
                FROM hourly_sentiment s
                LEFT JOIN crypto_prices p1 ON
                    DATE_TRUNC('hour', p1.timestamp) = s.hour
                    AND p1.symbol = %s
                LEFT JOIN crypto_prices p2 ON
                    DATE_TRUNC('hour', p2.timestamp) = s.hour + INTERVAL '%s hours'
                    AND p2.symbol = %s
                WHERE p1.price IS NOT NULL AND p2.price IS NOT NULL
                ORDER BY s.hour
                LIMIT 20
            """, (sentiment_threshold, symbol, hours_after, symbol))

            events = cur.fetchall()

            if not events:
                return [TextContent(type="text", text=f"No sentiment events found below {sentiment_threshold} with sufficient tweet volume.")]

            # Calculate success rate
            price_dropped = sum(1 for e in events if e['price_change_pct'] < 0)
            price_rallied = sum(1 for e in events if e['price_change_pct'] > 0)
            avg_change = sum(e['price_change_pct'] for e in events) / len(events)

            response = f"""# Sentiment Event Analysis: {symbol}

## Searching for: Sentiment < {sentiment_threshold}
## Looking {hours_after} hours ahead for price impact

### Found {len(events)} Events

### Outcome Summary
- **Price Dropped**: {price_dropped} times ({price_dropped/len(events)*100:.1f}%)
- **Price Rallied**: {price_rallied} times ({price_rallied/len(events)*100:.1f}%)
- **Average Price Change**: {avg_change:+.2f}%

### Event Details
| Date & Time | Sentiment | Tweets | Price at Event | Price {hours_after}h Later | Change |
|-------------|-----------|--------|----------------|---------------------------|--------|
"""
            for e in events:
                response += f"| {e['event_time'].strftime('%Y-%m-%d %H:00')} | {e['avg_sentiment']:.3f} | {e['tweet_count']} | ${e['price_at_event']:,.0f} | ${e['price_after']:,.0f} | {e['price_change_pct']:+.2f}% |\n"

            response += f"""

### Analysis
"""
            if price_dropped > len(events) * 0.6:
                response += f"✅ **Predictive Signal**: Price dropped {price_dropped/len(events)*100:.1f}% of the time after negative sentiment. However, this could be:\n"
                response += "   - Reverse causation (price dropped first, causing negative tweets)\n"
                response += "   - Common cause (news caused both sentiment and price drop)\n"
                response += f"   - Sample size ({len(events)} events) may be too small for statistical significance\n"
            elif price_rallied > len(events) * 0.6:
                response += f"⚠️ **Inverse Relationship**: Price actually rallied {price_rallied/len(events)*100:.1f}% of the time despite negative sentiment.\n"
                response += "   This suggests negative sentiment may be a contrarian indicator or lagging signal.\n"
            else:
                response += f"❌ **No Clear Pattern**: Results are close to random (50/50). Negative sentiment does not reliably predict price direction.\n"

            return [TextContent(type="text", text=response)]

    async def compare_crypto_sentiment_correlation(self, symbols: list, start_date: str, end_date: str) -> Sequence[TextContent]:
        """Compare correlation across multiple cryptos"""
        results = {}

        for symbol in symbols:
            conn = self._get_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        AVG(t.avg_score) as avg_sentiment,
                        AVG(p.price) as avg_price,
                        CORR(t.avg_score, p.price) as correlation
                    FROM tweets t
                    JOIN crypto_prices p ON
                        DATE_TRUNC('hour', p.timestamp) = DATE_TRUNC('hour', t.timestamp)
                    WHERE p.symbol = %s
                      AND t.timestamp >= %s::timestamp
                      AND t.timestamp < %s::timestamp + INTERVAL '1 day'
                """, (symbol, start_date, end_date))

                result = cur.fetchone()
                if result and result['correlation'] is not None:
                    results[symbol] = result

        if not results:
            return [TextContent(type="text", text="No correlation data available for the selected symbols and date range.")]

        # Sort by correlation strength
        sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)

        response = f"""# Cross-Crypto Sentiment Correlation Comparison

## Period: {start_date} to {end_date}

### Correlation Rankings (by strength)
| Rank | Symbol | Correlation | Interpretation |
|------|--------|-------------|----------------|
"""
        for i, (symbol, data) in enumerate(sorted_results, 1):
            corr = data['correlation']
            if abs(corr) < 0.2:
                interp = "Very Weak"
            elif abs(corr) < 0.4:
                interp = "Weak"
            elif abs(corr) < 0.6:
                interp = "Moderate"
            else:
                interp = "Strong"

            response += f"| {i} | {symbol} | {corr:.4f} | {interp} |\n"

        response += f"""

### Insights
"""
        strongest = sorted_results[0]
        weakest = sorted_results[-1]

        response += f"- **Most Correlated**: {strongest[0]} ({strongest[1]['correlation']:.4f})\n"
        response += f"- **Least Correlated**: {weakest[0]} ({weakest[1]['correlation']:.4f})\n"
        response += f"- **Spread**: {abs(strongest[1]['correlation'] - weakest[1]['correlation']):.4f}\n\n"

        if abs(strongest[1]['correlation']) < 0.3:
            response += "⚠️ All correlations are weak. Twitter sentiment may not be a strong indicator for any of these assets during this period.\n"

        return [TextContent(type="text", text=response)]

    async def get_hourly_sentiment_price(self, symbol: str, start_date: str, end_date: str, limit: int) -> Sequence[TextContent]:
        """Get hourly sentiment and price data"""
        conn = self._get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    DATE_TRUNC('hour', t.timestamp) as hour,
                    AVG(t.avg_score) as avg_sentiment,
                    COUNT(t.*) as tweet_count,
                    AVG(p.price) as avg_price,
                    MAX(p.price) as high_price,
                    MIN(p.price) as low_price
                FROM tweets t
                LEFT JOIN crypto_prices p ON
                    DATE_TRUNC('hour', p.timestamp) = DATE_TRUNC('hour', t.timestamp)
                    AND p.symbol = %s
                WHERE t.timestamp >= %s::timestamp
                  AND t.timestamp < %s::timestamp + INTERVAL '1 day'
                GROUP BY DATE_TRUNC('hour', t.timestamp)
                ORDER BY hour DESC
                LIMIT %s
            """, (symbol, start_date, end_date, limit))

            data = cur.fetchall()

            if not data:
                return [TextContent(type="text", text=f"No data found for {symbol} between {start_date} and {end_date}")]

            response = f"""# Hourly Sentiment & Price Data: {symbol}

## Period: {start_date} to {end_date}
## Showing: {len(data)} hours

| Hour | Sentiment | Tweets | Avg Price | Price Range |
|------|-----------|--------|-----------|-------------|
"""
            for d in reversed(data):  # Show oldest first
                price_str = f"${d['avg_price']:,.0f}" if d['avg_price'] else "N/A"
                range_str = f"${d['low_price']:,.0f}-${d['high_price']:,.0f}" if d['low_price'] else "N/A"
                sentiment_emoji = "🔴" if d['avg_sentiment'] < -0.1 else "🟢" if d['avg_sentiment'] > 0.1 else "🟡"

                response += f"| {d['hour'].strftime('%m-%d %H:00')} | {sentiment_emoji} {d['avg_sentiment']:.3f} | {d['tweet_count']} | {price_str} | {range_str} |\n"

            return [TextContent(type="text", text=response)]

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())

async def main():
    """Main entry point"""
    server = PriceSentimentMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
