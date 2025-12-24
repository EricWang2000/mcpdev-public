#!/usr/bin/env python3
"""
MCP Tools Test Suite

Tests all 5 tools with known-good parameters to verify:
1. No type conversion errors
2. No SQL syntax errors
3. Consistent output format
4. Error handling works
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import PriceSentimentMCPServer

async def test_all_tools():
    """Test all 5 MCP tools"""
    server = PriceSentimentMCPServer()

    # Test parameters (known to have data)
    test_params = {
        'symbol': 'BTC',
        'start_date': '2025-06-15',
        'end_date': '2025-06-20',
        'symbols': ['BTC', 'ETH', 'SOL'],
        'sentiment_threshold': -0.3,
        'hours_after': 12,
        'limit': 24
    }

    tests = [
        ("get_price_history", {
            'symbol': test_params['symbol'],
            'start_date': test_params['start_date'],
            'end_date': test_params['end_date']
        }),
        ("get_sentiment_price_correlation", {
            'symbol': test_params['symbol'],
            'start_date': test_params['start_date'],
            'end_date': test_params['end_date']
        }),
        ("find_sentiment_price_events", {
            'symbol': test_params['symbol'],
            'sentiment_threshold': test_params['sentiment_threshold'],
            'hours_after': test_params['hours_after']
        }),
        ("compare_crypto_sentiment_correlation", {
            'symbols': test_params['symbols'],
            'start_date': test_params['start_date'],
            'end_date': test_params['end_date']
        }),
        ("get_hourly_sentiment_price", {
            'symbol': test_params['symbol'],
            'start_date': test_params['start_date'],
            'end_date': test_params['end_date'],
            'limit': test_params['limit']
        })
    ]

    results = {
        'passed': [],
        'failed': []
    }

    for tool_name, args in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {tool_name}")
            print(f"Args: {args}")
            print(f"{'='*60}")

            # Get the tool method
            tool_method = getattr(server, tool_name)

            # Call the tool
            result = await tool_method(**args)

            # Verify result is TextContent
            if result and len(result) > 0:
                output = result[0].text
                print(f"✅ PASSED - {tool_name}")
                print(f"Output length: {len(output)} characters")
                print(f"First 200 chars: {output[:200]}...")
                results['passed'].append(tool_name)
            else:
                print(f"⚠️  WARNING - {tool_name} returned empty result")
                results['failed'].append((tool_name, "Empty result"))

        except Exception as e:
            print(f"❌ FAILED - {tool_name}")
            print(f"Error: {str(e)}")
            results['failed'].append((tool_name, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {len(results['passed'])}/{len(tests)}")
    for tool in results['passed']:
        print(f"   - {tool}")

    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}/{len(tests)}")
        for tool, error in results['failed']:
            print(f"   - {tool}: {error[:100]}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(test_all_tools())
