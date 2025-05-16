from binance.client import Client
from config import TESTNET_API_KEY, TESTNET_API_SECRET
import time
from datetime import datetime

def format_balance(value):
    """Format number for better readability"""
    return f"{float(value):.8f}".rstrip('0').rstrip('.')

def test_binance_connection():
    try:
        # Create Binance client for testnet
        client = Client(TESTNET_API_KEY, TESTNET_API_SECRET, testnet=True)

        # Test server time
        server_time = client.get_server_time()
        print("\n=== Server Time ===")
        print(f"Server Time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")

        # Test account info and show portfolio
        print("\n=== Account Portfolio ===")
        account = client.get_account()
        
        # Create list of currencies with balance > 0 and sort by value
        non_zero_balances = []
        for asset in account['balances']:
            free = float(asset['free'])
            locked = float(asset['locked'])
            total = free + locked
            if total > 0:
                non_zero_balances.append({
                    'asset': asset['asset'],
                    'free': free,
                    'locked': locked,
                    'total': total
                })
        
        # Sort by total value
        sorted_balances = sorted(non_zero_balances, 
                               key=lambda x: x['total'], 
                               reverse=True)
        
        # Show Top 5 Balance
        print("\n📊 Top 5 Balance:")
        print("-" * 50)
        for idx, bal in enumerate(sorted_balances[:5], 1):
            print(f"#{idx} {bal['asset']}:")
            print(f"  💰 Available: {format_balance(bal['free'])}")
            print(f"  🔒 In Order: {format_balance(bal['locked'])}")
            print(f"  📈 Total: {format_balance(bal['total'])}")
            print("-" * 50)

        # Get current BTC price
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        btc_price = float(ticker['price'])
        print(f"\n🔄 Current BTC/USDT Price: ${btc_price:,.2f}")

        # Get open orders
        open_orders = client.get_open_orders(symbol="BTCUSDT")
        if open_orders:
            print("\n📝 Open Orders:")
            print("-" * 50)
            for order in open_orders:
                print(f"Order ID: {order['orderId']}")
                print(f"Type: {order['type'].upper()} {order['side'].upper()}")
                print(f"Price: ${float(order['price']):,.2f}")
                print(f"Amount: {order['origQty']} {order['symbol'].replace('USDT', '')}")
                print("-" * 50)
        else:
            print("\n📝 No Open Orders")

        return True

    except Exception as e:
        print(f"\nError: {str(e)}")
        if "API-key" in str(e):
            print("\nSolutions:")
            print("1. Go to https://testnet.binance.vision/")
            print("2. Login with GitHub")
            print("3. Create new API Key")
            print("4. Enable Reading and Enable Spot & Margin Trading")
            print("5. Update API Key and Secret in config.py")
        elif "Timestamp" in str(e):
            print("\nSolutions:")
            print("1. Check if your computer time is synced with Internet time")
            print("2. Try again after a few seconds")
        return False

if __name__ == "__main__":
    print("Testing Binance Testnet Connection...")
    if test_binance_connection():
        print("\nConnection test successful! ✅")
    else:
        print("\nConnection test failed! ❌") 