from binance.client import Client
from config import TESTNET_API_KEY, TESTNET_API_SECRET
import time
from datetime import datetime
from order import Order

class Balance:
    def __init__(self):
        self.client = Client(TESTNET_API_KEY, TESTNET_API_SECRET, testnet=True)
        self.order = Order()
        
    def get_all_balances(self):
        """Get all asset balances"""
        try:
            account = self.client.get_account()
            balances = []
            # Always show BTC, USDC and USDT
            for asset in account['balances']:
                if asset['asset'] in ['BTC', 'USDC', 'USDT']:
                    free = float(asset['free'])
                    locked = float(asset['locked'])
                    total = free + locked
                    balances.append({
                        'asset': asset['asset'],
                        'free': free,
                        'locked': locked,
                        'total': total
                    })
            return balances
        except Exception as e:
            print(f"❌ Failed to get balances: {str(e)}")
            return None
            
    def get_asset_price(self, symbol):
        """Get current asset price"""
        try:
            # Check if symbol is supported
            if symbol == 'USDC':
                # Get USDC price from USDC/USDT pair
                ticker = self.client.get_symbol_ticker(symbol='USDCUSDT')
                return float(ticker['price'])
                
            # Check if trading pair exists
            exchange_info = self.client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            if f"{symbol}USDT" not in symbols:
                return None
                
            ticker = self.client.get_symbol_ticker(symbol=f"{symbol}USDT")
            return float(ticker['price'])
        except Exception as e:
            print(f"❌ Failed to get price for {symbol}: {str(e)}")
            return None
            
    def show_portfolio(self):
        """Show portfolio information"""
        try:
            balances = self.get_all_balances()
            if not balances:
                return
                
            # Create balance table
            total_value = 0
            
            print("\n" + "="*100)
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("💰 Portfolio Status:")
            print("="*100)
            
            print(f"\n📊 Total Assets: {len(balances)}")
            print("📊 Balance:")
            print("+" + "-"*98 + "+")
            print("| {:<8} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} |".format(
                "Asset", "Free", "Locked", "Total", "Price (USD)", "Value (USD)"
            ))
            print("+" + "-"*98 + "+")
            
            for balance in balances:
                asset = balance['asset']
                price = self.get_asset_price(asset)
                if not price:
                    continue
                    
                value = balance['total'] * price
                total_value += value
                
                print("| {:<8} | {:<12,.8f} | {:<12,.8f} | {:<12,.8f} | ${:<11,.2f} | ${:<11,.2f} |".format(
                    asset,
                    balance['free'],
                    balance['locked'],
                    balance['total'],
                    price,
                    value
                ))
            
            print("+" + "-"*98 + "+")
            print(f"\n💰 Total Portfolio Value: ${total_value:,.2f}")
                
            # Show recent orders
            recent_orders = self.order.get_recent_orders()
            if recent_orders:
                print("\n" + "="*100)
                print("📊 Recent Orders:")
                print("="*100)
                
                print("+" + "-"*98 + "+")
                print("| {:<8} | {:<12} | {:<8} | {:<12} | {:<10} | {:<20} |".format(
                    "ID", "Pair", "Type", "Amount", "Status", "Time"
                ))
                print("+" + "-"*98 + "+")
                
                for order in recent_orders:
                    print("| {:<8} | {:<12} | {:<8} | {:<12,.8f} | {:<10} | {:<20} |".format(
                        order['order_id'],
                        order['symbol'],
                        order['side'],
                        order['quantity'],
                        order['status'],
                        order['time']
                    ))
                
                print("+" + "-"*98 + "+")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def countdown(seconds):
    """Display countdown timer"""
    for i in range(seconds, 0, -1):
        print(f"\r⏳ Next update in {i} seconds...", end="")
        time.sleep(1)
    print("\r" + " " * 50 + "\r", end="")  # Clear the countdown line

def main():
    balance = Balance()
    print("\n🤖 Starting portfolio display...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            balance.show_portfolio()
            countdown(30)  # 30 seconds countdown
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped successfully")

if __name__ == "__main__":
    main() 