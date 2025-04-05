from binance.client import Client
from config import TESTNET_API_KEY, TESTNET_API_SECRET
import time
from datetime import datetime
from tabulate import tabulate

class SellAll:
    def __init__(self):
        self.client = Client(TESTNET_API_KEY, TESTNET_API_SECRET, testnet=True)
        
    def get_balances(self):
        """Get all asset balances"""
        try:
            account = self.client.get_account()
            balances = []
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                total = free + locked
                if total > 0 and asset['asset'] not in ['BTC', 'USDT']:
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
            # Check if trading pair exists
            exchange_info = self.client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            if f"{symbol}USDC" not in symbols:
                return None
                
            ticker = self.client.get_symbol_ticker(symbol=f"{symbol}USDC")
            return float(ticker['price'])
        except Exception as e:
            print(f"❌ Failed to get price for {symbol}: {str(e)}")
            return None
            
    def sell_asset(self, asset, quantity):
        """Sell asset to USDC"""
        try:
            symbol = f"{asset}USDC"
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
            return order
        except Exception as e:
            print(f"❌ Failed to sell {asset}: {str(e)}")
            return None
            
    def show_asset_info(self, balance):
        """Display single asset information"""
        try:
            asset = balance['asset']
            price = self.get_asset_price(asset)
            if price:
                value = balance['total'] * price
                balance_table = [[
                    asset,
                    f"{balance['free']:,.8f}",
                    f"{balance['locked']:,.8f}",
                    f"{balance['total']:,.8f}",
                    f"${price:,.2f}",
                    f"${value:,.2f}"
                ]]
                
                headers = ["Asset", "Free", "Locked", "Total", "Price (USDC)", "Value (USDC)"]
                print("\n" + "="*100)
                print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("💰 Asset to Sell:")
                print("="*100)
                print(tabulate(balance_table, headers=headers, tablefmt="grid"))
                print("="*100)
                
                return True
            return False
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
            
    def sell_all_assets(self):
        """Sell all assets to USDC one by one"""
        try:
            balances = self.get_balances()
            if not balances:
                print("No assets to sell")
                return
                
            print("\n🔄 Starting to sell assets one by one...")
            for balance in balances:
                asset = balance['asset']
                quantity = balance['free']
                if quantity > 0:
                    if self.show_asset_info(balance):
                        print(f"\nSelling {asset}...")
                        order = self.sell_asset(asset, quantity)
                        if order:
                            print(f"✅ Successfully sold {quantity} {asset}")
                        else:
                            print(f"❌ Failed to sell {asset}")
                        print(f"⏳ Waiting 10 seconds before next sale...")
                        time.sleep(10)  # Wait 10 seconds between orders
                
            print("\n✅ All assets sold successfully")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    sell_all = SellAll()
    print("\n🤖 Starting asset sale process...")
    print("This will sell all assets to USDC except BTC and USDT")
    print("Press Ctrl+C to stop")
    
    try:
        sell_all.sell_all_assets()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped successfully")

if __name__ == "__main__":
    main() 