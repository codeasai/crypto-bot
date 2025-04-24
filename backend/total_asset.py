from binance.client import Client
from config import TESTNET_API_KEY, TESTNET_API_SECRET
import time
from datetime import datetime

class TotalAsset:
    def __init__(self):
        self.client = Client(TESTNET_API_KEY, TESTNET_API_SECRET, testnet=True)
        
    def get_total_assets(self):
        """Count total assets in portfolio"""
        try:
            account = self.client.get_account()
            total = 0
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                if free + locked > 0:
                    total += 1
            return total
        except Exception as e:
            print(f"❌ Failed to get total assets: {str(e)}")
            return None
            
    def show_total_assets(self):
        """Display total assets count"""
        try:
            total = self.get_total_assets()
            if total is not None:
                print("\n" + "="*50)
                print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("💰 Total Assets in Portfolio:")
                print("="*50)
                print(f"\n📊 Total Assets: {total}")
                print("="*50)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def countdown(seconds):
    """Display countdown timer"""
    for i in range(seconds, 0, -1):
        print(f"\r⏳ Next update in {i} seconds...", end="")
        time.sleep(1)
    print("\r" + " " * 50 + "\r", end="")

def main():
    total_asset = TotalAsset()
    print("\n🤖 Starting total assets counter...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            total_asset.show_total_assets()
            countdown(30)  # 30 seconds countdown
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped successfully")

if __name__ == "__main__":
    main() 