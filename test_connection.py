import ccxt
import time
from config import TESTNET_API_KEY, TESTNET_API_SECRET

def format_balance(value):
    """ฟังก์ชันจัดรูปแบบตัวเลขให้อ่านง่าย"""
    return f"{float(value):.8f}".rstrip('0').rstrip('.')

def test_binance_connection():
    try:
        # สร้าง Binance instance สำหรับ testnet
        exchange = ccxt.binance({
            'apiKey': TESTNET_API_KEY,
            'secret': TESTNET_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # เปลี่ยนเป็น spot trading
                'testnet': True,
                'adjustForTimeDifference': True,  # เพิ่มการปรับ timestamp อัตโนมัติ
            },
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                }
            }
        })

        # ซิงค์เวลากับ server
        exchange.load_time_difference()
        print("\n=== Server Time Sync ===")
        print(f"Time difference with server: {exchange.options['timeDifference']} ms")

        # ทดสอบดึงข้อมูล account และแสดงข้อมูล portfolio
        print("\n=== Account Portfolio (Top 5) ===")
        balance = exchange.fetch_balance()
        
        # สร้างลิสต์ของ currencies ที่มียอดมากกว่า 0 และเรียงลำดับตามมูลค่า
        non_zero_balances = []
        for currency in balance['total']:
            if float(balance['total'][currency]) > 0:
                non_zero_balances.append({
                    'currency': currency,
                    'free': balance['free'][currency],
                    'used': balance['used'][currency],
                    'total': balance['total'][currency]
                })
        
        # เรียงลำดับตามยอดรวมจากมากไปน้อย
        sorted_balances = sorted(non_zero_balances, 
                               key=lambda x: float(x['total']), 
                               reverse=True)
        
        # แสดงข้อมูล Top 5 Balance
        print("\n📊 Top 5 Balance:")
        print("-" * 50)
        for idx, bal in enumerate(sorted_balances[:5], 1):
            print(f"#{idx} {bal['currency']}:")
            print(f"  💰 Available: {format_balance(bal['free'])}")
            print(f"  🔒 In Order: {format_balance(bal['used'])}")
            print(f"  📈 Total: {format_balance(bal['total'])}")
            print("-" * 50)

        # ดึงข้อมูลราคาปัจจุบันของ BTC/USDT
        ticker = exchange.fetch_ticker('BTC/USDT')
        btc_price = ticker['last']
        print(f"\n🔄 Current BTC/USDT Price: ${btc_price:,.2f}")

        # ดึงข้อมูล Open Orders
        open_orders = exchange.fetch_open_orders('BTC/USDT')
        if open_orders:
            print("\n📝 Open Orders:")
            print("-" * 50)
            for order in open_orders:
                print(f"Order ID: {order['id']}")
                print(f"Type: {order['type'].upper()} {order['side'].upper()}")
                print(f"Price: ${float(order['price']):,.2f}")
                print(f"Amount: {order['amount']} {order['symbol'].split('/')[0]}")
                print("-" * 50)
        else:
            print("\n📝 No Open Orders")

        return True

    except Exception as e:
        print(f"\nError: {str(e)}")
        if "Invalid API-key" in str(e):
            print("\nการแก้ไข:")
            print("1. ไปที่ https://testnet.binance.vision/")
            print("2. ล็อกอินด้วย GitHub")
            print("3. สร้าง API Key ใหม่")
            print("4. อย่าลืมเปิดสิทธิ์ Enable Reading และ Enable Spot & Margin Trading")
            print("5. นำ API Key และ Secret ใหม่มาใส่ในไฟล์ config.py")
        elif "Timestamp" in str(e):
            print("\nการแก้ไข:")
            print("1. ตรวจสอบว่าเวลาในเครื่องตรงกับเวลา Internet")
            print("2. ลองรันโปรแกรมอีกครั้ง หากยังไม่ได้ให้รอสักครู่แล้วลองใหม่")
        return False

if __name__ == "__main__":
    print("Testing Binance Testnet Connection...")
    if test_binance_connection():
        print("\nConnection test successful! ✅")
    else:
        print("\nConnection test failed! ❌") 