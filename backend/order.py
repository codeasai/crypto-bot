from binance.client import Client
from config.credentials import API_KEY, API_SECRET
import time
from datetime import datetime

class Order:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET, testnet=True)
        self.order_count = 0
        self.max_orders = 5
        self.orders = []  # เก็บประวัติคำสั่ง
        
    def get_balance(self, asset='USDT'):
        """แสดงยอดคงเหลือ"""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return {
                'asset': balance['asset'],
                'free': float(balance['free']),
                'locked': float(balance['locked'])
            }
        except Exception as e:
            print(f"❌ ไม่สามารถดึงยอดคงเหลือได้: {str(e)}")
            return None
            
    def check_balance(self, symbol, quantity):
        """ตรวจสอบยอดเงินก่อนส่งคำสั่ง"""
        try:
            # ดึงราคาปัจจุบัน
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # คำนวณมูลค่าที่ต้องการซื้อ
            total_value = current_price * quantity
            
            # ตรวจสอบยอด USDT
            usdt_balance = self.get_balance('USDT')
            if not usdt_balance:
                return False
                
            if usdt_balance['free'] < total_value:
                print(f"⚠️ ยอดเงินไม่พอ: ต้องการ {total_value:.2f} USDT แต่มี {usdt_balance['free']:.2f} USDT")
                return False
                
            return True
        except Exception as e:
            print(f"❌ ไม่สามารถตรวจสอบยอดเงินได้: {str(e)}")
            return False
            
    def place_market_order(self, symbol, side, quantity):
        """ส่งคำสั่งซื้อขายแบบตลาด"""
        try:
            if self.order_count >= self.max_orders:
                print(f"⚠️ ถึงขีดจำกัดจำนวนคำสั่งแล้ว ({self.max_orders} คำสั่ง)")
                return None
                
            # ตรวจสอบยอดเงินก่อนส่งคำสั่ง
            if not self.check_balance(symbol, quantity):
                return None
                
            if side.upper() == 'BUY':
                order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            else:
                order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
                
            self.order_count += 1
            order_info = {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': float(order['origQty']),
                'status': order['status'],
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.orders.append(order_info)
            return order_info
        except Exception as e:
            print(f"❌ ไม่สามารถส่งคำสั่งได้: {str(e)}")
            return None
            
    def get_order_status(self, symbol, order_id):
        """ตรวจสอบสถานะคำสั่ง"""
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': float(order['origQty']),
                'price': float(order['price']),
                'status': order['status'],
                'time': order['time']
            }
        except Exception as e:
            print(f"❌ ไม่สามารถตรวจสอบสถานะได้: {str(e)}")
            return None
            
    def reset_order_count(self):
        """รีเซ็ตจำนวนคำสั่ง"""
        self.order_count = 0
        self.orders = []
        print("🔄 รีเซ็ตจำนวนคำสั่งแล้ว")
        
    def get_recent_orders(self):
        """แสดงประวัติคำสั่งล่าสุด"""
        return self.orders

def main():
    order = Order()
    print("\n🤖 เริ่มส่งคำสั่งซื้อ BTC...")
    print("กด Ctrl+C เพื่อหยุดการทำงาน")
    
    try:
        while True:
            # ส่งคำสั่งซื้อ BTC
            result = order.place_market_order('BTCUSDT', 'BUY', 0.001)
            if result:
                print(f"\n📊 ส่งคำสั่งสำเร็จ:")
                print(f"ID: {result['order_id']}")
                print(f"คู่: {result['symbol']}")
                print(f"ประเภท: {result['side']}")
                print(f"จำนวน: {result['quantity']}")
                print(f"สถานะ: {result['status']}")
                print(f"เวลา: {result['time']}")
                
            # รอ 10 วินาที
            time.sleep(10)
            
            # รีเซ็ตจำนวนคำสั่งเมื่อครบ 5 คำสั่ง
            if order.order_count >= order.max_orders:
                order.reset_order_count()
            
    except KeyboardInterrupt:
        print("\n\n🛑 หยุดการทำงานเรียบร้อยแล้ว")

if __name__ == "__main__":
    main() 