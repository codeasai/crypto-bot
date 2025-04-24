import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
import datetime
import threading
import signal
import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple, Union

# กำหนดรูปแบบ logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostBasisCalculator:
    """
    คลาสสำหรับคำนวณราคาต้นทุนเฉลี่ยของสินทรัพย์คริปโต
    """
    @staticmethod
    def calculate_weighted_average(trades: List[Dict]) -> float:
        """
        คำนวณราคาต้นทุนเฉลี่ยถ่วงน้ำหนักตามปริมาณ
        """
        if not trades:
            return 0
            
        # กรองเฉพาะคำสั่งซื้อ
        buy_trades = [trade for trade in trades if trade['side'].upper() == 'BUY']
        
        if not buy_trades:
            return 0
            
        total_qty = sum(float(trade['amount']) for trade in buy_trades)
        
        if total_qty == 0:
            return 0
            
        total_cost = sum(float(trade['amount']) * float(trade['price']) for trade in buy_trades)
        return total_cost / total_qty
    
    @staticmethod
    def calculate_with_fees(trades: List[Dict]) -> float:
        """
        คำนวณราคาต้นทุนเฉลี่ยรวมค่าธรรมเนียม
        """
        if not trades:
            return 0
            
        # กรองเฉพาะคำสั่งซื้อ
        buy_trades = [trade for trade in trades if trade['side'].upper() == 'BUY']
        
        if not buy_trades:
            return 0
            
        total_qty = sum(float(trade['amount']) for trade in buy_trades)
        
        if total_qty == 0:
            return 0
            
        total_cost = sum(float(trade['amount']) * float(trade['price']) + (float(trade.get('fee', {}).get('cost', 0)) if trade.get('fee', {}).get('currency') == 'USDT' else 0) for trade in buy_trades)
        return total_cost / total_qty
    
    @staticmethod
    def calculate_fifo_cost_basis(trades: List[Dict]) -> float:
        """
        คำนวณราคาต้นทุนโดยใช้วิธี First-In-First-Out (FIFO)
        """
        if not trades:
            return 0
            
        # เรียงลำดับตามเวลา
        sorted_trades = sorted(trades, key=lambda t: t['datetime'])
        
        buy_queue = []
        remaining_qty = 0
        
        for trade in sorted_trades:
            side = trade['side'].upper()
            qty = float(trade['amount'])
            price = float(trade['price'])
            
            if side == 'BUY':
                buy_queue.append({'amount': qty, 'price': price})
                remaining_qty += qty
            elif side == 'SELL' and remaining_qty > 0:
                sell_qty = qty
                while sell_qty > 0 and buy_queue:
                    oldest_buy = buy_queue[0]
                    if oldest_buy['amount'] <= sell_qty:
                        # ขายหมดการซื้อครั้งแรก
                        sell_qty -= oldest_buy['amount']
                        remaining_qty -= oldest_buy['amount']
                        buy_queue.pop(0)
                    else:
                        # ขายเพียงบางส่วนของการซื้อครั้งแรก
                        oldest_buy['amount'] -= sell_qty
                        remaining_qty -= sell_qty
                        sell_qty = 0
        
        # คำนวณราคาต้นทุนจากการซื้อที่เหลือ
        if remaining_qty == 0:
            return 0
            
        total_cost = sum(trade['amount'] * trade['price'] for trade in buy_queue)
        return total_cost / remaining_qty


class TradingDecisionMaker:
    """
    คลาสสำหรับตัดสินใจซื้อขายตามกลยุทธ์ราคาต้นทุน
    """
    def __init__(self, profit_target_pct: float = 0.05, buy_dip_pct: float = 0.10):
        self.profit_target_pct = profit_target_pct
        self.buy_dip_pct = buy_dip_pct
    
    def decide_action(self, current_price: float, cost_basis: float, holdings: float) -> str:
        """
        ตัดสินใจว่าจะซื้อ ขาย หรือถือตามราคาต้นทุนและราคาปัจจุบัน
        """
        if cost_basis <= 0:
            return "ENTER_NEW" if holdings <= 0 else "HOLD"
        
        profit_pct = (current_price / cost_basis) - 1
        
        if profit_pct >= self.profit_target_pct:
            return "SELL_PROFIT"
        elif profit_pct <= -self.buy_dip_pct:
            return "BUY_MORE"
        else:
            return "HOLD"
    
    def adjust_thresholds_based_on_trend(self, trend: str) -> None:
        """
        ปรับเกณฑ์การตัดสินใจตามแนวโน้มตลาด
        """
        if trend == "STRONG_UP":
            # ในตลาดขาขึ้นแรง เพิ่มเป้าหมายกำไรและลดการซื้อเพิ่ม
            self.profit_target_pct = 0.08
            self.buy_dip_pct = 0.15
        elif trend == "MODERATE_UP":
            self.profit_target_pct = 0.05
            self.buy_dip_pct = 0.12
        elif trend == "NEUTRAL":
            # ในตลาดทรงตัว ใช้ค่าปกติ
            self.profit_target_pct = 0.05
            self.buy_dip_pct = 0.10
        elif trend == "MODERATE_DOWN":
            self.profit_target_pct = 0.03
            self.buy_dip_pct = 0.08
        elif trend == "STRONG_DOWN":
            # ในตลาดขาลงแรง ลดเป้าหมายกำไรและเพิ่มความระมัดระวังในการซื้อเพิ่ม
            self.profit_target_pct = 0.02
            self.buy_dip_pct = 0.20


class RiskManager:
    """
    คลาสสำหรับจัดการความเสี่ยงในการเทรด
    """
    def __init__(self, max_position_pct: float = 0.20, max_trade_pct: float = 0.05):
        self.max_position_pct = max_position_pct  # สัดส่วนสูงสุดของพอร์ตที่จะลงทุนในสินทรัพย์เดียว
        self.max_trade_pct = max_trade_pct  # สัดส่วนสูงสุดของพอร์ตต่อการเทรดครั้งเดียว
    
    def validate_trade(self, balance: float, position_value: float, trade_amount: float) -> Tuple[bool, float]:
        """
        ตรวจสอบว่าการเทรดอยู่ในขอบเขตความเสี่ยงที่ยอมรับได้หรือไม่
        """
        total_portfolio = balance + position_value
        
        # ตรวจสอบว่าการเทรดนี้จะทำให้ตำแหน่งเกินขีดจำกัดหรือไม่
        new_position_value = position_value + trade_amount
        new_position_pct = new_position_value / total_portfolio if total_portfolio > 0 else 0
        
        if new_position_pct > self.max_position_pct:
            # ปรับขนาดการเทรดให้อยู่ในขีดจำกัด
            max_allowed_position = total_portfolio * self.max_position_pct
            adjusted_trade = max(0, max_allowed_position - position_value)
            return False, adjusted_trade
        
        # ตรวจสอบว่าขนาดการเทรดเกินขีดจำกัดต่อครั้งหรือไม่
        trade_pct = trade_amount / total_portfolio if total_portfolio > 0 else 0
        
        if trade_pct > self.max_trade_pct:
            adjusted_trade = total_portfolio * self.max_trade_pct
            return False, adjusted_trade
        
        return True, trade_amount


class CryptoTradingBot:
    """
    คลาสหลักของ Crypto Trading Bot ที่ใช้กลยุทธ์ราคาต้นทุน
    """
    def __init__(self, config_file: str = 'bot_config.json'):
        self.config_file = config_file
        self.load_config()
        
        # ตั้งค่าเริ่มต้นสำหรับบอท
        self.exchange = None
        self.setup_exchange()
        
        self.cost_calculator = CostBasisCalculator()
        self.decision_maker = TradingDecisionMaker(
            profit_target_pct=self.config.get('profit_target_percentage', 0.05),
            buy_dip_pct=self.config.get('buy_dip_percentage', 0.10)
        )
        self.risk_manager = RiskManager(
            max_position_pct=self.config.get('max_position_percentage', 0.20),
            max_trade_pct=self.config.get('max_trade_percentage', 0.05)
        )
        
        # สถานะการทำงาน
        self.is_running = False
        self.stop_event = threading.Event()
        self.bot_thread = None
        
        # บันทึกข้อมูลการเทรด
        self.trade_history = []
        self.load_trade_history()
    
    def load_config(self) -> None:
        """
        โหลด config จากไฟล์ หรือสร้างไฟล์ใหม่ถ้ายังไม่มี
        """
        default_config = {
            'api_key': '',
            'api_secret': '',
            'is_testnet': True,
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'timeframe': '5m',
            'profit_target_percentage': 0.05,  # 5% เป้าหมายกำไร
            'buy_dip_percentage': 0.10,        # 10% สำหรับซื้อเมื่อราคาลด
            'max_position_percentage': 0.20,   # สูงสุด 20% ของพอร์ตต่อสินทรัพย์
            'max_trade_percentage': 0.05,      # สูงสุด 5% ของพอร์ตต่อการเทรด
            'check_interval': 60,              # ตรวจสอบทุก 60 วินาที
            'maturity_level': 1                # ระดับความเป็นอัตโนมัติ 1-4
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                    # ตรวจสอบว่า config มีครบทุกฟิลด์ ถ้าไม่มีให้ใช้ค่าเริ่มต้น
                    for key, value in default_config.items():
                        if key not in self.config:
                            self.config[key] = value
            else:
                self.config = default_config
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=4)
                
                logger.info(f"สร้างไฟล์ config ที่ {self.config_file} แล้ว กรุณากรอก API key และ API secret")
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลด config: {str(e)}")
            self.config = default_config
    
    def save_config(self) -> None:
        """
        บันทึก config ลงไฟล์
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"บันทึก config ไปยัง {self.config_file} เรียบร้อยแล้ว")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึก config: {str(e)}")
    
    def load_trade_history(self) -> None:
        """
        โหลดประวัติการเทรดจากไฟล์
        """
        history_file = 'trade_history.json'
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.trade_history = json.load(f)
            else:
                self.trade_history = []
                with open(history_file, 'w') as f:
                    json.dump(self.trade_history, f, indent=4)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดประวัติการเทรด: {str(e)}")
            self.trade_history = []
    
    def save_trade_history(self) -> None:
        """
        บันทึกประวัติการเทรดลงไฟล์
        """
        history_file = 'trade_history.json'
        try:
            with open(history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=4)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกประวัติการเทรด: {str(e)}")
    
    def setup_exchange(self) -> None:
        """
        ตั้งค่าการเชื่อมต่อกับ Exchange
        """
        try:
            if not self.config.get('api_key') or not self.config.get('api_secret'):
                logger.warning("ไม่พบ API key หรือ API secret กรุณาระบุใน config file")
                return
            
            exchange_config = {
                'apiKey': self.config['api_key'],
                'secret': self.config['api_secret'],
                'enableRateLimit': True,
            }
            
            # ตั้งค่า testnet ถ้าเปิดใช้งาน
            if self.config.get('is_testnet', True):
                exchange_config['options'] = {
                    'defaultType': 'spot',
                    'testnet': True,
                    'adjustForTimeDifference': True,
                }
                exchange_config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
            
            self.exchange = ccxt.binance(exchange_config)
            logger.info(f"เชื่อมต่อกับ Binance {'Testnet' if self.config.get('is_testnet', True) else ''} สำเร็จ")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า Exchange: {str(e)}")
    
    def fetch_my_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        ดึงประวัติการเทรดของตัวเองสำหรับเหรียญที่ระบุ
        """
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงประวัติการเทรด: {str(e)}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """
        ดึงราคาปัจจุบันของเหรียญที่ระบุ
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงราคาปัจจุบัน: {str(e)}")
            return 0
    
    def get_balance(self, asset: str) -> float:
        """
        ดึงยอดคงเหลือของสินทรัพย์ที่ระบุ
        """
        try:
            balance = self.exchange.fetch_balance()
            if asset in balance:
                return float(balance[asset]['free']) + float(balance[asset]['used'])
            return 0
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงยอดคงเหลือ: {str(e)}")
            return 0
    
    def calculate_position_value(self, symbol: str) -> float:
        """
        คำนวณมูลค่าปัจจุบันของตำแหน่งที่ถือ
        """
        try:
            # แยก base asset จาก symbol (เช่น 'BTC' จาก 'BTC/USDT')
            base_asset = symbol.split('/')[0]
            
            # ดึงยอดคงเหลือและราคาปัจจุบัน
            balance = self.get_balance(base_asset)
            current_price = self.get_current_price(symbol)
            
            # คำนวณมูลค่า
            return balance * current_price
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณมูลค่าตำแหน่ง: {str(e)}")
            return 0
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        วิเคราะห์เหรียญที่ระบุเพื่อตัดสินใจซื้อขาย
        """
        try:
            # ดึงข้อมูลที่จำเป็น
            trades = self.fetch_my_trades(symbol)
            current_price = self.get_current_price(symbol)
            
            # ดึงยอดคงเหลือของ base asset
            base_asset = symbol.split('/')[0]
            current_holdings = self.get_balance(base_asset)
            
            # คำนวณราคาต้นทุนด้วยวิธีต่างๆ
            weighted_avg = self.cost_calculator.calculate_weighted_average(trades)
            with_fees = self.cost_calculator.calculate_with_fees(trades)
            fifo_basis = self.cost_calculator.calculate_fifo_cost_basis(trades)
            
            # ใช้ weighted average เป็นหลัก
            cost_basis = weighted_avg if weighted_avg > 0 else fifo_basis
            
            # ตัดสินใจซื้อขาย
            action = self.decision_maker.decide_action(current_price, cost_basis, current_holdings)
            
            # คำนวณกำไร/ขาดทุนเป็น %
            profit_loss_pct = ((current_price / cost_basis) - 1) * 100 if cost_basis > 0 else 0
            
            # สร้าง dictionary ผลการวิเคราะห์
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'current_holdings': current_holdings,
                'position_value_usd': current_holdings * current_price,
                'cost_basis': {
                    'weighted_average': weighted_avg,
                    'with_fees': with_fees,
                    'fifo': fifo_basis
                },
                'action': action,
                'profit_loss_percentage': profit_loss_pct,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ {symbol}: {str(e)}")
            return {}
    
    def execute_trade(self, symbol: str, action: str, amount: Optional[float] = None) -> Dict:
        """
        ส่งคำสั่งซื้อขายตามการตัดสินใจ
        """
        result = {
            'symbol': symbol,
            'action': action,
            'status': 'FAILED',
            'amount': 0,
            'price': 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        try:
            if action == "HOLD" or not self.exchange:
                result['status'] = 'SKIPPED'
                return result
            
            # ดึงข้อมูลที่จำเป็น
            quote_asset = symbol.split('/')[1]  # เช่น 'USDT' จาก 'BTC/USDT'
            base_asset = symbol.split('/')[0]   # เช่น 'BTC' จาก 'BTC/USDT'
            
            current_price = self.get_current_price(symbol)
            quote_balance = self.get_balance(quote_asset)
            base_balance = self.get_balance(base_asset)
            
            # คำนวณมูลค่าตำแหน่งปัจจุบัน
            position_value = base_balance * current_price
            
            # คำนวณจำนวนที่จะซื้อหรือขาย
            if amount is None:
                if action == "BUY_MORE" or action == "ENTER_NEW":
                    # ใช้ % ที่กำหนดของยอด quote asset
                    trade_amount_usd = quote_balance * self.config.get('max_trade_percentage', 0.05)
                    trade_amount = trade_amount_usd / current_price
                elif action == "SELL_PROFIT":
                    # ขาย % ที่กำหนดของยอด base asset
                    trade_amount = base_balance * 0.5  # ขาย 50% ของยอดที่ถือ
                else:
                    result['status'] = 'INVALID_ACTION'
                    return result
            else:
                trade_amount = amount
            
            # ตรวจสอบความเสี่ยง
            if action == "BUY_MORE" or action == "ENTER_NEW":
                is_valid, adjusted_amount = self.risk_manager.validate_trade(
                    quote_balance, position_value, trade_amount * current_price
                )
                if not is_valid:
                    logger.warning(f"ปรับขนาดการเทรดจาก {trade_amount} เป็น {adjusted_amount / current_price} ตามข้อจำกัดความเสี่ยง")
                    trade_amount = adjusted_amount / current_price
            
            # ดำเนินการซื้อขาย
            order = None
            trade_amount = round(trade_amount, 8)  # ปัดเศษให้พอดีกับ precision ของ Binance
            
            if trade_amount <= 0:
                result['status'] = 'ZERO_AMOUNT'
                return result
            
            if action == "BUY_MORE" or action == "ENTER_NEW":
                order = self.exchange.create_market_buy_order(symbol, trade_amount)
            elif action == "SELL_PROFIT":
                order = self.exchange.create_market_sell_order(symbol, trade_amount)
            
            # บันทึกผลการซื้อขาย
            if order:
                result.update({
                    'status': 'SUCCESS',
                    'order_id': order.get('id', ''),
                    'amount': float(order.get('amount', 0)),
                    'price': float(order.get('price', current_price)),
                    'cost': float(order.get('cost', 0))
                })
                
                # เพิ่มข้อมูลเข้าประวัติการเทรด
                trade_record = result.copy()
                self.trade_history.append(trade_record)
                self.save_trade_history()
                
                logger.info(f"ส่งคำสั่ง {action} สำเร็จ: {trade_amount} {base_asset} @ {current_price} {quote_asset}")
            
            return result
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการส่งคำสั่งซื้อขาย: {str(e)}")
            result['error'] = str(e)
            return result
    
    def run_bot(self) -> None:
        """
        เริ่มการทำงานของบอทในโหมด daemon
        """
        if self.is_running:
            logger.warning("บอทกำลังทำงานอยู่แล้ว")
            return
        
        if not self.exchange:
            logger.error("ยังไม่ได้ตั้งค่า Exchange กรุณาตั้งค่า API Key ใน config")
            return
        
        logger.info(f"เริ่มการทำงานของบอทในโหมด background...")
        self.is_running = True
        self.stop_event.clear()
        
        # สร้างและเริ่ม thread
        self.bot_thread = threading.Thread(target=self._bot_worker)
        self.bot_thread.daemon = True
        self.bot_thread.start()
    
    def stop_bot(self) -> None:
        """
        หยุดการทำงานของบอท
        """
        if not self.is_running:
            logger.warning("บอทไม่ได้ทำงานอยู่")
            return
        
        logger.info("กำลังหยุดการทำงานของบอท...")
        self.stop_event.set()
        
        if self.bot_thread:
            self.bot_thread.join(timeout=10)
        
        self.is_running = False
        logger.info("หยุดการทำงานของบอทเรียบร้อยแล้ว")
    
    def _bot_worker(self) -> None:
        """
        ฟังก์ชันการทำงานของบอทในโหมด background
        """
        try:
            while not self.stop_event.is_set():
                # ตรวจสอบแต่ละเหรียญที่กำหนดไว้ใน config
                for symbol in self.config.get('symbols', []):
                    try:
                        logger.info(f"\n=== เริ่มการวิเคราะห์ {symbol} ===")
                        analysis = self.run_single_analysis(symbol)
                        
                        # แสดงสรุปผลการวิเคราะห์
                        if analysis:
                            if 'trade_result' in analysis and analysis['trade_result']['status'] == 'SUCCESS':
                                logger.info(f"🔄 ทำรายการ {analysis['trade_result']['action']} สำเร็จ: "
                                           f"{analysis['trade_result']['amount']} {symbol.split('/')[0]} @ "
                                           f"{analysis['trade_result']['price']} {symbol.split('/')[1]}")
                            else:
                                logger.info(f"💡 {symbol}: ราคา ${analysis['current_price']:.2f}, คำแนะนำ: {analysis['action']}")
                        
                    except Exception as e:
                        logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ {symbol}: {str(e)}")
                
                # ตรวจสอบว่าควรหยุดหรือไม่ก่อนที่จะรอ
                if self.stop_event.is_set():
                    break
                
                # รอจนกว่าจะถึงรอบถัดไป
                logger.info(f"\n⏰ รอ {self.config.get('check_interval', 60)} วินาทีจนถึงรอบตรวจสอบถัดไป...\n")
                for _ in range(int(self.config.get('check_interval', 60) / 5)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(5)
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดใน bot worker: {str(e)}")
            self.is_running = False
    
    def run_single_analysis(self, symbol: str) -> Dict:
        """
        ทำการวิเคราะห์และส่งคำสั่งซื้อขายสำหรับเหรียญที่ระบุ
        """
        try:
            # วิเคราะห์เหรียญ
            analysis = self.analyze_symbol(symbol)
            
            # แสดงผลการวิเคราะห์
            if analysis:
                logger.info(f"\n--- การวิเคราะห์ {symbol} ---")
                logger.info(f"ราคาปัจจุบัน: {analysis['current_price']:.8f} USDT")
                logger.info(f"จำนวนที่ถือ: {analysis['current_holdings']:.8f} {symbol.split('/')[0]}")
                logger.info(f"มูลค่า: {analysis['position_value_usd']:.2f} USDT")
                
                if analysis['cost_basis']['weighted_average'] > 0:
                    logger.info(f"ราคาต้นทุน: {analysis['cost_basis']['weighted_average']:.8f} USDT")
                    logger.info(f"กำไร/ขาดทุน: {analysis['profit_loss_percentage']:.2f}%")
                
                logger.info(f"คำแนะนำ: {analysis['action']}")
                
                # ส่งคำสั่งซื้อขายตามระดับความเป็นอัตโนมัติ
                maturity_level = self.config.get('maturity_level', 1)
                
                if maturity_level == 1:
                    # ระดับ 1: แสดงผลการวิเคราะห์อย่างเดียว ไม่ส่งคำสั่งซื้อขาย
                    logger.info("ระดับความเป็นอัตโนมัติ 1: เพียงแสดงผลการวิเคราะห์")
                    return analysis
                    
                elif maturity_level == 2:
                    # ระดับ 2: ส่งคำสั่งซื้อขายเมื่อมีสัญญาณชัดเจน (SELL_PROFIT)
                    if analysis['action'] == "SELL_PROFIT":
                        logger.info("ระดับความเป็นอัตโนมัติ 2: ส่งคำสั่งขายอัตโนมัติเมื่อมีกำไรถึงเป้าหมาย")
                        trade_result = self.execute_trade(symbol, analysis['action'])
                        analysis['trade_result'] = trade_result
                    else:
                        logger.info(f"ระดับความเป็นอัตโนมัติ 2: ไม่ส่งคำสั่งซื้อเนื่องจากยังไม่ถึงเกณฑ์ ({analysis['action']})")
                    
                elif maturity_level == 3:
                    # ระดับ 3: ส่งคำสั่งซื้อขายอัตโนมัติทั้งซื้อและขาย
                    if analysis['action'] != "HOLD":
                        logger.info(f"ระดับความเป็นอัตโนมัติ 3: ส่งคำสั่ง {analysis['action']} อัตโนมัติ")
                        trade_result = self.execute_trade(symbol, analysis['action'])
                        analysis['trade_result'] = trade_result
                    
                elif maturity_level == 4:
                    # ระดับ 4: ใช้ Reinforcement Learning (ในตัวอย่างนี้จำลองด้วยการซื้อขายอัตโนมัติที่ปรับตามสภาพตลาด)
                    # ปรับเปลี่ยนเกณฑ์การตัดสินใจตามสภาพตลาด (จำลอง RL)
                    market_trend = self._analyze_market_trend(symbol)
                    self.decision_maker.adjust_thresholds_based_on_trend(market_trend)
                    
                    # วิเคราะห์ใหม่ด้วยเกณฑ์ที่ปรับแล้ว
                    analysis = self.analyze_symbol(symbol)
                    
                    if analysis['action'] != "HOLD":
                        logger.info(f"ระดับความเป็นอัตโนมัติ 4 (RL): ส่งคำสั่ง {analysis['action']} อัตโนมัติ (แนวโน้มตลาด: {market_trend})")
                        trade_result = self.execute_trade(symbol, analysis['action'])
                        analysis['trade_result'] = trade_result
                
                return analysis
                
            return {}
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์ {symbol}: {str(e)}")
            return {}