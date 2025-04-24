import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
import datetime
import threading
import logging
from typing import Dict, List, Optional, Tuple, Union

from src.utils import CostBasisCalculator, RiskManager
from src.strategies import get_strategy

logger = logging.getLogger(__name__)

class CryptoTradingBot:
    """
    คลาสหลักของ Crypto Trading Bot
    """
    def __init__(self, bot_id: str = 'default', config_dir: str = 'configs', user_id: str = 'default'):
        self.bot_id = bot_id
        self.config_dir = config_dir
        self.user_id = user_id
        self.config_file = os.path.join(os.path.abspath(config_dir), f"{bot_id}.json")
        self.load_config()
        
        # ตั้งค่าเริ่มต้นสำหรับบอท
        self.exchange = None
        self.setup_exchange()
        
        self.cost_calculator = CostBasisCalculator()
        self.risk_manager = RiskManager(
            max_position_pct=self.config.get('max_position_percentage', 0.20),
            max_trade_pct=self.config.get('max_trade_percentage', 0.05)
        )
        
        # โหลดกลยุทธ์
        strategy_name = self.config.get('strategy', 'ema_crossover')
        strategy_params = self.config.get('strategy_params', {})
        try:
            self.strategy = get_strategy(strategy_name, strategy_params)
            logger.info(f"โหลดกลยุทธ์ {strategy_name} สำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดกลยุทธ์: {str(e)}")
            # ใช้กลยุทธ์ ema_crossover เป็นค่าเริ่มต้น
            self.strategy = get_strategy('ema_crossover', {})
        
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
            'strategy': 'ema_crossover',
            'strategy_params': {
                'short_period': 12,
                'long_period': 26
            },
            'max_position_percentage': 0.20,
            'max_trade_percentage': 0.05,
            'check_interval': 60,
            'maturity_level': 1
        }
        
        try:
            # ตรวจสอบว่า config_dir มีอยู่จริง
            if not os.path.exists(os.path.dirname(self.config_file)):
                os.makedirs(os.path.dirname(self.config_file))
                logger.info(f"Created config directory: {os.path.dirname(self.config_file)}")

            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                    # ตรวจสอบว่า config มีครบทุกฟิลด์ ถ้าไม่มีให้ใช้ค่าเริ่มต้น
                    for key, value in default_config.items():
                        if key not in self.config:
                            self.config[key] = value
            else:
                self.config = default_config
                self.save_config() # บันทึก default config ลงไฟล์
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
        history_file = os.path.join(os.path.dirname(self.config_file), f"{self.bot_id}_history.json")
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
        history_file = os.path.join(os.path.dirname(self.config_file), f"{self.bot_id}_history.json")
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
            api_key = self.config.get('api_key')
            api_secret = self.config.get('api_secret')
            is_testnet = self.config.get('is_testnet', True)

            if not api_key or not api_secret:
                logger.warning("ไม่พบ API key หรือ API secret กรุณาระบุใน config file")
                self.exchange = None
                return
            
            exchange_id = 'binance' # หรือ exchange อื่นๆ
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })

            if is_testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info(f"เชื่อมต่อกับ Binance {'Testnet' if self.config.get('is_testnet', True) else ''} สำเร็จ")
            else:
                 logger.info(f"เชื่อมต่อกับ Binance {'Mainnet' if self.config.get('is_testnet', True) else ''} สำเร็จ")

            self.exchange.load_markets()
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า Exchange: {str(e)}")
            self.exchange = None
    
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
            
            # คำนวณกำไร/ขาดทุนเป็น %
            profit_loss_pct = ((current_price / cost_basis) - 1) * 100 if cost_basis > 0 else 0
            
            # ใช้กลยุทธ์ที่โหลดไว้
            analysis_result = self.strategy.analyze({
                'symbol': symbol,
                'current_price': current_price,
                'current_holdings': current_holdings,
                'cost_basis': {
                    'weighted_average': weighted_avg,
                    'with_fees': with_fees,
                    'fifo': fifo_basis
                },
                'profit_loss_percentage': profit_loss_pct
            })
            
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
                'action': analysis_result.get('action', 'HOLD'),
                'reason': analysis_result.get('reason', ''),
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
                                if analysis.get('reason'):
                                    logger.info(f"เหตุผล: {analysis['reason']}")
                        
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
                if analysis.get('reason'):
                    logger.info(f"เหตุผล: {analysis['reason']}")
                
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
                    self.strategy.adjust_thresholds_based_on_trend(market_trend)
                    
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
    
    def _analyze_market_trend(self, symbol: str) -> str:
        """
        วิเคราะห์แนวโน้มตลาดเพื่อปรับเกณฑ์การตัดสินใจ (จำลอง RL)
        """
        try:
            # ดึงข้อมูลราคาย้อนหลัง
            timeframe = self.config.get('timeframe', '1h')
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=24)
            
            if not ohlcv or len(ohlcv) < 12:
                return "NEUTRAL"
                
            # แปลงเป็น DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # คำนวณอัตราการเปลี่ยนแปลงราคา
            df['price_change'] = df['close'].pct_change()
            
            # คำนวณ MA อย่างง่าย
            df['ma_short'] = df['close'].rolling(window=6).mean()
            df['ma_long'] = df['close'].rolling(window=12).mean()
            
            # ตรวจสอบแนวโน้ม
            last_price = df['close'].iloc[-1]
            ma_short = df['ma_short'].iloc[-1]
            ma_long = df['ma_long'].iloc[-1]
            
            price_changes = df['price_change'].dropna().tolist()
            avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
            
            # ตัดสินใจแนวโน้ม
            if ma_short > ma_long * 1.03 and avg_change > 0.01:
                return "STRONG_UP"
            elif ma_short > ma_long and avg_change > 0:
                return "MODERATE_UP"
            elif ma_short < ma_long * 0.97 and avg_change < -0.01:
                return "STRONG_DOWN"
            elif ma_short < ma_long and avg_change < 0:
                return "MODERATE_DOWN"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์แนวโน้มตลาด: {str(e)}")
            return "NEUTRAL"

    # เพิ่ม Method เพื่อให้ BotManager เรียกใช้ได้ง่ายขึ้น
    def get_status(self) -> Dict:
        return {
            'id': self.bot_id,
            'is_running': self.is_running,
            'config': self.config,
            'last_trade_time': self.trade_history[-1]['timestamp'] if self.trade_history else None
        }

# ตัวอย่างการใช้งาน (ถ้าต้องการทดสอบ bot.py โดยตรง)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running bot.py directly for testing...")

    # สร้าง config ทดสอบ (ควรใช้ไฟล์จริง)
    test_bot_id = "test_bot_01"
    test_config_dir = "./configs_test"
    if not os.path.exists(test_config_dir):
        os.makedirs(test_config_dir)

    # สร้างไฟล์ config ทดสอบ (ใส่ API Key ปลอม หรือใช้ Testnet Key จริง)
    test_config = {
        "api_key": "YOUR_TESTNET_API_KEY",
        "api_secret": "YOUR_TESTNET_API_SECRET",
        "is_testnet": True,
        "symbols": ["BTC/USDT"],
        "timeframe": "1m",
        "strategy": "ema_crossover",
        "strategy_params": { "short_period": 9, "long_period": 21 },
        "max_position_percentage": 0.1,
        "max_trade_percentage": 0.05,
        "check_interval": 15, # เช็คถี่ๆ สำหรับทดสอบ
        "maturity_level": 1 # ระดับ 1: แค่วิเคราะห์
    }
    with open(os.path.join(test_config_dir, f"{test_bot_id}.json"), 'w') as f:
        json.dump(test_config, f, indent=4)

    bot = CryptoTradingBot(bot_id=test_bot_id, config_dir=test_config_dir)

    if bot.exchange and bot.strategy:
        # bot.run_bot() # รันใน background
        # time.sleep(60) # รัน 1 นาที
        # bot.stop_bot()

        # หรือทดสอบรัน analysis ครั้งเดียว
        logger.info("Running single analysis test...")
        analysis_result = bot.run_single_analysis("BTC/USDT")
        logger.info(f"Analysis Result: {json.dumps(analysis_result, indent=2)}")

        # ทดสอบ trade execution (ระวัง! อาจจะเกิดการซื้อขายจริงถ้าใช้ key จริง)
        # if analysis_result.get('action') != 'HOLD':
        #     logger.info("Testing trade execution...")
        #     trade_result = bot.execute_trade("BTC/USDT", analysis_result['action'])
        #     logger.info(f"Trade Result: {json.dumps(trade_result, indent=2)}")

    else:
        logger.error("Bot initialization failed. Cannot run tests.")