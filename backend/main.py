import argparse
import signal
import sys
import logging
import os # เพิ่ม os
import threading # เพิ่ม threading
import time # เพิ่ม time
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from src.bot_manager import BotManager
from src.strategies import list_available_strategies

# ---- เพิ่ม import สำหรับ Google Auth และ JWT ----
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import jwt
import datetime
# ---------------------------------------------

app = Flask(__name__)
CORS(app, supports_credentials=True)

# ---- เพิ่ม Secret Key สำหรับ JWT ----
# ควรเก็บค่านี้เป็น environment variable ใน production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_very_secret_key_here')
# -----------------------------------

# ---- เพิ่ม Client ID ของ Google ----
# ควรเก็บค่านี้เป็น environment variable
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', "192088934065-0ourjhs2aso86vk890r2pfvsp6jmdo65.apps.googleusercontent.com")
# ------------------------------------

# กำหนดรูปแบบ logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- Global Variables ----
# ย้ายตัวแปรเหล่านี้มาที่นี่เพื่อให้เข้าถึงได้จาก routes ทั้งหมด
bot_manager: BotManager = None # กำหนด type hint
price_history = []
portfolio_data = {}
orders_data = []
bot_status_info = {
    'last_update': None,
    'errors': []
}
# ---------------------------

# ---- JWT Token Verification Decorator ----
from functools import wraps

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Bearer token malformed'}), 401

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            # สามารถเข้าถึงข้อมูล user จาก data['email'] หรือ data['user_id']
            # current_user = data # หรือดึงข้อมูล user จาก db
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(*args, **kwargs) # อาจจะส่ง current_user ไปด้วย f(current_user, *args, **kwargs)
    return decorated
# -----------------------------------------

# ---- Background Data Update Thread ----
def update_data():
    """อัพเดทข้อมูล API background"""
    global price_history, portfolio_data, orders_data, bot_status_info
    while True:
        try:
            if bot_manager is None:
                # logger.warning("Update thread: BotManager not initialized yet.")
                time.sleep(5)
                continue

            local_price_history = []
            bots = bot_manager.get_all_bots_status() # เปลี่ยนไปใช้ฟังก์ชันนี้ถ้ามี

            for bot_details in bots:
                 bot_id = bot_details.get('id')
                 bot_instance = bot_manager.get_bot(bot_id)
                 if bot_instance and bot_details.get('is_running'):
                    for symbol in bot_details.get('config', {}).get('symbols', []):
                        try:
                            current_price = bot_instance.get_current_price(symbol)
                            if current_price > 0:
                                local_price_history.append({
                                    'time': datetime.datetime.now().strftime('%H:%M:%S'),
                                    'symbol': symbol,
                                    'price': float(current_price)
                                })
                        except Exception as e:
                            logger.error(f"Error fetching price for {symbol} (bot {bot_id}): {str(e)}")

            # Update global price history (atomic update)
            price_history = local_price_history[-100:] # เก็บ 100 จุดล่าสุด

            # Update portfolio and orders
            portfolio_data = _internal_get_portfolio_data()
            orders_data = _internal_get_orders_data()

            # Update status timestamp
            bot_status_info['last_update'] = datetime.datetime.now().isoformat()
            # เคลียร์ errors ถ้าไม่มีปัญหา
            # bot_status_info['errors'] = [] # อาจจะไม่เคลียร์ที่นี่ รอให้ endpoint จัดการ

        except Exception as e:
            error_msg = f"Error in update_data thread: {str(e)}"
            logger.error(error_msg)
            bot_status_info['errors'].append({
                'time': datetime.datetime.now().isoformat(),
                'message': error_msg
            })
            if len(bot_status_info['errors']) > 10:
                 bot_status_info['errors'].pop(0)

        time.sleep(30)
# ------------------------------------

# ---- Signal Handler ---- (เหมือนเดิม)
def signal_handler(sig, frame):
    logger.info("Received stop signal, shutting down...")
    # อาจจะเพิ่มการ stop bots ก่อน exit
    if bot_manager:
        logger.info("Stopping all bots...")
        bot_manager.stop_all_bots()
    sys.exit(0)
# ----------------------

# ---- Helper Functions for Portfolio/Orders ----
# ย้าย logic การดึงข้อมูลมาเป็น internal function เพื่อใช้ใน update_data()
def _internal_get_portfolio_data():
    if bot_manager is None:
        return {'assets': [], 'totalValue': 0}
    try:
        total_value = 0
        assets = []
        all_balances = bot_manager.get_all_balances() # สมมติว่า BotManager มีฟังก์ชันนี้
        for bot_id, balance_info in all_balances.items():
            for asset, details in balance_info.get('total', {}).items():
                 if details > 0:
                    value = 0
                    if asset == 'USDT':
                        value = details
                    else:
                        try:
                             ticker = bot_manager.get_ticker(f'{asset}/USDT') # สมมติว่า BotManager มีฟังก์ชันนี้
                             if ticker and 'last' in ticker:
                                 value = details * ticker['last']
                             else:
                                 continue # Skip if ticker unavailable
                        except Exception as e:
                             logger.warning(f"Could not fetch ticker for {asset}/USDT: {e}")
                             continue

                    assets.append({
                        'symbol': asset,
                        'amount': float(details),
                        'value': float(value),
                        'bot_id': bot_id
                    })
                    total_value += value
        return {'assets': assets, 'totalValue': float(total_value)}
    except Exception as e:
        logger.error(f"Error in _internal_get_portfolio_data: {str(e)}")
        return {'assets': [], 'totalValue': 0, 'error': str(e)}

def _internal_get_orders_data():
    if bot_manager is None:
        return []
    try:
        orders_list = []
        all_open_orders = bot_manager.get_all_open_orders() # สมมติว่า BotManager มีฟังก์ชันนี้
        for bot_id, orders in all_open_orders.items():
            for order in orders:
                 orders_list.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['side'],
                    'amount': float(order['amount']),
                    'price': float(order.get('price', 0)), # Limit orders have price
                    'status': order['status'],
                    'bot_id': bot_id
                 })
        return orders_list
    except Exception as e:
        logger.error(f"Error in _internal_get_orders_data: {str(e)}")
        return []
# --------------------------------------------

# ---- API Routes ----

@app.route('/api/price')
@token_required # ปกป้อง Route นี้
def get_price_api(): # เปลี่ยนชื่อฟังก์ชันไม่ให้ซ้ำกับ global
    symbol = request.args.get('symbol', 'BTC/USDT')
    # Filter price history for the requested symbol (ใช้ global price_history)
    symbol_prices = [p for p in price_history if p.get('symbol') == symbol]
    return jsonify(symbol_prices)

@app.route('/api/portfolio')
@token_required # ปกป้อง Route นี้
def get_portfolio_api(): # เปลี่ยนชื่อฟังก์ชัน
    # Return data updated by the background thread
    return jsonify(portfolio_data)

@app.route('/api/orders')
@token_required # ปกป้อง Route นี้
def get_orders_api(): # เปลี่ยนชื่อฟังก์ชัน
    # Return data updated by the background thread
    return jsonify(orders_data)

@app.route('/api/status')
@token_required # ปกป้อง Route นี้
def get_status_api(): # เปลี่ยนชื่อฟังก์ชัน
    if bot_manager:
        # ดึงสถานะล่าสุดจาก BotManager แทนการใช้ global bot_status_info ที่อาจจะเก่า
        # เราอาจจะต้องการแค่สถานะ is_running และ config
        bots_summary = bot_manager.get_all_bots_status() # สมมติมีฟังก์ชันนี้
        return jsonify({
            'bots': bots_summary,
            'last_update': bot_status_info.get('last_update'), # ใช้เวลาอัพเดทล่าสุด
            'errors': bot_status_info.get('errors', [])
        })
    return jsonify({'error': 'BotManager not initialized'}), 500

@app.route('/api/strategies', methods=['GET'])
@token_required # ปกป้อง Route นี้ (ถ้าต้องการ)
def strategies():
    try:
        result = list_available_strategies()
        return jsonify({
            'success': True,
            'strategies': result,
            'message': 'ดึงกลยุทธ์สำเร็จ' if result else 'ไม่พบกลยุทธ์ที่ใช้งานได้'
        }), 200
    except Exception as e:
        logger.error(f"API Error in /api/strategies: {str(e)}")
        return jsonify({'success': False, 'error': 'Error fetching strategies', 'message': str(e)}), 500

# --- Bot Management Routes --- (ย้ายมาจาก api.py)
@app.route('/api/bot/create', methods=['POST'])
@token_required
def create_bot_api():
    if not bot_manager:
        return jsonify({'status': 'error', 'message': 'BotManager not initialized'}), 500
    try:
        data = request.json
        bot_id = data.get('id')
        config = data.get('config')
        if not bot_id or not config:
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        # ควร validate config เพิ่มเติม
        success = bot_manager.create_bot(bot_id, config)
        if success:
            return jsonify({'status': 'success', 'message': 'Bot created successfully'}), 201
        else:
            # อาจจะดึง error message จาก bot_manager
            return jsonify({'status': 'error', 'message': 'Failed to create bot'}), 500
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/bot/delete/<bot_id>', methods=['DELETE'])
@token_required
def delete_bot_api(bot_id):
    if not bot_manager:
        return jsonify({'status': 'error', 'message': 'BotManager not initialized'}), 500
    try:
        success = bot_manager.delete_bot(bot_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Bot deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete bot or bot not found'}), 404 # ใช้ 404 ถ้าไม่เจอ
    except Exception as e:
        logger.error(f"Error deleting bot {bot_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/bot/start/<bot_id>', methods=['POST'])
@token_required
def start_bot_api(bot_id):
    if not bot_manager:
        return jsonify({'status': 'error', 'message': 'BotManager not initialized'}), 500
    try:
        success = bot_manager.start_bot(bot_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Bot started successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start bot or bot not found'}), 404
    except Exception as e:
        logger.error(f"Error starting bot {bot_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/bot/stop/<bot_id>', methods=['POST'])
@token_required
def stop_bot_api(bot_id):
    if not bot_manager:
        return jsonify({'status': 'error', 'message': 'BotManager not initialized'}), 500
    try:
        success = bot_manager.stop_bot(bot_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Bot stopped successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to stop bot or bot not found'}), 404
    except Exception as e:
        logger.error(f"Error stopping bot {bot_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/bot/config/<bot_id>', methods=['POST'])
@token_required
def update_bot_config_api(bot_id):
    if not bot_manager:
        return jsonify({'status': 'error', 'message': 'BotManager not initialized'}), 500
    try:
        config = request.json
        if not config:
            return jsonify({'status': 'error', 'message': 'Missing config data'}), 400
        # ควร validate config เพิ่มเติม
        success = bot_manager.update_bot_config(bot_id, config)
        if success:
            return jsonify({'status': 'success', 'message': 'Config updated successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update config or bot not found'}), 404
    except Exception as e:
        logger.error(f"Error updating config for bot {bot_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
# ----------------------------

# --- Google Auth Route --- (เหมือนเดิม แต่เพิ่ม token_required ถ้าต้องการ)
@app.route('/api/auth/google', methods=['POST', 'OPTIONS'])
def google_auth():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    if request.method == 'POST':
        data = request.get_json()
        token = data.get('credential')
        if not token:
            return jsonify({'success': False, 'message': 'Credential missing'}), 400
        try:
            idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
            userid = idinfo['sub']
            email = idinfo['email']
            name = idinfo.get('name')
            picture = idinfo.get('picture')
            # TODO: Find or create user in database
            payload = {
                'user_id': userid,
                'email': email,
                'name': name,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }
            jwt_token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
            user_data = {'id': userid, 'email': email, 'name': name, 'picture': picture}
            logger.info(f"User {email} logged in.")
            return jsonify({'success': True, 'token': jwt_token, 'user': user_data, 'message': 'Login successful'}), 200
        except ValueError as e:
            logger.error(f"Invalid Google token: {str(e)}")
            return jsonify({'success': False, 'message': 'Invalid Google Token'}), 401
        except Exception as e:
            logger.error(f"Error during Google Auth: {str(e)}")
            return jsonify({'success': False, 'message': 'Internal server error during authentication'}), 500
# -------------------------

# 2. สร้าง API Endpoint สำหรับ updateUser
@app.route('/api/user/update', methods=['POST'])
@token_required # ปกป้อง route นี้
def update_user_profile():
    # ดึงข้อมูล user จาก token (ถ้าจำเป็น)
    # data = jwt.decode(request.headers['Authorization'].split(" ")[1], app.config['SECRET_KEY'], algorithms=["HS256"])
    # current_user_email = data['email']

    updates = request.get_json()
    if not updates:
        return jsonify({'success': False, 'message': 'No update data provided'}), 400

    # --- ส่วนนี้คือส่วนที่ต้อง implement การบันทึกข้อมูลจริง --- 
    # ตัวอย่าง: สมมติว่าบันทึกข้อมูล user ใน memory หรือ database
    # ต้องหาวิธีระบุ user ที่ต้องการอัปเดต (เช่น จาก email ใน JWT)
    # user_to_update = find_user_by_email(current_user_email)
    # if user_to_update:
    #     user_to_update.update(updates) # อัปเดตข้อมูล
    #     save_user(user_to_update)
    #     logger.info(f"User profile for {current_user_email} updated.")
    #     return jsonify({'success': True, 'message': 'Profile updated successfully'}), 200
    # else:
    #     return jsonify({'success': False, 'message': 'User not found'}), 404
    # -----------------------------------------------------------

    # ---- ถ้ายังไม่มี Database ให้จำลองว่าสำเร็จ ----
    logger.info(f"Received profile update request: {updates}")
    # ควรจะดึง email หรือ user_id จาก token มา log ด้วย
    return jsonify({'success': True, 'message': 'Profile update received (simulation)'}), 200
    # --------------------------------------------

# ---- CORS Preflight Helper ---- (เหมือนเดิม)
def _build_cors_preflight_response():
    response = make_response()
    # ควรระบุ origin ของ frontend แทน "*" ใน production
    response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response
# -------------------------------

# Endpoint for Frontend to fetch Google Client ID
@app.route('/api/config/google-client-id')
def get_google_client_id_config():
    client_id = app.config.get('GOOGLE_CLIENT_ID_CONFIG') # อ่านจาก app config
    if not client_id or client_id == "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com":
         logger.error("Google Client ID is not configured correctly in the backend environment.")
         return jsonify({"error": "Google Client ID not configured on server."}), 500
    return jsonify({"clientId": client_id})

# ---- Main Execution ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crypto Trading Bot Backend')
    parser.add_argument('--config-dir', type=str, default='configs',
                      help='Directory for bot config files')
    args = parser.parse_args()

    # ตั้งค่า signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # อ่าน GOOGLE_CLIENT_ID จาก env ตอนเริ่ม แล้วเก็บใน app.config
    # เพื่อให้ route ด้านบนอ่านได้ง่าย และเช็คตั้งแต่ตอนเริ่ม
    backend_google_client_id = os.environ.get('GOOGLE_CLIENT_ID', "192088934065-0ourjhs2aso86vk890r2pfvsp6jmdo65.apps.googleusercontent.com")
    if not backend_google_client_id or backend_google_client_id == "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com":
         logger.warning("GOOGLE_CLIENT_ID environment variable not set or using placeholder.")
         # อาจจะ exit ถ้าต้องการบังคับให้ตั้งค่า
         # sys.exit("Error: GOOGLE_CLIENT_ID environment variable is not set correctly.")
    app.config['GOOGLE_CLIENT_ID_CONFIG'] = backend_google_client_id
    # ตั้งค่า SECRET_KEY จาก env ด้วย
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_very_secret_key_here')
    if app.config['SECRET_KEY'] == 'your_very_secret_key_here':
         logger.warning("SECRET_KEY is using the default placeholder. Please set a secure environment variable.")

    try:
        # สร้าง BotManager และเก็บไว้ใน global variable
        bot_manager = BotManager(config_dir=args.config_dir)
        logger.info(f"BotManager initialized with config directory: {args.config_dir}")
        # โหลดบอทที่มีอยู่ตอนเริ่ม (อาจจะต้องเพิ่ม method ใน BotManager)
        # bot_manager.load_existing_bots()

        # เริ่ม background thread สำหรับอัพเดทข้อมูล
        update_thread = threading.Thread(target=update_data, daemon=True)
        update_thread.start()
        logger.info("Background data update thread started.")

        # รัน Flask app
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) # use_reloader=False สำคัญสำหรับ thread

    except Exception as e:
        logger.critical(f"Critical error during startup: {str(e)}", exc_info=True)
        sys.exit(1)
# ---------------------
