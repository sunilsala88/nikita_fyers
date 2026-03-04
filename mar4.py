
# ==================== CONFIGURATION VARIABLES ====================
# API Credentials
client_id = 'DF26CD57LX-100'
secret_key = '5SS5D9X5TR'

redirect_uri = 'https://fessorpro.com/'
response_type = "code"  
state = "sample_state"


secondary_data = False
kite_key = 'z59mkhj6yg8b6c81'
kite_secret = 'vd00dbutmiskmpelwfqf4o51s074d72h' 

# Trading Parameters
threshold = 12
take_type = {'FIXED': 1, 'PERCENTAGE': 2, 'RANGE': 3, 'ATR': 4}
T1 = 'RANGE'; v1 = 0.3
T2 = 'RANGE'; v2 = 1; v2_pos = 0.25
T3 = 'PERCENTAGE'; v3 = 20; v3_pos = 0.25
T4 = 'RANGE'; v4 = 2; v4_pos = 0.25
T5 = 'RANGE'; v5 = 3; v5_pos = 0.25
T6 = 'RANGE'; v6 = 4; v6_pos = 0

atr_sl_target='t3'

wait_buffer = 120
lot_size = {}
quantity_position = 4
candle_1 = '5'
candle_2 = '15'
candle_3 = '60'
capital = 800_000
min_15_condition = True
min_60_condition = True
index_15_condition = True

# Technical Indicators
rsi_1 = 14; rsi_smooth_1 = 14
rsi_2 = 14; rsi_smooth_2 = 14
rsi_3 = 14; rsi_smooth_3 = 14
atr_length = 14

# Strategy Settings
strategy_name = 'nikita'
account_type = 'PAPER'  # 'LIVE' or 'PAPER'
time_zone = "Asia/Kolkata"
start_hour, start_min = 9, 30
end_hour, end_min = 15, 10

# Symbol and Exchange Configuration
index_name = 'NIFTY50'
exchange = 'NSE'
type2 = 'INDEX'
fyers_underlying_index = f"{exchange}:{index_name}-{type2}"
kite_index = {'NIFTY50': 256265, 'NIFTYBANK': 260105}
kite_underlying_index_token = kite_index.get(index_name)
symbol_list = ['NIFTY2631024400PE','NIFTY2631024400CE']
fyers_initials = 'NSE:'
exchange_kite = 'NFO'



import signal
import threading

# Standard library imports
import os
import sys
import webbrowser
import pickle
import time
import asyncio
import logging
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

# Third-party imports
import pandas as pd
import pendulum as dt
import numpy as np
import certifi
import requests

# Trading API imports
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from kiteconnect import KiteTicker, KiteConnect

# Store original functions
_original_signal = signal.signal
_original_set_wakeup_fd = signal.set_wakeup_fd

def safe_signal(signalnum, handler):
    """Wrapper for signal.signal that only works in main thread"""
    if threading.current_thread() is threading.main_thread():
        return _original_signal(signalnum, handler)
    # In non-main thread, just return a dummy handler
    return handler

def safe_set_wakeup_fd(fd):
    """Wrapper for signal.set_wakeup_fd that only works in main thread"""
    if threading.current_thread() is threading.main_thread():
        return _original_set_wakeup_fd(fd)
    # In non-main thread, return -1 (indicating no previous fd)
    return -1

# Replace signal functions with our safe versions
signal.signal = safe_signal
signal.set_wakeup_fd = safe_set_wakeup_fd

# Disable logging for various modules
fyersModel.logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.CRITICAL)

# For Windows SSL error
os.environ['SSL_CERT_FILE'] = certifi.where()

# Global data storage
global final_data
final_data = {symbol: {} for symbol in symbol_list}
final_data[index_name] = {}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s || %(message)s",
    handlers=[
        logging.FileHandler(f'{strategy_name}_{dt.now(time_zone).date()}.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Log trading mode
if account_type == 'LIVE':
    logging.info("🔴 LIVE TRADING MODE - Real orders will be placed")
else:
    logging.info(f"📝 PAPER TRADING MODE ({account_type}) - Only logging trades to CSV")

# ==================== TIME SETUP ====================
# Get current time and trading hours
current_time = dt.now(time_zone)
start_time = dt.datetime(current_time.year, current_time.month, current_time.day, start_hour, start_min, tz=time_zone)
end_time = dt.datetime(current_time.year, current_time.month, current_time.day, end_hour, end_min, tz=time_zone)
print('start time:', start_time)
print('end time:', end_time)

# ==================== AUTHENTICATION SECTION ====================
def authenticate_fyers():
    """Handle Fyers authentication and token management"""
    # Check if access.txt exists, then read the file and get the access token
    if os.path.exists(f'fyers-access-{dt.now(time_zone).date()}.txt'):
        print('access token exists')
        with open(f'fyers-access-{dt.now(time_zone).date()}.txt', 'r') as f:
            access_token_fyers = f.read()
    else:
        # Define grant type for the session
        try:
            # Create a session model with the provided credentials
            session = fyersModel.SessionModel(
                client_id=client_id,
                secret_key=secret_key,
                redirect_uri=redirect_uri,
                response_type=response_type
            )

            # Generate the auth code using the session model
            response = session.generate_authcode()

            # Print the auth code received in the response
            print(response)

            # Open the auth code URL in a new browser window
            webbrowser.open(response, new=1)
            newurl = input("Enter the url: ")
            auth_code = newurl[newurl.index('auth_code=')+10:newurl.index('&state')]

            # Define grant type for the session
            grant_type = "authorization_code"
            session = fyersModel.SessionModel(
                client_id=client_id,
                secret_key=secret_key,
                redirect_uri=redirect_uri,
                response_type=response_type,
                grant_type=grant_type
            )

            # Set the authorization code in the session object
            session.set_token(auth_code)

            # Generate the access token using the authorization code
            response = session.generate_token()

            # Save the access token to access.txt
            access_token_fyers = response["access_token"]
            with open(f'fyers-access-{dt.now(time_zone).date()}.txt', 'w') as k:
                k.write(access_token_fyers)
        except Exception as e:
            # Print the exception and response for debugging
            print(e, response)
            print('unable to get access token')
            sys.exit()
    
    return access_token_fyers

def authenticate_kite():
    """Handle Kite authentication and token management"""
    if not secondary_data:
        return None
        
    api_key = kite_key
    api_secret = kite_secret

    # Check if Kite access token file exists
    if os.path.exists(f'kite_access-{dt.now(time_zone).date()}.txt'):
        print('Kite access token exists')
        with open(f'kite_access-{dt.now(time_zone).date()}.txt', 'r') as f:
            access_token_kite = f.read()
    else:
        # Generate Kite access token through OAuth flow
        try:
            kite = KiteConnect(api_key=api_key)
            response = kite.login_url()
            print("Kite Login URL:", response)
            
            # Open the login URL in browser
            webbrowser.open(response, new=1)
            
            newurl = input("Enter the redirect URL: ")
            print("Redirect URL:", newurl)
            
            # Extract request_token more dynamically to handle different URL parameter orders
            if 'request_token=' not in newurl:
                raise ValueError("request_token not found in the URL")
            
            start_index = newurl.index('request_token=') + 14
            end_index = newurl.find('&', start_index)
            
            if end_index == -1:  # If no '&' found after request_token, take until end
                request_token = newurl[start_index:]
            else:
                request_token = newurl[start_index:end_index]
            
            # Additional validation
            if not request_token or request_token.isspace():
                raise ValueError("Empty or invalid request_token extracted")
                
            print("Request Token:", request_token)
            
            # Generate session with request token
            data = kite.generate_session(request_token, api_secret=api_secret)
            print("Access Token:", data['access_token'])
            
            # Save the access token
            access_token_kite = data["access_token"]
            with open(f'kite_access-{dt.now(time_zone).date()}.txt', 'w') as k:
                k.write(access_token_kite)
                
        except Exception as e:
            print("Error generating Kite access token:", e)
            print("Unable to get Kite access token")
            sys.exit()
    
    return access_token_kite

# Authenticate and get tokens
access_token_fyers = authenticate_fyers()
print('access token fyers:', access_token_fyers)

access_token_kite = authenticate_kite()
if secondary_data:
    print('Kite access token:', access_token_kite)
else:
    print('Secondary data (Kite) disabled')


# ==================== API CLIENT INITIALIZATION ====================
# Initialize FyersModel instances for synchronous and asynchronous operations
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token_fyers, log_path=None)
fyers_asysc = fyersModel.FyersModel(client_id=client_id, is_async=True, token=access_token_fyers, log_path=None)

# Get available cash and set capital
available_cash = fyers.funds()
print('available cash:', available_cash.get('fund_limit')[-1].get('equityAmount'))
if capital == 0:
    capital = available_cash.get('fund_limit')[-1].get('equityAmount')
else:
    capital = capital

print('capital for trading:', capital)

# ==================== UTILITY FUNCTIONS ====================
def get_lot_size_for_symbols(symbol_list, fyers_initials):
    """Get lot sizes for given symbols from Fyers symbol master files"""
    try:
        urls = [
            "https://public.fyers.in/sym_details/NSE_CM.csv",
            "https://public.fyers.in/sym_details/NSE_FO.csv",
            "https://public.fyers.in/sym_details/MCX_COM.csv",
        ]

        column_names = [
            "Fytoken", "Symbol Details", "Exchange Instrument Type", "Minimum Lot Size",
            "Tick Size", "ISIN", "Trading Session", "Last Update Date", "Expiry Date",
            "Symbol Ticker", "Exchange", "Segment", "Scrip Code", "Underlying Symbol",
            "Underlying Scrip Code", "Strike Price", "Option Type", "Underlying FyToken",
            "Reserved Column 1", "Reserved Column 2", "Reserved Column 3",
        ]

        frames = []
        for url in urls:
            try:
                r = requests.get(url, timeout=30, verify=certifi.where())
                r.raise_for_status()
            except requests.exceptions.SSLError:
                r = requests.get(url, timeout=30, verify=False)
                r.raise_for_status()
            frames.append(pd.read_csv(StringIO(r.text), header=None, names=column_names))

        combined_df = pd.concat(frames, ignore_index=True)
        symbol_details = combined_df["Symbol Details"].astype(str)
        symbol_ticker = combined_df["Symbol Ticker"].astype(str)

        final_lot_size = {}
        for symbol in symbol_list:
            symbol_with_prefix = f"{fyers_initials}{symbol}"

            match = combined_df[symbol_details.eq(symbol_with_prefix)]
            if match.empty:
                match = combined_df[symbol_details.eq(symbol)]
            if match.empty:
                match = combined_df[symbol_ticker.eq(symbol_with_prefix)]
            if match.empty:
                match = combined_df[symbol_ticker.str.contains(symbol, na=False)]

            if match.empty:
                raise LookupError(f"Lot size not found for symbol: {symbol}")

            size = match["Minimum Lot Size"].iloc[0]
            if pd.isna(size):
                raise ValueError(f"Invalid lot size for symbol: {symbol}")

            final_lot_size[symbol] = int(size)

        return final_lot_size

    except Exception as e:
        raise SystemExit(f"Execution stopped: {e}")

def fetchOHLC(ticker, interval, duration):
    """Extract historical data and output as DataFrame"""
    from datetime import date, timedelta
    instrument = ticker
    data = {"symbol": instrument, "resolution": interval, "date_format": "1", 
            "range_from": date.today()-timedelta(duration), "range_to": date.today(), 
            "cont_flag": "1", 'oi_flag': "1"}
    sdata = fyers.history(data)
    sdata = pd.DataFrame(sdata['candles'])
    # Check if 'oi' column exists (7 columns)
    if sdata.shape[1] == 7:
        sdata.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi']
    else:
        sdata.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    sdata['date'] = pd.to_datetime(sdata['date'], unit='s')
    sdata.date = (sdata.date.dt.tz_localize('UTC').dt.tz_convert(time_zone))
    sdata['date'] = sdata['date'].dt.tz_localize(None)
    sdata = sdata.set_index('date')
    return sdata

def store(data, account_type):
    """Store TradeInfo objects as dictionaries for pickle serialization"""
    save_data = {symbol: trade_info.to_dict() for symbol, trade_info in data.items()}
    pickle.dump(save_data, open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'wb'))

def load(account_type):
    """Load trade data from pickle file"""
    return pickle.load(open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'rb'))

# Get lot sizes for all symbols
lot_size = get_lot_size_for_symbols(symbol_list, fyers_initials)
print('Lot sizes for symbols:', lot_size)

# ==================== TRADE LOGGING FUNCTIONS ====================
def log_trade_to_csv(symbol, action, quantity, price, timestamp, reason="", target_level="", pnl=0, targets=None):
    """Log trade information to CSV file with target values"""
    try:
        csv_file = 'trades.csv'
        file_exists = os.path.isfile(csv_file)
        
        # CSV headers with target values and hit timestamps
        headers = ['Timestamp', 'Symbol', 'Action', 'Quantity', 'Price', 'Total_Value', 'Reason', 'Target_Level', 'PnL',
                   'T1_Value', 'T2_Value', 'T3_Value', 'T4_Value', 'T5_Value', 'T6_Value', 'Stop_Loss',
                   'T1_Hit', 'T2_Hit', 'T3_Hit', 'T4_Hit', 'T5_Hit', 'T6_Hit', 'SL_Hit']
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if file doesn't exist
            if not file_exists:
                writer.writerow(headers)
            
            # Calculate total value
            total_value = quantity * lot_size.get(symbol, 1) * price if price != 'N/A' else 0
            
            # Target values
            t1_val = round(targets.get('t1', 0), 2) if targets else ''
            t2_val = round(targets.get('t2', 0), 2) if targets else ''
            t3_val = round(targets.get('t3', 0), 2) if targets else ''
            t4_val = round(targets.get('t4', 0), 2) if targets else ''
            t5_val = round(targets.get('t5', 0), 2) if targets else ''
            t6_val = round(targets.get('t6', 0), 2) if targets else ''
            sl_val = round(targets.get('stop_loss', 0), 2) if targets and targets.get('stop_loss') else ''
            
            # Write trade data
            writer.writerow([
                timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'N/A',
                symbol,
                action,
                quantity * lot_size.get(symbol, 1),  # Total quantity including lot size
                price,
                total_value,
                reason,
                target_level,
                pnl,
                t1_val, t2_val, t3_val, t4_val, t5_val, t6_val, sl_val,
                '', '', '', '', '', '', ''  # Hit timestamps - empty initially
            ])
            
        logging.info(f'Trade logged to CSV: {symbol} {action} {quantity} @ {price} - {reason}')
        
    except Exception as e:
        logging.error(f'Error logging trade to CSV: {e}')

def update_csv_target_hit(symbol, target_name, hit_price, hit_timestamp):
    """Update the last BUY row for a symbol in CSV to mark a target as hit"""
    try:
        csv_file = 'trades.csv'
        if not os.path.isfile(csv_file):
            logging.warning('trades.csv not found - cannot update target hit')
            return
        
        # Read all rows
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) < 2:
            return
        
        headers = rows[0]
        
        # Map target names to hit column names
        target_hit_map = {
            'T1': 'T1_Hit', 'T2': 'T2_Hit', 'T3': 'T3_Hit',
            'T4': 'T4_Hit', 'T5': 'T5_Hit', 'T6': 'T6_Hit',
            'STOP_LOSS': 'SL_Hit'
        }
        
        hit_col_name = target_hit_map.get(target_name)
        if not hit_col_name or hit_col_name not in headers:
            logging.warning(f'Target column {target_name} not found in CSV headers')
            return
        
        hit_col_idx = headers.index(hit_col_name)
        
        # Find the last BUY row for this symbol
        buy_row_idx = None
        for i in range(len(rows) - 1, 0, -1):
            if len(rows[i]) > 2 and rows[i][1] == symbol and rows[i][2] == 'BUY':
                buy_row_idx = i
                break
        
        if buy_row_idx is None:
            logging.warning(f'No BUY row found for {symbol} in CSV')
            return
        
        # Extend row if needed
        while len(rows[buy_row_idx]) <= hit_col_idx:
            rows[buy_row_idx].append('')
        
        # Update the hit timestamp and price
        hit_time_str = hit_timestamp.strftime('%Y-%m-%d %H:%M:%S') if hit_timestamp else 'N/A'
        rows[buy_row_idx][hit_col_idx] = f'{hit_time_str} @ {hit_price}'
        
        # Write back
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        logging.info(f'CSV updated: {symbol} {target_name} hit at {hit_price} ({hit_time_str})')
        
    except Exception as e:
        logging.error(f'Error updating CSV target hit: {e}')

# ==================== TECHNICAL INDICATORS ====================
def rma(series, length):
    """Rolling Mean Average (EMA with alpha=1/length)"""
    return series.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def rsi_ma(close, rsi_length=14, ma_length=14, scalar=100, drift=1, offset=0, ma_type='sma', **kwargs):
    """
    Calculate the moving average of an RSI series.
    rsi_series: RSI values as a Pandas Series
    ma_length: Moving average period
    ma_type: 'ema' (default) or 'sma'
    offset: shift the result if needed
    """
    if ma_type == 'ema':
        ma = close.ewm(span=ma_length, min_periods=ma_length, adjust=False).mean()
    else:
        ma = close.rolling(window=ma_length, min_periods=ma_length).mean()
    if offset:
        ma = ma.shift(offset)
    ma.name = f"RSI_MA_{ma_length}"
    return ma.round(2)

def atr(df, length=14):
    """Calculate Average True Range (ATR)"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=length, min_periods=length).mean()
    return atr

def rsi(close, length=14, scalar=100, drift=1, offset=0, **kwargs):
    """Relative Strength Index (RSI) without pandas_ta or talib"""
    import pandas as pd
    # Ensure input is a Series
    close = close if isinstance(close, pd.Series) else pd.Series(close)
    length = int(length) if length and length > 0 else 14
    scalar = float(scalar) if scalar else 100
    drift = int(drift) if drift else 1
    offset = int(offset) if offset else 0

    # Calculate differences
    diff = close.diff(drift)
    positive = diff.clip(lower=0)
    negative = -diff.clip(upper=0)

    # Calculate rolling mean average (rma)
    positive_avg = rma(positive, length)
    negative_avg = rma(negative, length)

    # Calculate RSI
    rsi = scalar * positive_avg / (positive_avg + negative_avg)
    rsi.name = f"RSI_{length}"
    rsi.category = "momentum"

    # Offset
    if offset != 0:
        rsi = rsi.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rsi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rsi = rsi.ffill() if kwargs["fill_method"] == "ffill" else rsi.bfill()

    return rsi.round(2)


# ==================== TRADING CLASSES AND LOGIC ====================
# Global dataframes for strategy
global symbol_dataframes, index_15

# Initialize dataframes dictionary for each symbol
symbol_dataframes = {symbol: {'df_5': None, 'df_15': None, 'df_60': None} for symbol in symbol_list}
index_5 = None
index_15 = None
index_60 = None

class TradeInfo:
    """Manage trading information for each symbol"""
    def __init__(self):
        self.trade_flag = 0
        self.buy_price = 0
        self.quantity = 0
        self.pnl = 0
        self.condition = False
        self.wait_until = None
        self.wait_until_flag = False
        self.high = 0
        self.low = 0
        self.range = 0
        self.targets = {}
        self.crossover_active = False
        self.trade_taken_on_crossover = False
        self.order_time = None
        self.t1_executed = False
        self.t2_executed = False
        self.t3_executed = False
        self.t4_executed = False
        self.t5_executed = False
        self.t6_executed = False
        self.t1_trailing_active = False
        self.t1_trailing_start_time = None
    
    def reset(self, keep_crossover_state=False):
        """Reset trade info, optionally keeping crossover state"""
        self.trade_flag = 0
        self.buy_price = 0
        self.quantity = 0
        self.pnl = 0
        self.condition = False
        self.wait_until = None
        self.wait_until_flag = False
        self.high = 0
        self.low = 0
        self.range = 0
        self.targets = {}
        if not keep_crossover_state:
            self.crossover_active = False
        self.trade_taken_on_crossover = True  # Prevent immediate re-entry
        self.order_time = None
        self.t1_executed = False
        self.t2_executed = False
        self.t3_executed = False
        self.t4_executed = False
        self.t5_executed = False
        self.t6_executed = False
        self.t1_trailing_active = False
        self.t1_trailing_start_time = None
    
    def to_dict(self):
        """Convert to dictionary for pickle serialization"""
        return {
            'trade_flag': self.trade_flag,
            'buy_price': self.buy_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'condition': self.condition,
            'wait_until': self.wait_until,
            'wait_until_flag': self.wait_until_flag,
            'high': self.high,
            'low': self.low,
            'range': self.range,
            'targets': self.targets,
            'crossover_active': self.crossover_active,
            'trade_taken_on_crossover': self.trade_taken_on_crossover,
            'order_time': self.order_time,
            't1_executed': self.t1_executed,
            't2_executed': self.t2_executed,
            't3_executed': self.t3_executed,
            't4_executed': self.t4_executed,
            't5_executed': self.t5_executed,
            't6_executed': self.t6_executed,
            't1_trailing_active': self.t1_trailing_active,
            't1_trailing_start_time': self.t1_trailing_start_time
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create TradeInfo from dictionary (for loading from pickle)"""
        trade_info = cls()
        trade_info.trade_flag = data.get('trade_flag', 0)
        trade_info.buy_price = data.get('buy_price', 0)
        trade_info.quantity = data.get('quantity', 0)
        trade_info.pnl = data.get('pnl', 0)
        trade_info.condition = data.get('condition', False)
        trade_info.wait_until = data.get('wait_until', None)
        trade_info.wait_until_flag = data.get('wait_until_flag', False)
        trade_info.high = data.get('high', 0)
        trade_info.low = data.get('low', 0)
        trade_info.range = data.get('range', 0)
        trade_info.targets = data.get('targets', {})
        trade_info.crossover_active = data.get('crossover_active', False)
        trade_info.trade_taken_on_crossover = data.get('trade_taken_on_crossover', False)
        trade_info.order_time = data.get('order_time', None)
        trade_info.t1_executed = data.get('t1_executed', False)
        trade_info.t2_executed = data.get('t2_executed', False)
        trade_info.t3_executed = data.get('t3_executed', False)
        trade_info.t4_executed = data.get('t4_executed', False)
        trade_info.t5_executed = data.get('t5_executed', False)
        trade_info.t6_executed = data.get('t6_executed', False)
        trade_info.t1_trailing_active = data.get('t1_trailing_active', False)
        trade_info.t1_trailing_start_time = data.get('t1_trailing_start_time', None)
        return trade_info

# Initialize trade info for all symbols
try:
    loaded_data = load(account_type)
    real_info = {}
    
    # Convert loaded dictionary to TradeInfo objects
    for symbol in symbol_list:
        if symbol in loaded_data:
            # Load from existing data (handles both dict and TradeInfo)
            if isinstance(loaded_data[symbol], dict):
                real_info[symbol] = TradeInfo.from_dict(loaded_data[symbol])
                logging.info(f"Loaded existing trade info for {symbol}")
            else:
                real_info[symbol] = loaded_data[symbol]
                logging.info(f"Loaded existing TradeInfo object for {symbol}")
        else:
            # Initialize new TradeInfo if symbol not found
            logging.info(f"No existing data found for {symbol} - initializing new TradeInfo")
            real_info[symbol] = TradeInfo()

except Exception:
    logging.info("No saved data found or error loading - initializing new TradeInfo for all symbols")
    real_info = {symbol: TradeInfo() for symbol in symbol_list}

def check_active_position(ticker):
    """Check if there's an active position for the given ticker"""
    try:
        positions = fyers.positions()
        for pos in positions.get('netPositions', []):
            if pos.get('symbol') == f"{fyers_initials}{ticker}" and pos.get('quantity', 0) != 0:
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking active position: {e}")
        return False

def check_active_order(ticker):
    """Check if there's an active order for the given ticker"""
    try:
        orders = fyers.orderbook()
        for order in orders.get('orderBook', []):
            if (order.get('symbol') == f"{fyers_initials}{ticker}" and 
                order.get('status') in [1, 2, 4, 5, 6]):  # 1=Pending, 2=Placed, 4=Transit, 5=Partially Filled, 6=Pending Modification
                # logging.info(f"Active order found: {order.get('id')} - Status: {order.get('status')}")
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking active order: {e}")
        return False

def target_helper(range, buy_price, take_type, v,atr_value=None):
    """Calculate target price based on take type"""
    if take_type == 'FIXED':
        return buy_price + v
    elif take_type == 'PERCENTAGE':
        return buy_price + (buy_price * v / 100)
    elif take_type == 'RANGE':
        return buy_price + (range * v)
    elif take_type == 'ATR':
        if atr_value is None:
            raise ValueError("ATR value must be provided for ATR take type")
        return buy_price + (abs(atr_value) * v)
    else:
        raise ValueError("Invalid take_type. Must be 'FIXED', 'PERCENTAGE', 'RANGE', or 'ATR'.")

def get_targets(range, buy_price,atr):
    """Calculate all target levels"""
    if T1 == 'RANGE':
        if range < threshold:
            t1 = buy_price + (range * v1)
        elif range >= threshold:
            t1 = buy_price + 5
    else:
        t1 = target_helper(range, buy_price, T1, v1,atr)

    if T2 == 'RANGE':
        if range < threshold:
            t2 = buy_price + (range * v2)
        elif range >= threshold:
            t2 = buy_price + (0.5 * range)
    else:
        t2 = target_helper(range, buy_price, T2, v2,atr)
    
    t3 = target_helper(range, buy_price, T3, v3,atr)
    t4 = target_helper(range, buy_price, T4, v4,atr)
    t5 = target_helper(range, buy_price, T5, v5,atr)
    t6 = target_helper(range, buy_price, T6, v6,atr)
    return t1, t2, t3, t4, t5, t6

def calculate_margin(ticker, quantity, side=1, product_type="INTRADAY", limit_price=0.0, stop_loss=0.0):
    """Calculate margin required for a trade using Fyers span_margin API.
    
    Args:
        ticker: Symbol name (without exchange prefix)
        quantity: Total quantity (lots * lot_size)
        side: 1 for BUY, -1 for SELL
        product_type: 'INTRADAY' or 'CNC' or 'MARGIN'
        limit_price: Limit price (0.0 for market order)
        stop_loss: Stop loss price (0.0 for no SL)
    
    Returns:
        dict with keys:
            'total_margin': Total margin required
            'available_margin': Available margin in account
            'margin_ok': True if sufficient margin available
            'details': Full API response
        Returns None on error.
    """
    try:
        payload = {
            "data": [{
                "symbol": f"{fyers_initials}{ticker}",
                "qty": int(quantity),
                "side": side,
                "type": 2,  # Market order
                "productType": product_type,
                "limitPrice": limit_price,
                "stopLoss": stop_loss
            }]
        }
        
        headers = {
            "Authorization": f"{client_id}:{access_token_fyers}",
            "Content-Type": "application/json"
        }
        
        url = "https://api.fyers.in/api/v2/span_margin"
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        response = resp.json()
        logging.info(f'{ticker}: Margin API response: {response}')
        
        if response.get('s') == 'ok' or response.get('code') == 200:
            # Extract margin details from response
            resp_data = response.get('data', {})
            total_margin = resp_data.get('total', 0)  # 'total' = span + exposure margin
            
            # Get available funds
            available_cash_response = fyers.funds()
            available_margin = available_cash_response.get('fund_limit')[-1].get('equityAmount', 0)
            
            margin_ok = available_margin >= total_margin
            
            result = {
                'total_margin': total_margin,
                'available_margin': available_margin,
                'margin_ok': margin_ok,
                'details': response
            }
            
            logging.info(f'{ticker}: Margin required={total_margin}, Available={available_margin}, Sufficient={margin_ok}')
            return result
        else:
            logging.error(f'{ticker}: Margin API error response: {response}')
            return None
            
    except Exception as e:
        logging.error(f'{ticker}: Error calculating margin: {e}')
        return None

def get_current_quantity(ticker, ticker_price):
    """Calculate the quantity to buy based on capital, lot size, and margin requirements"""
    try:
        lot_size1 = lot_size.get(ticker)
        if lot_size1 is None:
            raise ValueError(f"Lot size not found for ticker: {ticker}")
        
        # In PAPER mode, use specified capital directly without querying live account
        if account_type == 'PAPER':
            if capital > 0:
                trading_capital = capital
            else:
                trading_capital = 100000  # Default paper trading capital
            print('Paper trading capital:', trading_capital)
        else:
            available_cash_response = fyers.funds()
            current_available = available_cash_response.get('fund_limit')[-1].get('equityAmount')
            print('available cash:', current_available)
            # Use specified capital if non-zero, otherwise use available cash
            if capital > 0:
                trading_capital = capital
            else:
                trading_capital = current_available

        if len(symbol_list) > 1:
            capital_per_symbol = trading_capital / len(symbol_list)
        else:
            capital_per_symbol = trading_capital
        quantity = int(capital_per_symbol // (ticker_price * lot_size1))
        # Ensure quantity is always even
        if quantity % 2 != 0:
            quantity -= 1
        
        if quantity < 2:
            logging.info(f'{ticker}: Capital-based quantity={quantity}, too low')
            return quantity
        
        # Verify margin requirement only for LIVE trading
        if account_type == 'LIVE':
            margin_result = calculate_margin(ticker, quantity * lot_size1, side=1)
            if margin_result is not None:
                while not margin_result['margin_ok'] and quantity >= 2:
                    logging.warning(f'{ticker}: Margin insufficient for qty={quantity} (need={margin_result["total_margin"]}, have={margin_result["available_margin"]}). Reducing quantity.')
                    quantity -= 2  # Reduce by 2 to keep even
                    if quantity < 2:
                        break
                    margin_result = calculate_margin(ticker, quantity * lot_size1, side=1)
                    if margin_result is None:
                        break
                
                if margin_result and margin_result['margin_ok']:
                    logging.info(f'{ticker}: Margin verified OK for qty={quantity} (margin={margin_result["total_margin"]}, available={margin_result["available_margin"]})')
                elif quantity < 2:
                    logging.warning(f'{ticker}: Cannot find viable quantity within margin limits')
                    return 0
            else:
                logging.warning(f'{ticker}: Margin check failed (API error) - proceeding with capital-based quantity={quantity}')
        else:
            logging.info(f'{ticker}: PAPER mode - skipping margin check, using full capital-based quantity={quantity}')
        
        logging.info(f'{ticker}: Capital={trading_capital}, Price={ticker_price}, LotSize={lot_size1}, Quantity={quantity}')
        return quantity
    except Exception as e:
        logging.error(f"Error calculating quantity for {ticker}: {e}")
        return 0

def sell_partial_quantity(ticker, quantity_to_sell, exit_all=False, reason="Partial exit", target_level=""):
    """Sell a partial quantity of the position for the given ticker"""
    try:
        ticker_price = final_data.get(ticker, {}).get('ltp', 'N/A')
        
        if exit_all:
            # Calculate PnL for full exit
            sell_qty = real_info[ticker].quantity if ticker in real_info else 0
            buy_price = real_info[ticker].buy_price if ticker in real_info else 0
            pnl = round((ticker_price - buy_price) * sell_qty * lot_size.get(ticker, 1), 2) if ticker_price != 'N/A' and buy_price > 0 else 0
            
            # Log full exit to CSV
            log_trade_to_csv(
                symbol=ticker,
                action='SELL_ALL',
                quantity=sell_qty,
                price=ticker_price,
                timestamp=dt.now(time_zone),
                reason=reason,
                target_level=target_level,
                pnl=pnl
            )
            
            # Only place actual order if account_type is LIVE
            if account_type == 'LIVE':
                p=fyers.exit_positions({'id': fyers_initials + ticker + "-INTRADAY"})
                print(p)
                logging.info(f'LIVE Exit position response for {ticker}: {p}')
            else:
                logging.info(f'{ticker}: PAPER TRADE - SELL ALL position simulated (qty: {real_info[ticker].quantity if ticker in real_info else 0}, price: {ticker_price})')
            return

        lot_size1=lot_size.get(ticker)
        quantity_to_sell = int(quantity_to_sell)
        
        # Calculate PnL for partial exit
        buy_price = real_info[ticker].buy_price if ticker in real_info else 0
        pnl = round((ticker_price - buy_price) * quantity_to_sell * lot_size.get(ticker, 1), 2) if ticker_price != 'N/A' and buy_price > 0 else 0
        
        # Log partial sell to CSV
        log_trade_to_csv(
            symbol=ticker,
            action='SELL_PARTIAL',
            quantity=quantity_to_sell,
            price=ticker_price,
            timestamp=dt.now(time_zone),
            reason=reason,
            target_level=target_level,
            pnl=pnl
        )
        
        # Only place actual order if account_type is LIVE
        if account_type == 'LIVE':
            data = {
                "symbol": f"{fyers_initials}{ticker}",
                "qty": quantity_to_sell*lot_size1,
                "type": 2,  # Market order
                "side": -1,  # Sell
                "productType": "INTRADAY",
                "limitPrice": 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False,
                "orderTag": "algoexit",
                "isSliceOrder": False
            }
            response = fyers.place_order(data=data)
            print(response)
            logging.info(f'LIVE Partial sell order response: {response}')
        else:
            logging.info(f'{ticker}: PAPER TRADE - SELL PARTIAL position simulated (qty: {quantity_to_sell * lot_size1}, price: {ticker_price})')
        
    except Exception as e:
        logging.error(f'Error placing partial sell order: {e}')

def take_position(ticker, quantity_position, targets=None):
        try:
                lot_size1=lot_size.get(ticker)
                ticker_price = final_data.get(ticker, {}).get('ltp', 'N/A')
                
                # Always log the trade to CSV
                log_trade_to_csv(
                    symbol=ticker,
                    action='BUY',
                    quantity=quantity_position,
                    price=ticker_price,
                    timestamp=dt.now(time_zone),
                    reason='Entry on RSI crossover and breakout',
                    targets=targets
                )
                
                # Only place actual order if account_type is LIVE
                if account_type == 'LIVE':
                    data = {
                                    "symbol":fyers_initials+ticker,
                                    "qty":lot_size1*quantity_position,
                                    "type":2,
                                    "side":1,
                                    "productType":"INTRADAY",
                                    "limitPrice":0,
                                    "stopPrice":0,
                                    "validity":"DAY",
                                    "disclosedQty":0,
                                    "offlineOrder":False,
                                    "orderTag":f"{int(33.99)}d{int(44.99)}",
                                    "isSliceOrder":False
                            }

                    response = fyers.place_order(data=data)
                    print(response)
                    logging.info(f'{ticker}: LIVE Order response: {response}')
                else:
                    logging.info(f'{ticker}: PAPER TRADE - BUY order simulated (qty: {quantity_position * lot_size1}, price: {ticker_price})')
                
        except Exception as e:
                print(e)
                logging.error(f'Error placing order for {ticker}: {e}')


# ==================== MAIN STRATEGY LOGIC ====================
trade_flag = 0
first = 1
last_high = 0

def main_strategy_second():
    """Main strategy logic that processes trading signals"""
    global first, last_high, real_info
    global symbol_dataframes, index_15
    current_time = dt.now(time_zone)
        
    if (current_time.second == 1 and current_time.minute % 5 == 0) or first == 1:
        first = 0
        # Fetch OHLC data for all symbols
        for symbol in symbol_list:
            symbol_fyers = fyers_initials + symbol
            logging.info(f"\n--- Fetching data for {symbol} ---")
            df_5 = fetchOHLC(symbol_fyers, candle_1, 10)
            df_5['rsi'] = rsi(df_5['close'], length=rsi_1)
            df_5['rsi_smooth'] = rsi_ma(df_5['rsi'])
            df_5['rsi_flag'] = (df_5['rsi'] > df_5['rsi_smooth']).astype(bool)
            df_5['atr'] = atr(df_5, length=atr_length)
            df_15 = fetchOHLC(symbol_fyers, candle_2, 10)
            df_15['rsi'] = rsi(df_15['close'], length=rsi_2)
            df_15['rsi_smooth'] = rsi_ma(df_15['rsi'])
            df_15['rsi_flag'] = (df_15['rsi'] > df_15['rsi_smooth']).astype(bool)
            df_60 = fetchOHLC(symbol_fyers, candle_3, 10)
            df_60['rsi'] = rsi(df_60['close'], length=rsi_3)
            df_60['rsi_smooth'] = rsi_ma(df_60['rsi'])
            df_60['rsi_flag'] = (df_60['rsi'] > df_60['rsi_smooth']).astype(bool)
            
            # Store dataframes per symbol
            symbol_dataframes[symbol] = {'df_5': df_5, 'df_15': df_15, 'df_60': df_60}
            
            print(f"\n{symbol} - df_5 tail:\n{df_5.tail()}")
            print(f"{symbol} - df_15 tail:\n{df_15.tail()}")
            print(f"{symbol} - df_60 tail:\n{df_60.tail()}")

        index_15 = fetchOHLC(fyers_underlying_index, candle_2, 10)
        index_15['rsi'] = rsi(index_15['close'], length=rsi_2)
        index_15['rsi_smooth'] = rsi_ma(index_15['rsi'])
        index_15['rsi_flag'] = (index_15['rsi'] > index_15['rsi_smooth']).astype(bool)


    # Process trading logic for each symbol independently
    for ticker in symbol_list:
        ticker_price = final_data.get(ticker, {}).get('ltp', 'N/A')
        print(dt.now(time_zone), ticker, ticker_price)
        trade_flag = real_info[ticker].trade_flag  # local per-symbol flag

        # Get symbol-specific dataframes
        df_5 = symbol_dataframes[ticker]['df_5']
        df_15 = symbol_dataframes[ticker]['df_15']
        df_60 = symbol_dataframes[ticker]['df_60']

        # Validate dataframes are initialized and have enough data
        if df_5 is None or len(df_5) < 2:
            logging.info(f"{ticker}: df_5 not initialized or insufficient data")
            continue
        if df_15 is None or len(df_15) < 2:
            logging.info(f"{ticker}: df_15 not initialized or insufficient data")
            continue
        if df_60 is None or len(df_60) < 2:
            logging.info(f"{ticker}: df_60 not initialized or insufficient data")
            continue
        if index_15 is None or len(index_15) < 2:
            logging.info(f"{ticker}: index_15 not initialized or insufficient data")
            continue

        # Detect RSI crossover (fresh crossover = RSI crosses above RSI_MA)
        rsi_above_ma_prev = df_5['rsi'].iloc[-3] > df_5['rsi_smooth'].iloc[-3] if len(df_5) >= 3 else False
        rsi_above_ma_current = df_5['rsi'].iloc[-2] > df_5['rsi_smooth'].iloc[-2] if len(df_5) >= 2 else False
        
        # Fresh crossover detected: RSI crosses from below to above MA
        if not rsi_above_ma_prev and rsi_above_ma_current:
            # logging.info(f'{ticker}: Fresh RSI crossover detected!')
            real_info[ticker].crossover_active = True
            real_info[ticker].trade_taken_on_crossover = False
        
        # Crossover ended: RSI crosses back below MA
        elif rsi_above_ma_prev and not rsi_above_ma_current:
            # logging.info(f'{ticker}: RSI crossed back below MA - resetting crossover flags')
            real_info[ticker].crossover_active = False
            real_info[ticker].trade_taken_on_crossover = False
        
        # Detect ongoing crossover (for startup/restart scenarios)
        # If crossover_active is False but RSI is currently above MA, activate it
        elif not real_info[ticker].crossover_active and rsi_above_ma_current:
            # logging.info(f'{ticker}: Ongoing RSI crossover detected (startup/restart) - activating crossover flag')
            print(f"{ticker}: Ongoing RSI crossover detected (startup/restart) - activating crossover flag")
            real_info[ticker].crossover_active = True
            real_info[ticker].trade_taken_on_crossover = False

        # Entry condition - only trade on fresh crossover
        if (trade_flag == 0 and 
            real_info[ticker].crossover_active and 
            not real_info[ticker].trade_taken_on_crossover and 
            df_5['rsi_flag'].iloc[-2] and 
            ticker_price != 'N/A'):
            
            current_high = df_5['high'].iloc[-2]
            current_low = df_5['low'].iloc[-2]
            print(f'{ticker}: rsi is above ma, current_high={current_high}, ticker_price={ticker_price}')
            
            if ticker_price >= current_high and ((df_15['rsi_flag'].iloc[-1] == True and min_15_condition) or 
                                               (df_60['rsi_flag'].iloc[-1] == True and min_60_condition) or 
                                               (index_15['rsi_flag'].iloc[-1] == True and index_15_condition)):
                logging.info(f'{ticker}: Placing buy order - ticker_price: {ticker_price} >= current_high: {current_high}')

                quantity_position=get_current_quantity(ticker,ticker_price)
                if quantity_position < 2:
                    logging.warning(f'{ticker}: Calculated quantity is {quantity_position} (minimum 2 required) - insufficient capital. Skipping order.')
                    continue

                # Compute targets before placing order so they can be logged in CSV
                candle_range = current_high - current_low
                t1, t2, t3, t4, t5, t6 = get_targets(candle_range, ticker_price, df_5['atr'].iloc[-1])
                entry_targets = {'t1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5, 't6': t6, 'stop_loss': current_low}

                # # Final margin check before placing order
                # margin_check = calculate_margin(ticker, quantity_position * lot_size.get(ticker, 1), side=1)
                # if margin_check is not None and not margin_check['margin_ok']:
                #     logging.warning(f'{ticker}: Final margin check FAILED - Required: {margin_check["total_margin"]}, Available: {margin_check["available_margin"]}. Skipping order.')
                #     continue
                # elif margin_check is not None:
                #     logging.info(f'{ticker}: Final margin check PASSED - Required: {margin_check["total_margin"]}, Available: {margin_check["available_margin"]}')

                take_position(ticker, quantity_position, targets=entry_targets)


                real_info[ticker].trade_flag = 1
                trade_flag = 1
                real_info[ticker].quantity = quantity_position
                real_info[ticker].buy_price = ticker_price
                real_info[ticker].high = current_high
                real_info[ticker].low = current_low
                real_info[ticker].range = candle_range
                real_info[ticker].trade_taken_on_crossover = True  # Mark that trade was taken on this crossover
                real_info[ticker].order_time = dt.now(time_zone)  # Track when order was placed
                real_info[ticker].targets = entry_targets
                logging.info(f'{ticker}: Trade taken on crossover - Targets: t1={t1}, t2={t2}, t3={t3}, t4={t4}, t5={t5}, t6={t6},atr={df_5["atr"].iloc[-1]}')
                logging.info(f'{ticker}: Crossover flag locked until RSI crosses back below MA')
                logging.info(f"{ticker} - Buy Price: {real_info[ticker].buy_price}, High: {real_info[ticker].high}, Low: {real_info[ticker].low}, Range: {real_info[ticker].range},quantity: {real_info[ticker].quantity}")

        # Already open position
        elif trade_flag == 1 and ticker_price != 'N/A':
            print(f"{ticker}: Trade already placed - t1: {real_info[ticker].targets.get('t1')}, t2: {real_info[ticker].targets.get('t2')}, t3: {real_info[ticker].targets.get('t3')}, current price: {ticker_price}")
            # For LIVE trading, check actual positions/orders. For PAPER trading, simulate position tracking
            if account_type == 'LIVE':
                if trade_flag == 1 and (check_active_position(ticker) or check_active_order(ticker)):
                    print(f"{ticker}: Active position or order detected - current price: {ticker_price}")
                else:
                    # Allow 30 seconds buffer after order placement before resetting trade_flag
                    order_time = real_info[ticker].order_time
                    if order_time and (dt.now(time_zone) - order_time).total_seconds() < 10:
                        logging.info(f"{ticker}: Waiting for order to be filled... ({(dt.now(time_zone) - order_time).total_seconds():.1f}s elapsed)")
                        continue
                    else:
                        logging.info(f"{ticker}: No active position detected after 10s - resetting trade flag")
                        real_info[ticker].trade_flag = 0
                        real_info[ticker].reset(keep_crossover_state=True)
                        continue
            else:
                # Paper trading mode - simulate position holding
                print(f"{ticker}: PAPER TRADE position active - current price: {ticker_price}")

            t3_target = real_info[ticker].targets.get('t3')
            t2_target = real_info[ticker].targets.get('t2')
            t1_target = real_info[ticker].targets.get('t1')
            t4_target = real_info[ticker].targets.get('t4')
            t5_target = real_info[ticker].targets.get('t5')
            t6_target = real_info[ticker].targets.get('t6')

            if t6_target is not None and ticker_price > t6_target and not real_info[ticker].t6_executed:
                logging.info(f'{ticker}: Target t6 reached at {ticker_price} - closing entire position')
                sell_partial_quantity(ticker, real_info[ticker].quantity, exit_all=True, reason="Target T6 reached - full exit", target_level="T6")
                update_csv_target_hit(ticker, 'T6', ticker_price, dt.now(time_zone))
                real_info[ticker].trade_flag = 0
                real_info[ticker].reset(keep_crossover_state=True)
                logging.info(f'{ticker}: Position fully closed at T6')
                real_info[ticker].t6_executed = True
                continue  # Skip remaining target checks — position is fully closed

            if t5_target is not None and ticker_price > t5_target and not real_info[ticker].t5_executed:
                logging.info(f'{ticker}: Target t5 reached at {ticker_price}')
                if v5_pos > 0 and real_info[ticker].quantity > 0:
                    partial_quantity = max(1, int(real_info[ticker].quantity * v5_pos))
                    if partial_quantity > real_info[ticker].quantity:
                        partial_quantity = real_info[ticker].quantity
                    sell_partial_quantity(ticker, partial_quantity, reason="Target T5 reached", target_level="T5")
                    real_info[ticker].quantity -= partial_quantity
                    logging.info(f'{ticker}: Remaining quantity after T5: {real_info[ticker].quantity}')
                else:
                    logging.info(f'{ticker}: T5 target hit at {ticker_price} (v5_pos=0 or qty=0, logging only)')
                update_csv_target_hit(ticker, 'T5', ticker_price, dt.now(time_zone))
                real_info[ticker].t5_executed = True

            if t4_target is not None and ticker_price > t4_target and not real_info[ticker].t4_executed:
                logging.info(f'{ticker}: Target t4 reached at {ticker_price}')
                if v4_pos > 0 and real_info[ticker].quantity > 0:
                    partial_quantity = max(1, int(real_info[ticker].quantity * v4_pos))
                    if partial_quantity > real_info[ticker].quantity:
                        partial_quantity = real_info[ticker].quantity
                    sell_partial_quantity(ticker, partial_quantity, reason="Target T4 reached", target_level="T4")
                    real_info[ticker].quantity -= partial_quantity
                    logging.info(f'{ticker}: Remaining quantity after T4: {real_info[ticker].quantity}')
                else:
                    logging.info(f'{ticker}: T4 target hit at {ticker_price} (v4_pos=0 or qty=0, logging only)')
                update_csv_target_hit(ticker, 'T4', ticker_price, dt.now(time_zone))
                real_info[ticker].t4_executed = True

            if t3_target is not None and ticker_price > t3_target and not real_info[ticker].t3_executed:
                logging.info(f'{ticker}: Target t3 reached - consider partial profit booking at {ticker_price}')
                if real_info[ticker].quantity > 0:
                    partial_quantity = max(1, int(real_info[ticker].quantity * v3_pos))  # At least 1 lot
                    if partial_quantity > real_info[ticker].quantity:
                        partial_quantity = real_info[ticker].quantity
                    sell_partial_quantity(ticker, partial_quantity, reason=f"Target T3 reached", target_level="T3")
                    real_info[ticker].quantity -= partial_quantity  # Track remaining quantity
                    logging.info(f'{ticker}: Remaining quantity after T3: {real_info[ticker].quantity}')
                real_info[ticker].t3_executed = True
                update_csv_target_hit(ticker, 'T3', ticker_price, dt.now(time_zone))

            if t2_target is not None and ticker_price > t2_target and not real_info[ticker].t2_executed:
                # Close position
                logging.info(f'{ticker}: Target t2 reached - closing partial at {ticker_price}')
                if real_info[ticker].quantity > 0:
                    partial_quantity = max(1, int(real_info[ticker].quantity * v2_pos))  # At least 1 lot
                    if partial_quantity > real_info[ticker].quantity:
                        partial_quantity = real_info[ticker].quantity
                    sell_partial_quantity(ticker, partial_quantity, reason=f"Target T2 reached", target_level="T2")
                    real_info[ticker].quantity -= partial_quantity  # Track remaining quantity
                    logging.info(f'{ticker}: Remaining quantity after T2: {real_info[ticker].quantity}')
                real_info[ticker].t2_executed = True
                update_csv_target_hit(ticker, 'T2', ticker_price, dt.now(time_zone))

            if t1_target is not None and ticker_price > t1_target and not real_info[ticker].t1_executed:
                logging.info(f'{ticker}: Target t1 reached at {ticker_price}')
                real_info[ticker].t1_executed = True
                update_csv_target_hit(ticker, 'T1', ticker_price, dt.now(time_zone))

            # T1 trailing stop: after T1 hit, if price drops below entry high, start wait_buffer timer
            # If price stays below entry high for wait_buffer seconds, close position
            # If price recovers above entry high, reset the timer
            if real_info[ticker].t1_executed and real_info[ticker].high > 0:
                if ticker_price < real_info[ticker].high:
                    if not real_info[ticker].t1_trailing_active:
                        # Price just dropped below entry high — start the timer
                        real_info[ticker].t1_trailing_active = True
                        real_info[ticker].t1_trailing_start_time = dt.now(time_zone)
                        logging.info(f"{ticker}: Price {ticker_price} dropped below entry high {real_info[ticker].high} after T1 — starting {wait_buffer}s trailing timer")
                    else:
                        # Timer already running — check if wait_buffer has elapsed
                        elapsed = (dt.now(time_zone) - real_info[ticker].t1_trailing_start_time).total_seconds()
                        logging.info(f"{ticker}: Price {ticker_price} still below entry high {real_info[ticker].high} — trailing timer {elapsed:.1f}s / {wait_buffer}s")
                        if elapsed >= wait_buffer:
                            logging.info(f"{ticker}: Price stayed below entry high for {wait_buffer}s — closing position")
                            try:
                                sell_partial_quantity(ticker, real_info[ticker].quantity, exit_all=True, reason=f"Price below entry high for {wait_buffer}s after T1", target_level="T1_TRAILING_STOP")
                                update_csv_target_hit(ticker, 'STOP_LOSS', ticker_price, dt.now(time_zone))
                                real_info[ticker].trade_flag = 0
                                real_info[ticker].reset(keep_crossover_state=True)
                                logging.info(f'{ticker}: Position closed — T1 trailing stop')
                            except Exception as e:
                                logging.error(f'{ticker}: Error closing position on T1 trailing stop: {e}')
                            continue  # Position fully closed, skip remaining checks
                else:
                    # Price is back above entry high — reset the trailing timer
                    if real_info[ticker].t1_trailing_active:
                        logging.info(f"{ticker}: Price {ticker_price} recovered above entry high {real_info[ticker].high} — resetting trailing timer")
                        real_info[ticker].t1_trailing_active = False
                        real_info[ticker].t1_trailing_start_time = None

            # Trailing stop: if price crossed above t2+ and falls back below entry candle high, close position
            any_upper_target_hit = any([
                real_info[ticker].t2_executed,
                real_info[ticker].t3_executed,
                real_info[ticker].t4_executed,
                real_info[ticker].t5_executed,
                real_info[ticker].t6_executed
            ])
            if any_upper_target_hit and ticker_price < real_info[ticker].high and real_info[ticker].high > 0:
                logging.info(f"{ticker}: Price {ticker_price} fell back below entry high {real_info[ticker].high} after hitting upper target — closing position")
                try:
                    sell_partial_quantity(ticker, real_info[ticker].quantity, exit_all=True, reason="Price fell below entry high after target hit", target_level="TRAILING_STOP")
                    update_csv_target_hit(ticker, 'STOP_LOSS', ticker_price, dt.now(time_zone))
                    real_info[ticker].trade_flag = 0
                    real_info[ticker].reset(keep_crossover_state=True)
                    logging.info(f'{ticker}: Position closed — trailing stop (fell below entry high)')
                except Exception as e:
                    logging.error(f'{ticker}: Error closing position on trailing stop: {e}')
                continue  # Position fully closed, skip remaining checks

            # Stop loss check — independent of target checks so it always runs
            if ticker_price < (real_info[ticker].low) and real_info[ticker].low > 0:
                # If wait_until is active, skip SL until buffer expires
                if real_info[ticker].wait_until_flag and real_info[ticker].wait_until and dt.now(time_zone) < real_info[ticker].wait_until:
                    logging.info(f"{ticker}: Price {ticker_price} below SL {real_info[ticker].low} but wait_until active until {real_info[ticker].wait_until} — holding")
                else:
                    logging.info(f"{ticker}: Price {ticker_price} went below low {real_info[ticker].low} - closing position")
                    try:
                        sell_partial_quantity(ticker, real_info[ticker].quantity, exit_all=True, reason="Stop Loss Hit", target_level="STOP_LOSS")
                        update_csv_target_hit(ticker, 'STOP_LOSS', ticker_price, dt.now(time_zone))
                        real_info[ticker].trade_flag = 0
                        real_info[ticker].reset(keep_crossover_state=True)
                        logging.info(f'{ticker}: position closed at stop loss')
                    except Exception as e:
                        logging.error(f'{ticker}: Error exiting position at stop loss: {e}')  
                        logging.error(f'{ticker}: position close failed at stop loss')

    store(real_info, account_type)












async def strategy_logic():
    """Strategy logic that runs every second"""
    while True:
        try:
            global final_data
            main_strategy_second()            
            await asyncio.sleep(1)  # Run every second
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error in strategy logic: {e}")


# ==================== SOCKET DATA MANAGEMENT ====================
class SocketData:
    """Async-safe data storage for both sockets"""
    def __init__(self):
        self.fyers_data = None
        self.kite_data = None
        self.fyers_timestamp = 0
        self.kite_timestamp = 0
        self.last_source = None  # Track which source was last used
        self.data_queue = None  # Will be initialized in async context
    
    def initialize(self):
        """Initialize asyncio queue (call in async context)"""
        if self.data_queue is None:
            self.data_queue = asyncio.Queue()
        
    async def update_fyers(self, data, timestamp):
        """Update Fyers data and queue for processing"""
        self.fyers_data = data
        self.fyers_timestamp = timestamp
        await self.data_queue.put(('fyers', data, timestamp))
    
    async def update_kite(self, data, timestamp):
        """Update Kite data and queue for processing"""
        self.kite_data = data
        self.kite_timestamp = timestamp
        await self.data_queue.put(('kite', data, timestamp))
    
    async def process_data_stream(self):
        """Continuously process incoming data from both sockets"""
        while True:
            try:
                # Wait for data from either socket
                source_name, data, timestamp = await self.data_queue.get()
                
                # Determine which socket has the latest data
                if secondary_data and self.fyers_timestamp > 0 and self.kite_timestamp > 0:
                    if self.fyers_timestamp >= self.kite_timestamp:
                        source = "FYERS (Primary)"
                        display_data = self.fyers_data
                        display_timestamp = self.fyers_timestamp
                    else:
                        source = "KITE (Secondary)"
                        display_data = self.kite_data
                        display_timestamp = self.kite_timestamp
                elif self.fyers_timestamp > 0:
                    source = "FYERS (Primary)"
                    display_data = self.fyers_data
                    display_timestamp = self.fyers_timestamp
                elif secondary_data and self.kite_timestamp > 0:
                    source = "KITE (Secondary)"
                    display_data = self.kite_data
                    display_timestamp = self.kite_timestamp
                else:
                    continue
                
                # Only print if source changed
                if self.last_source != source:
                    self.last_source = source
                
                # Display the latest data
                self._display_data(display_data, source, display_timestamp)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error processing data stream: {e}")
    
    def _display_data(self, data, source, timestamp):
        """Display formatted data and save to final_data"""
        global final_data
        
        # Convert timestamp to readable format
        if isinstance(timestamp, (int, float)):
            time_str = dt.from_timestamp(timestamp, tz=time_zone).strftime('%Y-%m-%d %H:%M:%S')
        else:
            time_str = str(timestamp)
        
        # Extract symbol from data to determine which symbol's data to update
        symbol_key = data.get('symbol', 'N/A')
        
        # Extract just the symbol name from Fyers format (e.g., "MCX:CRUDEOILM26FEBFUT" -> "CRUDEOILM26FEBFUT")
        if ':' in str(symbol_key):
            symbol_key = symbol_key.split(':')[-1]
        
        # Check if it's the index (handle NIFTY50-INDEX -> NIFTY50 mapping)
        is_index = False
        if '-INDEX' in str(symbol_key):
            base_symbol = symbol_key.replace('-INDEX', '')
            if base_symbol == index_name:
                is_index = True
                symbol_key = index_name  # Use index_name as key
        
        # Also check if symbol_key is directly the index_name
        if symbol_key == index_name:
            is_index = True
        
        # Only save if this symbol is in our symbol_list or is the index
        if symbol_key in symbol_list or is_index:
            # Save data to final_data
            final_data[symbol_key] = {
                'source': source,
                'timestamp': timestamp,
                'time_str': time_str,
                'ltp': data.get('ltp', 'N/A'),
                'volume': data.get('volume', 'N/A'),
                'buy_price': data.get('buy_price', 'N/A'),
                'buy_quantity': data.get('buy_quantity', 'N/A'),
                'sell_price': data.get('sell_price', 'N/A'),
                'sell_quantity': data.get('sell_quantity', 'N/A'),
                'change': data.get('change', 'N/A'),
                'open': data.get('open', 'N/A'),
                'high': data.get('high', 'N/A'),
                'low': data.get('low', 'N/A'),
                'close': data.get('close', 'N/A'),
                'symbol': symbol_key
            }

# Create shared data store
shared_data = SocketData()

# ==================== WEBSOCKET HANDLERS ====================
def start_fyers_socket(loop):
    """Start Fyers WebSocket connection (runs in executor thread)"""
    def onmessage(message):
        if isinstance(message, dict):
            # Extract exchange time (epoch timestamp)
            exchange_time = message.get('exch_feed_time', 0)
            
            # Prepare data in standardized format
            data = {
                'ltp': message.get('ltp', 'N/A'),
                'volume': message.get('vol_traded_today', 'N/A'),
                'buy_quantity': message.get('bid_size', 'N/A'),
                'sell_quantity': message.get('ask_size', 'N/A'),
                'buy_price': message.get('bid_price', 'N/A'),
                'sell_price': message.get('ask_price', 'N/A'),
                'change': message.get('ch', 'N/A'),
                'symbol': message.get('symbol', 'N/A'),
                'open': message.get('open_price', 'N/A'),
                'high': message.get('high_price', 'N/A'),
                'low': message.get('low_price', 'N/A'),
                'close': message.get('prev_close_price', 'N/A')
            }
            
            # Schedule update in the event loop
            asyncio.run_coroutine_threadsafe(
                shared_data.update_fyers(data, exchange_time), 
                loop
            )
    
    def onerror(message):
        logging.error(f"Fyers Error: {message}")
    
    def onclose(message):
        logging.info(f"Fyers Connection closed: {message}")
    
    def onopen():
        logging.info("Fyers Connection established")
        data_type = "SymbolUpdate"
        # Subscribe to all symbols in symbol_list
        fyers_symbols = [f"{fyers_initials}{s}" for s in symbol_list]
        # Add underlying index to subscription
        fyers_symbols.append(fyers_underlying_index)
        fyers_ws.subscribe(symbols=fyers_symbols, data_type=data_type)
        logging.info(f"Subscribed to {len(fyers_symbols)} symbols on Fyers: {fyers_symbols}")
    
    # Create Fyers WebSocket instance
    fyers_ws = data_ws.FyersDataSocket(
        access_token=f"{client_id}:{access_token_fyers}",
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=True,
        on_connect=onopen,
        on_close=onclose,
        on_error=onerror,
        on_message=onmessage
    )
    
    # Connect (blocking call in this thread)
    fyers_ws.connect()

def start_kite_socket(loop):
    """Start Kite WebSocket connection (runs in executor thread)"""
    if not secondary_data:
        logging.info("Kite socket disabled (secondary_data=False)")
        return
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=kite_key)
    kite.set_access_token(access_token_kite)
    
    # Get instrument tokens for all symbols
    def get_instrument_tokens(symbol_names, exchange='NFO'):
        try:
            instruments = kite.instruments(exchange)
            token_map = {}
            for symbol_name in symbol_names:
                for instrument in instruments:
                    if instrument['tradingsymbol'] == symbol_name:
                        token_map[instrument['instrument_token']] = symbol_name
                        logging.info(f"Found instrument: {instrument['tradingsymbol']} (Token: {instrument['instrument_token']})")
                        break
                else:
                    logging.error(f"Symbol {symbol_name} not found in {exchange}")
            return token_map
        except Exception as e:
            logging.error(f"Error fetching instruments: {e}")
            return {}
    
    token_symbol = get_instrument_tokens(symbol_list, exchange_kite)
    
    # Add underlying index token (always add this even if symbols not found)
    token_symbol[kite_underlying_index_token] = index_name
    
    if not token_symbol:
        logging.error("Cannot start Kite socket - no instrument tokens found")
        return
    
    instrument_tokens = list(token_symbol.keys())
    print("kite tokens", instrument_tokens)
    
    # Initialize KiteTicker
    kws = KiteTicker(kite_key, access_token_kite)

    def on_ticks(ws, ticks):
        for tick in ticks:
            # Extract exchange time
            exchange_time = tick.get('exchange_timestamp', None)
            
            # Convert to epoch timestamp if it's a datetime object (do this FIRST)
            if isinstance(exchange_time, datetime):
                exchange_time = exchange_time.timestamp()
            elif exchange_time is None:
                exchange_time = 0
            
            token = tick.get('instrument_token', None)
            name = token_symbol.get(token, 'N/A')

            if name == index_name:
                # Prepare index data in standardized format
                index_data = {
                    'ltp': tick.get('last_price', 'N/A'),
                    'change': tick.get('change', 'N/A'),
                    'symbol': index_name,
                    'open': tick['ohlc'].get('open', 'N/A'),
                    'high': tick['ohlc'].get('high', 'N/A'),
                    'low': tick['ohlc'].get('low', 'N/A'),
                    'close': tick['ohlc'].get('close', 'N/A')
                }
                
                asyncio.run_coroutine_threadsafe(
                    shared_data.update_kite(index_data, exchange_time),
                    loop
                )
            
            # Extract depth data
            buy_quantity = 'N/A'
            buy_price = 'N/A'
            sell_quantity = 'N/A'
            sell_price = 'N/A'
            
            if 'depth' in tick and tick['depth']:
                if tick['depth'].get('buy') and len(tick['depth']['buy']) > 0:
                    buy_quantity = tick['depth']['buy'][0].get('quantity', 'N/A')
                    buy_price = tick['depth']['buy'][0].get('price', 'N/A')
                if tick['depth'].get('sell') and len(tick['depth']['sell']) > 0:
                    sell_quantity = tick['depth']['sell'][0].get('quantity', 'N/A')
                    sell_price = tick['depth']['sell'][0].get('price', 'N/A')
            
            # Extract OHLC data
            open_price = 'N/A'
            high_price = 'N/A'
            low_price = 'N/A'
            close_price = 'N/A'
            
            if 'ohlc' in tick:
                open_price = tick['ohlc'].get('open', 'N/A')
                high_price = tick['ohlc'].get('high', 'N/A')
                low_price = tick['ohlc'].get('low', 'N/A')
                close_price = tick['ohlc'].get('close', 'N/A')
            
            # Prepare data in standardized format
            data = {
                'ltp': tick.get('last_price', 'N/A'),
                'volume': tick.get('volume_traded', 'N/A'),
                'buy_quantity': buy_quantity,
                'sell_quantity': sell_quantity,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'change': tick.get('change', 'N/A'),
                'symbol': token_symbol.get(tick['instrument_token'], 'N/A'),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            }
            
            # Schedule update in the event loop
            asyncio.run_coroutine_threadsafe(
                shared_data.update_kite(data, exchange_time),
                loop
            )
    
    def on_connect(ws, response):
        logging.info("Kite Connected to websocket!")
        ws.subscribe(instrument_tokens)
        ws.subscribe([kite_underlying_index_token])  # Ensure underlying index is subscribed
        ws.set_mode(ws.MODE_FULL, instrument_tokens)
        logging.info(f"Subscribed to {len(instrument_tokens)} symbols on Kite: {list(token_symbol.values())}")
    
    def on_close(ws, code, reason):
        logging.info(f"Kite Connection closed: {code} - {reason}")
    
    def on_error(ws, code, reason):
        logging.error(f"Kite Error: {code} - {reason}")
    
    def on_reconnect(ws, attempts_count):
        logging.info(f"Kite Reconnecting... Attempt {attempts_count}")
    
    def on_noreconnect(ws):
        logging.error("Kite Max reconnection attempts reached")
    
    # Assign callbacks
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.on_reconnect = on_reconnect
    kws.on_noreconnect = on_noreconnect
    
    # Connect (blocking call in this thread)
    kws.connect(threaded=False)


# ==================== MAIN EXECUTION ====================
async def main():
    """Main async function to run both sockets"""
    print("\n" + "="*60)
    print("🚀 DUAL SOCKET HANDLER - STARTED (Asyncio)")
    print("="*60)
    print(f"Symbols: {symbol_list}")
    print(f"Primary Source: Fyers")
    if secondary_data:
        print(f"Secondary Source: Kite")
        print(f"Strategy: Use whichever has latest exchange time")
    else:
        print(f"Secondary Source: Disabled")
        print(f"Strategy: Use Fyers only")
    print("="*60 + "\n")
    
    # Initialize shared data in async context
    shared_data.initialize()
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    
    # Create thread pool executor for blocking websocket operations
    max_workers = 2 if secondary_data else 1
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    # Start data processor task
    processor_task = asyncio.create_task(shared_data.process_data_stream())
    
    # Start strategy logic task (runs every second)
    strategy_task = asyncio.create_task(strategy_logic())
    
    # Start websockets in executor threads
    # Give Fyers a slight head start (primary source)
    fyers_task = loop.run_in_executor(executor, start_fyers_socket, loop)
    
    tasks = [fyers_task, processor_task, strategy_task]
    
    if secondary_data:
        await asyncio.sleep(2)
        kite_task = loop.run_in_executor(executor, start_kite_socket, loop)
        tasks.append(kite_task)
    
    try:
        # Run websockets, processor, and strategy concurrently
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("\n\n" + "="*60)
        print("🛑 SHUTTING DOWN DUAL SOCKET HANDLER")
        print("="*60)
        processor_task.cancel()
        strategy_task.cancel()
        executor.shutdown(wait=False)
        print("Connections closed successfully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 SHUTTING DOWN DUAL SOCKET HANDLER")
        print("="*60)
        print("Connections closed successfully")

