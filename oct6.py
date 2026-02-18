client_id = 'MD9QWRB74V-100'
secret_key = '9WHU5CV5DO'
redirect_uri ='https://fessorpro.com/'



lot_size=75
candle_time='5'
super_candle_1='5'
super_candle_2='5'
super_atr1=1  # Use proper ATR period for smoothing
super_atr2=14  # Standard ATR period
super_mul1=4   # Standard multiplier
super_mul2=2   # Conservative multiplier
target=5
stop_point=15
trail_point=0

daily_target_pnl=1500
track_pnl_strategy=True
money=20000

strategy_name='ravi'

#strategy parameters
index_name='NIFTY50'
exchange='NSE'
type2='INDEX'
ticker=f"{exchange}:{index_name}-{type2}"
underlying_ticker=f"{exchange}:{index_name}-{type2}"
# underlying_ticker='MCX:CRUDEOIL25AUGFUT'
# ticker='MCX:CRUDEOIL21JULFUT'
strike_count=10
strike_diff=50
account_type='LIVE'

time_zone="Asia/Kolkata"


start_hour,start_min=9,20
end_hour,end_min=15,15
# quantity=1  # Removed hardcoded quantity - will be calculated based on available capital
lot_size=75




# Import the required module from the fyers_apiv3 package
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import pandas as pd
import pendulum as dt
import asyncio
import pickle
import time
import webbrowser
import os
import sys
import certifi
import pytz
import numpy as np

# @title

#for windows ssl error
os.environ['SSL_CERT_FILE'] = certifi.where()


#disable fyersApi and Fyers Request logs
import logging

#disable logging for fyersApi
fyersModel.logging.getLogger().setLevel(logging.CRITICAL)

# Disable HTTP connection logs from requests and urllib3
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.CRITICAL)


# Configure logging with custom formatter
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s || %(message)s",
    handlers=[
        logging.FileHandler(f'{strategy_name}_{dt.now(time_zone).date()}.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Create a custom formatter with pendulum
class PendulumFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt1 = dt.from_timestamp(record.created, tz=time_zone)
        return dt1.format("YYYY-MM-DD HH:mm:ss")

# Create logger and set custom formatter
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# Create custom handlers with the pendulum formatter
file_handler = logging.FileHandler(f'{strategy_name}_{dt.now(time_zone).date()}.log', mode='a')
console_handler = logging.StreamHandler()

# Set formatter for both handlers
formatter = PendulumFormatter("%(asctime)s || %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Clear any existing handlers and add our custom ones
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to root logger to avoid conflicts
logger.propagate = False

# Additional suppression of HTTP connection logs
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("requests.packages.urllib3.util.retry").setLevel(logging.CRITICAL)

# Disable propagation for these loggers
for logger_name in ["requests", "urllib3", "urllib3.connectionpool", "requests.packages.urllib3", "requests.packages.urllib3.connectionpool"]:
    log = logging.getLogger(logger_name)
    log.setLevel(logging.CRITICAL)
    log.propagate = False

logger.info("This is an info message")

# def disable_fyers_files_logging():
#     """
#     Cross-platform function to disable Fyers logging files.
#     Works on Windows, macOS, and Linux systems.
#     """
#     import platform
#     import tempfile
    
#     # Get the null device path based on the operating system
#     def get_null_device():
#         """Returns the appropriate null device for the current OS"""
#         system = platform.system().lower()
#         if system == 'windows':
#             return 'nul'
#         else:
#             return '/dev/null'
    
#     null_device = get_null_device()
    
#     # Prevent Fyers log file creation by redirecting to null device
#     def prevent_fyers_file_creation():
#         """Prevent creation of Fyers log files by monkey-patching file opening"""
#         import builtins
#         original_open = builtins.open
        
#         def patched_open(file, mode='r', **kwargs):
#             # Check if this is a Fyers log file being opened for writing
#             if isinstance(file, str) and any(name in file.lower() for name in ['fyersapi.log', 'fyersdatasocket.log', 'fyersrequests.log']):
#                 if 'w' in mode or 'a' in mode:
#                     try:
#                         # Try to redirect to null device
#                         return original_open(null_device, mode, **kwargs)
#                     except (OSError, IOError):
#                         # If null device fails, create a temporary file that gets deleted
#                         import tempfile
#                         temp_file = tempfile.NamedTemporaryFile(mode=mode, delete=True, **kwargs)
#                         return temp_file
#             return original_open(file, mode, **kwargs)
        
#         builtins.open = patched_open

#     # Apply file creation prevention
#     prevent_fyers_file_creation()

#     # Comprehensive function to disable all Fyers logging
#     def disable_all_fyers_logging():
#         """Disable all Fyers-related logging to prevent log file creation while maintaining API functionality"""
#         try:
#             # Main Fyers logger - disable but keep structure intact
#             if hasattr(fyersModel, 'logging'):
#                 fyers_main_logger = fyersModel.logging.getLogger()
#                 fyers_main_logger.setLevel(logging.CRITICAL)
#                 # Don't disable completely, just set to critical level
#                 # fyers_main_logger.disabled = True
#         except Exception:
#             pass
        
#         # List of all possible Fyers and HTTP-related loggers
#         logger_names = [
#             'fyersApi', 'fyersDataSocket', 'fyersRequests', 'fyers_apiv3',
#             'requests', 'urllib3', 'urllib3.connectionpool', 'requests.packages.urllib3',
#             'requests.packages.urllib3.connectionpool', 'websocket', 'fyers_websocket',
#             'FyersWebsocket', 'data_ws', 'fyers_apiv3.FyersWebsocket.data_ws'
#         ]
        
#         for logger_name in logger_names:
#             try:
#                 logger_obj = logging.getLogger(logger_name)
#                 logger_obj.setLevel(logging.CRITICAL)
#                 logger_obj.disabled = True
#                 logger_obj.propagate = False
                
#                 # Remove only file handlers, keep at least one handler
#                 file_handlers_to_remove = []
#                 for handler in logger_obj.handlers:
#                     if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
#                         file_handlers_to_remove.append(handler)
                
#                 for handler in file_handlers_to_remove:
#                     try:
#                         logger_obj.removeHandler(handler)
#                         handler.close()
#                     except Exception:
#                         pass
                    
#                 # If no handlers left, add a null handler to prevent issues
#                 if not logger_obj.handlers:
#                     logger_obj.addHandler(logging.NullHandler())
                    
#             except Exception:
#                 pass
        
#         # Disable websocket logging if available
#         try:
#             if hasattr(data_ws, 'logging'):
#                 ws_logger = data_ws.logging.getLogger()
#                 ws_logger.setLevel(logging.CRITICAL)
#                 # Don't disable completely for API functionality
#         except Exception:
#             pass

#     # Apply initial logging disabling
#     disable_all_fyers_logging()

#     # Override the logging module's FileHandler for Fyers files only
#     original_file_handler_init = logging.FileHandler.__init__
#     def patched_file_handler_init(self, filename, mode='a', encoding=None, delay=False, errors=None):
#         # Check if this is a Fyers log file
#         if isinstance(filename, str) and any(name in filename.lower() for name in ['fyersapi.log', 'fyersdatasocket.log', 'fyersrequests.log']):
#             try:
#                 # Redirect to null device
#                 filename = null_device
#             except Exception:
#                 # If null device fails, use a temporary file
#                 import tempfile
#                 filename = tempfile.NamedTemporaryFile(delete=True).name
#         return original_file_handler_init(self, filename, mode, encoding, delay, errors)

#     # Safely patch the FileHandler
#     try:
#         logging.FileHandler.__init__ = patched_file_handler_init
#     except Exception:
#         pass

#     # Additional function to comprehensively disable all Fyers-related logging
#     def disable_all_fyers_logging_extended():
#         """Comprehensively disable all Fyers-related logging with extended coverage"""
#         # List of all possible Fyers logger names
#         fyers_loggers = [
#             'fyersApi', 'fyersDataSocket', 'fyersRequests', 
#             'requests', 'urllib3', 'urllib3.connectionpool',
#             'fyers_apiv3', 'FyersWebsocket', 'websocket',
#             'requests.packages.urllib3.connectionpool',
#             'requests.packages.urllib3.util.retry',
#             'websockets', 'asyncio'
#         ]
        
#         for logger_name in fyers_loggers:
#             try:
#                 logger_obj = logging.getLogger(logger_name)
#                 logger_obj.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL
#                 logger_obj.disabled = True
#                 logger_obj.propagate = False
#                 # Remove all existing handlers safely
#                 for handler in logger_obj.handlers[:]:
#                     try:
#                         logger_obj.removeHandler(handler)
#                         if hasattr(handler, 'close'):
#                             handler.close()
#                     except Exception:
#                         pass
#             except Exception:
#                 pass

#     # Call the extended function to disable logging
#     disable_all_fyers_logging_extended()
    
#     # Additional protection: Override root logger file handlers for Fyers-related logs
#     def protect_root_logger():
#         """Add protection at root logger level"""
#         try:
#             root_logger = logging.getLogger()
#             for handler in root_logger.handlers[:]:
#                 if isinstance(handler, logging.FileHandler):
#                     if hasattr(handler, 'baseFilename') and handler.baseFilename:
#                         filename = handler.baseFilename.lower()
#                         if any(name in filename for name in ['fyersapi', 'fyersdatasocket', 'fyersrequests']):
#                             root_logger.removeHandler(handler)
#                             handler.close()
#         except Exception:
#             pass
    
#     protect_root_logger()
    
#     print(f"✅ Fyers logging disabled successfully for {platform.system()} platform")

# disable_fyers_files_logging()

# Check if access.txt exists, then read the file and get the access token
if os.path.exists(f'access-{dt.now(time_zone).date()}.txt'):
    print('access token exists')
    with open(f'access-{dt.now(time_zone).date()}.txt', 'r') as f:
        access_token = f.read()

else:
    # Define response type and state for the session
    response_type = "code"
    state = "sample_state"
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
        access_token = response["access_token"]
        with open(f'access-{dt.now(time_zone).date()}.txt', 'w') as k:
            k.write(access_token)
    except Exception as e:
        # Print the exception and response for debugging
        print(e, response)
        print('unable to get access token')
        sys.exit()

# Print the access token
print('access token:', access_token)


# Get the current time
current_time=dt.now(time_zone)
start_time=dt.datetime(current_time.year,current_time.month,current_time.day,start_hour,start_min,tz=time_zone)
end_time=dt.datetime(current_time.year,current_time.month,current_time.day,end_hour,end_min,tz=time_zone)
print('start time:', start_time)
print('end time:', end_time)




# Initialize FyersModel instances for synchronous and asynchronous operations
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path=None)
fyers_asysc = fyersModel.FyersModel(client_id=client_id, is_async=True, token=access_token, log_path=None)

available_cash= fyers.funds()
print('available cash:', available_cash.get('fund_limit')[-1].get('equityAmount'))



# Define the data for the option chain request
data = {
    "symbol": underlying_ticker,
    "strikecount": strike_count,
    "timestamp": ""
}

# Get the expiry data from the option chain
response = fyers.optionchain(data=data)['data']
expiry = response['expiryData'][0]['date']
print("current_expiry selected", expiry)
expiry_e = response['expiryData'][0]['expiry']

# Define the data for the option chain request with expiry
data = {
    "symbol": underlying_ticker,
    "strikecount": strike_count,
    "timestamp": expiry_e
}

# Get the option chain data
response = fyers.optionchain(data=data)['data']
option_chain = pd.DataFrame(response['optionsChain'])
symbols = option_chain['symbol'].to_list()

# Get the current spot price
spot_price = option_chain['ltp'].iloc[0]
print('current spot price is', spot_price)

# Separate the symbols into call and put lists
call_list = []
put_list = []
for s in symbols:
    if s.endswith('CE'):
        call_list.append(s)
    else:
        put_list.append(s)

# Combine the put and call lists
symbols = put_list + call_list
print(symbols)

# Initialize the DataFrame for storing option data
df = pd.DataFrame(columns=['name', 'ltp', 'ch', 'chp', 'avg_trade_price', 'open_price', 'high_price', 'low_price', 'prev_close_price', 'vol_traded_today', 'oi', 'pdoi', 'oipercent', 'bid_price', 'ask_price', 'last_traded_time', 'exch_feed_time', 'bid_size', 'ask_size', 'last_traded_qty', 'tot_buy_qty', 'tot_sell_qty', 'lower_ckt', 'upper_ckt', 'type', 'symbol', 'expiry'])
df['name'] = symbols
df.set_index('name', inplace=True)
print(df)


new_symbols_list=[underlying_ticker]

f = dt.now(time_zone).date() - dt.duration( days=5)
p = dt.now(time_zone).date()

data = {
    "symbol": underlying_ticker,
    "resolution": candle_time,
    "date_format": "1",
    "range_from": f.strftime('%Y-%m-%d'),
    "range_to": p.strftime('%Y-%m-%d'),
    "cont_flag": "1"
}


# Fetch historical data
response2 =fyers.history(data=data)
hist_data = pd.DataFrame(response2['candles'])
hist_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

ist = pytz.timezone('Asia/Kolkata')
hist_data['date'] = pd.to_datetime(hist_data['date'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ist)
# print(hist_data)
# hist_data=hist_data[hist_data['date'].dt.date<dt.now(time_zone).date()]
print(hist_data)



def supertrend2(high, low, close, length=10, multiplier=3):
    """
    Correct Supertrend implementation based on the standard algorithm.
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of close prices
        length (int): The ATR period. Default: 10
        multiplier (float): The ATR multiplier. Default: 3.0
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            SUPERT - The trend value
            SUPERTd - The direction (1 for long, -1 for short)
            SUPERTl - The long values
            SUPERTs - The short values
    """
    # Convert to pandas Series if needed
    high = pd.Series(high) if not isinstance(high, pd.Series) else high
    low = pd.Series(low) if not isinstance(low, pd.Series) else low
    close = pd.Series(close) if not isinstance(close, pd.Series) else close
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # Combine True Range components
    tr = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = tr.max(axis=1)
    
    # Handle first row (NaN values due to shift)
    true_range.iloc[0] = tr1.iloc[0]
    
    # Calculate ATR
    if length == 1:
        atr = true_range
    else:
        atr = true_range.rolling(window=length, min_periods=1).mean()
    
    # Calculate HL2 (median price)
    hl2 = (high + low) / 2
    
    # Calculate basic bands
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    # Initialize series for final bands and supertrend
    final_upper = pd.Series(index=close.index, dtype=float)
    final_lower = pd.Series(index=close.index, dtype=float)
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    # Set first values
    final_upper.iloc[0] = basic_upper.iloc[0]
    final_lower.iloc[0] = basic_lower.iloc[0]
    
    # Initial supertrend and direction
    if close.iloc[0] <= final_lower.iloc[0]:
        supertrend.iloc[0] = final_upper.iloc[0]
        direction.iloc[0] = -1
    else:
        supertrend.iloc[0] = final_lower.iloc[0]
        direction.iloc[0] = 1
    
    # Calculate for remaining bars
    for i in range(1, len(close)):
        # Calculate final upper band
        if (basic_upper.iloc[i] < final_upper.iloc[i-1]) or (close.iloc[i-1] > final_upper.iloc[i-1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        
        # Calculate final lower band
        if (basic_lower.iloc[i] > final_lower.iloc[i-1]) or (close.iloc[i-1] < final_lower.iloc[i-1]):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # Calculate supertrend
        if (supertrend.iloc[i-1] == final_upper.iloc[i-1]) and (close.iloc[i] <= final_upper.iloc[i]):
            supertrend.iloc[i] = final_upper.iloc[i]
        elif (supertrend.iloc[i-1] == final_upper.iloc[i-1]) and (close.iloc[i] > final_upper.iloc[i]):
            supertrend.iloc[i] = final_lower.iloc[i]
        elif (supertrend.iloc[i-1] == final_lower.iloc[i-1]) and (close.iloc[i] >= final_lower.iloc[i]):
            supertrend.iloc[i] = final_lower.iloc[i]
        elif (supertrend.iloc[i-1] == final_lower.iloc[i-1]) and (close.iloc[i] < final_lower.iloc[i]):
            supertrend.iloc[i] = final_upper.iloc[i]
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
        
        # Calculate direction
        if supertrend.iloc[i] == final_upper.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = 1
    
    # Create output DataFrame
    df = pd.DataFrame({
        "SUPERT": supertrend,
        "SUPERTd": direction,
        "SUPERTl": np.where(direction == 1, supertrend, np.nan),
        "SUPERTs": np.where(direction == -1, supertrend, np.nan),
    }, index=close.index)
    
    return df


def supertrend(high, low, close, length=10, multiplier=3):
    """
    Supertrend function that matches pandas_ta.supertrend output.
    
    Args:
        high (pd.Series): Series of high prices
        low (pd.Series): Series of low prices
        close (pd.Series): Series of close prices
        length (int): The ATR period. Default: 7
        multiplier (float): The ATR multiplier. Default: 3.0
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            SUPERT - The trend value
            SUPERTd - The direction (1 for long, -1 for short)
            SUPERTl - The long values
            SUPERTs - The short values
    """
    # Calculate ATR using the pandas_ta method (RMA - Rolling Moving Average)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/length, adjust=False).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Initialize direction and trend
    direction = [1]  # Start with long
    trend = [lowerband.iloc[0]]  # Start with lowerband
    long = [lowerband.iloc[0]]
    short = [np.nan]

    # Iterate through the data to calculate the Supertrend
    for i in range(1, len(close)):
        if close.iloc[i] > upperband.iloc[i - 1]:
            direction.append(1)
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            direction.append(-1)
        else:
            direction.append(direction[i - 1])
            if direction[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if direction[i] == -1 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        if direction[i] == 1:
            trend.append(lowerband.iloc[i])
            long.append(lowerband.iloc[i])
            short.append(np.nan)
        else:
            trend.append(upperband.iloc[i])
            long.append(np.nan)
            short.append(upperband.iloc[i])

    

    # Create DataFrame to return
    df = pd.DataFrame({
        "SUPERT": trend,
        "SUPERTd": direction,
        "SUPERTl": long,
        "SUPERTs": short,
    }, index=close.index)

    #if this column contains even a single negative value call supertrend2 function
    if (df['SUPERT'] < 0).any():
        df = supertrend2(high, low, close, length, multiplier)



    return df





super = supertrend(hist_data['high'], hist_data['low'], hist_data['close'])
print(super)


# Function to get the OTM option based on spot price and side (CE/PE)
def get_otm_option(spot_price, side, points=100):
    if side == 'CE':
        otm_strike = (round(spot_price / strike_diff) * strike_diff) + points
    else:
        otm_strike = (round(spot_price / strike_diff) * strike_diff) - points
    otm_option = option_chain[(option_chain['strike_price'] == otm_strike) & (option_chain['option_type'] == side)]['symbol'].squeeze()
    return otm_option, otm_strike



call_option, call_buy_strike = get_otm_option(spot_price, 'CE', 0)
put_option, put_buy_strike = get_otm_option(spot_price, 'PE', 0)
print('call option:', call_option)
print('put option:', put_option)

# Cache for historical data to reduce API calls
historical_data_cache = {}
cache_timestamps = {}

def fetchOHLC(ticker,interval,duration):
    """extracts historical data and outputs in the form of dataframe"""
    instrument = ticker
    # print(ticker)
    data = {"symbol":instrument,"resolution":interval,"date_format":"1","range_from":(dt.now(time_zone)-dt.duration(days=duration)).date(),"range_to":dt.now(time_zone).date(),"cont_flag":"1"}
    sdata=fyers.history(data)
    # print(sdata)
    sdata=pd.DataFrame(sdata['candles'])
    sdata.columns=['date','open','high','low','close','volume']
    sdata['date']=pd.to_datetime(sdata['date'], unit='s')
    sdata.date=(sdata.date.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata'))
    sdata['date'] = sdata['date'].dt.tz_localize(None)
    sdata=sdata.set_index('date')
    
    # Remove duplicate timestamps, keeping the last occurrence
    sdata = sdata[~sdata.index.duplicated(keep='last')]
    
    # sdata=sdata.iloc[:-1]
    # print(sdata)
    return sdata

def fetchOHLC_cached(ticker, interval, duration):
    """Cached version of fetchOHLC to reduce API calls and latency"""
    cache_key = f"{ticker}_{interval}_{duration}"
    current_time = dt.now(time_zone)
    
    # Check if we have cached data that's less than 30 seconds old
    if (cache_key in historical_data_cache and 
        cache_key in cache_timestamps and 
        (current_time - cache_timestamps[cache_key]).total_seconds() < 30):
        return historical_data_cache[cache_key]
    
    # Fetch fresh data
    try:
        data = fetchOHLC(ticker, interval, duration)
        historical_data_cache[cache_key] = data
        cache_timestamps[cache_key] = current_time
        return data
    except Exception as e:
        # Return cached data if fresh fetch fails
        if cache_key in historical_data_cache:
            logger.warning(f"Using cached data due to fetch error: {e}")
            return historical_data_cache[cache_key]
        raise e







# Log the start of the strategy
logger.info('🚀 Strategy starting - logging system initialized')
logger.info('📊 Starting main strategy execution')

new_symbols_list.extend([call_option,put_option])

# Add call and put options to the DataFrame if they're not already there
for new_symbol in [call_option, put_option]:
    if new_symbol not in df.index:
        # Add the symbol as a new row with NaN values initially
        new_row = pd.Series([None] * len(df.columns), index=df.columns, name=new_symbol)
        df = pd.concat([df, new_row.to_frame().T])
        logger.info(f"Added {new_symbol} to DataFrame")

print(f"DataFrame now has {len(df)} symbols including options")

# Function to store data using pickle
def store(data, account_type):
    pickle.dump(data, open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'wb'))

# Function to load data using pickle
def load(account_type):
    return pickle.load(open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'rb'))

# # Function to place a limit order
# def take_mkt_position(ticker, action, quantity, limit_price):
#     try:
#         data = {
#             "symbol": ticker,
#             "qty": quantity,
#             "type": 2,
#             "side": action,
#             "productType": "INTRADAY",
#             "limitPrice": 0,
#             "stopPrice": 0,
#             "validity": "DAY",
#             "disclosedQty": 0,
#             "offlineOrder": False,
#             "stopLoss": 0,
#             "takeProfit": 0
#         }
#         response3 = fyers.place_order(data=data)
#         logger.info(response3)
#         print(response3)
#     except Exception as e:
#         logger.info(e)
#         print(e)
#         print('unable to place order for some reason')





hist1=fetchOHLC(call_option,super_candle_1,100)
print(hist1.tail(30))
# hist1.to_csv('data.csv')
hist2=fetchOHLC(call_option,super_candle_2,100)
print(hist2.tail(30))





super1 = supertrend(hist1['high'], hist1['low'], hist1['close'],super_atr1,super_mul1)
print(super1.tail(30))
super2=supertrend(hist2['high'], hist2['low'], hist2['close'],super_atr2,super_mul2)
print(super2.tail(30))

t1=super1['SUPERTd'].iloc[-1]
t2=super2['SUPERTd'].iloc[-1]
print(t1,t2)

hist1=fetchOHLC(put_option,super_candle_1,100)
hist2=fetchOHLC(put_option,super_candle_2,100)

super1 = supertrend(hist1['high'], hist1['low'], hist1['close'],super_atr1,super_mul1)
print(super1.tail(30))
super2=supertrend(hist2['high'], hist2['low'], hist2['close'],super_atr2,super_mul2)
print(super2.tail(30))
t3=super1['SUPERTd'].iloc[-1]
t4=super2['SUPERTd'].iloc[-1]
print(t3,t4)

def get_supertrend_value(ticker):
        # Use cached data to reduce latency
        hist1=fetchOHLC_cached(ticker,super_candle_1,100)
        hist2=fetchOHLC_cached(ticker,super_candle_2,100)

        # Remove duplicates based on index to avoid non-unique label issues
        hist1 = hist1[~hist1.index.duplicated(keep='last')]
        hist2 = hist2[~hist2.index.duplicated(keep='last')]

        super1 = supertrend(hist1['high'], hist1['low'], hist1['close'],super_atr1,super_mul1)
        print(super1)
        super2=supertrend(hist2['high'], hist2['low'], hist2['close'],super_atr2,super_mul2)
        print(super2)

        #merge hist1 and super1
        hist_sup1= pd.merge(hist1, super1, left_index=True, right_index=True)
        hist_sup2= pd.merge(hist2, super2, left_index=True, right_index=True)
        
        # Ensure no duplicates in merged dataframes
        hist_sup1 = hist_sup1[~hist_sup1.index.duplicated(keep='last')]
        hist_sup2 = hist_sup2[~hist_sup2.index.duplicated(keep='last')]
        
        print(hist_sup1)
        print(hist_sup2)

        try:
            trend_changes = hist_sup1['SUPERTd'] != hist_sup1['SUPERTd'].shift(1)
            if trend_changes.any():
                last_trend_change_idx = hist_sup1[trend_changes].index[-1]
                # Get the highest high since the last trend change
                high1 = hist_sup1.loc[last_trend_change_idx:, 'high'].max()
            else:
                # If no trend changes, use overall max
                high1 = hist_sup1['high'].max()
        except Exception as e:
            print(f"Error calculating high1: {e}")
            high1 = hist_sup1['high'].max()

        print(f"Highest high since last trend change: {high1}")

        try:
            trend_changes = hist_sup2['SUPERTd'] != hist_sup2['SUPERTd'].shift(1)
            if trend_changes.any():
                last_trend_change_idx = hist_sup2[trend_changes].index[-1]
                # Get the highest high since the last trend change
                high2 = hist_sup2.loc[last_trend_change_idx:, 'high'].max()
            else:
                # If no trend changes, use overall max
                high2 = hist_sup2['high'].max()
        except Exception as e:
            print(f"Error calculating high2: {e}")
            high2 = hist_sup2['high'].max()
        
        print(f"Highest high since last trend change: {high2}")


        trend1=super1['SUPERTd'].iloc[-1]
        trend2=super2['SUPERTd'].iloc[-1]
        return trend1, trend2,high1,high2

t1,t2,ch1,ch2=get_supertrend_value(call_option)
t3,t4,ph1,ph2=get_supertrend_value(put_option)



# Load or initialize paper trading information
if account_type == 'PAPER':

    try:
        paper_info = load(account_type)
        # Add wait_until field if it doesn't exist (backward compatibility)
        if 'wait_until' not in paper_info.get(underlying_ticker, {}):
            paper_info[underlying_ticker]['wait_until'] = None
   
    except:
        column_names = ['time', 'ticker', 'price', 'action', 'stop_price', 'take_profit', 'spot_price', 'quantity']
        filled_df = pd.DataFrame(columns=column_names)
        filled_df.set_index('time', inplace=True)
        paper_info = {  underlying_ticker:{'call_buy':{'option_name':call_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':0,'pnl':0,'trend1':t1,'trend2':t2,'high1':ch1,'high2':ch2},
                        'put_buy':{'option_name':put_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':0,'pnl':0,'trend1':t3,'trend2':t4,'high1':ph1,'high2':ph2},
                        'condition':False,
                        'spot_buy':0,
                        'next_min':0,
                        'trend1':None,
                        'trend2':None,
                        'exit_instrument':[],
                        'pos_instrument':[],
                        'filled_df':filled_df,
                        'wait_until':None
                        
                        }
                    }

else:
    try:
        real_info = load(account_type)
        # Add wait_until field if it doesn't exist (backward compatibility)
        if 'wait_until' not in real_info.get(underlying_ticker, {}):
            real_info[underlying_ticker]['wait_until'] = None
   
    except:
        column_names = ['time', 'ticker', 'price', 'action', 'stop_price', 'take_profit', 'spot_price', 'quantity']
        filled_df = pd.DataFrame(columns=column_names)
        filled_df.set_index('time', inplace=True)
        real_info = {  underlying_ticker:{'call_buy':{'option_name':call_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':0,'pnl':0,'trend1':t1,'trend2':t2,'high1':ch1,'high2':ch2},
                        'put_buy':{'option_name':put_option,'trade_flag':0,'buy_price':0,'current_stop_price':0,'current_profit_price':0,'filled_df':filled_df.copy(),'underlying_price_level':0,'quantity':0,'pnl':0,'trend1':t3,'trend2':t4,'high1':ph1,'high2':ph2},
                        'condition':False,
                        'spot_buy':0,
                        'next_min':0,
                        'trend1':None,
                        'trend2':None,
                        'exit_instrument':[],
                        'pos_instrument':[],
                        'filled_df':filled_df,
                        'wait_until':None
                        
                        }
                    }


# Optimized order function with reduced latency
def take_position_fast(ticker, action, quantity=None):
    """Fast order placement with minimal processing overhead"""
    global lot_size
    
    if quantity is None:
        quantity = lot_size

    # Pre-construct order data for speed
    order_data = {
        "symbol": ticker,
        "qty": quantity,
        "type": 2,  # Market order for fastest execution
        "side": action,
        "productType": "INTRADAY",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss": 0,
        "takeProfit": 0
    }
    
    try:
        # Direct order placement without extra logging during critical periods
        response = fyers.place_order(data=order_data)
        return response
    except Exception as e:
        logger.error(f"Fast order failed: {e}")
        return None

# Function to place a limit order
def take_position(ticker, action, quantity=None):
    global lot_size
    
    # Use provided quantity or default to 1*lot_size
    if quantity is None:
        quantity = 1 * lot_size

    try:
        data = {
            "symbol": ticker,
            "qty": quantity,  # Use the calculated quantity instead of hardcoded 75
            "type": 2,
            "side": action,
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        response3 = fyers.place_order(data=data)
        logger.info(response3)
        print(response3)
        return response3
    except Exception as e:
        logger.info(e)
        print(e)
        print('unable to place order for some reason')
        return None



if account_type=='PAPER':
    t1,t2,ch1,ch2=get_supertrend_value(call_option)
    paper_info.get(underlying_ticker).get('call_buy').update({'trend1':t1,'trend2':t2,'high1':ch1,'high2':ch2})

    call_trend1=t1
    call_trend2=t2

    t1,t2,ph1,ph2=get_supertrend_value(put_option)
    paper_info.get(underlying_ticker).get('put_buy').update({'trend1':t1,'trend2':t2,'high1':ph1,'high2':ph2})

    put_trend1=t1
    put_trend2=t2
else:
    t1,t2,ch1,ch2=get_supertrend_value(call_option)
    real_info.get(underlying_ticker).get('call_buy').update({'trend1':t1,'trend2':t2,'high1':ch1,'high2':ch2})

    call_trend1=t1
    call_trend2=t2

    t1,t2,ph1,ph2=get_supertrend_value(put_option)
    real_info.get(underlying_ticker).get('put_buy').update({'trend1':t1,'trend2':t2,'high1':ph1,'high2':ph2})

    put_trend1=t1
    put_trend2=t2


def onmessage(ticks):
    global df
    # Optimized data processing - reduce unnecessary operations
    symbol = ticks.get('symbol')
    if not symbol:
        return
    
    try:
        # Fast path for existing symbols - avoid expensive operations
        if symbol in df.index:
            # Bulk update only changed values to minimize DataFrame operations
            updated_data = {k: v for k, v in ticks.items() if v is not None and k in df.columns}
            if updated_data:
                # Use direct assignment instead of iterating
                for key, value in updated_data.items():
                    df.loc[symbol, key] = value
        else:
            # Add missing symbol only if in our watch list
            if symbol in new_symbols_list:
                # Faster DataFrame expansion using dict constructor
                new_data = {col: ticks.get(col) for col in df.columns}
                df.loc[symbol] = new_data
                logger.info(f"Added symbol: {symbol}")
    except Exception as e:
        logger.error(f"Error in onmessage: {e}")

def onerror(message):
    logger.error(f"WebSocket Error: {message}")
    print("WebSocket Error:", message)

def onclose(message):
    logger.warning(f"WebSocket Connection closed: {message}")
    print("WebSocket Connection closed:", message)

def onopen():
    global symbols, new_symbols_list
    # Specify the data type and symbols you want to subscribe to
    data_type = "SymbolUpdate"
    print(f"WebSocket connected. Subscribing to symbols: {new_symbols_list}")
    try:
        fyers_socket.subscribe(symbols=new_symbols_list, data_type=data_type)
        print('WebSocket subscription complete')
        logger.info(f"WebSocket connected and subscribed to {len(new_symbols_list)} symbols")
    except Exception as e:
        logger.error(f"Failed to subscribe to symbols: {e}")
        print(f"Subscription error: {e}")

print(new_symbols_list)

# Initialize reconnection tracking
reconnect_attempts = 0

# Create a FyersDataSocket instance with optimized parameters for low latency
fyers_socket = data_ws.FyersDataSocket(
    access_token=f"{client_id}:{access_token}",
    log_path=None,
    litemode=True,  # Enable lite mode for faster data processing
    write_to_file=False,
    reconnect=True,
    on_connect=onopen,
    on_close=onclose,
    on_error=onerror,
    on_message=onmessage
)

fyers_socket.connect()

# Keep the socket running to receive real-time data (moved here from onopen)
# Note: This should be called in a separate thread to avoid blocking
import threading

def start_websocket():
    global fyers_socket, reconnect_attempts
    max_reconnect_attempts = 5
    reconnect_attempts = 0
    
    while reconnect_attempts < max_reconnect_attempts:
        try:
            logger.info(f"Starting WebSocket connection (attempt {reconnect_attempts + 1}/{max_reconnect_attempts})")
            fyers_socket.keep_running()
            break  # If successful, break out of the loop
        except Exception as e:
            reconnect_attempts += 1
            logger.error(f"WebSocket keep_running error (attempt {reconnect_attempts}): {e}")
            if reconnect_attempts < max_reconnect_attempts:
                logger.info(f"Attempting to reconnect in 5 seconds...")
                time.sleep(5)
                try:
                    # Recreate the WebSocket connection
                    fyers_socket = data_ws.FyersDataSocket(
                        access_token=f"{client_id}:{access_token}",
                        log_path=None,
                        litemode=False,
                        write_to_file=False,
                        reconnect=True,
                        on_connect=onopen,
                        on_close=onclose,
                        on_error=onerror,
                        on_message=onmessage
                    )
                    fyers_socket.connect()
                except Exception as reconnect_error:
                    logger.error(f"Failed to recreate WebSocket connection: {reconnect_error}")
            else:
                logger.error("Max reconnection attempts reached. WebSocket connection failed.")
                break

# Start websocket in a separate thread
websocket_thread = threading.Thread(target=start_websocket, daemon=True)
websocket_thread.start()

print('WebSocket connection initiated')


async def paper_order(name):
    global lot_size
    global paper_info
    global real_info
    global df
    global spot_price
    
    # Select the appropriate info dictionary based on trading mode
    info = real_info if is_real_trading else paper_info
    trading_prefix = "[REAL TRADING]" if is_real_trading else ""


    # Get option names
    call_name = info.get(name)['call_buy']['option_name']
    put_name = info.get(name)['put_buy']['option_name']
    
    # Get the current spot price
    try:
        spot_price = df.loc[name, 'ltp']
        call_price = df.loc[call_name, 'ltp'] if call_name in df.index else None
        put_price = df.loc[put_name, 'ltp'] if put_name in df.index else None
        
        # Check if we have valid prices
        if pd.isna(spot_price) or spot_price <= 0:
            logger.warning(f"{trading_prefix} Invalid spot price: {spot_price}")
            return
            
        if call_price is None or pd.isna(call_price) or call_price <= 0:
            logger.warning(f"{trading_prefix} Missing or invalid call price for {call_name}: {call_price}")
            # Try to resubscribe to missing symbols
            if call_name not in df.index:
                fyers_socket.subscribe(symbols=[call_name], data_type="SymbolUpdate")
            return
            
        if put_price is None or pd.isna(put_price) or put_price <= 0:
            logger.warning(f"{trading_prefix} Missing or invalid put price for {put_name}: {put_price}")
            # Try to resubscribe to missing symbols
            if put_name not in df.index:
                fyers_socket.subscribe(symbols=[put_name], data_type="SymbolUpdate")
            return

        print(f"{trading_prefix} Spot price: {spot_price}, call option: {call_name} price: {call_price} , put option: {put_name} price: {put_price}")
        
    except KeyError as e:
        logger.error(f"{trading_prefix} Missing symbol in DataFrame: {e}")
        # Try to resubscribe to all required symbols
        missing_symbols = [sym for sym in [name, call_name, put_name] if sym not in df.index]
        if missing_symbols:
            fyers_socket.subscribe(symbols=missing_symbols, data_type="SymbolUpdate")
            logger.info(f"{trading_prefix} Resubscribed to missing symbols: {missing_symbols}")
        return
    except Exception as e:
        logger.error(f"{trading_prefix} Error getting price data: {e}")
        return

    # Get the current time
    ct = dt.now(time_zone)

    # Check if the current time is greater than the start time
    if ct > start_time:
        
        # Get trade flags
        call_flag = info.get(name)['call_buy']['trade_flag']
        put_flag = info.get(name)['put_buy']['trade_flag']

        call_buy_price = info.get(name)['call_buy']['buy_price']
        put_buy_price = info.get(name)['put_buy']['buy_price']

        # Get stop prices
        call_stop_price = info.get(name)['call_buy']['current_stop_price']
        put_stop_price = info.get(name)['put_buy']['current_stop_price']


        call_trend1 = info.get(name).get('call_buy').get('trend1')
        call_trend2 = info.get(name).get('call_buy').get('trend2')

        call_high1=info.get(name).get('call_buy').get('high1')
        call_high2=info.get(name).get('call_buy').get('high2')

        put_trend1 = info.get(name).get('put_buy').get('trend1')
        put_trend2 = info.get(name).get('put_buy').get('trend2')

        put_high1=info.get(name).get('put_buy').get('high1')
        put_high2=info.get(name).get('put_buy').get('high2')

        print(call_trend1, call_trend2, put_trend1, put_trend2,call_high1,call_high2,put_high1,put_high2)
        # Get condition
        condition = info.get(name)['condition']
        print('call price',call_price,'put price', put_price)

        

        # Check if the current time is greater than the end time
        if ct > end_time:
            print(f'{trading_prefix} Closing all positions due to end time')

            # Close call buy position if trade flag is set
            if call_flag == 1:

                logger.info(f'{trading_prefix} [END TIME CLOSURE] Closing CALL position due to end time reached at {end_time}')
                
                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(call_name, -1, info.get(name)['call_buy']['quantity'])
                        logger.info(f'{trading_prefix} [CALL SELL] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to close call position: {e}')
                
                # Update tracking
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, -info.get(name)['call_buy']['quantity']]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['call_buy']['quantity'] = 0  # Update quantity
                info.get(name)['call_buy']['trade_flag'] = 0 # Update flag
                info.get(name)['condition'] = False
                info.get(name)['next_min'] = False
                logger.info(f'{trading_prefix} [CALL SELL] Position closed at end time: {call_name} | Price: ₹{call_price} | Qty: {info.get(name)["call_buy"]["quantity"]} | Spot: ₹{spot_price}')
                info.get(name)['filled_df'].to_csv(f'{strategy_name}_{dt.now(time_zone).date()}.csv')

            # Close put buy position if trade flag is set
            if put_flag == 1:

                logger.info(f'{trading_prefix} [END TIME CLOSURE] Closing PUT position due to end time reached at {end_time}')
                
                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(put_name, -1, info.get(name)['put_buy']['quantity'])
                        logger.info(f'{trading_prefix} [PUT SELL] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to close put position: {e}')
                
                # Update tracking
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, -info.get(name)['put_buy']['quantity'] ]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['put_buy']['quantity'] = 0  # Update quantity
                info.get(name)['put_buy']['trade_flag'] = 0  # Update flag
                info.get(name)['condition'] = False
                info.get(name)['next_min'] = False
                logger.info(f'{trading_prefix} [PUT SELL] Position closed at end time: {put_name} | Price: ₹{put_price} | Qty: {info.get(name)["put_buy"]["quantity"]} | Spot: ₹{spot_price}')
                info.get(name)['filled_df'].to_csv(f'{strategy_name}_{dt.now(time_zone).date()}.csv')

            return 0


        if (ct.second ==1 and  ct.minute in range(0,60,int(candle_time))):

            # Use cached data for supertrend calculations
            t1,t2,ch1,ch2=get_supertrend_value(call_name)
            info.get(name).get('call_buy').update({'trend1':t1,'trend2':t2,'high1':ch1,'high2':ch2})

            call_trend1=t1
            call_trend2=t2

            t1,t2,ph1,ph2=get_supertrend_value(put_name)
            info.get(name).get('put_buy').update({'trend1':t1,'trend2':t2,'high1':ph1,'high2':ph2})

            put_trend1=t1
            put_trend2=t2
            if info.get(name)['condition']==False :

                # Check if we need to wait for next candle close
                wait_until = info.get(name).get('wait_until')
                if wait_until and ct < wait_until:
                    return

                if call_trend1==1 and call_trend2==1 and (call_price>ch1 or call_price>ch2):
                    info.get(name)['spot_buy']=spot_price
                    #calculate quantity
                    q=int(fyers.funds().get('fund_limit')[-1].get('equityAmount'))
                    if q==0:
                        q=money
                    # Prevent division by zero
                    if call_price <= 0:
                        logger.error(f'{trading_prefix} Invalid call price: {call_price}')
                        return
                    quantity = int(q / call_price)
                    #multiple of lotsize
                    quantity = (quantity // lot_size) * lot_size
                    
                    # Minimum quantity check
                    if quantity <= 0:
                        logger.error(f'{trading_prefix} Insufficient funds for trade. Available: ₹{q}, Option price: ₹{call_price}')
                        return

                    info.get(name)['call_buy']['quantity'] = quantity
                    logger.info(f'{trading_prefix} [QUANTITY CALC] CALL quantity calculated: {quantity} lots | Available funds: ₹{q} | Option price: ₹{call_price}')
                    info.get(name)['call_buy']['buy_price'] = call_price
                    # Initialize stop loss based on option price, not spot price
                    call_stop_price = call_price - stop_point
                    info.get(name)['call_buy']['current_stop_price'] = call_stop_price
                    
                    # Execute real order if in real trading mode
                    if is_real_trading:
                        try:
                            response = take_position(call_name, 1, quantity)
                            logger.info(f'{trading_prefix} [CALL BUY] Order placed: {response}')
                        except Exception as e:
                            logger.error(f'{trading_prefix} Failed to place call buy order: {e}')
                            return

                    a = [call_name, call_price, 'BUY', call_stop_price, 0, spot_price, quantity]
                    info.get(name)['filled_df'].loc[ct] = a  # Save to dataframe
                    info.get(name)['call_buy']['trade_flag'] = 1  # Update call flag
                    info.get(name)['put_buy']['trade_flag'] = 3  # Update put flag
                    #current profit price
                    info.get(name)['call_buy']['current_profit_price'] = call_price + target
                    # Clear the wait_until flag
                    info.get(name)['wait_until'] = None
         
                    logger.info(f'{trading_prefix} [CALL BUY] ✅ Entry signal: {call_name} | Price: ₹{call_price} | Qty: {quantity} | Spot: ₹{spot_price} | Target: ₹{call_price + target} | Trends: {call_trend1},{call_trend2} | Highs: ₹{ch1},₹{ch2}')
                    info.get(name)['condition'] = True 
                    
                elif put_trend1==1 and put_trend2==1 and (put_price>ph1 or put_price>ph2):
                    info.get(name)['spot_buy']=spot_price

                    q=int(fyers.funds().get('fund_limit')[-1].get('equityAmount'))
                    if q==0:
                        q=money
                    # Prevent division by zero
                    if put_price <= 0:
                        logger.error(f'{trading_prefix} Invalid put price: {put_price}')
                        return
                    quantity = int(q / put_price)
                    #multiple of lotsize
                    quantity = (quantity // lot_size) * lot_size
                    
                    # Minimum quantity check
                    if quantity <= 0:
                        logger.error(f'{trading_prefix} Insufficient funds for trade. Available: ₹{q}, Option price: ₹{put_price}')
                        return

                    info.get(name)['put_buy']['quantity'] = quantity
                    logger.info(f'{trading_prefix} [QUANTITY CALC] PUT quantity calculated: {quantity} lots | Available funds: ₹{q} | Option price: ₹{put_price}')
                    info.get(name)['put_buy']['buy_price'] = put_price

                    # Initialize stop loss based on option price, not spot price
                    put_stop_price = put_price - stop_point
                    info.get(name)['put_buy']['current_stop_price'] = put_stop_price

                    # Execute real order if in real trading mode
                    if is_real_trading:
                        try:
                            response = take_position(put_name, 1, quantity)
                            logger.info(f'{trading_prefix} [PUT BUY] Order placed: {response}')
                        except Exception as e:
                            logger.error(f'{trading_prefix} Failed to place put buy order: {e}')
                            return

                    a = [put_name, put_price, 'BUY', put_stop_price, 0, spot_price, quantity]
                    info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                    info.get(name)['put_buy']['trade_flag'] = 1  # Update put flag
                    info.get(name)['call_buy']['trade_flag'] = 3  # Update call flag
                    #current profit price
                    info.get(name)['put_buy']['current_profit_price'] = put_price + target
                    # Clear the wait_until flag
                    info.get(name)['wait_until'] = None
                    logger.info(f'{trading_prefix} [PUT BUY] ✅ Entry signal: {put_name} | Price: ₹{put_price} | Qty: {quantity} | Spot: ₹{spot_price} | Target: ₹{put_price + target} | Trends: {put_trend1},{put_trend2} | Highs: ₹{ph1},₹{ph2}')
                    print(f'Put buy condition satisfied: {put_name} at {put_price}')
                
                    info.get(name)['condition'] = True 
                
                
            elif info.get(name)['condition']==True and ( (call_flag==1 and call_price>call_buy_price) or (put_flag==1 and put_price>put_buy_price) ) :
                #close position 
                logger.info(f'{trading_prefix} [PROFITABLE EXIT] Closing current position due to favorable price movement before opening new position')

                if call_flag==1:
                    # Execute real order if in real trading mode
                    if is_real_trading:
                        try:
                            response = take_position(call_name, -1, info.get(name)['call_buy']['quantity'])
                            logger.info(f'{trading_prefix} [CALL SELL] Order placed: {response}')
                        except Exception as e:
                            logger.error(f'{trading_prefix} Failed to close call position: {e}')
                    
                    a = [call_name, call_price, 'SELL', 0, 0, spot_price, -info.get(name)['call_buy']['quantity']]
                    info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                    info.get(name)['call_buy']['quantity'] = 0  # Update quantity
                    info.get(name)['call_buy']['trade_flag'] = 0 # Update flag
                    info.get(name)['condition'] = False
                    info.get(name)['next_min'] = False
                    # Set wait for next candle close based on candle_time
                    candle_interval = int(candle_time)

                    next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
                    if next_candle >= 60:
                        next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
                    else:
                        next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
                    info.get(name)['wait_until'] = next_candle_time
                    profit_pct = ((call_price - call_buy_price) / call_buy_price) * 100
                    logger.info(f'{trading_prefix} [CALL SELL] 📈 Profitable exit: {call_name} | Buy: ₹{call_buy_price} → Sell: ₹{call_price} | Profit: {profit_pct:.2f}% | Next trade after: {next_candle_time}')

                if put_flag==1:
                    # Execute real order if in real trading mode
                    if is_real_trading:
                        try:
                            response = take_position(put_name, -1, info.get(name)['put_buy']['quantity'])
                            logger.info(f'{trading_prefix} [PUT SELL] Order placed: {response}')
                        except Exception as e:
                            logger.error(f'{trading_prefix} Failed to close put position: {e}')
                    
                    a = [put_name, put_price, 'SELL', 0, 0, spot_price, -info.get(name)['put_buy']['quantity'] ]
                    info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                    info.get(name)['put_buy']['quantity'] = 0  # Update quantity
                    info.get(name)['put_buy']['trade_flag'] = 0  # Update flag
                    info.get(name)['condition'] = False
                    info.get(name)['next_min'] = False
                    # Set wait for next candle close based on candle_time
                    candle_interval = int(candle_time)

                    next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
                    if next_candle >= 60:
                        next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
                    else:
                        next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
                    info.get(name)['wait_until'] = next_candle_time
                    profit_pct = ((put_price - put_buy_price) / put_buy_price) * 100
                    logger.info(f'{trading_prefix} [PUT SELL] 📈 Profitable exit: {put_name} | Buy: ₹{put_buy_price} → Sell: ₹{put_price} | Profit: {profit_pct:.2f}% | Next trade after: {next_candle_time}')




            else :
                logger.info(f'{trading_prefix} [5-MIN UPDATE] Updating trailing stop-loss prices and setting next_min flag')
                

                if call_flag==1:
                    # Calculate new trailing stop based on current option price
                    new_call_stop = call_price - stop_point
                    if info.get(name)['call_buy']['current_stop_price']:
                        # Use max for trailing stop (move stop up, never down)
                        t = max(new_call_stop, info.get(name)['call_buy']['current_stop_price'])
                    else:
                        t = new_call_stop
                
                    info.get(name)['call_buy']['current_stop_price'] = t
                    info.get(name)['next_min'] = True 
                    logger.info(f'{trading_prefix} [CALL STOP UPDATE] New trailing SL: ₹{t} | Option: {call_name} | Current price: ₹{call_price} | New calculated stop: ₹{new_call_stop}')

                if put_flag==1:
                    # Calculate new trailing stop based on current option price
                    new_put_stop = put_price - stop_point
                    if info.get(name)['put_buy']['current_stop_price']:
                        # Use max for trailing stop (move stop up, never down)
                        t = max(new_put_stop, info.get(name)['put_buy']['current_stop_price'])
                    else:
                        t = new_put_stop
                    info.get(name)['put_buy']['current_stop_price'] = t
                    info.get(name)['next_min'] = True 
                    logger.info(f'{trading_prefix} [PUT STOP UPDATE] New trailing SL: ₹{t} | Option: {put_name} | Current price: ₹{put_price} | New calculated stop: ₹{new_put_stop}')

            
            await asyncio.sleep(1)

    
        # Check Take Profit FIRST - highest priority
        elif condition and call_flag==1 and info.get(name)['call_buy']['current_profit_price'] < call_price:
            logger.info(f'{trading_prefix} [CALL TARGET HIT] Closing position due to profit target achieved')
            
            # Execute real order if in real trading mode
            if is_real_trading:
                try:
                    response = take_position(call_name, -1, info.get(name)['call_buy']['quantity'])
                    logger.info(f'{trading_prefix} [CALL SELL] Order placed: {response}')
                except Exception as e:
                    logger.error(f'{trading_prefix} Failed to close call position at target: {e}')
            
            a= [call_name, call_price, 'SELL', 0, 0, spot_price, -info.get(name)['call_buy']['quantity']]
            info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
            info.get(name)['call_buy']['quantity'] = 0  # Update quantity
            info.get(name)['call_buy']['trade_flag'] = 0 # Update flag
            info.get(name)['condition'] = False
            info.get(name)['next_min'] = False
            # Set wait for next candle close
            candle_interval = int(candle_time)

            next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
            if next_candle >= 60:
                next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
            info.get(name)['wait_until'] = next_candle_time
            profit_pct = ((call_price - call_buy_price) / call_buy_price) * 100
            logger.info(f'{trading_prefix} [CALL SELL] 🎯 TARGET HIT: {call_name} | Buy: ₹{call_buy_price} → Sell: ₹{call_price} | Target: ₹{info.get(name)["call_buy"]["current_profit_price"]} | Profit: {profit_pct:.2f}% | Next trade: {next_candle_time}')
            
        elif condition and put_flag==1 and info.get(name)['put_buy']['current_profit_price'] < put_price:
            logger.info(f'{trading_prefix} [PUT TARGET HIT] Closing position due to profit target achieved')
            
            # Execute real order if in real trading mode
            if is_real_trading:
                try:
                    response = take_position(put_name, -1, info.get(name)['put_buy']['quantity'])
                    logger.info(f'{trading_prefix} [PUT SELL] Order placed: {response}')
                except Exception as e:
                    logger.error(f'{trading_prefix} Failed to close put position at target: {e}')
            
            a= [put_name, put_price, 'SELL', 0, 0, spot_price, -info.get(name)['put_buy']['quantity'] ]
            info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
            info.get(name)['put_buy']['quantity'] = 0  # Update quantity
            info.get(name)['put_buy']['trade_flag'] = 0  # Update flag
            info.get(name)['condition'] = False
            info.get(name)['next_min'] = False
            # Set wait for next candle close
            candle_interval = int(candle_time)

            next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
            if next_candle >= 60:
                next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
            else:
                next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
            info.get(name)['wait_until'] = next_candle_time
            profit_pct = ((put_price - put_buy_price) / put_buy_price) * 100
            logger.info(f'{trading_prefix} [PUT SELL] 🎯 TARGET HIT: {put_name} | Buy: ₹{put_buy_price} → Sell: ₹{put_price} | Target: ₹{info.get(name)["put_buy"]["current_profit_price"]} | Profit: {profit_pct:.2f}% | Next trade: {next_candle_time}')
 
        # Check call sell condition - AFTER take profit
        elif condition and call_flag == 1 and info.get(name)['next_min']:
            current_stop_price=info.get(name)['call_buy']['current_stop_price']
            if call_price < current_stop_price:
                # Pure stop loss hit
                logger.info(f'{trading_prefix} [CALL EXIT] Closing due to stop loss hit')
                
                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(call_name, -1, info.get(name)['call_buy']['quantity'])
                        logger.info(f'{trading_prefix} [CALL SELL] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to close call position: {e}')
                
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, -info.get(name)['call_buy']['quantity']]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['call_buy']['quantity'] = 0  # Update quantity
                info.get(name)['call_buy']['trade_flag'] = 0 # Update flag
                info.get(name)['condition'] = False
                info.get(name)['next_min'] = False
                # Set wait for next candle close
                candle_interval = int(candle_time)

                next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
                if next_candle >= 60:
                    next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
                info.get(name)['wait_until'] = next_candle_time
                pnl = ((call_price - call_buy_price) / call_buy_price) * 100
                logger.info(f'{trading_prefix} [CALL SELL] 🛑 STOP LOSS: {call_name} | Buy: ₹{call_buy_price} → Sell: ₹{call_price} | SL: ₹{current_stop_price} | P&L: {pnl:.2f}% | Next trade: {next_candle_time}')
            elif call_price >= call_buy_price:
                # Breakeven or profit - separate condition for clarity
                logger.info(f'{trading_prefix} [CALL EXIT] Closing due to breakeven/profit reached')
                
                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(call_name, -1, info.get(name)['call_buy']['quantity'])
                        logger.info(f'{trading_prefix} [CALL SELL] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to close call position: {e}')
                
                a = [call_name, call_price, 'SELL', 0, 0, spot_price, -info.get(name)['call_buy']['quantity']]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['call_buy']['quantity'] = 0  # Update quantity
                info.get(name)['call_buy']['trade_flag'] = 0 # Update flag
                info.get(name)['condition'] = False
                info.get(name)['next_min'] = False
                # Set wait for next candle close
                candle_interval = int(candle_time)

                next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
                if next_candle >= 60:
                    next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
                info.get(name)['wait_until'] = next_candle_time
                pnl = ((call_price - call_buy_price) / call_buy_price) * 100
                logger.info(f'{trading_prefix} [CALL SELL] ⚖️ BREAKEVEN/PROFIT: {call_name} | Buy: ₹{call_buy_price} → Sell: ₹{call_price} | SL: ₹{current_stop_price} | P&L: {pnl:.2f}% | Next trade: {next_candle_time}')


        # Check put sell condition
        elif condition and put_flag == 1 and info.get(name)['next_min']:
            current_stop_price=info.get(name)['put_buy']['current_stop_price']
            if put_price < current_stop_price:
                # Pure stop loss hit
                logger.info(f'{trading_prefix} [PUT EXIT] Closing due to stop loss hit')
                
                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(put_name, -1, info.get(name)['put_buy']['quantity'])
                        logger.info(f'{trading_prefix} [PUT SELL] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to close put position: {e}')
                
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, -info.get(name)['put_buy']['quantity'] ]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['put_buy']['quantity'] = 0  # Update quantity
                info.get(name)['put_buy']['trade_flag'] = 0  # Update flag
                info.get(name)['condition'] = False
                info.get(name)['next_min'] = False
                # Set wait for next candle close
                candle_interval = int(candle_time)

                next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
                if next_candle >= 60:
                    next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
                info.get(name)['wait_until'] = next_candle_time
                pnl = ((put_price - put_buy_price) / put_buy_price) * 100
                logger.info(f'{trading_prefix} [PUT SELL] 🛑 STOP LOSS: {put_name} | Buy: ₹{put_buy_price} → Sell: ₹{put_price} | SL: ₹{current_stop_price} | P&L: {pnl:.2f}% | Next trade: {next_candle_time}')
            elif put_price >= put_buy_price:
                # Breakeven or profit - separate condition for clarity
                logger.info(f'{trading_prefix} [PUT EXIT] Closing due to breakeven/profit reached')
                
                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(put_name, -1, info.get(name)['put_buy']['quantity'])
                        logger.info(f'{trading_prefix} [PUT SELL] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to close put position: {e}')
                
                a = [put_name, put_price, 'SELL', 0, 0, spot_price, -info.get(name)['put_buy']['quantity'] ]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['put_buy']['quantity'] = 0  # Update quantity
                info.get(name)['put_buy']['trade_flag'] = 0  # Update flag
                info.get(name)['condition'] = False
                info.get(name)['next_min'] = False
                # Set wait for next candle close
                candle_interval = int(candle_time)

                next_candle = ((ct.minute // candle_interval) + 1) * candle_interval
                if next_candle >= 60:
                    next_candle_time = ct.replace(hour=ct.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_candle_time = ct.replace(minute=next_candle, second=0, microsecond=0)
                info.get(name)['wait_until'] = next_candle_time
                pnl = ((put_price - put_buy_price) / put_buy_price) * 100
                logger.info(f'{trading_prefix} [PUT SELL] ⚖️ BREAKEVEN/PROFIT: {put_name} | Buy: ₹{put_buy_price} → Sell: ₹{put_price} | SL: ₹{current_stop_price} | P&L: {pnl:.2f}% | Next trade: {next_candle_time}')


        if info.get(name)['condition']==False :

            # Check if we need to wait for next candle close
            wait_until = info.get(name).get('wait_until')
            if wait_until and ct < wait_until:
                return
            
            if call_trend1==1 and call_trend2==1 and( call_price>call_high1 or call_price>call_high2):
                info.get(name)['spot_buy']=spot_price
                #calculate quantity
                q=int(fyers.funds().get('fund_limit')[-1].get('equityAmount'))
                if q==0:
                    q=money
                # Prevent division by zero
                if call_price <= 0:
                    logger.error(f'{trading_prefix} Invalid call price: {call_price}')
                    return
                quantity = int(q / call_price)
                #multiple of lotsize
                quantity = (quantity // lot_size) * lot_size
                
                # Minimum quantity check
                if quantity <= 0:
                    logger.error(f'{trading_prefix} Insufficient funds for trade. Available: ₹{q}, Option price: ₹{call_price}')
                    return

                info.get(name)['call_buy']['quantity'] = quantity
                logger.info(f'{trading_prefix} [QUANTITY CALC] CALL quantity calculated: {quantity} lots | Available funds: ₹{q} | Option price: ₹{call_price}')
                info.get(name)['call_buy']['buy_price'] = call_price
                # Initialize stop loss based on option price, not spot price
                call_stop_price = call_price - stop_point
                info.get(name)['call_buy']['current_stop_price'] = call_stop_price
                

                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(call_name, 1, quantity)
                        logger.info(f'{trading_prefix} [CALL BUY] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to place call buy order: {e}')
                        return

                a = [call_name, call_price, 'BUY', call_stop_price, 0, spot_price, quantity]
                info.get(name)['filled_df'].loc[ct] = a  # Save to dataframe
                info.get(name)['call_buy']['trade_flag'] = 1  # Update call flag
                info.get(name)['put_buy']['trade_flag'] = 3  # Update put flag
                #current profit price
                info.get(name)['call_buy']['current_profit_price'] = call_price + target
                # Clear the wait_until flag
                info.get(name)['wait_until'] = None

                logger.info(f'{trading_prefix} [CALL BUY] ✅ Entry signal: {call_name} | Price: ₹{call_price} | Qty: {quantity} | Spot: ₹{spot_price} | Target: ₹{call_price + target} | Trends: {call_trend1},{call_trend2} | Highs: ₹{call_high1},₹{call_high2}')
                info.get(name)['condition'] = True 
                
            elif put_trend1==1 and put_trend2==1 and (put_price>put_high1 or put_price>put_high2):
                info.get(name)['spot_buy']=spot_price

                q=int(fyers.funds().get('fund_limit')[-1].get('equityAmount'))
                if q==0:
                    q=money
                # Prevent division by zero
                if put_price <= 0:
                    logger.error(f'{trading_prefix} Invalid put price: {put_price}')
                    return
                quantity = int(q / put_price)
                #multiple of lotsize
                quantity = (quantity // lot_size) * lot_size
                
                # Minimum quantity check
                if quantity <= 0:
                    logger.error(f'{trading_prefix} Insufficient funds for trade. Available: ₹{q}, Option price: ₹{put_price}')
                    return

                info.get(name)['put_buy']['quantity'] = quantity
                logger.info(f'{trading_prefix} [QUANTITY CALC] PUT quantity calculated: {quantity} lots | Available funds: ₹{q} | Option price: ₹{put_price}')
                info.get(name)['put_buy']['buy_price'] = put_price

                # Initialize stop loss based on option price, not spot price
                put_stop_price = put_price - stop_point
                info.get(name)['put_buy']['current_stop_price'] = put_stop_price

                # Execute real order if in real trading mode
                if is_real_trading:
                    try:
                        response = take_position(put_name, 1, quantity)
                        logger.info(f'{trading_prefix} [PUT BUY] Order placed: {response}')
                    except Exception as e:
                        logger.error(f'{trading_prefix} Failed to place put buy order: {e}')
                        return

                a = [put_name, put_price, 'BUY', put_stop_price, 0, spot_price, quantity]
                info.get(name)['filled_df'].loc[ct] = a  # Update dataframe
                info.get(name)['put_buy']['trade_flag'] = 1  # Update put flag
                info.get(name)['call_buy']['trade_flag'] = 3  # Update call flag
                #current profit price
                info.get(name)['put_buy']['current_profit_price'] = put_price + target
                # Clear the wait_until flag
                info.get(name)['wait_until'] = None
                logger.info(f'{trading_prefix} [PUT BUY] ✅ Entry signal: {put_name} | Price: ₹{put_price} | Qty: {quantity} | Spot: ₹{spot_price} | Target: ₹{put_price + target} | Trends: {put_trend1},{put_trend2} | Highs: ₹{put_high1},₹{put_high2}')
                print(f'Put buy condition satisfied: {put_name} at {put_price}')
            
                info.get(name)['condition'] = True 
            



        if (ct.second ==1 and  ct.minute in range(0,60,2)):
            # Reduced frequency for supertrend updates (every 2 minutes instead of every minute)
            # Use cached data for faster processing
            t1,t2,ch1,ch2=get_supertrend_value(call_name)
            info.get(name).get('call_buy').update({'trend1':t1,'trend2':t2,'high1':ch1,'high2':ch2})
            t1,t2,ph1,ph2=get_supertrend_value(put_name)
            info.get(name).get('put_buy').update({'trend1':t1,'trend2':t2,'high1':ph1,'high2':ph2})
 


        # Save filled dataframes to CSV files
        if not info.get(name)['filled_df'].empty:
            info.get(name)['filled_df'].to_csv(f'{strategy_name}_{dt.now(time_zone).date()}.csv')

        # Store info using pickle
        store(info, account_type)












pnl=0




async def check_close_position(position_df):
    global real_info
    names=real_info.get('exit_instrument')
    current_pos=position_df['symbol'].to_list()
    new_list=[]
    for n in names:
        if n in current_pos:
            d1 = {"id": n + "-INTRADAY"}
            response = fyers.exit_positions(data=d1)
            new_list.append(n)
            logger.info(f'closing position {n}')
    real_info['exit_instrument']=new_list

async def track_pnl():
    if track_pnl_strategy:
        response = await fyers_asysc.positions()
        current_pnl=response.get('overall').get('pl_total')
        print('current pnl is',current_pnl)
        if current_pnl>daily_target_pnl:
            try:
                logger.info(f'current pnl {current_pnl} is greater than target pnl {daily_target_pnl}, exiting all positions and stopping the strategy')
                #close all position and stop the strategy
                pos1 = await fyers_asysc.positions()
                if pos1['netPositions']:
                    position_df=pd.DataFrame(pos1['netPositions'])
                    await check_close_position(position_df)
                logger.info('all positions closed, stopping the strategy')
                sys.exit()
            except Exception as e:
                logging.info(e)
                data = {}
                response = fyers.exit_positions(data=data)
                logging.info(response)
                print('stopping the entire code')
                sys.exit()




async def main_strategy_code():
    global df
    global paper_info
    global fyers_socket
    global new_symbols_list
    logger.info('🔄 Starting main strategy code execution')
    
    # await paper_order(f"{exchange}:{index_name}-INDEX")
    while True:
        ct = dt.now(time_zone)  # Get the current time
        print('current_system time',ct)
        # print(df)

        
        # Check if we have exchange data
        if underlying_ticker in df.index:
            try:
                exch_time = df.loc[underlying_ticker, 'exch_feed_time']
                
                if pd.isna(exch_time) or exch_time == 0:
                    print('exchange data time: Waiting for live data from exchange...')
                else:
                    print('exchange data time:', dt.from_timestamp(exch_time, tz=time_zone))
            except (KeyError, TypeError, ValueError) as e:
                print(f'exchange data time: Error processing timestamp - {e}')
        else:
            print(f'underlying_ticker {underlying_ticker} not found in DataFrame')
            print(f'Available symbols in df: {list(df.index)[:5]}...')


        # Check for missing symbols and resubscribe if needed (reduced frequency)
        if ct.second == 0 and ct.minute % 2 == 0:  # Every 2 minutes instead of every minute
            missing_symbols = []
            for symbol in new_symbols_list:
                if symbol not in df.index or pd.isna(df.loc[symbol, 'ltp']) or df.loc[symbol, 'ltp'] <= 0:
                    missing_symbols.append(symbol)
            
            if missing_symbols:
                logger.warning(f'Missing or invalid data for symbols: {missing_symbols}')
                try:
                    # Resubscribe to missing symbols
                    fyers_socket.subscribe(symbols=missing_symbols, data_type="SymbolUpdate")
                    logger.info(f'Resubscribed to missing symbols: {missing_symbols}')
                except Exception as e:
                    logger.error(f'Failed to resubscribe to symbols: {e}')

        # Get current PnL and chase order every 5 seconds
        # if ct.second in range(0, 59, 5):
        #     try:
        #         # Fetch positions asynchronously
        #         pos1 = await fyers_asysc.positions()
        #         if pos1['netPositions']:
        #             position_df=pd.DataFrame(pos1['netPositions'])
        #             await check_close_position(position_df)



        #     except:
        #         # Print error message if unable to fetch PnL or chase order
        #         print('unable to fetch pnl or chase order')

        #     # Print the current PnL
        #     # print("current_pnl", pnl)

        # Run strategy if DataFrame is not empty and has required symbols
        if df.shape[0] != 0 and underlying_ticker in df.index:
            # logger.info(f'📈 DataFrame has data ({df.shape[0]} symbols), proceeding with strategy')
            # Check if we have price data for the underlying and options
            try:
                spot_price = df.loc[underlying_ticker, 'ltp']
                
                # Optimized stale data check - only check every 10 seconds instead of every iteration
                if ct.second % 10 == 0:
                    try:
                        await track_pnl()
                        exchange_time = dt.from_timestamp(df.loc[underlying_ticker, 'exch_feed_time'], tz=time_zone)
                        time_diff = (ct - exchange_time).total_seconds()
                        if time_diff > 300:  # 5 minutes
                            logger.warning(f'Stale data detected. Time diff: {time_diff}s')
                            await asyncio.sleep(0.1)
                            continue
                    except (KeyError, TypeError, ValueError):
                        pass  # Skip stale data check if exchange time not available
                
                if pd.isna(spot_price) or spot_price <= 0:
                    logger.warning(f'Invalid spot price: {spot_price}')
                    await asyncio.sleep(1)
                    return

                # Execute paper order or real order based on account type
                if account_type == 'PAPER':
                    await paper_order(underlying_ticker)
                    # await paper_order1(f"{exchange}:{index_name}-INDEX")
                    # sys.exit(0)
                else:
                 
                    await real_order(underlying_ticker)
                    


            except KeyError as e:
                logger.warning(f'Missing data for symbol: {e}')
            except Exception as e:
                logger.error(f'Error in strategy execution: {e}')
        else:
            logger.warning(f'📊 Waiting for data - DataFrame shape: {df.shape}, Has underlying: {underlying_ticker in df.index if len(df) > 0 else False}')


        # Sleep for 0.1 seconds for faster response time (instead of 1 second)
        await asyncio.sleep(1)

time.sleep(5)



async def main():
    while True:
        await main_strategy_code()

asyncio.run(main())




