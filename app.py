


# Replace these values with your actual API credentials
client_id = 'DF26CD57LX-100'
secret_key = '5SS5D9X5TR'
redirect_uri = 'https://fessorpro.com/'
response_type = "code"  
state = "sample_state"

totp_key='6ITMUZAD7POI44YBWQFEGODOQAUAF3ZH'
import pyotp as tp
t=tp.TOTP(totp_key).now()
print(t)

index_name='NIFTY50'
exchange='NSE'
type2='INDEX'
ticker=f"{exchange}:{index_name}-{type2}"
underlying_ticker=f"{exchange}:{index_name}-{type2}"

capital=12000
lot_size=65
candle_1='5'
candle_2='15'
candle_3='60'
rsi_1=14
rsi_smooth_1=14
rsi_2=14
rsi_smooth_2=14
rsi_3=14
rsi_smooth_3=14
atr_length=14
track_pnl_strategy=True
strategy_name='nikita'
account_type='LIVE'
time_zone="Asia/Kolkata"
start_hour,start_min=9,20
end_hour,end_min=15,10

access_token_fyers = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcGtyRHJZZmRQT2lHTXhGc1FIUDhlRFZNMFR3T1hXQkwzVTJlWE1qenVQMGE5YVlOZVJqUUo3dEZLREFaV2VXRTZTamVQMlFGMGc4eVZmYmZjYTM4QlNBemRJUEhybkxkQkYzUWhjVHcybHVpZ0xwST0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5ZTZlMGVkMzU5OTY1MTU0MTllOGQ3YmE5MjE0NzYwNjc2YTNkMGZhNDI2NmY5MmM5NzRhNDBlMyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWFM0NTQ3NCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzcxMjg4MjAwLCJpYXQiOjE3NzEyMjEyMjcsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc3MTIyMTIyNywic3ViIjoiYWNjZXNzX3Rva2VuIn0.nUMcpG7uD8BNjc-1_TntSQc-7YSlTCpiEx0vt3GqSJc'


api_key = 'z59mkhj6yg8b6c81'
access_token_kite = 'mdi5tINmKt6Csy5NmEHBvntHYWRtKQk6'

# Symbol to track
symbol = 'CRUDEOILM26FEBFUT'
symbol_fyers = "MCX:" + symbol
exchange_kite='MCX'
symbol_kite = symbol    


# Kite credentials (Secondary)
api_key = 'z59mkhj6yg8b6c81'
access_token_kite = 'mdi5tINmKt6Csy5NmEHBvntHYWRtKQk6'


global final_data
final_data={symbol:{}}


# CRITICAL: Patch signal module to prevent errors in non-main threads
# This must be done before importing fyers_apiv3 (which uses Twisted)
import signal
import threading

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
import numpy as np
import certifi
from kiteconnect import KiteTicker, KiteConnect
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

# For Windows SSL error
os.environ['SSL_CERT_FILE'] = certifi.where()


# Get the current time
current_time=dt.now(time_zone)
start_time=dt.datetime(current_time.year,current_time.month,current_time.day,start_hour,start_min,tz=time_zone)
end_time=dt.datetime(current_time.year,current_time.month,current_time.day,end_hour,end_min,tz=time_zone)
print('start time:', start_time)
print('end time:', end_time)




# Initialize FyersModel instances for synchronous and asynchronous operations
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token_fyers, log_path=None)
fyers_asysc = fyersModel.FyersModel(client_id=client_id, is_async=True, token=access_token_fyers, log_path=None)

available_cash= fyers.funds()
print('available cash:', available_cash.get('fund_limit')[-1].get('equityAmount'))



def fetchOHLC(ticker,interval,duration):
    from datetime import date, timedelta
    """extracts historical data and outputs in the form of dataframe"""
    instrument = ticker
    data = {"symbol":instrument,"resolution":interval,"date_format":"1","range_from":date.today()-timedelta(duration),"range_to":date.today(),"cont_flag":"1",'oi_flag':"1"}
    sdata=fyers.history(data)
    # print(sdata)
    sdata=pd.DataFrame(sdata['candles'])
    # Check if 'oi' column exists (7 columns)
    if sdata.shape[1] == 7:
        sdata.columns=['date','open','high','low','close','volume','oi']
    else:
        sdata.columns=['date','open','high','low','close','volume']
    sdata['date']=pd.to_datetime(sdata['date'], unit='s')
    sdata.date=(sdata.date.dt.tz_localize('UTC').dt.tz_convert(time_zone))
    sdata['date'] = sdata['date'].dt.tz_localize(None)
    sdata=sdata.set_index('date')
    return sdata

def rma(series, length):
    """Rolling Mean Average (EMA with alpha=1/length)"""
    return series.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

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
        rsi.fillna(method=kwargs["fill_method"], inplace=True)

    return rsi.round(1)

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
    return ma.round(1)




async def strategy_logic():
    """Strategy logic that runs every second"""
    while True:
        try:
            global final_data
            if dt.now(time_zone).second in range(0,60,5):
                # Access final_data here for your strategy
                print(symbol,final_data.get(symbol).get('ltp', 'N/A'))
                
                # Convert epoch timestamp to string format
                if isinstance(final_data[symbol].get('timestamp'), (int, float)):
                    timestamp_str = dt.from_timestamp(final_data[symbol]['timestamp'], tz=time_zone).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp_str = str(final_data[symbol].get('timestamp', 'N/A'))
                print('system   time:',dt.now(time_zone).strftime('%Y-%m-%d %H:%M:%S'))
                print('exchange time:', timestamp_str)

    
            if dt.now(time_zone).second==1:
                df_5=fetchOHLC(symbol_fyers,candle_1,10)
                df_5['rsi_5']=rsi(df_5['close'], length=rsi_1)
                df_5['rsi_smooth_5']=rsi_ma(df_5['rsi_5'])
                df_15=fetchOHLC(symbol_fyers,candle_2,10)
                df_15['rsi_15']=rsi(df_15['close'], length=rsi_2)
                df_15['rsi_smooth_15']=rsi_ma(df_15['rsi_15'])
                df_60=fetchOHLC(symbol_fyers,candle_3,10)
                df_60['rsi_60']=rsi(df_60['close'], length=rsi_3)
                df_60['rsi_smooth_60']=rsi_ma(df_60['rsi_60'])
                print(df_5.tail())
                print(df_15.tail())
                print(df_60.tail())


            
            await asyncio.sleep(1)  # Run every second
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error in strategy logic: {e}")









# ==================== SHARED DATA ====================
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
                if self.fyers_timestamp > 0 and self.kite_timestamp > 0:
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
                elif self.kite_timestamp > 0:
                    source = "KITE (Secondary)"
                    display_data = self.kite_data
                    display_timestamp = self.kite_timestamp
                else:
                    continue
                
                # Only print if source changed
                if self.last_source != source:
                    # print(f"\n{'='*60}")
                    # print(f"SWITCHED TO: {source}")
                    # print(f"{'='*60}")
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
        
        # Save data to final_data
        final_data[symbol] = {
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
            'symbol': data.get('symbol', 'N/A')
        }
        
        # print(f"\n{'='*60}")
        # print(f"📊 DATA SOURCE: {source}")
        # print(f"{'='*60}")
        # print(f"Exchange Time: {time_str}")
        # print(f"Last Price: {data.get('ltp', 'N/A')}")
        # print(f"Volume: {data.get('volume', 'N/A')}")
        # print(f"Buy Price: {data.get('buy_price', 'N/A')}")
        # print(f"Buy Quantity: {data.get('buy_quantity', 'N/A')}")
        # print(f"Sell Price: {data.get('sell_price', 'N/A')}")
        # print(f"Sell Quantity: {data.get('sell_quantity', 'N/A')}")
        # print(f"Change: {data.get('change', 'N/A')}")
        # print(f"Open: {data.get('open', 'N/A')}")
        # print(f"High: {data.get('high', 'N/A')}")
        # print(f"Low: {data.get('low', 'N/A')}")
        # print(f"Close: {data.get('close', 'N/A')}")
        # print(f"Symbol: {data.get('symbol', 'N/A')}")
        # print(f"{'='*60}")

# Create shared data store
shared_data = SocketData()

# ==================== FYERS SOCKET (PRIMARY) ====================
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
        fyers_ws.subscribe(symbols=[symbol_fyers], data_type=data_type)
        logging.info(f"Subscribed to {symbol_fyers} on Fyers")
    
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

# ==================== KITE SOCKET (SECONDARY) ====================
def start_kite_socket(loop):
    """Start Kite WebSocket connection (runs in executor thread)"""
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token_kite)
    
    # Get instrument token
    def get_instrument_token(symbol_name, exchange='NFO'):
        try:
            instruments = kite.instruments(exchange)
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol_name:
                    logging.info(f"Found instrument: {instrument['tradingsymbol']} (Token: {instrument['instrument_token']})")
                    return instrument['instrument_token']
            logging.error(f"Symbol {symbol_name} not found in {exchange}")
            return None
        except Exception as e:
            logging.error(f"Error fetching instrument: {e}")
            return None
    
    instrument_token = get_instrument_token(symbol_kite,exchange_kite)
    if instrument_token is None:
        logging.error("Cannot start Kite socket - instrument token not found")
        return
    
    token_symbol = {instrument_token: f"{symbol_kite}"}
    
    # Initialize KiteTicker
    kws = KiteTicker(api_key, access_token_kite)
    
    def on_ticks(ws, ticks):
   
        for tick in ticks:
            # Extract exchange time
            exchange_time = tick.get('exchange_timestamp', None)
            
            # Convert to epoch timestamp if it's a datetime object
            if isinstance(exchange_time, datetime):
                exchange_time = exchange_time.timestamp()
            elif exchange_time is None:
                exchange_time = 0
            
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
        ws.subscribe([instrument_token])
        ws.set_mode(ws.MODE_FULL, [instrument_token])
        logging.info(f"Subscribed to {symbol_kite} on Kite")
    
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


# ==================== MAIN ====================
async def main():
    """Main async function to run both sockets"""
    print("\n" + "="*60)
    print("🚀 DUAL SOCKET HANDLER - STARTED (Asyncio)")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Primary Source: Fyers")
    print(f"Secondary Source: Kite")
    print(f"Strategy: Use whichever has latest exchange time")
    print("="*60 + "\n")
    
    # Initialize shared data in async context
    shared_data.initialize()
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    
    # Create thread pool executor for blocking websocket operations
    executor = ThreadPoolExecutor(max_workers=2)
    
    # Start data processor task
    processor_task = asyncio.create_task(shared_data.process_data_stream())
    
    # Start strategy logic task (runs every second)
    strategy_task = asyncio.create_task(strategy_logic())
    
    # Start both websockets in executor threads
    # Give Fyers a slight head start (primary source)
    fyers_task = loop.run_in_executor(executor, start_fyers_socket, loop)
    await asyncio.sleep(2)
    kite_task = loop.run_in_executor(executor, start_kite_socket, loop)

    
    try:
        # Run both websockets, processor, and strategy concurrently
        await asyncio.gather(fyers_task, kite_task, processor_task, strategy_task)
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

