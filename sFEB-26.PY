# Replace these values with your actual API credentials
client_id = 'DF26CD57LX-100'
secret_key = '5SS5D9X5TR'
redirect_uri = 'https://fessorpro.com/'
response_type = "code"  
state = "sample_state"


secondary_data=True
if secondary_data:
    api_key = 'z59mkhj6yg8b6c81'
    access_token_kite = 'jnQRoinTwJNGYPrCoa5k0nuVKCtdGzXY'
threshold=12
take_type={'FIXED':1,'PERCENTAGE':2,'RANGE':3}
T1 = 'RANGE'; v1 = 0.3
T2 = 'RANGE'; v2 = 1    ;v2_pos=0.5
T3 = 'RANGE'; v3 = 1.5  ;v3_pos=0.5
T4 = 'RANGE'; v4 = 2    ;v4_pos=0
T5 = 'RANGE'; v5 = 3    ;v5_pos=0
wait_buffer=120
lot_size=65
quantity_position=2
candle_1='5'
candle_2='15'
candle_3='60'

min_15_condition=True
min_60_condition=True
index_15_condition=True

rsi_1=14;rsi_smooth_1=14
rsi_2=14;rsi_smooth_2=14
rsi_3=14;rsi_smooth_3=14
atr_length=14
strategy_name='nikita'
account_type='LIVE'
time_zone="Asia/Kolkata"
start_hour,start_min=9,30
end_hour,end_min=15,5

index_name='NIFTY50'
exchange='NSE'
type2='INDEX'
fyers_underlying_index=f"{exchange}:{index_name}-{type2}"
kite_index={'NIFTY50':256265,'NIFTYBANK':260105}
kite_underlying_index_token=kite_index.get(index_name)
symbol_list=['NIFTY2630225550CE']
fyers_initials='NSE:'
exchange_kite='NFO'   
access_token_fyers = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcG4tTVVVS3ZPMjljRXNJcmRVczl3RjhaX0M2NzRVQmd1RVk1WDJ2UUhSYzNSX2FHdS15al9JZzg5OHNwOUFQOUVVaXpyWlNKV0doaXo3dC1VTXZYbEx2cUxsb1RMSkVXRVR1aEw0aFZMR3cyQjdHTT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiJmM2IwNzA2NmQ5Yjc5ODc5MWU0OTBjMjA0NmIyNjI5OWEyNGEyOWU3NDUwODNjNjZmYzI5Y2NiZCIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWFM0NTQ3NCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzcyMTUyMjAwLCJpYXQiOjE3NzIwODYwMzYsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc3MjA4NjAzNiwic3ViIjoiYWNjZXNzX3Rva2VuIn0.bgZHr1njLEF0TQN5N8_fuUq4T55UwscmMVAzUdol_gE"



global final_data
final_data={symbol:{} for symbol in symbol_list}
final_data[index_name]={}

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

    return rsi.round(2)

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

global df_5,df_15,df_60,index_5,index_15,index_60

# Initialize dataframes as None
df_5 = None
df_15 = None
df_60 = None
index_5 = None
index_15 = None
index_60 = None

# Function to store data using pickle
def store(data, account_type):
    pickle.dump(data, open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'wb'))

# Function to load data using pickle
def load(account_type):
    return pickle.load(open(f'data-{dt.now(time_zone).date()}-{account_type}.pickle', 'rb'))

try:
        real_info = load(account_type)
        if real_info.get(symbol_list[0]) is None:
            logging.info("No existing data found for symbol - initializing new real_info structure")
            print("No existing data found for symbol - initializing new real_info structure")
            real_info = { symbol_list[0]:{
                'trade_flag':0,
                'buy_price':0,
                'quantity':0,
                'pnl':0,        
                'condition':False,
                'wait_until':None,
                'wait_until_flag':False,
                'high':0,
                'low':0,
                'range':0,
                "targets":{},
                'crossover_active':False,
                'trade_taken_on_crossover':False,
                'order_time':None,
                't1_executed':False,
                't2_executed':False,
                't3_executed':False
            }}
        else:
            # Backwards compatibility: add missing flags if they don't exist
            if 't1_executed' not in real_info[symbol_list[0]]:
                real_info[symbol_list[0]]['t1_executed'] = False
            if 't2_executed' not in real_info[symbol_list[0]]:
                real_info[symbol_list[0]]['t2_executed'] = False
            if 't3_executed' not in real_info[symbol_list[0]]:
                real_info[symbol_list[0]]['t3_executed'] = False
        

except:

            real_info = { symbol_list[0]:{
                'trade_flag':0,
                'buy_price':0,
                'quantity':0,
                'pnl':0,        
                'condition':False,
                'wait_until':None,
                'wait_until_flag':False,
                'high':0,
                'low':0,
                'range':0,
                "targets":{},
                'crossover_active':False,
                'trade_taken_on_crossover':False,
                'order_time':None,
                't1_executed':False,
                't2_executed':False,
                't3_executed':False
            }}


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
                logging.info(f"Active order found: {order.get('id')} - Status: {order.get('status')}")
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking active order: {e}")
        return False
    
def target_helper(range, buy_price, take_type, v):
    if take_type == 'FIXED':
        return buy_price + v
    elif take_type == 'PERCENTAGE':
        return buy_price + (buy_price * v / 100)
    elif take_type == 'RANGE':
        return buy_price + (range * v)
    else:
        raise ValueError("Invalid take_type. Must be 'FIXED', 'PERCENTAGE', or 'RANGE'.")

def get_targets(range,buy_price):
    if T1=='RANGE':
        if range<threshold:
            t1=buy_price+(range*v1)
        elif range>=threshold:
            t1=buy_price+5
    else:
        t1=target_helper(range, buy_price, T1, v1)


    if T2=='RANGE':
        if range<threshold:
            t2=buy_price+(range*v2)
        elif range>=threshold:
            t2=buy_price+(0.5*range)
    else:
        t2=target_helper(range, buy_price, T2, v2)
    t3=target_helper(range, buy_price, T3, v3)
    t4=target_helper(range, buy_price, T4, v4)
    t5=target_helper(range, buy_price, T5, v5)
    return t1,t2,t3,t4,t5

def sell_partial_quantity(ticker, quantity_to_sell):
    """Sell a partial quantity of the position for the given ticker"""
    try:
        data = {
            "symbol": f"{fyers_initials}{ticker}",
            "qty": quantity_to_sell*lot_size,
            "type": 2,  # Market order
            "side": -1,  # Sell
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "orderTag": f"partial_exit_{quantity_to_sell}",
            "isSliceOrder": False
        }
        response = fyers.place_order(data=data)
        logging.info(f'Partial sell order response: {response}')
    except Exception as e:
        logging.error(f'Error placing partial sell order: {e}')

trade_flag=0
first=1
last_high=0

def main_strategy_second():
    global trade_flag,first,last_high,real_info
    global df_5,df_15,df_60,index_5,index_15,index_60
    current_time=dt.now(time_zone)
        
    if (current_time.second==1 and current_time.minute % 5 == 0) or first==1:
        first=0
        # Fetch OHLC data for all symbols
        for symbol in symbol_list:
            symbol_fyers = fyers_initials + symbol
            logging.info(f"\n--- Fetching data for {symbol} ---")
            df_5=fetchOHLC(symbol_fyers,candle_1,10)
            df_5['rsi']=rsi(df_5['close'], length=rsi_1)
            df_5['rsi_smooth']=rsi_ma(df_5['rsi'])
            df_5['rsi_flag'] = (df_5['rsi'] > df_5['rsi_smooth']).astype(bool)
            df_5.to_csv('5min.csv')
            df_15=fetchOHLC(symbol_fyers,candle_2,10)
            df_15['rsi']=rsi(df_15['close'], length=rsi_2)
            df_15['rsi_smooth']=rsi_ma(df_15['rsi'])
            df_15['rsi_flag'] = (df_15['rsi'] > df_15['rsi_smooth']).astype(bool)
            df_60=fetchOHLC(symbol_fyers,candle_3,10)
            df_60['rsi']=rsi(df_60['close'], length=rsi_3)
            df_60['rsi_smooth']=rsi_ma(df_60['rsi'])
            df_60['rsi_flag'] = (df_60['rsi'] > df_60['rsi_smooth']).astype(bool)
            print(f"df_5 tail:\n{df_5.tail()}")
            print(f"df_15 tail:\n{df_15.tail()}")
            print(f"df_60 tail:\n{df_60.tail()}")
        #underlying index data
        index_5=fetchOHLC(fyers_underlying_index,candle_1,10)
        index_5['rsi']=rsi(index_5['close'], length=rsi_1)
        index_5['rsi_smooth']=rsi_ma(index_5['rsi'])
        index_5['rsi_flag']=index_5['rsi']>index_5['rsi_smooth']
        index_15=fetchOHLC(fyers_underlying_index,candle_2,10)
        index_15['rsi']=rsi(index_15['close'], length=rsi_2)
        index_15['rsi_smooth']=rsi_ma(index_15['rsi'])
        index_15['rsi_flag'] = (index_15['rsi'] > index_15['rsi_smooth']).astype(bool)
        index_60=fetchOHLC(fyers_underlying_index,candle_3,10)
        index_60['rsi']=rsi(index_60['close'], length=rsi_3)
        index_60['rsi_smooth']=rsi_ma(index_60['rsi'])
        index_60['rsi_flag'] = (index_60['rsi'] > index_60['rsi_smooth']).astype(bool)  
        # logging.info(f"index_5 tail:\n{index_5.tail()}")
        # logging.info(f"index_15 tail:\n{index_15.tail()}")
        # logging.info(f"index_60 tail:\n{index_60.tail()}")

    

    ticker=symbol_list[0]
    ticker_price=final_data[ticker].get('ltp', 'N/A')
    print(dt.now(time_zone),ticker,ticker_price)
    trade_flag=real_info[ticker]['trade_flag']

    # Validate dataframes are initialized and have enough data
    if df_5 is None or len(df_5) < 2:
        logging.info("df_5 not initialized or insufficient data")
        return

    # Detect RSI crossover (fresh crossover = RSI crosses above RSI_MA)
    rsi_above_ma_prev = df_5['rsi'].iloc[-3] > df_5['rsi_smooth'].iloc[-3] if len(df_5) >= 3 else False
    rsi_above_ma_current = df_5['rsi'].iloc[-2] > df_5['rsi_smooth'].iloc[-2] if len(df_5) >= 2 else False
    
    # Fresh crossover detected: RSI crosses from below to above MA
    if not rsi_above_ma_prev and rsi_above_ma_current:
        logging.info('Fresh RSI crossover detected!')
        real_info[ticker]['crossover_active'] = True
        real_info[ticker]['trade_taken_on_crossover'] = False
    
    # Crossover ended: RSI crosses back below MA
    elif rsi_above_ma_prev and not rsi_above_ma_current:
        logging.info('RSI crossed back below MA - resetting crossover flags')
        real_info[ticker]['crossover_active'] = False
        real_info[ticker]['trade_taken_on_crossover'] = False
    
    # Detect ongoing crossover (for startup/restart scenarios)
    # If crossover_active is False but RSI is currently above MA, activate it
    elif not real_info[ticker]['crossover_active'] and rsi_above_ma_current:
        logging.info('Ongoing RSI crossover detected (startup/restart) - activating crossover flag')
        print("Ongoing RSI crossover detected (startup/restart) - activating crossover flag")
        real_info[ticker]['crossover_active'] = True
        real_info[ticker]['trade_taken_on_crossover'] = False

    #entry condition - only trade on fresh crossover
    if (trade_flag == 0 and 
        real_info[ticker]['crossover_active'] and 
        not real_info[ticker]['trade_taken_on_crossover'] and 
        df_5['rsi_flag'].iloc[-2] and 
        ticker_price != 'N/A'):
        
        current_high=df_5['high'].iloc[-2]
        current_low=df_5['low'].iloc[-2]
        print('rsi is above ma',current_high,ticker_price)
        # logging.info(f'Entry condition met - current_high: {current_high}, current_low: {current_low}')
        
        if ticker_price>=current_high  and  ((df_15['rsi_flag'].iloc[-1]==True and min_15_condition) or (df_60['rsi_flag'].iloc[-1]==True and min_60_condition) or (index_15['rsi_flag'].iloc[-1]==True and index_15_condition)):
            logging.info(f'Placing buy order - ticker_price: {ticker_price} >= current_high: {current_high}')

          
            data = {
                "symbol":fyers_initials+ticker,
                "qty":lot_size*quantity_position,
                "type":2,
                "side":1,
                "productType":"INTRADAY",
                "limitPrice":0,
                "stopPrice":0,
                "validity":"DAY",
                "disclosedQty":0,
                "offlineOrder":False,
                "orderTag":f"{int(current_high)}d{int(current_low)}",
                "isSliceOrder":False
            }
            try:
                response = fyers.place_order(data=data)
                logging.info(f'Order response: {response}')
                trade_flag=1
                real_info[ticker]['trade_flag']=1
                real_info[ticker]['buy_price']=ticker_price
                real_info[ticker]['high']=current_high
                real_info[ticker]['low']=current_low
                real_info[ticker]['range']=current_high - current_low
                real_info[ticker]['trade_taken_on_crossover']=True  # Mark that trade was taken on this crossover
                real_info[ticker]['order_time']=dt.now(time_zone)  # Track when order was placed
                real_info[ticker]['t1_executed']=False
                real_info[ticker]['t2_executed']=False
                real_info[ticker]['t3_executed']=False
                t1,t2,t3,t4,t5=get_targets(real_info[ticker]['range'],real_info[ticker]['buy_price'])
                real_info[ticker]['targets']={'t1':t1,'t2':t2,'t3':t3,'t4':t4,'t5':t5}
                logging.info(f'Trade taken on crossover - Targets: t1={t1}, t2={t2}, t3={t3}, t4={t4}, t5={t5}')
                logging.info(f'Crossover flag locked until RSI crosses back below MA')
            except Exception as e:
                logging.error(f'Error placing buy order: {e}')

    #already open position
    elif trade_flag==1 and ticker_price != 'N/A':
        print(f"Trade already placed - ticker_price: {ticker_price}, t2: {real_info[ticker]['targets'].get('t2')}, low: {real_info[ticker]['low']}")
        if trade_flag == 1 and (check_active_position(ticker) or check_active_order(ticker)):
            print(f"Active position or order detected for {ticker} - current price: {ticker_price}")
        else:
            # Allow 30 seconds buffer after order placement before resetting trade_flag
            order_time = real_info[ticker].get('order_time')
            if order_time and (dt.now(time_zone) - order_time).total_seconds() < 30:
                logging.info(f"Waiting for order to be filled... ({(dt.now(time_zone) - order_time).total_seconds():.1f}s elapsed)")
                return 0
            else:
                logging.info(f"No active position detected for {ticker} after 30s - resetting trade flag")
                trade_flag = 0
                real_info[ticker].update({
                    'trade_flag': 0,
                    'buy_price': 0,
                    'quantity': 0,'high': 0,'low': 0,'range': 0,'targets':{},'order_time':None,
                    't1_executed':False,'t2_executed':False,'t3_executed':False})
                return


        if ticker_price>real_info[ticker]['targets'].get('t3') and not real_info[ticker].get('t3_executed', False):
            logging.info(f'Target t3 reached - consider partial profit booking at {ticker_price}')
            # Implement partial profit booking logic here (e.g., exit half position)
            partial_quantity = quantity_position * v3_pos  # Calculate quantity to sell based on v3_pos
            sell_partial_quantity(ticker, int(partial_quantity))
            real_info[ticker]['t3_executed'] = True

        elif ticker_price>real_info[ticker]['targets'].get('t2') and not real_info[ticker].get('t2_executed', False):
            #close position
            logging.info(f'Target t2 reached - closing partial at {ticker_price}')
            partial_quantity = quantity_position * v2_pos  # Calculate quantity to sell based on v2_pos
            sell_partial_quantity(ticker, int(partial_quantity))
            real_info[ticker]['t2_executed'] = True
            
        elif ticker_price>real_info[ticker]['targets'].get('t1') and not real_info[ticker].get('t1_executed', False):
            logging.info(f'Target t1 reached - setting wait_until_flag at {ticker_price}')
            real_info[ticker]['wait_until_flag']=True
            real_info[ticker]['wait_until']=dt.now(time_zone)+dt.duration(seconds=wait_buffer)
            real_info[ticker]['t1_executed'] = True


        if ticker_price<(real_info[ticker]['low']):
            logging.info(f"Price {ticker_price} went below low {real_info[ticker]['low']} - closing position")
            try:
                fyers.exit_positions({'id':fyers_initials+ticker+"-INTRADAY"})
                trade_flag = 0  # Update global trade_flag
                real_info[ticker].update({
                    'trade_flag':0,
                    'buy_price':0,
                    'quantity':0,
                    'pnl':0,        
                    'condition':False,
                    'wait_until':None,
                    'wait_until_flag':False,
                    'high':0,
                    'low':0,
                    'range':0,
                    "targets":{},
                    'trade_taken_on_crossover':True,  # Prevent re-entry until fresh crossover
                    'order_time':None,
                    't1_executed':False,
                    't2_executed':False,
                    't3_executed':False
                })
                logging.info('position closed at stop loss')
            except Exception as e:
                logging.error(f'Error exiting position at stop loss: {e}')  
                logging.error('position close failed at stop loss')

    store(real_info,account_type)



 

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

        # print('fyers',message)
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
            # print('fyers message received')
            # print(message)
            
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

# ==================== KITE SOCKET (SECONDARY) ====================
def start_kite_socket(loop):
    """Start Kite WebSocket connection (runs in executor thread)"""
    if not secondary_data:
        logging.info("Kite socket disabled (secondary_data=False)")
        return
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
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
    print("kite tokens",instrument_tokens)
    
    # Initialize KiteTicker
    kws = KiteTicker(api_key, access_token_kite)

# kite data [{'tradable': False, 'mode': 'full', 'instrument_token': 256265, 'last_price': 25769.55, 'ohlc': {'high': 25885.3, 'low': 25763.0, 'open': 25873.35, 'close': 25819.35}, 'change': -0.19287859686630096, 'exchange_timestamp': datetime.datetime(2026, 2, 19, 10, 13, 16)}]

    def on_ticks(ws, ticks):
        # print('kite data',ticks)
   
        for tick in ticks:
            # Extract exchange time
            exchange_time = tick.get('exchange_timestamp', None)
            
            # Convert to epoch timestamp if it's a datetime object (do this FIRST)
            if isinstance(exchange_time, datetime):
                exchange_time = exchange_time.timestamp()
            elif exchange_time is None:
                exchange_time = 0
            
            token=tick.get('instrument_token', None)
            name=token_symbol.get(token, 'N/A')
            # print(name)

            if name==index_name:
                # print('index tick',tick)
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
                # Update final_data with index data
                # final_data[index_name] = index_data
                # print(index_data)
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
            # print('kite message received')
            # print(data)
            
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


# ==================== MAIN ====================
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

