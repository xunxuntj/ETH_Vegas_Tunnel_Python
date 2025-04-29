import pandas as pd
import os
import gate_api
from gate_api.exceptions import ApiException

# Configure Gate.io Futures API client
api_key = os.getenv("API_KEY")  # Read-only API Key
api_secret = os.getenv("API_SECRET")  # Read-only API Secret

# Set up Futures API client
configuration = gate_api.Configuration(
    host="https://api.gateio.ws/api/v4",
    key=api_key,
    secret=api_secret
)

api_client = gate_api.ApiClient(configuration)

# Function to fetch ETH/USDT perpetual contract data
def get_market_data(timeframe='5m'):
    api_instance = gate_api.FuturesApi(api_client)
    settle = 'usdt'  # Settle currency
    contract = 'ETH_USDT'  # Trading pair (can be changed as needed)
    limit = 500  # Fetch the latest 500 candlestick data
    try:
        # Validate the contract name by fetching available contracts
        contracts = api_instance.list_futures_contracts(settle=settle)

        # Debugging: Print available contracts
        print("Available contracts:")
        for c in contracts:
            print(c.name)  # Print the name of each contract

        if contract not in [c.name for c in contracts]:
            print(f"Error: Contract '{contract}' not found. Please check the contract name.")
            return None

        # Get futures candlesticks with proper parameters
        from_timestamp = 1546905600  # Example start time in Unix timestamp
        to_timestamp = 1546935600    # Example end time in Unix timestamp
        interval = '5m'              # Example interval

        try:
            # Fetch candlestick data with the correct parameters
            api_response = api_instance.list_futures_candlesticks(
                settle=settle,
                contract=contract,
                _from=from_timestamp,
                to=to_timestamp,
                interval=interval
            )
            return api_response
        except ApiException as e:
            print(f"Exception when calling list_futures_candlesticks: {e}")
            return None
    except ApiException as e:
        print(f"Exception when calling list_futures_candlesticks: {e}")
        return None

# Replace talib.EMA with pandas' rolling mean for EMA calculation
def calculate_ema(series, timeperiod):
    return series.ewm(span=timeperiod, adjust=False).mean()

# Replace talib.MACD with manual calculation
def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = calculate_ema(series, fastperiod)
    ema_slow = calculate_ema(series, slowperiod)
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# Replace talib.RSI with manual calculation
def calculate_rsi(series, timeperiod=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Replace talib.ATR with manual calculation
def calculate_atr(high, low, close, timeperiod=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    return atr

# 更新 calculate_indicators 以使用新函数
def calculate_indicators(data):
    df = pd.DataFrame(data)

    # 将各列数据类型转换为float以便进行计算
    df['last'] = df['last'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['vol'] = df['vol'].astype(float)

    # 计算EMA
    df['EMA21'] = calculate_ema(df['last'], timeperiod=21)
    df['EMA34'] = calculate_ema(df['last'], timeperiod=34)
    df['EMA144'] = calculate_ema(df['last'], timeperiod=144)
    df['EMA169'] = calculate_ema(df['last'], timeperiod=169)

    # 计算MACD
    df['macd'], df['signal'], df['hist'] = calculate_macd(df['last'])

    # 计算RSI
    df['RSI'] = calculate_rsi(df['last'])

    return df

# 计算打分
def calculate_scores(df):
    # 初始化打分
    longScore = 0
    shortScore = 0

    # 多单打分（1小时K线）
    if df['last'].iloc[-1] > df['EMA144'].iloc[-1] and df['last'].iloc[-1] > df['EMA169'].iloc[-1]:  # 价格在隧道上方
        longScore += 30
        print("多单打分：价格在隧道上方，得分 +30")
    
    # 空单打分（1小时K线）
    if df['last'].iloc[-1] < df['EMA144'].iloc[-1] and df['last'].iloc[-1] < df['EMA169'].iloc[-1]:  # 价格在隧道下方
        shortScore += 30
        print("空单打分：价格在隧道下方，得分 +30")
    
    # 15分钟K线
    if df['macd'].iloc[-1] > df['signal'].iloc[-1]:  # MACD金叉
        longScore += 20
        print("多单打分：MACD金叉，得分 +20")
    if df['RSI'].iloc[-1] > 40 and df['RSI'].iloc[-1] < 70:  # RSI 上勾
        longScore += 20
        print("多单打分：RSI上勾，得分 +20")
    
    if df['macd'].iloc[-1] < df['signal'].iloc[-1]:  # MACD死叉
        shortScore += 20
        print("空单打分：MACD死叉，得分 +20")
    if df['RSI'].iloc[-1] < 50:  # RSI 下弯
        shortScore += 20
        print("空单打分：RSI下弯，得分 +20")

    return longScore, shortScore

# 计算止损、止盈和风报比
def calculate_sl_tp(df, risk_factor=1):
    atr = calculate_atr(df['high'], df['low'], df['last']).iloc[-1]
    stopLossLong = df['last'].iloc[-1] - atr * risk_factor
    stopLossShort = df['last'].iloc[-1] + atr * risk_factor
    
    tp1Long = df['last'].iloc[-1] + atr * 2 * risk_factor
    tp2Long = df['last'].iloc[-1] + atr * 3 * risk_factor
    
    tp1Short = df['last'].iloc[-1] - atr * 2 * risk_factor
    tp2Short = df['last'].iloc[-1] - atr * 3 * risk_factor
    
    rrLong = (tp1Long - df['last'].iloc[-1]) / (df['last'].iloc[-1] - stopLossLong)
    rrShort = (df['last'].iloc[-1] - tp1Short) / (stopLossShort - df['last'].iloc[-1])
    
    return stopLossLong, stopLossShort, tp1Long, tp2Long, tp1Short, tp2Short, rrLong, rrShort

# 主函数
def main():
    # 获取5分钟K线数据
    market_data_5min = get_market_data(timeframe='5m')
    if market_data_5min is None:
        return
    
    # 获取1小时K线数据
    market_data_hour = get_market_data(timeframe='1h')
    if market_data_hour is None:
        return
    
    # 获取15分钟K线数据
    market_data_15min = get_market_data(timeframe='15m')
    if market_data_15min is None:
        return
    
    # 提取价格数据（1小时K线和15分钟K线）
    data_5min = [
        {
            "timestamp": market_data_5min[0].t,  # Corrected attribute for timestamp
            "low": market_data_5min[0].l,       # Corrected attribute for low price
            "high": market_data_5min[0].h,      # Corrected attribute for high price
            "last": market_data_5min[0].c,      # Corrected attribute for close price
            "change": None,                     # Placeholder, as change is not directly available
            "vol": market_data_5min[0].v,       # Corrected attribute for volume
            "close": market_data_5min[0].c      # Corrected attribute for close price
        }
    ]
    
    data_hour = [
        {
            "timestamp": market_data_hour[0].t,  # Corrected attribute for timestamp
            "low": market_data_hour[0].l,       # Corrected attribute for low price
            "high": market_data_hour[0].h,      # Corrected attribute for high price
            "last": market_data_hour[0].c,      # Corrected attribute for close price
            "change": None,                     # Placeholder, as change is not directly available
            "vol": market_data_hour[0].v,       # Corrected attribute for volume
            "close": market_data_hour[0].c      # Corrected attribute for close price
        }
    ]
    
    data_15min = [
        {
            "timestamp": market_data_15min[0].timestamp,
            "low": market_data_15min[0].low,
            "high": market_data_15min[0].high,
            "last": market_data_15min[0].last,
            "change": market_data_15min[0].change,
            "vol": market_data_15min[0].vol,
            "close": market_data_15min[0].last
        }
    ]
    
    # 计算技术指标（1小时和15分钟K线数据）
    df_5min = calculate_indicators(data_5min)
    df_hour = calculate_indicators(data_hour)
    df_15min = calculate_indicators(data_15min)
    
    # 打分（基于5分钟K线、1小时和15分钟K线）
    longScore, shortScore = calculate_scores(df_15min)
    
    # 止损、止盈、风报比（基于5分钟K线）
    stopLossLong, stopLossShort, tp1Long, tp2Long, tp1Short, tp2Short, rrLong, rrShort = calculate_sl_tp(df_5min)
    
    # 输出打分和开仓建议
    print(f"Long Score: {longScore}")
    if longScore >= 70:
        print(f"建议开多单")
        print(f"开单价：{df_5min['last'].iloc[-1]}")
        print(f"止损: {stopLossLong}")
        print(f"TP1: {tp1Long}  风报比: {rrLong}")
        print(f"TP2: {tp2Long}  风报比: {rrLong * 1.2}")
        print(f"建议TP1，因为是震荡市场")
    
    print(f"\nShort Score: {shortScore}")
    if shortScore >= 70:
        print(f"\n建议开空单")
        print(f"开单价：{df_5min['last'].iloc[-1]}")
        print(f"止损: {stopLossShort}")
        print(f"TP1: {tp1Short}  风报比: {rrShort}")
        print(f"TP2: {tp2Short}  风报比: {rrShort * 1.2}")
        print(f"建议TP1，因为是震荡市场")

if __name__ == "__main__":
    main()
