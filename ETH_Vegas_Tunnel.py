import pandas as pd
import os
import gate_api
from gate_api.exceptions import ApiException
import logging

# 日志等级设置
LOGLEVEL = os.getenv("LOGLEVEL", "DEBUG").upper()  # 可通过环境变量控制
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=getattr(logging, LOGLEVEL, logging.DEBUG)
)
logger = logging.getLogger(__name__)

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
# 更新 get_market_data 函数，确保时间范围覆盖足够的数据点

def get_market_data(timeframe='5m'):
    api_instance = gate_api.FuturesApi(api_client)
    settle = 'usdt'  # Settle currency
    contract = 'ETH_USDT'  # Trading pair (can be changed as needed)
    limit = 500  # Fetch the latest 500 candlestick data
    try:
        # Validate the contract name by fetching available contracts
        contracts = api_instance.list_futures_contracts(settle=settle)
        if contract not in [c.name for c in contracts]:
            logger.error("Contract '%s' not found. Please check the contract name.", contract)
            return None

        # Get futures candlesticks with proper parameters
        import time
        current_time = int(time.time())  # Current time in Unix timestamp

        # 动态调整时间范围以满足数据点需求
        if timeframe == '1h':
            interval_seconds = 60 * 60  # 1 hour in seconds
            required_points = 100
        elif timeframe == '15m':
            interval_seconds = 15 * 60  # 15 minutes in seconds
            required_points = 150
        elif timeframe == '5m':
            interval_seconds = 5 * 60  # 5 minutes in seconds
            required_points = 200
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        from_timestamp = current_time - (interval_seconds * required_points)
        to_timestamp = current_time

        # Fetch candlestick data with the correct parameters
        api_response = api_instance.list_futures_candlesticks(
            settle=settle,
            contract=contract,
            _from=from_timestamp,
            to=to_timestamp,
            interval=timeframe
        )

        # 验证返回的数据点数量
        if len(api_response) < required_points:
            logger.error("Insufficient data points for %s. Required: %d, Returned: %d", timeframe, required_points, len(api_response))
            return None

        # 打印调试信息
        logger.debug("Fetched %d data points for timeframe %s.", len(api_response), timeframe)

        return api_response
    except ApiException as e:
        logger.error("Exception when calling list_futures_candlesticks: %s", e)
        return None

# 在 parse_market_data 函数中添加对原始数据的详细调试信息

def parse_market_data(api_response):
    """
    解析 API 返回的市场数据为 DataFrame。
    :param api_response: API 返回的市场数据
    :return: 解析后的 DataFrame
    """
    try:
        logger.debug("Raw API response (first 5 items): %s", api_response[:5])
        logger.debug("Total items in API response: %d", len(api_response))

        data = []
        for item in api_response:
            if isinstance(item, dict):
                data.append({
                    "timestamp": float(item.get("t", 0)),
                    "low": float(item.get("l", 0)),
                    "high": float(item.get("h", 0)),
                    "last": float(item.get("c", 0)),
                    "vol": float(item.get("v", 0))
                })
            else:
                data.append({
                    "timestamp": float(getattr(item, "t", 0)),
                    "low": float(getattr(item, "l", 0)),
                    "high": float(getattr(item, "h", 0)),
                    "last": float(getattr(item, "c", 0)),
                    "vol": float(getattr(item, "v", 0))
                })
        df = pd.DataFrame(data)
        logger.debug("Parsed DataFrame row count: %d", len(df))
        logger.debug("Parsed DataFrame (first 5 rows):\n%s", df.head())
        return df
    except Exception as e:
        logger.error("Error parsing market data: %s", e)
        return pd.DataFrame()

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

# 在 calculate_indicators 函数中添加更多调试信息

def calculate_indicators(data):
    df = pd.DataFrame(data)
    logger.debug("输入数据框行数：%d", len(df))  # 打印数据框行数
    logger.debug("输入数据框内容（前几行）：\n%s", df.head())  # 打印数据框内容

    # 验证数据点数量
    if len(df) < 14:
        logger.error("Insufficient data points for indicator calculations. Required: 14, Available: %d", len(df))
        return pd.DataFrame()

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
    # 如果没有RSI等关键指标，先返回0分并报错
    required_cols = ['RSI', 'EMA144', 'EMA169', 'macd', 'signal', 'last']
    for col in required_cols:
        if col not in df.columns:
            logger.error("'%s' column missing in DataFrame. Please check indicator calculation.", col)
            return 0, 0

    # 初始化打分
    longScore = 0
    shortScore = 0

    # 打印数据框的最后几行以调试数据问题
    logger.debug("调试信息：数据框最后几行\n%s", df.tail())

    # 确保获取到足够的数据点以进行指标计算
    if len(df) < 14:
        logger.error("Insufficient data points for indicator calculations. Please fetch more data.")
        return 0, 0

    # 只检查最新一行RSI是否为NaN
    if pd.isna(df['RSI'].iloc[-1]):
        logger.error("Latest RSI is NaN. Please ensure sufficient data points.")
        return 0, 0

    # 优化打分逻辑以避免冲突
    logger.debug("多单打分：价格在隧道上方")
    if df['last'].iloc[-1] > df['EMA144'].iloc[-1] and df['last'].iloc[-1] > df['EMA169'].iloc[-1]:
        longScore += 30
        logger.debug("指标满足: last=%f > EMA144=%f and last=%f > EMA169=%f，得分 +30", df['last'].iloc[-1], df['EMA144'].iloc[-1], df['last'].iloc[-1], df['EMA169'].iloc[-1])
    else:
        logger.debug("指标不满足: last=%f <= EMA144=%f or last=%f <= EMA169=%f，得分 0", df['last'].iloc[-1], df['EMA144'].iloc[-1], df['last'].iloc[-1], df['EMA169'].iloc[-1])

    # 空单打分（1小时K线）
    logger.debug("空单打分：价格在隧道下方")
    if df['last'].iloc[-1] < df['EMA144'].iloc[-1] and df['last'].iloc[-1] < df['EMA169'].iloc[-1]:
        shortScore += 30
        logger.debug("指标满足: last=%f < EMA144=%f and last=%f < EMA169=%f，得分 +30", df['last'].iloc[-1], df['EMA144'].iloc[-1], df['last'].iloc[-1], df['EMA169'].iloc[-1])
    else:
        logger.debug("指标不满足: last=%f >= EMA144=%f or last=%f >= EMA169=%f，得分 0", df['last'].iloc[-1], df['EMA144'].iloc[-1], df['last'].iloc[-1], df['EMA169'].iloc[-1])

    # 15分钟K线
    logger.debug("多单打分：MACD金叉")
    if df['macd'].iloc[-1] > df['signal'].iloc[-1]:  # MACD金叉
        longScore += 20
        logger.debug("指标满足: macd=%f > signal=%f，得分 +20", df['macd'].iloc[-1], df['signal'].iloc[-1])
    else:
        logger.debug("指标不满足: macd=%f <= signal=%f，得分 0", df['macd'].iloc[-1], df['signal'].iloc[-1])

    logger.debug("多单打分：RSI上勾")
    if df['RSI'].iloc[-1] > 40 and df['RSI'].iloc[-1] < 70:  # RSI 上勾
        longScore += 20
        logger.debug("指标满足: RSI=%f > 40 and RSI=%f < 70，得分 +20", df['RSI'].iloc[-1], df['RSI'].iloc[-1])
    else:
        logger.debug("指标不满足: RSI=%f <= 40 or RSI=%f >= 70，得分 0", df['RSI'].iloc[-1], df['RSI'].iloc[-1])

    logger.debug("空单打分：MACD死叉")
    if df['macd'].iloc[-1] < df['signal'].iloc[-1]:  # MACD死叉
        shortScore += 20
        logger.debug("指标满足: macd=%f < signal=%f，得分 +20", df['macd'].iloc[-1], df['signal'].iloc[-1])
    else:
        logger.debug("指标不满足: macd=%f >= signal=%f，得分 0", df['macd'].iloc[-1], df['signal'].iloc[-1])

    logger.debug("空单打分：RSI下弯")
    if df['RSI'].iloc[-1] < 50:  # RSI 下弯
        shortScore += 20
        logger.debug("指标满足: RSI=%f < 50，得分 +20", df['RSI'].iloc[-1])
    else:
        logger.debug("指标不满足: RSI=%f >= 50，得分 0", df['RSI'].iloc[-1])

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

# 动态滚动止盈止损建议
def suggest_trailing_sl_tp(current_price, atr, risk_factor=1):
    trailing_sl = current_price - atr * risk_factor  # 滚动止损
    trailing_tp = current_price + atr * risk_factor * 2  # 滚动止盈
    return trailing_sl, trailing_tp

# 市场状态检测模块
def detect_market_state(df):
    atr = calculate_atr(df['high'], df['low'], df['last']).iloc[-1]
    bollinger_width = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()).iloc[-1]
    if bollinger_width > atr * 2:
        return "trend"  # 趋势市场
    else:
        return "range"  # 震荡市场

# 动量背离检测模块
def detect_momentum_divergence(df):
    # 检查关键列是否存在
    if 'macd' not in df.columns or 'last' not in df.columns:
        logger.error("'macd' or 'last' column missing in DataFrame. Please check indicator calculation.")
        return 0  # 缺失关键指标，无法检测
    if len(df) < 2:
        return 0  # 数据不足，无法检测
    price_diff = df['last'].iloc[-1] - df['last'].iloc[-2]
    macd_diff = df['macd'].iloc[-1] - df['macd'].iloc[-2]
    if price_diff > 0 and macd_diff < 0:
        logger.debug("检测到看空背离，附加空单得分 +10")
        return -10  # 看空背离
    elif price_diff < 0 and macd_diff > 0:
        logger.debug("检测到看多背离，附加多单得分 +10")
        return 10  # 看多背离
    return 0

# 更新 Vegas 隧道判断逻辑，增加区间扩张和收窄趋势判断

def analyze_vegas_tunnel(df):
    # 检查关键列是否存在
    required_cols = ['EMA144', 'EMA169', 'last']
    for col in required_cols:
        if col not in df.columns:
            logger.error("'%s' column missing in DataFrame. Please check indicator calculation.", col)
            return "insufficient_data", "insufficient_data"
    # 确保数据足够
    if len(df) < 100:
        logger.error("Insufficient data points for Vegas Tunnel analysis.")
        return "insufficient_data", "insufficient_data"

    # 判断价格与隧道的位置关系
    price_above_tunnel = (df['last'] > df['EMA144']) & (df['last'] > df['EMA169'])
    price_below_tunnel = (df['last'] < df['EMA144']) & (df['last'] < df['EMA169'])

    # 判断隧道形态
    ema_diff = df['EMA144'] - df['EMA169']
    ema_diff_positive = ema_diff > 0
    ema_diff_negative = ema_diff < 0

    # 判断隧道方向
    ema144_slope = df['EMA144'].diff().mean()
    ema169_slope = df['EMA169'].diff().mean()

    if ema144_slope > 0 and ema169_slope > 0 and ema_diff_positive.all():
        trend = "uptrend"  # 上涨趋势
    elif ema144_slope < 0 and ema169_slope < 0 and ema_diff_negative.all():
        trend = "downtrend"  # 下跌趋势
    elif ema_diff_positive.any() and ema_diff_negative.any():
        trend = "cross"  # 隧道交叉
    else:
        trend = "sideways"  # 横盘震荡

    # 判断区间变化（扩张或收窄）
    ema_diff_slope = ema_diff.diff().mean()
    if ema_diff_slope > 0:
        range_change = "expanding"  # 区间扩张
    elif ema_diff_slope < 0:
        range_change = "contracting"  # 区间收窄
    else:
        range_change = "stable"  # 区间稳定

    return trend, range_change

# 更新数据验证和错误处理逻辑

def validate_data(df, required_points):
    """
    验证数据点是否足够。
    :param df: 数据框
    :param required_points: 所需的最小数据点数量
    :return: 如果数据足够返回 True，否则返回 False
    """
    if len(df) < required_points:
        logger.error("Insufficient data points. Required: %d, Available: %d", required_points, len(df))
        return False
    return True

# 主函数
def main():
    # 获取5分钟K线数据
    market_data_5min = get_market_data(timeframe='5m')
    if market_data_5min is None:
        return
    if not market_data_5min:
        logger.error("No data returned for 5-minute candlesticks. Please check the API parameters or data availability.")
        return
    
    # 获取1小时K线数据
    market_data_hour = get_market_data(timeframe='1h')
    if market_data_hour is None:
        return
    
    # 获取15分钟K线数据
    market_data_15min = get_market_data(timeframe='15m')
    if market_data_15min is None:
        return
    
    # 解析全部K线数据
    df_5min = parse_market_data(market_data_5min)
    df_hour = parse_market_data(market_data_hour)
    df_15min = parse_market_data(market_data_15min)

    # 计算技术指标
    df_5min = calculate_indicators(df_5min)
    df_hour = calculate_indicators(df_hour)
    df_15min = calculate_indicators(df_15min)

    # 在主函数中添加数据验证
    if not validate_data(df_5min, 14):
        logger.error("5分钟数据不足，无法计算指标。")
        return

    if not validate_data(df_hour, 100):
        logger.error("1小时数据不足，无法进行 Vegas 隧道分析。")
        return

    if not validate_data(df_15min, 14):
        logger.error("15分钟数据不足，无法计算指标。")
        return
    
    # 打分（基于5分钟K线、1小时和15分钟K线）
    longScore, shortScore = calculate_scores(df_15min)

    # 在打分逻辑中加入动量背离检测
    longScore += detect_momentum_divergence(df_15min)
    shortScore += detect_momentum_divergence(df_15min)

    # 检测市场状态并动态调整策略
    market_state = detect_market_state(df_15min)
    logger.info("当前市场状态：%s", market_state)
    if 'macd' in df_15min.columns and 'signal' in df_15min.columns:
        if market_state == "trend":
            logger.info("趋势市场：优先使用突破策略")
            # 在趋势市场中，增加对突破信号的权重
            if df_15min['macd'].iloc[-1] > df_15min['signal'].iloc[-1]:
                longScore += 10
                logger.debug("趋势市场中，MACD金叉附加多单得分 +10")
            if df_15min['macd'].iloc[-1] < df_15min['signal'].iloc[-1]:
                shortScore += 10
                logger.debug("趋势市场中，MACD死叉附加空单得分 +10")
        else:
            logger.info("震荡市场：优先使用反转策略")
            # 在震荡市场中，增加对反转信号的权重
            if 'RSI' in df_15min.columns:
                if df_15min['RSI'].iloc[-1] > 70:
                    shortScore += 10
                    logger.debug("震荡市场中，RSI超买附加空单得分 +10")
                if df_15min['RSI'].iloc[-1] < 30:
                    longScore += 10
                    logger.debug("震荡市场中，RSI超卖附加多单得分 +10")
    else:
        logger.warning("'macd' or 'signal' column missing in df_15min, 跳过趋势/震荡市场加分逻辑。")

    # 止损、止盈、风报比（基于5分钟K线）
    stopLossLong, stopLossShort, tp1Long, tp2Long, tp1Short, tp2Short, rrLong, rrShort = calculate_sl_tp(df_5min)
    
    # 在主函数中调用 Vegas 隧道分析（提前，保证变量已定义）
    trend_direction, range_change = analyze_vegas_tunnel(df_hour)

    # 输出打分和开仓建议（格式化输出，便于市场操作参考）
    logger.info("\n===== 策略信号输出 =====")
    logger.info("当前市场状态: %s", market_state)
    logger.info("1H 趋势方向: %s | 1H 区间变化: %s", trend_direction, range_change)
    logger.info("15m Long Score: %d | Short Score: %d", longScore, shortScore)
    logger.info("最新价格: %f", df_15min['last'].iloc[-1])
    if longScore >= 70:
        logger.info("【建议开多单】")
        logger.info("开仓价: %f", df_15min['last'].iloc[-1])
        logger.info("止损: %f", stopLossLong)
        logger.info("TP1: %f  风报比: %.2f", tp1Long, rrLong)
        logger.info("TP2: %f  风报比: %.2f", tp2Long, rrLong * 1.2)
        trailing_sl, trailing_tp = suggest_trailing_sl_tp(df_15min['last'].iloc[-1], calculate_atr(df_15min['high'], df_15min['low'], df_15min['last']).iloc[-1])
        logger.info("滚动止损: %f", trailing_sl)
        logger.info("滚动止盈: %f", trailing_tp)
    if shortScore >= 70:
        logger.info("【建议开空单】")
        logger.info("开仓价: %f", df_15min['last'].iloc[-1])
        logger.info("止损: %f", stopLossShort)
        logger.info("TP1: %f  风报比: %.2f", tp1Short, rrShort)
        logger.info("TP2: %f  风报比: %.2f", tp2Short, rrShort * 1.2)
        trailing_sl, trailing_tp = suggest_trailing_sl_tp(df_15min['last'].iloc[-1], calculate_atr(df_15min['high'], df_15min['low'], df_15min['last']).iloc[-1])
        logger.info("滚动止损: %f", trailing_sl)
        logger.info("滚动止盈: %f", trailing_tp)
    if longScore < 70 and shortScore < 70:
        logger.info("当前无明确开仓信号，请耐心等待下一信号。")
    logger.info("===== 策略信号输出结束 =====\n")

if __name__ == "__main__":
    main()
