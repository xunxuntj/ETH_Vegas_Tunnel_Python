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
def candle_pattern_score(df):
    # 只看最后两根或三根K线
    if len(df) < 3:
        return 0, "无形态"
    # 取最后三根K线的开收
    close1, close2, close3 = df['last'].iloc[-3], df['last'].iloc[-2], df['last'].iloc[-1]
    open1, open2, open3 = close1, close2, close3  # 若有open列可替换
    high1, high2, high3 = df['high'].iloc[-3], df['high'].iloc[-2], df['high'].iloc[-1]
    low1, low2, low3 = df['low'].iloc[-3], df['low'].iloc[-2], df['low'].iloc[-1]
    # 阳吞阴
    if close3 > open3 and close2 < open2 and close3 > open2 and open3 < close2:
        return 2, "阳吞阴"
    # 阴吞阳
    if close3 < open3 and close2 > open2 and close3 < open2 and open3 > close2:
        return 2, "阴吞阳"
    # 锤子线
    if close3 > open3 and (open3 - low3) > 2 * abs(close3 - open3) and (high3 - close3) < abs(close3 - open3):
        return 2, "锤子线"
    # 上吊线
    if close3 < open3 and (open3 - low3) > 2 * abs(close3 - open3) and (high3 - open3) < abs(close3 - open3):
        return 2, "上吊线"
    # 十字星
    if abs(close3 - open3) < (high3 - low3) * 0.1:
        return 2, "十字星"
    # 早晨之星
    if close1 < open1 and abs(close2 - open2) < (high2 - low2) * 0.2 and close3 > open3 and close3 > (open1 + close1) / 2:
        return 2, "早晨之星"
    # 黄昏之星
    if close1 > open1 and abs(close2 - open2) < (high2 - low2) * 0.2 and close3 < open3 and close3 < (open1 + close1) / 2:
        return 2, "黄昏之星"
    # 乌云盖顶
    if close2 > open2 and close3 < open3 and open3 > close2 and close3 < (open2 + close2) / 2:
        return 2, "乌云盖顶"
    # 刺透形态
    if close2 < open2 and close3 > open3 and open3 < close2 and close3 > (open2 + close2) / 2:
        return 2, "刺透形态"
    # 三连阳
    if close1 < close2 < close3 and close1 > open1 and close2 > open2 and close3 > open3:
        return 2, "三连阳"
    # 三连阴
    if close1 > close2 > close3 and close1 < open1 and close2 < open2 and close3 < open3:
        return 2, "三连阴"
    return 0, "无形态"

def calculate_scores(df, trend_score):
    """
    统一采用小分制（每项2分，满分10分），与README一致。
    1H趋势、MACD、RSI、K线形态、动量背离各占2分。
    7分及以上为强信号。
    EMA21过滤：15m收盘价需在EMA21上方（多头）或下方（空头）才允许加分，否则直接返回0分。
    """
    required_cols = ['RSI', 'EMA144', 'EMA169', 'macd', 'signal', 'last', 'EMA21']
    for col in required_cols:
        if col not in df.columns:
            logger.error("'%s' column missing in DataFrame. Please check indicator calculation.", col)
            return 0, 0, {}
    longScore = 0
    shortScore = 0
    details = {}
    if pd.isna(df['RSI'].iloc[-1]):
        logger.error("Latest RSI is NaN. Please ensure sufficient data points.")
        return 0, 0, details
    # EMA21过滤
    last = df['last'].iloc[-1]
    ema21 = df['EMA21'].iloc[-1]
    if last < ema21:
        details['ema21_filter'] = '15m收盘价未站上EMA21，long信号无效'
        return 0, shortScore, details
    if last > ema21:
        details['ema21_filter'] = '15m收盘价未跌破EMA21，short信号无效'
        return longScore, 0, details
    # 1H趋势分数
    if trend_score == 2:
        longScore += 2
        details['1h_trend'] = '上涨'
    elif trend_score == -2:
        shortScore += 2
        details['1h_trend'] = '下跌'
    else:
        details['1h_trend'] = '震荡'
    # MACD
    if df['macd'].iloc[-1] > df['signal'].iloc[-1]:
        longScore += 2
        details['macd'] = '金叉'
    else:
        shortScore += 2
        details['macd'] = '死叉'
    # RSI
    if df['RSI'].iloc[-1] > 50:
        longScore += 2
        details['rsi'] = '上钩'
    else:
        shortScore += 2
        details['rsi'] = '下弯'
    # K线形态
    candle_score, candle_type = candle_pattern_score(df)
    if candle_type == '阳吞阴':
        longScore += 2
    elif candle_type == '阴吞阳':
        shortScore += 2
    details['candle'] = candle_type
    # 动量背离
    momentum = detect_momentum_divergence(df)
    if momentum > 0:
        longScore += 2
        details['momentum_divergence'] = True
    elif momentum < 0:
        shortScore += 2
        details['momentum_divergence'] = True
    else:
        details['momentum_divergence'] = False
    return longScore, shortScore, details

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
        logger.debug("检测到看空背离，附加空单得分 +2")
        return -2  # 看空背离
    elif price_diff < 0 and macd_diff > 0:
        logger.debug("检测到看多背离，附加多单得分 +2")
        return 2  # 看多背离
    return 0

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

# 更新 Vegas 隧道判断逻辑，增加区间扩张和收窄趋势判断

def analyze_vegas_tunnel(df):
    required_cols = ['EMA144', 'EMA169', 'last']
    for col in required_cols:
        if col not in df.columns:
            logger.error("'%s' column missing in DataFrame. Please check indicator calculation.", col)
            return "insufficient_data", "insufficient_data", 0
    if len(df) < 100:
        logger.error("Insufficient data points for Vegas Tunnel analysis.")
        return "insufficient_data", "insufficient_data", 0
    ema_diff = df['EMA144'] - df['EMA169']
    ema144_slope = df['EMA144'].diff().mean()
    ema169_slope = df['EMA169'].diff().mean()
    price = df['last'].iloc[-1]
    ema144 = df['EMA144'].iloc[-1]
    ema169 = df['EMA169'].iloc[-1]
    # 趋势分数
    trend_score = 0
    if ema144_slope > 0 and ema169_slope > 0 and (price > ema144 and price > ema169) and (ema_diff.iloc[-1] > ema_diff.iloc[-20]):
        trend = "uptrend"
        trend_score = 2
    elif ema144_slope < 0 and ema169_slope < 0 and (price < ema144 and price < ema169) and (ema_diff.iloc[-1] > ema_diff.iloc[-20]):
        trend = "downtrend"
        trend_score = -2
    elif abs(ema_diff.iloc[-1]) < 2 or (abs(ema144_slope) < 0.1 and abs(ema169_slope) < 0.1):
        trend = "sideways"
        trend_score = 0
    else:
        trend = "cross"
        trend_score = 0
    ema_diff_slope = ema_diff.diff().mean()
    if ema_diff_slope > 0:
        range_change = "expanding"
    elif ema_diff_slope < 0:
        range_change = "contracting"
    else:
        range_change = "stable"
    return trend, range_change, trend_score

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

# 5分钟周期信号确认。direction: 'long' or 'short'
# 以5m最后一根K线的macd与signal判断方向。

def confirm_5m_signal(df_5min, direction):
    """
    5分钟周期信号确认。direction: 'long' or 'short'
    以5m最后一根K线的macd与signal判断方向。
    """
    if len(df_5min) < 2:
        return False
    if 'macd' not in df_5min.columns or 'signal' not in df_5min.columns:
        return False
    if direction == 'long' and df_5min['macd'].iloc[-1] > df_5min['signal'].iloc[-1]:
        return True
    if direction == 'short' and df_5min['macd'].iloc[-1] < df_5min['signal'].iloc[-1]:
        return True
    return False

# 多周期MACD形态判断

def macd_cross_signal(df):
    """
    判断MACD金叉/死叉形态：
    - 刚刚金叉：前一根死叉，这一根金叉
    - 刚刚死叉：前一根金叉，这一根死叉
    - 持续金叉/死叉
    返回：'just_golden', 'just_dead', 'golden', 'dead', 'none'
    """
    if 'macd' not in df.columns or 'signal' not in df.columns or len(df) < 2:
        return 'none'
    prev_macd, prev_signal = df['macd'].iloc[-2], df['signal'].iloc[-2]
    curr_macd, curr_signal = df['macd'].iloc[-1], df['signal'].iloc[-1]
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        return 'just_golden'  # 刚刚金叉
    elif prev_macd >= prev_signal and curr_macd < curr_signal:
        return 'just_dead'    # 刚刚死叉
    elif curr_macd > curr_signal:
        return 'golden'      # 持续金叉
    elif curr_macd < curr_signal:
        return 'dead'        # 持续死叉
    else:
        return 'none'

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
    trend_direction, range_change, trend_score = analyze_vegas_tunnel(df_hour)
    longScore, shortScore, signal_details = calculate_scores(df_15min, trend_score)

    # 5m确认
    long_5m = confirm_5m_signal(df_5min, 'long')
    short_5m = confirm_5m_signal(df_5min, 'short')

    # 输出打分和开仓建议（格式化输出，便于市场操作参考）
    output = {
        "time": pd.to_datetime(df_15min['timestamp'].iloc[-1], unit='s').strftime('%Y-%m-%d %H:%M'),
        "price": float(df_15min['last'].iloc[-1]),
        "long_score": int(longScore),
        "short_score": int(shortScore),
        "recommendation": "",
        "take_profit": None,
        "stop_loss": None,
        "signal_details": signal_details
    }
    # 止盈止损
    stopLossLong, stopLossShort, tp1Long, tp2Long, tp1Short, tp2Short, rrLong, rrShort = calculate_sl_tp(df_5min)
    # 推荐建议
    if longScore >= 7 and long_5m:
        output["recommendation"] = "多头强势，建议开多单"
        output["take_profit"] = float(tp1Long)
        output["stop_loss"] = float(stopLossLong)
    elif shortScore >= 7 and short_5m:
        output["recommendation"] = "空头强势，建议开空单"
        output["take_profit"] = float(tp1Short)
        output["stop_loss"] = float(stopLossShort)
    else:
        output["recommendation"] = "无明确信号，观望为主"
    # 在主函数中分别判断1h、15m、5m的MACD形态，并综合决策
    macd_1h = macd_cross_signal(df_hour)
    macd_15m = macd_cross_signal(df_15min)
    macd_5m = macd_cross_signal(df_5min)
    macd_multi = {
        '1h': macd_1h,
        '15m': macd_15m,
        '5m': macd_5m
    }
    output['macd_multi_signal'] = macd_multi
    # 多周期共振决策
    if macd_1h in ['just_golden', 'golden'] and macd_15m in ['just_golden', 'golden'] and macd_5m in ['just_golden', 'golden']:
        output['multi_timeframe_macd'] = '三周期金叉共振，强多信号'
    elif macd_1h in ['just_dead', 'dead'] and macd_15m in ['just_dead', 'dead'] and macd_5m in ['just_dead', 'dead']:
        output['multi_timeframe_macd'] = '三周期死叉共振，强空信号'
    else:
        output['multi_timeframe_macd'] = '多周期信号不一致，观望为主'

    # ===== 增强可读性输出 =====
    logger.info("\n===== 策略信号结构化输出（小分制，满分10分，7分及以上为强信号） =====")
    logger.info("时间: %s", output['time'])
    logger.info("最新价格: %.2f", output['price'])
    logger.info("1H趋势: %s | 区间: %s", trend_direction, range_change)
    logger.info("15m Long Score: %d | Short Score: %d", output['long_score'], output['short_score'])
    logger.info("MACD形态: 1H=%s, 15m=%s, 5m=%s", macd_1h, macd_15m, macd_5m)
    logger.info("多周期MACD共振: %s", output['multi_timeframe_macd'])
    logger.info("信号细节: %s", output['signal_details'])
    if output['recommendation'] == "多头强势，建议开多单":
        logger.info("【建议开多单】 开仓价: %.2f 止损: %.2f 止盈: %.2f", output['price'], output['stop_loss'], output['take_profit'])
    elif output['recommendation'] == "空头强势，建议开空单":
        logger.info("【建议开空单】 开仓价: %.2f 止损: %.2f 止盈: %.2f", output['price'], output['stop_loss'], output['take_profit'])
    else:
        logger.info("【建议】%s", output['recommendation'])
    logger.info("===== END =====\n")

if __name__ == "__main__":
    main()
