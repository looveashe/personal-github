# -*- coding: utf-8 -*-
"""
板块龙头共振识别策略（仅识别信号）
功能: 每日收盘后识别符合条件的个股，输出候选列表

策略核心逻辑:
  第一步: 筛选主线板块（近3日涨幅前5% + 日均涨停>=3 + 有连板龙头）
  第二步: 圈定候选跟风股（与龙头高相关 + 异动 + 近高点 + 小市值）
  第三步: 技术信号共振确认（MACD金叉 + 布林/维加斯买点）
"""

import datetime
import time
import random
import json
import os
import sys
import io
import warnings
import re

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import akshare as ak

# ============================================================
# 多数据源支持（Tushare / Baostock 作为备用或增强）
# ============================================================
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    ts = None

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False
    bs = None

TUSHARE_PRO = None
_BS_LOGINED = False


def init_tushare(token: str) -> bool:
    """初始化 Tushare 接口，返回是否成功"""
    global TUSHARE_PRO
    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        TUSHARE_PRO = pro
        print(f"  Tushare 初始化成功")
        return True
    except Exception as e:
        print(f"  Tushare 初始化失败: {e}")
        return False

def init_baostock():
    """初始化 Baostock 登录"""
    global _BS_LOGINED
    if BAOSTOCK_AVAILABLE and not _BS_LOGINED:
        bs.login()
        _BS_LOGINED = True
    return _BS_LOGINED

warnings.filterwarnings('ignore')

# ============================================================
# TA-Lib 降级机制
# ============================================================
try:
    import talib

    HAS_TALIB = True
    print("TA-Lib 已加载，使用原生C加速计算")
except ImportError:
    HAS_TALIB = False
    print("TA-Lib 未安装，使用 pandas 降级实现，功能不受影响")


# ============================================================
# 策略参数配置
# ============================================================
@dataclass
class StrategyConfig:
    """策略参数汇总"""
    # 市场环境过滤
    use_index_filter: bool = True          # 是否开启大盘过滤
    index_code: str = "000300"             # 沪深300
    index_ma_period: int = 20              # 均线周期
    index_start_offset: int = 60           # 指数拉取天数

    # 仓位管理
    position_method: str = "equal"         # "equal" 或 "inverse_volatility"
    vol_lookback: int = 20                 # 波动率计算回看天数

    # 出场规则
    exit_use_macd: bool = True
    exit_use_boll_lower: bool = True
    exit_use_ma5: bool = True
    exit_ma5_period: int = 5  # 保留作为默认，实际由下面参数覆盖

    # 差异化出场：龙头用较长均线，跟风股用较短均线
    leader_exit_ma_period: int = 10       # 龙头10日均线
    follower_exit_ma_period: int = 5      # 跟风股5日均线

    # 龙头断板联动止损
    leader_break_exit_enabled: bool = True

    # 移动止盈（盈利 > threshold 触发，回撤 > drawdown 止盈）
    trailing_profit_threshold: float = 0.15   # 盈利15%以上
    trailing_drawdown_limit: float = 0.08     # 从最高点回撤8%止盈

    # 成分股活跃度预过滤（日均成交额、换手率）
    min_daily_amount: float = 5e7             # 5000万
    min_daily_turnover: float = 0.01          # 1%

    # 板块筛选（龙头路径）
    board_return_top_pct: float = 0.08         # 前8%涨幅阈值
    board_daily_limit_up_min: int = 2          # 日均涨停≥2
    leader_consecutive_days: int = 3           # 保留字段（实际龙头认定使用下方两个参数）
    top_sectors_count: int = 2                 # 取前N个板块

    # 龙头认定：近5日涨停≥2天 或 近10日涨停≥3天
    leader_zt_days_min_5: int = 2
    leader_zt_days_min_10: int = 3

    # 趋势路径阈值
    board_return_top_pct_trend: float = 0.30   # 前30%涨幅
    board_daily_limit_up_min_trend: float = 1.0 # 日均涨停≥1 或 近3日有过涨停（逻辑内判断）

    # 跟风股筛选
    corr_threshold: float = 0.8
    corr_lookback: int = 60
    follower_change_threshold: float = 0.03
    follower_volume_ratio: float = 1.0
    high_price_distance: float = 0.15
    high_price_lookback: int = 63
    market_cap_limit: float = 500

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_diff_threshold: float = -1.0

    # 布林带
    bbands_period: int = 20
    bbands_nbdev: int = 2
    bbands_volume_ratio: float = 1.0

    # 维加斯通道
    vegas_periods: Tuple[int, int, int] = (12, 144, 169)

    # 数据
    data_lookback_days: int = 400
    request_delay: float = 5.0           # 进一步提高延迟，避免频率限制
    max_retries: int = 3

    # 板块拥挤度过滤
    board_crowding_limit_ratio: float = 0.30      # 当日涨停成分股占比上限
    min_active_starters: int = 3                   # 未封板但已明显启动的最少标的数
    active_starter_min_rise: float = 0.05          # 明显启动的最低近3日涨幅

    # 筹码安全垫（前期涨幅上限）
    pre_event_days: int = 20
    pre_event_max_gain: float = 0.20


CONFIG = StrategyConfig()
_sector_rotation_history = []  # 记录最近3天的板块名称集合，用于轮动预警

DAILY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "daily")
BOARD_DAILY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "board_daily")



# 独立缓存文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CONCEPT_INDEX_PATH = os.path.join(DATA_DIR, "stock_concept_index.json")
INDUSTRY_SW_INDEX_PATH = os.path.join(DATA_DIR, "stock_industry_sw_index.json")


# ============================================================
# 技术指标函数（支持 talib / pandas 双模式）
# ============================================================
def calc_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD，支持talib和pandas两种实现"""
    if HAS_TALIB:
        diff, dea, macd_hist = talib.MACD(
            close.values,
            fastperiod=CONFIG.macd_fast,
            slowperiod=CONFIG.macd_slow,
            signalperiod=CONFIG.macd_signal,
        )
        idx = close.index
        return pd.Series(diff, index=idx), pd.Series(dea, index=idx), pd.Series(macd_hist, index=idx)
    else:
        ema_fast = close.ewm(span=CONFIG.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=CONFIG.macd_slow, adjust=False).mean()
        diff = ema_fast - ema_slow
        dea = diff.ewm(span=CONFIG.macd_signal, adjust=False).mean()
        macd_hist = 2 * (diff - dea)
        return diff, dea, macd_hist


def calc_bbands(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带"""
    if HAS_TALIB:
        upper, mid, lower = talib.BBANDS(
            close.values,
            timeperiod=CONFIG.bbands_period,
            nbdevup=CONFIG.bbands_nbdev,
            nbdevdn=CONFIG.bbands_nbdev,
        )
        idx = close.index
        return pd.Series(upper, index=idx), pd.Series(mid, index=idx), pd.Series(lower, index=idx)
    else:
        mid = close.rolling(window=CONFIG.bbands_period).mean()
        std = close.rolling(window=CONFIG.bbands_period).std()
        upper = mid + CONFIG.bbands_nbdev * std
        lower = mid - CONFIG.bbands_nbdev * std
        return upper, mid, lower


def calc_vegas_emas(close: pd.Series) -> Dict[int, pd.Series]:
    """计算维加斯通道EMA"""
    result = {}
    for period in CONFIG.vegas_periods:
        result[period] = close.ewm(span=period, adjust=False).mean()
    return result


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算KDJ指标，返回 K, D, J"""
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    k = rsv.ewm(com=(m1 - 1), adjust=False).mean()
    d = k.ewm(com=(m2 - 1), adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============================================================
# 日期工具函数
# ============================================================
def to_date_str(date) -> str:
    """统一转换为 YYYYMMDD 格式"""
    if isinstance(date, datetime.date):
        return date.strftime("%Y%m%d")
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y%m%d")
    return str(date).replace("-", "")


def to_dash_date(date) -> str:
    """统一转换为 YYYY-MM-DD 格式"""
    if isinstance(date, (datetime.date, pd.Timestamp)):
        return date.strftime("%Y-%m-%d")
    s = str(date).replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def get_trade_dates(end_date: str, count: int) -> List[str]:
    """获取最近N个交易日日期列表"""
    try:
        df = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(df["trade_date"]).sort_values()
        end_dt = pd.to_datetime(end_date)
        mask = dates <= end_dt
        return dates[mask].tail(count).dt.strftime("%Y%m%d").tolist()
    except Exception:
        end_dt = pd.to_datetime(end_date)
        return [(end_dt - pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range(count * 2)][::-1]


def get_latest_trade_date(date_str: str) -> str:
    """返回 <= date_str 的最近一个交易日，格式 YYYY-MM-DD"""
    try:
        df = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(df["trade_date"]).sort_values()
        target = pd.to_datetime(date_str)
        valid = dates[dates <= target]
        if len(valid) > 0:
            return valid.iloc[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    return date_str


# ============================================================
# 数据获取函数（带重试机制 + 指数退避）
# ============================================================
def _safe_ak_call(func, *args, description="", silent=False, **kwargs):
    """
    带指数退避重试的 akshare 请求包装器。
    silent=True 时，非网络错误不打印（用于非关键数据，如板块指数）。
    """
    last_error = None
    for attempt in range(1, CONFIG.max_retries + 1):
        try:
            time.sleep(CONFIG.request_delay + random.uniform(0, 0.2))
            result = func(*args, **kwargs)
            return result
        except (
            requests.exceptions.RequestException,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as e:
            last_error = e
            if attempt < CONFIG.max_retries:
                wait = CONFIG.request_delay * (2 ** attempt)
                desc = f" ({description})" if description else ""
                print(f"    ⚠ 网络异常{desc}，{wait:.1f}s 后重试 ({attempt}/{CONFIG.max_retries})")
                time.sleep(wait)
        except Exception as e:
            last_error = e
            break

    if last_error:
        desc = f" ({description})" if description else ""
        if not silent:
            print(f"    ❌ 请求失败{desc}: {last_error}")
    return None


def fetch_stock_pool() -> pd.DataFrame:
    """获取选股池：全A股 - ST - 上市不满60日"""
    print("  获取全A股列表...")
    df = _safe_ak_call(ak.stock_zh_a_spot_em, description="全A股列表")
    if df is None:
        return pd.DataFrame()
    df = df[~df["名称"].str.contains("ST", na=False)]
    df = df[~df["名称"].str.contains("退", na=False)]
    df = df[df["代码"].str.match(r"^\d{6}$")]
    print(f"  选股池股票数量: {len(df)}")
    return df


def fetch_board_list() -> pd.DataFrame:
    """
    获取概念板块列表（使用 akshare 同花顺概念，速度快）。
    申万三级行业由 build_industry_index_sw 单独构建并合入缓存。
    """
    try:
        print("  使用 akshare 获取同花顺概念板块列表...")
        df = _safe_ak_call(ak.stock_board_concept_name_ths, description="同花顺概念板块列表")
        if df is not None and not df.empty:
            # 列名适配：可能为中文名称/代码，板块名称/板块代码，或英文 name/code
            if '名称' in df.columns and '代码' in df.columns:
                df.rename(columns={'名称': '概念名称', '代码': '概念代码'}, inplace=True)
            elif '板块名称' in df.columns and '板块代码' in df.columns:
                df.rename(columns={'板块名称': '概念名称', '板块代码': '概念代码'}, inplace=True)
            elif 'name' in df.columns and 'code' in df.columns:
                df.rename(columns={'name': '概念名称', 'code': '概念代码'}, inplace=True)
            else:
                print("  ⚠ 概念板块列表列名异常，实际列名:", df.columns.tolist())
                return pd.DataFrame()
            print(f"  同花顺概念板块数量: {len(df)}")
            return df[['概念名称', '概念代码']]
    except Exception as e:
        print(f"  ❌ 同花顺概念板块列表获取失败: {e}")

    print("  ⚠ 无概念板块数据（申万三级行业将在后续步骤单独构建）")
    return pd.DataFrame()


# 预定义常见 User-Agent，避免每次请求使用同一个
_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/109.0",
]

# 全局 Session，用于复用 Cookie
_THS_SESSION = None

def _get_ths_session():
    """初始化同花顺请求会话，获取必要 Cookie"""
    global _THS_SESSION
    if _THS_SESSION is not None:
        return _THS_SESSION
    _THS_SESSION = requests.Session()
    _THS_SESSION.headers.update({
        "User-Agent": random.choice(_UA_LIST),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    try:
        # 先访问同花顺主页，获取初始 Cookie
        _THS_SESSION.get("http://www.10jqka.com.cn", timeout=10)
    except Exception:
        pass
    return _THS_SESSION


def _scrape_ths_concept_page(concept_code_num: str) -> pd.DataFrame:
    """
    获取同花顺概念成分股（使用纯数字代码，如 "308614"）。
    优先使用同花顺 JSON 接口，备用 HTML 页面解析。
    包含重试与延迟，避免被反爬。
    返回列：['代码','名称']，失败返回空 DataFrame。
    """
    session = _get_ths_session()
    base_referer = f"http://q.10jqka.com.cn/gn/detail/code/{concept_code_num}/"

    # 共享的函数：执行带重试的请求
    def _attempt_request(method, url, **kwargs):
        for attempt in range(1, 4):  # 最多3次
            try:
                time.sleep(random.uniform(0.8, 2.0))   # 防止请求过快
                headers = dict(session.headers)
                headers["User-Agent"] = random.choice(_UA_LIST)
                headers["Referer"] = base_referer
                # 根据 url 调整 Accept
                if "json" in url or "/ajax/" in url:
                    headers["Accept"] = "application/json, text/plain, */*"
                else:
                    headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                kwargs["headers"] = headers
                if method == "get":
                    resp = session.get(url, timeout=15, **kwargs)
                elif method == "post":
                    resp = session.post(url, timeout=15, **kwargs)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.RequestException, Exception) as e:
                if attempt < 3:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    return None

    # 方法1：同花顺 JSON 接口（最稳定）
    json_url = f"http://q.10jqka.com.cn/gn/detail/ajax/stock?code={concept_code_num}"
    r = _attempt_request("get", json_url)
    if r is not None:
        try:
            data = r.json()
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                records = []
                for item in data["data"]:
                    code = str(item.get("code", "")).strip()
                    name = str(item.get("name", "")).strip()
                    if code and len(code) == 6:
                        records.append({"代码": code, "名称": name})
                if records:
                    return pd.DataFrame(records)
        except Exception:
            pass
        # 方法2：HTML 详情页解析
        r = _attempt_request("get", html_url)
        if r is not None:
            try:
                dfs = _parse_html_tables(r.text)
                for df in dfs:
                    # 跳过仍为 MultiIndex 的 DataFrame（理论上已展平，兜底）
                    if isinstance(df.columns, pd.MultiIndex):
                        continue
                    if "代码" in df.columns and "名称" in df.columns:
                        return df[["代码", "名称"]].copy()
            except Exception:
                pass

        # 方法3：页面内嵌的 var list 提取
        try:
            match = re.search(r'var\s+list\s*=\s*(\[.*?\]);', r.text, re.DOTALL)
            if match:
                arr = json.loads(match.group(1))
                records = []
                for item in arr:
                    code = item.get("code", "")
                    name = item.get("name", "")
                    if code and len(code) == 6:
                        records.append({"代码": code, "名称": name})
                if records:
                    return pd.DataFrame(records)
        except Exception:
            pass

    return pd.DataFrame()

def fetch_concept_constituents(concept_name: str, concept_code: str = "") -> pd.DataFrame:
    """
    获取概念成分股（通过爬取同花顺概念详情页）。
    返回 DataFrame，列：['代码','名称','概念代码','概念名称']
    """
    # 提取数字代码（去掉可能的“GN”等前缀）
    code_num = str(concept_code).replace("GN", "").replace("gn", "")
    df_raw = _scrape_ths_concept_page(code_num)
    if df_raw.empty:
        return pd.DataFrame()

    # 清洗代码格式（只保留6位数字）
    df_raw["代码"] = df_raw["代码"].astype(str).str.extract(r'(\d{6})', expand=False)
    df_raw = df_raw.dropna(subset=["代码"])

    df_raw["概念代码"] = concept_code
    df_raw["概念名称"] = concept_name
    return df_raw[["代码", "名称", "概念代码", "概念名称"]]


def fetch_industry_list_sw() -> pd.DataFrame:
    """
    获取申万三级行业列表，返回列名为 '行业名称'、'行业代码'。
    """
    try:
        df = ak.sw_index_third_info()
        if df is not None and not df.empty:
            if '行业代码' in df.columns and '行业名称' in df.columns:
                df = df.rename(columns={'行业代码': '行业代码', '行业名称': '行业名称'})
            elif 'industry_code' in df.columns and 'industry_name' in df.columns:
                df = df.rename(columns={'industry_code': '行业代码', 'industry_name': '行业名称'})
            else:
                print("  sw_index_third_info 返回的列名不符合预期，请升级 akshare。")
                print("  实际列名:", df.columns.tolist())
                return pd.DataFrame()
            df['行业名称'] = '申万三级-' + df['行业名称'].astype(str)
            return df[['行业名称', '行业代码']].copy()
        else:
            print("  sw_index_third_info 返回空 DataFrame，请检查网络。")
    except Exception as e:
        print(f"  获取申万三级行业列表失败: {e}")
        print("  请尝试执行: pip install akshare --upgrade")
    return pd.DataFrame()


def fetch_industry_constituents_sw(board_code: str, board_name: str = "") -> pd.DataFrame:
    """
    使用 akshare 获取申万三级行业成分股，返回列 '代码','名称'。
    """
    try:
        df = _safe_ak_call(
            ak.sw_index_third_cons,
            symbol=board_code,
            description=f"申万行业成分股 {board_name or board_code}",
            silent=True
        )
        if df is not None and not df.empty:
            col_map = {}
            code_col = None
            name_col = None
            for col in df.columns:
                col_clean = str(col).strip().lower()
                if col_clean in ('代码', '股票代码', 'code', 'stock_code', 'ts_code', 'symbol',
                                 'ticker', 'secucode', 'stockcode'):
                    code_col = col
                elif col_clean in ('名称', '股票名称', 'name', 'stock_name', 'secuname',
                                   'stockname', 'companyname', 'sec_name'):
                    name_col = col
            if code_col and name_col:
                df = df.rename(columns={code_col: '代码', name_col: '名称'})
            elif len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: '代码', df.columns[1]: '名称'})
            else:
                return pd.DataFrame()

            # 确保代码为6位数字
            df['代码'] = df['代码'].astype(str).str.extract(r'(\d{6})', expand=False)
            df = df.dropna(subset=['代码'])
            if '代码' in df.columns and '名称' in df.columns:
                return df[['代码', '名称']]
    except Exception as e:
        print(f"    获取申万行业成分股 {board_code} 失败: {e}")
    return pd.DataFrame()

def fetch_board_index(board_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取板块指数K线（同花顺）"""
    result = _safe_ak_call(
        ak.stock_board_concept_index_ths,
        symbol=board_code,
        start_date=start_date,
        end_date=end_date,
        description=f"板块指数 {board_code}",
        silent=True,  # 板块指数拉取失败是常见情况（某些板块无历史数据），静默处理
    )
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        result["日期"] = pd.to_datetime(result["日期"])
        result = result.sort_values("日期").reset_index(drop=True)
    return result


def _parse_html_tables(text: str) -> List[pd.DataFrame]:
    """安全解析 HTML 表格，抑制 lxml/html5lib 的解析输出，并统一处理多级标题"""
    if "<table" not in text.lower():
        return []

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        dfs = pd.read_html(io.StringIO(text))
        # 如果列索引是 MultiIndex（多级标题），展平为单一字符串
        for i, df in enumerate(dfs):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    '_'.join(filter(None, map(str, col))).strip('_')
                    for col in df.columns
                ]
        return dfs
    except ValueError:
        return []
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr





def fetch_board_constituents(board_code: str, board_name: str = "") -> pd.DataFrame:
    """
    获取板块成分股（概念或行业），不再使用任何东方财富接口。
    - 概念板块：Tushare concept_detail
    - 行业板块：Tushare index_member
    - 若均失败返回空 DataFrame（不再爬虫）
    """
    if TUSHARE_PRO is None:
        return pd.DataFrame()

    # 判断板块类型：概念代码一般为数字，行业代码以 'BK' 开头或名称含 '行业-'
    is_industry = board_name.startswith('行业-') or board_code.startswith('BK')
    try:
        if is_industry:
            # 申万行业成分股
            return fetch_industry_constituents_sw(board_code, board_name)
        else:
            # 概念板块
            detail = TUSHARE_PRO.concept_detail(concept_code=board_code, fields='ts_code,name')
            if detail is not None and not detail.empty:
                detail['代码'] = detail['ts_code'].str[:6]
                detail['名称'] = detail['name']
                detail['板块代码'] = board_code
                print(f"    ✅ 概念成分股 {board_name or board_code}: {len(detail)} 只")
                return detail[['代码', '名称', '板块代码']]
    except Exception as e:
        print(f"    ⚠ Tushare 成分股获取失败 ({board_code}): {e}")

    return pd.DataFrame()


def fetch_limit_up_pool(trade_date: str) -> pd.DataFrame:
    """获取某日涨停板数据（仅使用 akshare 东方财富接口，Tushare 暂不可用）"""
    result = _safe_ak_call(
        ak.stock_zt_pool_em,
        date=trade_date,
        description=f"涨停板 {trade_date}",
    )
    if result is not None and not result.empty:
        # ak.stock_zt_pool_em 返回列: 代码, 名称, 涨跌幅, 最新价, 成交额, ...
        # 可能不含“连板数”，设为 0 兼容后续逻辑
        if "连板数" not in result.columns:
            result["连板数"] = 0
        result["日期"] = trade_date
        return result[["代码", "名称", "日期", "连板数"]].copy()
    return pd.DataFrame()


def fetch_stock_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取个股日K线（前复权，新浪数据源），带本地缓存"""
    # 1. 检查缓存文件
    cache_file = os.path.join(DAILY_CACHE_DIR, f"{code}.csv")
    cached_df = None
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=["日期"])
            cached_df = cached_df.sort_values("日期").reset_index(drop=True)
            if not cached_df.empty:
                cached_start = cached_df["日期"].min().strftime("%Y%m%d")
                cached_end = cached_df["日期"].max().strftime("%Y%m%d")
                if cached_start <= start_date and cached_end >= end_date:
                    mask = (cached_df["日期"] >= start_date) & (cached_df["日期"] <= end_date)
                    return cached_df.loc[mask].reset_index(drop=True)
        except Exception:
            pass  # 缓存损坏则重新获取

    # 2. 多数据源获取（按优先级：Tushare → akshare → Baostock）
    result = None

    # 2.1 Tushare
    if TUSHARE_PRO is not None:
        try:
            ts_code = f"{code}.{'SH' if code.startswith(('6', '9')) else 'SZ'}"
            # 使用 Tushare 前复权日线
            ts_df = TUSHARE_PRO.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                                      adj='qfq', factors=False)
            if ts_df is not None and not ts_df.empty:
                ts_df.rename(columns={
                    'trade_date': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'vol': '成交量',
                    'amount': '成交额',
                    'turnover_rate': '换手率'
                }, inplace=True)
                ts_df['日期'] = pd.to_datetime(ts_df['日期'])
                result = ts_df.sort_values('日期').reset_index(drop=True)
        except Exception as e:
            print(f"    Tushare 获取 {code} 失败: {e}，降级到 akshare")

    # 2.2 akshare（新浪）
    if result is None:
        result = _safe_ak_call(
            ak.stock_zh_a_daily,
            symbol=f"{'sh' if code.startswith(('6', '9')) else 'sz'}{code}",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
            description=f"个股日线(新浪) {code}",
        )
        if result is not None and not result.empty:
            # 新浪接口返回英文列名，统一映射为中文
            col_mapping = {
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "volume": "成交量",
            }
            rename_map = {}
            for col in result.columns:
                col_lower = col.lower()
                if col_lower in col_mapping:
                    rename_map[col] = col_mapping[col_lower]
            if rename_map:
                result.rename(columns=rename_map, inplace=True)

    # 2.3 Baostock（作为最后备选）
    if result is None and BAOSTOCK_AVAILABLE:
        init_baostock()
        try:
            bs_code = f"{'sh' if code.startswith(('6','9')) else 'sz'}.{code}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2"  # 前复权
            )
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                bs_df = pd.DataFrame(data_list, columns=['日期','开盘','最高','最低','收盘','成交量','成交额','换手率'])
                bs_df['日期'] = pd.to_datetime(bs_df['日期'])
                for c in ['开盘','最高','最低','收盘','成交量','成交额','换手率']:
                    bs_df[c] = pd.to_numeric(bs_df[c], errors='coerce')
                result = bs_df
        except Exception as e:
            print(f"    Baostock 获取 {code} 失败: {e}")

    if result is None or result.empty:
        return pd.DataFrame()

    result["日期"] = pd.to_datetime(result["日期"])
    result = result.sort_values("日期").reset_index(drop=True)

    # 3. 合并旧缓存，更新本地文件
    os.makedirs(DAILY_CACHE_DIR, exist_ok=True)
    if cached_df is not None and not cached_df.empty:
        combined = pd.concat([cached_df, result]).drop_duplicates(subset=["日期"]).sort_values("日期")
        combined.to_csv(cache_file, index=False)
    else:
        result.to_csv(cache_file, index=False)

    return result


# ============================================================
# 股票→概念反向索引（本地缓存，30天刷新一次）
# ============================================================
STOCK_CONCEPT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stock_concept_index.json")


def _invert_stock_concept_index() -> Dict[str, set]:
    """
    从本地缓存 stock_concept_index.json 反转为 board_code→constituent_codes 映射。
    若缓存不存在或为空，返回空字典。
    """
    import json
    if not os.path.exists(STOCK_CONCEPT_CACHE):
        return {}
    with open(STOCK_CONCEPT_CACHE, "r", encoding="utf-8") as f:
        stock_concept = json.load(f)
    board_map: Dict[str, set] = {}
    for stock_code, concepts in stock_concept.items():
        for c in concepts:
            board_code = c["概念代码"]
            if board_code not in board_map:
                board_map[board_code] = set()
            board_map[board_code].add(stock_code)
    return board_map


def get_stock_concepts(code: str) -> List[Dict]:
    """
    获取某只股票的概念/行业标签，先查概念缓存，若不存在再查行业缓存。
    概念缓存条目键名：'概念名称'、'概念代码'
    行业缓存条目键名：'行业名称'、'行业代码'
    """
    # 尝试加载概念缓存
    if os.path.exists(CONCEPT_INDEX_PATH):
        with open(CONCEPT_INDEX_PATH, 'r', encoding='utf-8') as f:
            concept_data = json.load(f)
        if code in concept_data:
            return concept_data[code]

    # 尝试加载行业缓存
    if os.path.exists(INDUSTRY_SW_INDEX_PATH):
        with open(INDUSTRY_SW_INDEX_PATH, 'r', encoding='utf-8') as f:
            industry_data = json.load(f)
        return industry_data.get(code, [])

    return []

def _get_stock_market_cap(code: str, date_str: str) -> Optional[float]:
    """获取某只股票在某交易日的流通市值（单位：元）"""
    # 优先 Tushare
    if TUSHARE_PRO is not None:
        try:
            # daily_basic 可获取流通市值
            df = TUSHARE_PRO.daily_basic(ts_code=f"{code}.{'SH' if code.startswith(('6','9')) else 'SZ'}",
                                         trade_date=date_str.replace('-', ''),
                                         fields='ts_code,trade_date,circ_mv')
            if df is not None and not df.empty:
                circ_mv = df['circ_mv'].iloc[0]
                if circ_mv and not pd.isna(circ_mv):
                    return float(circ_mv) * 1e4  # Tushare 流通市值单位是万元，转为元
        except Exception as e:
            pass
    # akshare 备用
    try:
        spot = ak.stock_zh_a_spot_em()
        if spot is not None and not spot.empty:
            if '代码' in spot.columns and '流通市值' in spot.columns:
                row = spot[spot['代码'] == code]
                if not row.empty:
                    return row['流通市值'].iloc[0]
    except Exception:
        pass
    return None


def _read_stock_3d_ret(code: str, cache_dir: str = DAILY_CACHE_DIR, eval_date: str = None) -> Optional[float]:
    """从本地缓存读取股票近3个交易日涨幅；若缓存缺失且指定了eval_date则自动拉取"""
    cache_file = os.path.join(cache_dir, f"{code}.csv")
    if not os.path.exists(cache_file) and eval_date is not None:
        # 自动补拉最近100个交易日
        end_dt = eval_date
        start_dt = (pd.to_datetime(eval_date) - pd.Timedelta(days=100)).strftime("%Y%m%d")
        fetch_stock_daily(code, start_dt, end_dt)
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_csv(cache_file, parse_dates=["日期"])
        if df.empty or len(df) < 4:
            return None
        close_col = "收盘价" if "收盘价" in df.columns else "收盘"
        closes = df[close_col].values
        return (closes[-1] - closes[-4]) / closes[-4]
    except Exception:
        return None


def get_constituent_ret_median(codes: set, eval_date: str) -> float:
    """计算给定股票集合的涨幅中位数（采样前N只，取近3日涨幅）"""
    filtered = [c for c in codes if c[:2] in ('60', '00', '30')]
    if not filtered:
        return None
    sample = filtered[:CONFIG.board_sample_top_n]
    rets = []
    for code in sample:
        ret = _read_stock_3d_ret(code, eval_date=to_date_str(eval_date))
        if ret is not None:
            rets.append(ret)
    if not rets:
        return None
    return float(np.median(rets))


def calc_board_up_ratio(constituent_codes: set, trade_dates: List[str]) -> Tuple[float, float]:
    """
    计算板块近N日平均上涨家数占比，以及连续N日都满足阈值的条件。
    返回 (avg_up_ratio, consecutive_up_days)
    """
    # 先从缓存获取每只股票在给定日期的涨跌方向
    up_counts = []
    total = 0
    dates = trade_dates  # 格式 YYYYMMDD
    valid_stocks = [c for c in constituent_codes if c[:2] in ('60', '00', '30')]
    if not valid_stocks:
        return 0.0, 0

    # 为了避免重复拉取，预先获取所有成分股在所需日期的日线（仅拉取一次）
    # 简单处理：逐个股票检查，从缓存读取涨跌幅
    daily_ratios = []
    for d in dates:
        up = 0
        total_stocks = 0
        for code in valid_stocks:
            cache_file = os.path.join(DAILY_CACHE_DIR, f"{code}.csv")
            if not os.path.exists(cache_file):
                # 缺失则尝试补拉最近一段时间
                start_pull = (pd.to_datetime(d) - pd.Timedelta(days=20)).strftime("%Y%m%d")
                fetch_stock_daily(code, start_pull, d)
            if not os.path.exists(cache_file):
                continue
            try:
                df = pd.read_csv(cache_file, parse_dates=["日期"])
                close_col = "收盘价" if "收盘价" in df.columns else "收盘"
                # 找到该日期的行
                df['日期'] = pd.to_datetime(df['日期'])
                today_row = df[df['日期'] == pd.to_datetime(d)]
                if today_row.empty:
                    continue
                close_today = today_row[close_col].iloc[0]
                # 获取前一交易日收盘价
                prev_day = (pd.to_datetime(d) - pd.Timedelta(days=1))
                prev_row = df[df['日期'] == prev_day]
                if prev_row.empty:
                    continue
                close_prev = prev_row[close_col].iloc[0]
                change = (close_today - close_prev) / close_prev
                if change > 0:
                    up += 1
                total_stocks += 1
            except Exception:
                continue
        if total_stocks > 0:
            daily_ratios.append(up / total_stocks)
        else:
            daily_ratios.append(0.0)

    avg_ratio = np.mean(daily_ratios) if daily_ratios else 0.0
    # 连续满足>0.5的天数
    consecutive = 0
    max_consecutive = 0
    for r in daily_ratios:
        if r > CONFIG.board_up_ratio_line:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    return avg_ratio, max_consecutive


def fetch_index_daily(index_code: str = "000300", start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """获取大盘指数日K线（Tushare 优先，akshare/Baostock 备用）"""
    if start_date is None:
        start_date = (pd.to_datetime("today") - pd.Timedelta(days=CONFIG.index_start_offset)).strftime("%Y%m%d")
    if end_date is None:
        end_date = pd.to_datetime("today").strftime("%Y%m%d")

    result = None

    # Tushare 指数日线
    if TUSHARE_PRO is not None:
        try:
            # 映射指数代码到 Tushare 的指数代码（如 000300.SH）
            ts_index_map = {
                "000300": "000300.SH",
                "000001": "000001.SH",
                "399001": "399001.SZ",
            }
            ts_code = ts_index_map.get(index_code, f"{index_code}.SH" if index_code.startswith(('0','3')) else f"{index_code}.SZ")
            ts_df = TUSHARE_PRO.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if ts_df is not None and not ts_df.empty:
                ts_df.rename(columns={'trade_date':'date','close':'close'}, inplace=True)
                ts_df['date'] = pd.to_datetime(ts_df['date'])
                ts_df = ts_df.sort_values('date')
                result = ts_df[['date','open','high','low','close','vol']]
        except Exception as e:
            print(f"    Tushare 指数数据 {index_code} 失败: {e}")

    # akshare
    if result is None:
        symbol_map = {
            "000300": "sh000300",
            "000001": "sh000001",
            "399001": "sz399001",
        }
        symbol = symbol_map.get(index_code, index_code)
        ak_result = _safe_ak_call(
            ak.stock_zh_index_daily,
            symbol=symbol,
            description=f"大盘指数 {index_code}",
        )
        if ak_result is not None and not ak_result.empty:
            ak_result["date"] = pd.to_datetime(ak_result["date"])
            result = ak_result.sort_values("date").reset_index(drop=True)

    # Baostock
    if result is None and BAOSTOCK_AVAILABLE:
        init_baostock()
        try:
            bs_index_map = {
                "000300": "sh.000300",
                "000001": "sh.000001",
                "399001": "sz.399001",
            }
            bs_code = bs_index_map.get(index_code, f"sh.{index_code}" if index_code.startswith(('0','3')) else f"sz.{index_code}")
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2"
            )
            data_list = [rs.get_row_data() for _ in range(len(rs.get_data()))]
            if data_list:
                bs_df = pd.DataFrame(data_list, columns=['date','open','high','low','close','volume'])
                bs_df['date'] = pd.to_datetime(bs_df['date'])
                for c in ['open','high','low','close','volume']:
                    bs_df[c] = pd.to_numeric(bs_df[c], errors='coerce')
                result = bs_df
        except Exception as e:
            print(f"    Baostock 指数数据 {index_code} 失败: {e}")

    if result is None or result.empty:
        return pd.DataFrame()
    mask = (result["date"] >= pd.to_datetime(start_date)) & (result["date"] <= pd.to_datetime(end_date))
    return result.loc[mask]


def check_index_trend(eval_date: str) -> bool:
    """
    市场环境过滤：沪深300收盘价是否高于其20日均线。
    返回 True 表示多头环境，可以操作；False 表示空头，建议空仓。
    """
    if not CONFIG.use_index_filter:
        return True  # 未开启过滤则默认通过

    print(f"  🌐 市场趋势过滤：检查 {CONFIG.index_code} ...")
    end_date = eval_date
    start_date = (pd.to_datetime(eval_date) - pd.Timedelta(days=CONFIG.index_start_offset)).strftime("%Y%m%d")
    idx_df = fetch_index_daily(CONFIG.index_code, start_date, end_date)
    if idx_df.empty or len(idx_df) < CONFIG.index_ma_period:
        print(f"    ⚠ 指数数据不足，默认通过过滤")
        return True

    close = idx_df["close"]
    ma = close.rolling(CONFIG.index_ma_period).mean()
    latest_close = close.iloc[-1]
    latest_ma = ma.iloc[-1]
    trend = latest_close >= latest_ma
    status = "🟢 多头" if trend else "🔴 空头"
    print(f"    {status} | 收盘 {latest_close:.2f} vs MA{CONFIG.index_ma_period} {latest_ma:.2f}")
    return trend


def allocate_position_hierarchical(signals: List[Dict], sectors: List[Dict]) -> List[Dict]:
    """
    按主线S/A/B等级分层分配仓位。
    signals 中每个元素需包含 '板块等级' 和 '日线数据'
    """
    if not signals:
        return signals

    # 等级基数权重
    base_weights = {"S": 0.60, "A": 0.30, "B": 0.10}
    groups = {"S": [], "A": [], "B": []}
    for sig in signals:
        level = sig.get("板块等级", "B")
        groups[level].append(sig)

    # 仅考虑实际有信号的等级，动态归一化基础权重
    active_levels = [lvl for lvl in ("S", "A", "B") if groups[lvl]]
    total_active_weight = sum(base_weights[lvl] for lvl in active_levels)
    if total_active_weight == 0:
        return signals

    level_weights = {lvl: base_weights[lvl] / total_active_weight for lvl in active_levels}

    # 各等级内部再使用等权或倒数波动率分配
    for lvl in active_levels:
        group_signals = groups[lvl]
        n = len(group_signals)
        if n == 0:
            continue
        if CONFIG.position_method == "equal":
            weight_per_stock = level_weights[lvl] / n
            for s in group_signals:
                s["建议仓位"] = round(weight_per_stock, 4)
        else:  # 倒数波动率
            volatilities = []
            for s in group_signals:
                df = s.get("日线数据")
                if df is not None and len(df) >= CONFIG.vol_lookback:
                    ret = df["收盘"].pct_change().dropna().tail(CONFIG.vol_lookback)
                    vol = ret.std()
                else:
                    vol = 0.02
                volatilities.append(max(vol, 0.001))
            inv_vol = [1.0/v for v in volatilities]
            total_inv = sum(inv_vol)
            for i, s in enumerate(group_signals):
                s["建议仓位"] = round(level_weights[lvl] * inv_vol[i] / total_inv, 4)

    return signals


def allocate_position(signals: List[Dict]) -> List[Dict]:
    """
    对最终信号分配建议仓位比例。
    - 等权重: 1/N
    - 倒数波动率: 1/volatility，归一化
    """
    if not signals:
        return signals

    n = len(signals)
    if CONFIG.position_method == "equal":
        weight = 1.0 / n if n else 0.0
        for s in signals:
            s["建议仓位"] = round(weight, 4)
        return signals

    # 倒数波动率加权
    volatilities = []
    for s in signals:
        df = s.get("日线数据")
        if df is not None and len(df) >= CONFIG.vol_lookback:
            returns = df["收盘"].pct_change().dropna().tail(CONFIG.vol_lookback)
            vol = returns.std()
        else:
            vol = 0.02  # 默认波动率
        volatilities.append(max(vol, 0.001))  # 防止除零

    inv_vol = [1.0 / v for v in volatilities]
    total_inv = sum(inv_vol)
    for i, s in enumerate(signals):
        s["建议仓位"] = round(inv_vol[i] / total_inv, 4)
    return signals


def build_stock_concept_index(force_refresh: bool = False) -> Dict[str, list]:
    """
    构建股票→概念反向索引（仅同花顺概念），写入独立缓存文件。
    首次或超过30天自动全量更新。
    """
    cache_path = CONCEPT_INDEX_PATH
    # 检查是否需要刷新
    need_update = force_refresh
    if not need_update and os.path.exists(cache_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.date.today() - mtime).days >= 30:
            need_update = True
        else:
            print(f"  ✓ 概念缓存有效 ({mtime})，直接加载")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                return cached
            need_update = True

    if need_update:
        print("  ⏳ 构建股票→概念反向索引（使用同花顺概念，月度更新）...")
        boards = fetch_board_list()
        if boards.empty:
            return {}

        stock_concept_map = {}
        total = len(boards)
        for idx, (_, row) in enumerate(boards.iterrows()):
            board_name = row["概念名称"]
            board_code = row["概念代码"]

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"    进度: {idx+1}/{total}")

            # 使用 akshare 同花顺概念成分股（已修复为 ak.stock_board_concept_cons_ths）
            # 使用概念名称作为 symbol 获取成分股
            constituents = fetch_concept_constituents(board_name, board_code)
            if constituents.empty:
                continue

            for _, stock_row in constituents.iterrows():
                code = str(stock_row["代码"]).strip()[:6]
                if not code.isdigit():
                    continue
                if code not in stock_concept_map:
                    stock_concept_map[code] = []
                stock_concept_map[code].append({
                    "概念名称": board_name,
                    "概念代码": board_code,
                })

        # 写入独立缓存文件
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(stock_concept_map, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 概念索引已缓存至: {cache_path}，覆盖 {len(stock_concept_map)} 只股票")

        return stock_concept_map

    # 如果缓存有效，从文件加载返回
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_industry_index_sw():
    """
    构建申万三级行业索引，写入独立缓存文件。
    首次或超过365天自动全量更新。
    缓存键名：'行业名称'、'行业代码'（与概念缓存分离）。
    """
    cache_path = INDUSTRY_SW_INDEX_PATH
    need_update = False
    if os.path.exists(cache_path):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.date.today() - mtime).days >= 365:
            need_update = True
        else:
            print(f"  ✓ 行业缓存有效 ({mtime})，跳过更新")
            return
    else:
        need_update = True

    if not need_update:
        return

    print("===== 构建申万三级行业索引（写入独立缓存）=====")

    industry_boards = fetch_industry_list_sw()
    if industry_boards.empty:
        print("❌ 无法获取申万三级行业列表")
        return

    print(f"  申万三级行业板块数量: {len(industry_boards)}")

    # 初始化行业字典
    industry_map = {}

    total = len(industry_boards)
    for idx, (_, row) in enumerate(industry_boards.iterrows()):
        industry_name = row['行业名称']
        industry_code = row['行业代码']
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"    进度: {idx + 1}/{total} — {industry_name}")

        const = fetch_industry_constituents_sw(industry_code, industry_name)
        if const.empty:
            continue

        for _, stock_row in const.iterrows():
            code = stock_row['代码']
            if code not in industry_map:
                industry_map[code] = []
            if not any(c['行业代码'] == industry_code for c in industry_map[code]):
                industry_map[code].append({
                    "行业名称": industry_name,
                    "行业代码": industry_code,
                })

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(industry_map, f, ensure_ascii=False, indent=2)
    print(f"✅ 行业索引构建完成，覆盖 {len(industry_map)} 只股票")


# ============================================================
# 第一步：筛选主线板块
# ============================================================
def screen_main_sectors(eval_date: str) -> List[Dict]:
    """
    新逻辑：龙头先行，反查板块
    第1步：找出潜在龙头（近5日涨停>=4天）
    第2步：检索龙头所属的全部概念板块
    第3步：计算板块近3日涨幅和日均涨停家数
    第4步：涨幅排名前5%
    第5步：筛选同时满足条件的板块（涨幅>=阈值 且 日均涨停>=3）
    """
    print("\n===== 第一步：筛选主线板块 =====")

    date_str = to_date_str(eval_date)
    trade_dates = get_trade_dates(eval_date, 10)
    recent_5_dates = trade_dates[-5:]
    recent_3_dates = trade_dates[-3:]
    recent_10_dates = trade_dates[-10:]  # 取全部10日

    print(f"  评估日期: {eval_date}")
    print(f"  近5个交易日: {recent_5_dates}")
    print(f"  近10个交易日: {recent_10_dates}")

    # ----------------------------------------------------------------
    # 第1步：找出潜在龙头（近5日涨停≥2天 或 近10日涨停≥3天）
    # ----------------------------------------------------------------
    print("\n  [第1步] 找出潜在龙头（近5日≥2 或 近10日≥3）...")

    all_limit_up_data = {}
    for d in trade_dates:
        lt_data = fetch_limit_up_pool(d)
        if lt_data is not None and not lt_data.empty:
            lt_data["日期"] = d
            lt_data = lt_data.drop_duplicates(subset=["代码", "日期"])
        all_limit_up_data[d] = lt_data

    stock_zt_count_5 = {}
    stock_zt_count_10 = {}
    stock_zt_detail = {}

    for d in trade_dates:
        lt_data = all_limit_up_data.get(d)
        if lt_data is None or lt_data.empty:
            continue
        is_in_5 = d in recent_5_dates
        for _, row in lt_data.iterrows():
            code = row["代码"]
            if is_in_5:
                stock_zt_count_5[code] = stock_zt_count_5.get(code, 0) + 1
            stock_zt_count_10[code] = stock_zt_count_10.get(code, 0) + 1

            if code not in stock_zt_detail:
                stock_zt_detail[code] = {
                    "代码": code,
                    "名称": row.get("名称", ""),
                    "连板数": 0,
                    "首次封板时间": "99:99:99",
                    "首次涨停日期": d,
                }
            lb = row.get("连板数", 0)
            if pd.isna(lb):
                lb = 0
            lb = int(lb)
            ft = str(row.get("首次封板时间", "99:99:99"))
            if lb > stock_zt_detail[code]["连板数"] or (
                lb == stock_zt_detail[code]["连板数"] and ft < stock_zt_detail[code]["首次封板时间"]
            ):
                stock_zt_detail[code]["连板数"] = lb
                stock_zt_detail[code]["首次封板时间"] = ft

    potential_leaders = []
    for code, cnt10 in stock_zt_count_10.items():
        cnt5 = stock_zt_count_5.get(code, 0)
        if cnt5 >= 2 or cnt10 >= 3:
            info = stock_zt_detail[code].copy()
            info["涨停天数_5"] = cnt5
            info["涨停天数_10"] = cnt10
            info["涨停天数"] = cnt5   # 兼容后续原有逻辑
            potential_leaders.append(info)

    if not potential_leaders:
        print("  ❌ 无潜在龙头（近5日≥2 或 近10日≥3），路径A无结果")
        return []

    potential_leaders.sort(key=lambda x: (x["涨停天数_5"], x["连板数"]), reverse=True)
    leader_codes_set = {ldr["代码"] for ldr in potential_leaders}

    print(f"  潜在龙头数量: {len(potential_leaders)}")
    for ldr in potential_leaders[:10]:
        print(f"    {ldr['名称']}({ldr['代码']}) 近5日{ldr['涨停天数_5']}板 近10日{ldr['涨停天数_10']}板 连板{ldr['连板数']}")

    # ----------------------------------------------------------------
    # 第2步：板块映射——每只龙头只保留贡献最大的板块
    # ----------------------------------------------------------------
    print("\n  [第2步] 板块映射：龙头→贡献最大板块...")

    stock_concept_index = build_stock_concept_index()
    if not stock_concept_index:
        print("  ❌ 概念索引为空")
        return []

    # 2.1 收集每个龙头所属的全部板块（缺失时实时查询 akshare）
    leader_board_map = {}   # leader_code -> [(board_name, board_code), ...]
    for ldr in potential_leaders:
        code = ldr["代码"]
        concepts = stock_concept_index.get(code, [])
        if not concepts:
            # 不再使用东方财富接口，若缓存缺失则尝试反向遍历概念索引（耗时，仅尝试一次）
            print(f"    ⚠ {ldr['名称']}({code}) 无概念数据，尝试从索引反查...")
            # 遍历所有概念，检查该股是否在成分股中（仅当强制重建时可能补齐）
            # 此处不执行实时查询，直接跳过
            print(f"    ❌ {ldr['名称']}({code}) 跳过（无板块归属）")
        if not concepts:
            continue
        leader_board_map[code] = [(c["概念名称"], c["概念代码"]) for c in concepts]

    if not leader_board_map:
        print("  ❌ 无概念板块包含龙头")
        return []

    # 2.2 汇总所有涉及到的板块（去重）
    all_board_codes = set()
    for boards in leader_board_map.values():
        for _, bcode in boards:
            all_board_codes.add(bcode)

    # 2.3 拉取这些板块的成分股，并计算近3日日均涨停家数（作为贡献度指标）
    board_temp_info = {}   # board_code -> {"成分股代码集": set, "日均涨停家数": float}
    for bcode in all_board_codes:
        constituents = fetch_board_constituents(bcode, "")
        if constituents.empty:
            board_temp_info[bcode] = {"成分股代码集": set(), "日均涨停家数": 0.0}
            continue
        codes_set = set(constituents["代码"].tolist())
        daily_counts = []
        for d in recent_3_dates:
            lt_data = all_limit_up_data.get(d)
            if lt_data is not None and not lt_data.empty:
                lt_codes = set(lt_data["代码"].tolist())
                daily_counts.append(len(codes_set & lt_codes))
            else:
                daily_counts.append(0)
        avg_limit = float(np.mean(daily_counts)) if daily_counts else 0.0
        board_temp_info[bcode] = {"成分股代码集": codes_set, "日均涨停家数": avg_limit}

    # 2.4 为每个龙头选择贡献度最大的板块（按日均涨停家数）
    leader_to_board = {}   # leader_code -> (board_name, board_code)
    for code, boards in leader_board_map.items():
        best_board = None
        best_score = -1
        for bname, bcode in boards:
            score = board_temp_info.get(bcode, {}).get("日均涨停家数", 0.0)
            if score > best_score:
                best_score = score
                best_board = (bname, bcode)
        if best_board:
            leader_to_board[code] = best_board

    # 2.5 构建 board_candidate_pool（多个龙头可能指向同一板块）
    board_candidate_pool = {}
    for ldr in potential_leaders:
        code = ldr["代码"]
        if code not in leader_to_board:
            continue
        bname, bcode = leader_to_board[code]
        if bname not in board_candidate_pool:
            board_candidate_pool[bname] = {
                "板块名称": bname,
                "板块代码": bcode,
                "龙头列表": [],
                "成分股代码集": board_temp_info[bcode]["成分股代码集"],
            }
        board_candidate_pool[bname]["龙头列表"].append(ldr)

    if not board_candidate_pool:
        print("  ❌ 无概念板块符合条件")
        return []

    print(f"  板块映射后候选板块数量: {len(board_candidate_pool)}")

    # ----------------------------------------------------------------
    # 第3步：计算板块热度（涨幅中位数、上涨家数占比、日均涨停家数）
    # ----------------------------------------------------------------
    print("\n  [第3步] 计算板块热度...")
    MAX_SAMPLE = 50   # 采样前50只成分股用于热度指标计算

    # ---------- 动态上涨家数占比阈值：根据大盘环境调整 ----------
    threshold_up_ratio = 0.6   # 默认上升周期60%
    env_valid = False
    try:
        idx_df = fetch_index_daily(
            CONFIG.index_code,
            (pd.to_datetime(eval_date) - pd.Timedelta(days=60)).strftime("%Y%m%d"),
            date_str
        )
        if not idx_df.empty and len(idx_df) >= 20:
            close = idx_df["close"]
            ma20 = close.rolling(20).mean()
            on_ma20 = close.iloc[-1] >= ma20.iloc[-1]
            # 连续下跌天数（从最近往前数）
            pct = close.pct_change()
            consecutive_down = 0
            for i in range(len(pct) - 1, 0, -1):
                if pct.iloc[i] < 0:
                    consecutive_down += 1
                else:
                    break
            env_valid = True
            if not on_ma20:
                if consecutive_down >= 3:
                    threshold_up_ratio = 0.3
                    print(f"  📉 大盘位于20日线下方且连续下跌{consecutive_down}天，上涨家数占比阈值降至30%")
                else:
                    threshold_up_ratio = 0.4
                    print(f"  📉 大盘位于20日线下方，上涨家数占比阈值降至40%")
            else:
                print(f"  📈 大盘位于20日线上方，阈值保持60%")
    except Exception:
        pass  # 异常则默认60%

    # ---------- 动态日均涨停家数阈值 ----------
    if env_valid:
        if on_ma20:
            limit_up_threshold = 3.0
            print(f"  📈 上升周期，日均涨停家数阈值设为 {limit_up_threshold}")
        elif consecutive_down >= 3:
            limit_up_threshold = None  # 寒冬周期跳过涨停家数检查
            print(f"  🧊 寒冬周期，日均涨停家数条件跳过")
        else:
            limit_up_threshold = 1.5
            print(f"  📉 下跌周期，日均涨停家数阈值降至 {limit_up_threshold}")
    else:
        limit_up_threshold = CONFIG.board_daily_limit_up_min
        print(f"  使用默认日均涨停家数阈值 {limit_up_threshold}")

    boards_to_remove = []
    for board_name, board_info in board_candidate_pool.items():
        constituent_codes = board_info["成分股代码集"]
        if not constituent_codes:
            boards_to_remove.append(board_name)
            continue

        # 过滤有效代码（沪深主板、中小板、创业板）
        codes = [c for c in constituent_codes if c[:2] in ('60', '00', '30')]
        if not codes:
            boards_to_remove.append(board_name)
            continue

        # 限制采样数量
        sample_codes = codes[:MAX_SAMPLE]

        # ----- 计算每只样本股近3日涨幅，备用 -----
        stock_ret_list = []  # (code, ret)
        for code in sample_codes:
            ret = _read_stock_3d_ret(code, eval_date=date_str)
            if ret is not None:
                stock_ret_list.append((code, ret))
        # ----- 计算板块近3日涨幅：优先以流通市值前10的涨幅中位数 -----
        if stock_ret_list:
            # 获取样本股市值
            code_mcaps = {}
            for code in sample_codes:
                mcap = _get_stock_market_cap(code, date_str)
                if mcap is not None and mcap > 0:
                    code_mcaps[code] = mcap
            if len(code_mcaps) >= 10:
                top_codes = sorted(code_mcaps, key=code_mcaps.get, reverse=True)[:10]
                top_ret_list = [r for c, r in stock_ret_list if c in top_codes]
                ret_3d = float(np.median(top_ret_list)) if top_ret_list else -999.0
            else:
                # 市值数据不足，回退至涨幅前10
                stock_ret_list.sort(key=lambda x: x[1], reverse=True)
                top_n = min(10, len(stock_ret_list))
                top_rets = [x[1] for x in stock_ret_list[:top_n]]
                ret_3d = float(np.median(top_rets))
        else:
            ret_3d = -999.0
        board_info["近3日涨幅"] = ret_3d

        # ----- 计算上涨家数占比（基于样本股在近3日的涨跌） -----
        up_ratios = []
        for day_str in recent_3_dates:
            up_count = 0
            total = 0
            for code, _ in stock_ret_list:  # 复用已有涨幅的股票列表（确保有日线缓存）
                # 从缓存读取该日涨跌方向
                cache_file = os.path.join(DAILY_CACHE_DIR, f"{code}.csv")
                if not os.path.exists(cache_file):
                    # 缺失则尝试补拉最近20天
                    start_pull = (pd.to_datetime(day_str) - pd.Timedelta(days=20)).strftime("%Y%m%d")
                    fetch_stock_daily(code, start_pull, day_str)
                if not os.path.exists(cache_file):
                    continue
                try:
                    df = pd.read_csv(cache_file, parse_dates=["日期"])
                    close_col = "收盘价" if "收盘价" in df.columns else "收盘"
                    df['日期'] = pd.to_datetime(df['日期'])
                    today_row = df[df['日期'] == pd.to_datetime(day_str)]
                    if today_row.empty:
                        continue
                    close_today = today_row[close_col].iloc[0]
                    # 前一交易日
                    prev_day = (pd.to_datetime(day_str) - pd.Timedelta(days=1))
                    prev_row = df[df['日期'] == prev_day]
                    if prev_row.empty:
                        continue
                    close_prev = prev_row[close_col].iloc[0]
                    if close_today > close_prev:
                        up_count += 1
                    total += 1
                except Exception:
                    continue
            if total > 0:
                up_ratios.append(up_count / total)
        if up_ratios:
            avg_up_ratio = np.mean(up_ratios)
        else:
            avg_up_ratio = 0.0
        board_info["近3日上涨家数占比"] = avg_up_ratio

        # ----- 近3日每日涨停家数 -----
        daily_limit_count = []
        for d in recent_3_dates:
            lt_data = all_limit_up_data.get(d)
            if lt_data is not None and not lt_data.empty:
                lt_codes = set(lt_data["代码"].tolist())
                daily_limit_count.append(len(constituent_codes & lt_codes))
            else:
                daily_limit_count.append(0)
        board_info["涨停家数列表"] = daily_limit_count
        board_info["日均涨停家数"] = float(np.mean(daily_limit_count)) if daily_limit_count else 0.0

        # ----- 领涨龙头：连板数最高 -----
        sorted_leaders = sorted(
            board_info["龙头列表"],
            key=lambda x: (x.get("连板数", 0), x.get("涨停天数", 0)),
            reverse=True,
        )
        board_info["领涨龙头"] = sorted_leaders[0] if sorted_leaders else {}

        # ----- 上涨家数占比过滤（动态阈值 + 涨停家数兜底） -----
        avg_limit = board_info.get("日均涨停家数", 0)
        if avg_up_ratio < threshold_up_ratio:
            # 兜底条件：板块日均涨停家数 >= 3，即使占比低也保留
            if avg_limit >= 3:
                print(f"    💡 {board_name} 上涨家数占比 {avg_up_ratio:.1%} < {threshold_up_ratio:.0%}，但日均涨停 {avg_limit:.1f} ≥ 3，兜底保留")
            else:
                print(f"    ❌ {board_name} 上涨家数占比 {avg_up_ratio:.1%} < {threshold_up_ratio:.0%}，且日均涨停 {avg_limit:.1f} < 3，剔除")
                boards_to_remove.append(board_name)

    # 清除不满足热度条件的板块
    for bname in boards_to_remove:
        if bname in board_candidate_pool:
            del board_candidate_pool[bname]
    print(f"  上涨家数占比过滤后剩余板块: {len(board_candidate_pool)}")

    # ----------------------------------------------------------------
    # 第4步：涨幅排名前5%，计算板块涨幅排名分
    # ----------------------------------------------------------------
    board_returns_list = sorted(
        [(n, i["近3日涨幅"]) for n, i in board_candidate_pool.items() if i["近3日涨幅"] > -999],
        key=lambda x: x[1], reverse=True,
    )
    if not board_returns_list:
        print("  ❌ 无板块有有效涨幅数据")
        return []

    top_n = max(1, int(len(board_returns_list) * CONFIG.board_return_top_pct))
    top_threshold = board_returns_list[min(top_n, len(board_returns_list)) - 1][1]
    print(f"  前5%涨幅阈值: {top_threshold:.4f}（有效板块{len(board_returns_list)}个，前{top_n}个）")

    # 板块涨幅排名分（百分位：最高100，最低0）
    N_boards = len(board_returns_list)
    percentile_map = {}
    for rank_idx, (bname, ret) in enumerate(board_returns_list):
        if N_boards > 1:
            percentile = 100.0 * (N_boards - 1 - rank_idx) / (N_boards - 1)
        else:
            percentile = 100.0
        percentile_map[bname] = percentile

    # ----------------------------------------------------------------
    # 第5步：筛选同时满足条件的板块，评分公式: 日均涨停家数×10 + 连板数×5 + 板块涨幅排名分
    # ----------------------------------------------------------------
    qualified = []
    for board_name, info in board_candidate_pool.items():
        return_3d = info.get("近3日涨幅", -999)
        avg_limit = info.get("日均涨停家数", 0)
        leader = info.get("领涨龙头", {})
        # 涨幅达标 + 涨停家数条件（寒冬周期跳过涨停家数检查）
        if return_3d >= top_threshold:
            if limit_up_threshold is not None and avg_limit < limit_up_threshold:
                continue
            leader_lb = leader.get("连板数", 0) if leader else 0
            perc = percentile_map.get(board_name, 0)
            total_score = avg_limit * 10 + leader_lb * 5 + perc
            qualified.append({
                "板块代码": info["板块代码"],
                "板块名称": board_name,
                "近3日涨幅": return_3d,
                "日均涨停家数": avg_limit,
                "领涨龙头": leader,
                "成分股代码集": info["成分股代码集"],
                "排序分": total_score,          # 新评分公式
                "板块涨幅排名分": perc,
            })

    qualified.sort(key=lambda x: x["排序分"], reverse=True)
    main_sectors = qualified[: CONFIG.top_sectors_count]

    # 给每个主线板块设定优先级（S/A/B）
    for s in main_sectors:
        ldr = s["领涨龙头"]
        lb = ldr.get("连板数", 0)
        daily_limits = s.get("涨停家数列表", [])
        is_increasing = (
            len(daily_limits) >= 2 and all(
                daily_limits[i] < daily_limits[i+1] for i in range(len(daily_limits)-1)
            )
        )
        if lb >= 5 and is_increasing:
            s["priority"] = "S"
        elif lb >= 3:
            s["priority"] = "A"
        else:
            s["priority"] = "B"

    print(f"\n  主线板块筛选结果 ({len(main_sectors)}个):")
    for s in main_sectors:
        ldr = s["领涨龙头"]
        print(f"    {s['板块名称']} | 涨幅:{s['近3日涨幅']:.2%} | "
              f"日均涨停:{s['日均涨停家数']:.1f} | "
              f"龙头:{ldr.get('名称','?')}({ldr.get('代码','?')}) "
              f"涨停{ldr.get('涨停天数',0)}天 连板{ldr.get('连板数',0)} | "
              f"等级:{s['priority']}")

    return main_sectors


# ============================================================
# 新增：趋势型主线识别路径（无关龙头）
# ============================================================
def _fetch_board_index_cached(board_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """带缓存的板块指数日线拉取"""
    import os
    cache_file = os.path.join(BOARD_DAILY_CACHE_DIR, f"{board_code}.csv")
    os.makedirs(BOARD_DAILY_CACHE_DIR, exist_ok=True)

    # 尝试从缓存读取完整区间
    if os.path.exists(cache_file):
        try:
            cached = pd.read_csv(cache_file, parse_dates=["日期"])
            if not cached.empty:
                cached_start = cached["日期"].min().strftime("%Y%m%d")
                cached_end = cached["日期"].max().strftime("%Y%m%d")
                if cached_start <= start_date and cached_end >= end_date:
                    mask = (cached["日期"] >= start_date) & (cached["日期"] <= end_date)
                    return cached.loc[mask].reset_index(drop=True)
        except Exception:
            pass

    # 拉取并合并缓存
    board_idx = fetch_board_index(board_code, start_date, end_date)
    if board_idx.empty:
        return board_idx

    if os.path.exists(cache_file):
        try:
            old = pd.read_csv(cache_file, parse_dates=["日期"])
            combined = pd.concat([old, board_idx]).drop_duplicates(subset=["日期"]).sort_values("日期")
            combined.to_csv(cache_file, index=False)
        except Exception:
            board_idx.to_csv(cache_file, index=False)
    else:
        board_idx.to_csv(cache_file, index=False)

    return board_idx


def screen_trend_sectors(eval_date: str) -> List[Dict]:
    """
    趋势型主线识别（无龙头前置条件，纯本地缓存版本）。
    1. 获取所有概念板块。
    2. 板块3日涨幅：采样该板块最多5只成分股，直接读本地日线缓存计算均值（无网络请求）。
    3. 板块日均涨停家数：基于全部成分股和涨停池（无网络请求）。
    4. 截取涨幅前10%且日均涨停≥2的板块。
    """
    print("\n===== 趋势型主线识别（并行路径，仅用本地缓存） =====")
    date_str = to_date_str(eval_date)
    # 近3个交易日
    recent_3_dates = get_trade_dates(eval_date, 10)[-3:]  # 最近3个交易日

    # 获取所有概念板块列表
    boards = fetch_board_list()
    if boards.empty:
        return []

    # 准备涨停池数据
    print("  获取近3日涨停池...")
    all_limit_up_data = {}
    for d in recent_3_dates:
        lt = fetch_limit_up_pool(d)
        if lt is not None and not lt.empty:
            # 统一列名，避免不同数据源代码列不一致
            rename_map = {}
            for col in lt.columns:
                col_lower = str(col).lower()
                if col_lower in ("股票代码", "ts_code", "symbol") and "代码" not in lt.columns:
                    rename_map[col] = "代码"
            if rename_map:
                lt = lt.rename(columns=rename_map)
            all_limit_up_data[d] = lt.drop_duplicates(subset=["代码"])

    # 板块成分股映射
    board_constituents_map = _invert_stock_concept_index()
    if not board_constituents_map:
        print("  ⚠ 无成分股数据，趋势路径无法运行。请先执行每日构建缓存。")
        return []

    # 本地股票缓存目录（与 fetch_stock_daily 一致）
    stock_cache_dir = DAILY_CACHE_DIR

    print("  预加载全部股票近3日数据至内存（一次IO）...")
    # 收集所有板块需要考察的股票代码（去重）
    all_need_codes = set()
    for raw_code in board_constituents_map:
        codes = board_constituents_map[raw_code]
        filtered = {c for c in codes if c[:2] in ('60', '00', '30')}
        all_need_codes.update(filtered)
    print(f"  涉及股票数量: {len(all_need_codes)}")

    # 内存缓存：{code: {'ret_3d': float|None, 'directions': {date: 1/0}}}
    stock_mem_cache = {}
    for code in all_need_codes:
        cache_file = os.path.join(DAILY_CACHE_DIR, f"{code}.csv")
        record = {"ret_3d": None, "directions": {}}
        if not os.path.exists(cache_file):
            # 自动补拉最近20天
            start_pull = (pd.to_datetime(date_str) - pd.Timedelta(days=20)).strftime("%Y%m%d")
            fetch_stock_daily(code, start_pull, date_str)
        if not os.path.exists(cache_file):
            stock_mem_cache[code] = record
            continue
        try:
            df = pd.read_csv(cache_file, parse_dates=["日期"])
            close_col = "收盘价" if "收盘价" in df.columns else "收盘"
            df['日期'] = pd.to_datetime(df['日期'])
            closes = df[close_col].values
            if len(closes) >= 4:
                record["ret_3d"] = (closes[-1] - closes[-4]) / closes[-4]
            # 预计算3个交易日的涨跌方向（1涨，0跌）
            for d in recent_3_dates:
                try:
                    today_mask = df['日期'] == pd.to_datetime(d)
                    prev_day = pd.to_datetime(d) - pd.Timedelta(days=1)
                    prev_mask = df['日期'] == prev_day
                    if today_mask.any() and prev_mask.any():
                        close_today = df.loc[today_mask, close_col].iloc[0]
                        close_prev = df.loc[prev_mask, close_col].iloc[0]
                        record["directions"][d] = 1 if close_today > close_prev else 0
                    else:
                        record["directions"][d] = None
                except Exception:
                    record["directions"][d] = None
        except Exception:
            pass
        stock_mem_cache[code] = record

    # 预加载流通市值（可选，若耗时太长可跳过）
    code_mcap_cache = {}
    # 仅对市值函数可能返回非空做批量预取（本处暂保留单次调用，后续可加入tushare批量优化）
    # 为加速，这里简单跳过预取，市值信息在板块计算中按需要实时调用，但会减慢少量速度
    # 如需进一步提速，可去掉市值挑选逻辑，直接使用涨幅前10

    print(f"  遍历 {len(boards)} 个板块计算涨幅与涨停数...")
    board_records = []
    for idx, (_, row) in enumerate(boards.iterrows()):
        board_name = row["概念名称"]
        raw_code = str(row.get("概念代码", row.get("code", "")))
        if "/" in raw_code:
            board_code = raw_code.rstrip("/").split("/")[-1]
        else:
            board_code = raw_code

        constituent_codes = board_constituents_map.get(board_code, set())
        if not constituent_codes:
            continue

        filtered_codes = [c for c in constituent_codes if c[:2] in ('60', '00', '30')]
        if not filtered_codes:
            continue
        MAX_SAMPLE = 50
        sample_codes = filtered_codes[:MAX_SAMPLE]

        # 从内存获取近3日涨幅
        stock_ret_list = []
        for code in sample_codes:
            ret = stock_mem_cache.get(code, {}).get("ret_3d")
            if ret is not None:
                stock_ret_list.append((code, ret))
        if not stock_ret_list:
            continue

        # 市值中位数方案（保留原逻辑，但市值仍可能慢，此处先保留简单版本以提速）
        # 如果市值获取慢，可跳过市值环节，直接用涨幅前10
        # 这里提供快速版：直接涨幅前10中位数
        stock_ret_list.sort(key=lambda x: x[1], reverse=True)
        top_n = min(10, len(stock_ret_list))
        top_rets = [x[1] for x in stock_ret_list[:top_n]]
        ret_3d = float(np.median(top_rets))

        # 日均涨停家数
        daily_limit_counts = []
        for d in recent_3_dates:
            lt_data = all_limit_up_data.get(d)
            if lt_data is not None and not lt_data.empty:
                lt_codes = set(lt_data["代码"].tolist())
                daily_limit_counts.append(len(constituent_codes & lt_codes))
            else:
                daily_limit_counts.append(0)
        avg_limit = float(np.mean(daily_limit_counts))
        any_limit = any(c > 0 for c in daily_limit_counts)

        board_records.append({
            "板块代码": board_code,
            "板块名称": board_name,
            "近3日涨幅": ret_3d,
            "日均涨停家数": avg_limit,
            "近3日有过涨停": any_limit,
            "成分股代码集": constituent_codes,
        })

        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(boards)} ...")

    if not board_records:
        return []

    # 前10%涨幅阈值
    returns = [b["近3日涨幅"] for b in board_records]
    pct = (1 - CONFIG.board_return_top_pct_trend) * 100
    top_threshold = np.percentile(returns, pct)
    print(f"  全市场板块近3日涨幅前{CONFIG.board_return_top_pct_trend*100:.0f}%阈值: {top_threshold:.4f}")

    prelim_sectors = [
        b for b in board_records
        if b["近3日涨幅"] >= top_threshold and (b["日均涨停家数"] >= CONFIG.board_daily_limit_up_min_trend or b["近3日有过涨停"])
    ]
    # 趋势确认过滤
    print("  趋势确认过滤...")
    confirmed = []
    for s in prelim_sectors:
        bcode = s["板块代码"]
        # 1. 板块指数站上20日均线，成交量，跌幅
        idx_start = (pd.to_datetime(eval_date) - pd.Timedelta(days=60)).strftime("%Y%m%d")
        idx_df = _fetch_board_index_cached(bcode, idx_start, date_str)
        if idx_df.empty or len(idx_df) < 20:
            continue
        # 统一列名
        close_col = None
        vol_col = None
        for col in idx_df.columns:
            if col in ('收盘', '收盘价'):
                close_col = col
            if col in ('成交量', 'vol'):
                vol_col = col
        if close_col is None or vol_col is None:
            continue
        close = idx_df[close_col]
        vol = idx_df[vol_col]

        # 站上20日均线
        ma20 = close.rolling(20).mean()
        if close.iloc[-1] < ma20.iloc[-1]:
            continue

        # 近5日成交量均值 > 前20日均值 （要求至少有20+5的数据）
        if len(vol) < 25:
            continue
        vol5_mean = vol.iloc[-5:].mean()
        vol20_mean = vol.iloc[-20:].mean()
        if vol5_mean <= vol20_mean:
            continue

        # 近3日无单日跌幅超过3%
        pct_chg = close.pct_change()
        recent3 = pct_chg.iloc[-3:]
        if (recent3 < -0.03).any():
            continue

        # 2. 板块内上涨家数占比连续3天 > 50%
        constituents = s["成分股代码集"]
        up_ratio_days = []
        for d in recent_3_dates:
            up = 0
            total = 0
            for code in constituents:
                if code[:2] not in ('60', '00', '30'):
                    continue
                cache_file = os.path.join(DAILY_CACHE_DIR, f"{code}.csv")
                if not os.path.exists(cache_file):
                    continue
                try:
                    df = pd.read_csv(cache_file, parse_dates=["日期"])
                    close_col_stock = "收盘价" if "收盘价" in df.columns else "收盘"
                    df['日期'] = pd.to_datetime(df['日期'])
                    today_row = df[df['日期'] == pd.to_datetime(d)]
                    prev_day = pd.to_datetime(d) - pd.Timedelta(days=1)
                    prev_row = df[df['日期'] == prev_day]
                    if today_row.empty or prev_row.empty:
                        continue
                    close_today = today_row[close_col_stock].iloc[0]
                    close_prev = prev_row[close_col_stock].iloc[0]
                    if close_today > close_prev:
                        up += 1
                    total += 1
                except:
                    continue
            ratio = up / total if total > 0 else 0
            up_ratio_days.append(ratio)
        if not all(r > 0.5 for r in up_ratio_days):
            continue

        confirmed.append(s)

    trend_sectors = confirmed
    trend_sectors.sort(key=lambda x: x["近3日涨幅"], reverse=True)
    for s in trend_sectors:
        s["领涨龙头"] = {}
        s["priority"] = "B"   # 趋势型默认 B 级

    print(f"  趋势型主线板块数量: {len(trend_sectors)}")
    for s in trend_sectors:
        print(f"    {s['板块名称']} | 涨幅:{s['近3日涨幅']:.2%} | 日均涨停:{s['日均涨停家数']:.1f} | 等级:{s['priority']}")
    return trend_sectors


# ============================================================
# 第二步：圈定候选跟风股
# ============================================================
def screen_followers(main_sectors: List[Dict], eval_date: str) -> List[Dict]:
    """
    圈定候选跟风股：
    1.txt. 与龙头近60日收益率相关系数 >= 0.7
    2. 龙头首个涨停日，个股涨幅 >= 5% 或 量比 >= 1.txt.5
    3. 当前价距近3个月最高价 < 15%
    4. 流通市值 <= 300亿
    """
    print("\n===== 第二步：圈定候选跟风股 =====")

    date_str = to_date_str(eval_date)
    start_date = (pd.to_datetime(eval_date) - pd.Timedelta(days=CONFIG.data_lookback_days)).strftime("%Y%m%d")

    # 获取近3日涨停数据（用于板块涨停家数连续下降判断）
    recent_3_dates = get_trade_dates(eval_date, 10)[-3:]
    all_limit_up_data = {}
    for d in recent_3_dates:
        lt = fetch_limit_up_pool(d)
        if lt is not None and not lt.empty:
            # 统一列名
            rename_map = {}
            for col in lt.columns:
                col_lower = str(col).lower()
                if col_lower in ("股票代码", "ts_code", "symbol") and "代码" not in lt.columns:
                    rename_map[col] = "代码"
            if rename_map:
                lt = lt.rename(columns=rename_map)
            all_limit_up_data[d] = lt.drop_duplicates(subset=["代码"])

    trade_dates_all = get_trade_dates(eval_date, 20)  # 用于轮动识别
    candidates = []

    for sector in main_sectors:
        leader = sector.get("领涨龙头", {})
        leader_code = leader.get("代码", "")
        leader_name = leader.get("名称", "")
        board_name = sector["板块名称"]
        constituent_codes = sector.get("成分股代码集", set())

        use_sector_as_leader = (not leader_code)  # trend-based path: no individual leader

        # ----- 失效退出1：板块涨停家数连续两天下降 -----
        daily_limit_counts_board = []
        for d in recent_3_dates:
            lt_data = all_limit_up_data.get(d)
            if lt_data is not None and not lt_data.empty:
                lt_codes = set(lt_data["代码"].tolist())
                daily_limit_counts_board.append(len(constituent_codes & lt_codes))
            else:
                daily_limit_counts_board.append(0)
        if len(daily_limit_counts_board) >= 2 and len(daily_limit_counts_board) >= 3:
            # 检查最近三天：若后两天依次下降，则剔除
            if daily_limit_counts_board[-1] < daily_limit_counts_board[-2] < daily_limit_counts_board[-3]:
                print(f"    ❌ {board_name} 涨停家数连续下降 ({daily_limit_counts_board})，跳过该板块")
                continue

        print(f"\n  板块: {board_name}, 龙头: {leader_name if leader_name else '（无龙头，使用板块指数）'}"
              f"{'(' + leader_code + ')' if leader_code else ''}")
        print(f"  成分股数量: {len(constituent_codes)}")

        # Get proxy returns for correlation
        if use_sector_as_leader:
            sector_idx_df = _fetch_board_index_cached(sector["板块代码"], start_date, date_str)
            if sector_idx_df.empty or len(sector_idx_df) < 60:
                # 板块指数不可用 → 用成分股合成平均收益率序列作为代理
                print(f"    板块指数无数据，改用成分股合成代理...")
                synth_returns = None
                synth_codes = list(constituent_codes)[:5]  # 采样5只成分股
                for scode in synth_codes:
                    sdf = fetch_stock_daily(scode, start_date, date_str)
                    if sdf.empty or len(sdf) < 60:
                        continue
                    sdf = sdf.set_index("日期")
                    if "收盘" not in sdf.columns:
                        continue
                    ret_series = sdf["收盘"].pct_change().dropna()
                    if synth_returns is None:
                        synth_returns = ret_series
                    else:
                        common = synth_returns.index.intersection(ret_series.index)
                        synth_returns = (synth_returns.loc[common] + ret_series.loc[common]) / 2.0  # 简单平均
                if synth_returns is not None and len(synth_returns) >= 60:
                    proxy_returns = synth_returns
                    leader_first_zt_dt = pd.to_datetime(eval_date)
                    leader_lb = 1
                else:
                    print(f"    ⚠ 合成代理数据仍不足，跳过该板块")
                    continue
            else:
                sector_idx_df = sector_idx_df.set_index("日期")
                close_col = "收盘价" if "收盘价" in sector_idx_df.columns else "收盘"
                proxy_returns = sector_idx_df[close_col].pct_change().dropna()
                leader_first_zt_dt = pd.to_datetime(eval_date)
                leader_lb = 1
        else:
            leader_df = fetch_stock_daily(leader_code, start_date, date_str)
            if leader_df.empty or len(leader_df) < 60:
                print(f"    ⚠ 龙头数据不足，跳过该板块")
                continue
            leader_df = leader_df.set_index("日期")
            proxy_returns = leader_df["收盘"].pct_change().dropna()

            # 计算龙头近3日涨幅，用于相对强度过滤
            leader_ret_3d = _read_stock_3d_ret(leader_code, eval_date=date_str)

            leader_first_zt_date = leader["首次涨停日期"]
            leader_lb = leader["连板数"]
            if leader_lb > 1:
                try:
                    idx = trade_dates_all.index(leader_first_zt_date)
                    first_idx = max(0, idx - leader_lb + 1)
                    leader_first_zt_date = trade_dates_all[first_idx]
                except (ValueError, IndexError):
                    pass

            leader_first_zt_dt = pd.to_datetime(leader_first_zt_date)

            # ----- 失效退出2：龙头跌破5日线 -----
            if "收盘" not in leader_df.columns:
                continue
            leader_ma5 = leader_df["收盘"].rolling(5).mean()
            if len(leader_ma5) >= 5 and leader_df["收盘"].iloc[-1] < leader_ma5.iloc[-1]:
                print(f"    ❌ 龙头 {leader_name}({leader_code}) 跌破5日线，跳过该板块")
                continue

            # 计算龙头首次回调超3%的日期和涨幅（用于抗跌性验证）
            leader_callback_date = None
            leader_callback_pct = None
            leader_pct = leader_df["收盘"].pct_change()
            leader_pct = leader_pct[leader_pct.index >= leader_first_zt_dt]
            drop_mask = leader_pct < -0.03
            if drop_mask.any():
                leader_callback_date = drop_mask.idxmax()  # 第一个满足条件的索引
                leader_callback_pct = leader_pct.loc[leader_callback_date]

        stock_count = 0
        matched_count = 0
        for code in constituent_codes:
            if code[:2] not in ('60', '00', '30'):
                continue   # 跳过北交所等非沪深股票
            if code == leader_code:
                continue
            stock_count += 1

            stock_df = fetch_stock_daily(code, start_date, date_str)
            if stock_df.empty or len(stock_df) < 60:
                continue
            # 价格列统一为“收盘”
            if "收盘" not in stock_df.columns and "收盘价" in stock_df.columns:
                stock_df.rename(columns={"收盘价": "收盘"}, inplace=True)
            if "收盘" not in stock_df.columns:
                continue

            # 成分股活跃度预过滤：日均成交额、换手率
            if "成交额" in stock_df.columns and "换手率" in stock_df.columns:
                recent20 = stock_df.tail(20)
                avg_amount = recent20["成交额"].mean()
                avg_turnover = recent20["换手率"].mean()
                if avg_amount < CONFIG.min_daily_amount or avg_turnover < CONFIG.min_daily_turnover:
                    continue

            stock_df = stock_df.set_index("日期")
            stock_returns = stock_df["收盘"].pct_change().dropna()

            common_idx = proxy_returns.index.intersection(stock_returns.index)
            if len(common_idx) < 20:
                continue

            # ---- 事件型相关性：只统计龙头启动日之后 ----
            post_mask = common_idx >= leader_first_zt_dt
            if post_mask.sum() >= 5:
                common_idx_post = common_idx[post_mask]
                corr = proxy_returns.loc[common_idx_post].corr(
                    stock_returns.loc[common_idx_post]
                )
            else:
                corr = proxy_returns.loc[common_idx].corr(stock_returns.loc[common_idx])
            if pd.isna(corr) or corr < CONFIG.corr_threshold:
                continue

            if use_sector_as_leader:
                # 无龙头板块：跳过启动日异动和轮动级别
                zt_day_change = CONFIG.follower_change_threshold
                vol_ratio = CONFIG.follower_volume_ratio
                rotation_level = 1  # 默认一线
                ab_vol_ratio_5 = 0.0
            else:
                # ---- 启动日异动与轮动识别（实盘时间窗口适配） ----
                if leader_first_zt_dt not in stock_df.index:
                    continue
                if '开盘' not in stock_df.columns or '收盘' not in stock_df.columns or '成交量' not in stock_df.columns:
                    continue
                # 计算龙头涨停日与评估日的交易日差
                try:
                    leader_idx = trade_dates_all.index(leader_first_zt_date)
                    eval_idx = trade_dates_all.index(date_str)
                    days_since_break = eval_idx - leader_idx
                except ValueError:
                    days_since_break = 999  # 无法计算，保守处理跳过
                    continue

                # 确定观察窗口：龙头涨停日 ~ 评估日，最多3个交易日
                effective_dates = []
                for d in trade_dates_all[leader_idx:]:
                    if d in stock_df.index:
                        effective_dates.append(pd.to_datetime(d))
                    if len(effective_dates) >= 3:
                        break
                if not effective_dates:
                    continue

                # 仅当日窗口下，异动条件简化
                only_today = (days_since_break == 0)
                first_abnormal_dt = None
                cum_ret_final = 0.0
                max_vol_ratio_final = 0.0
                for i, d in enumerate(effective_dates):
                    # 当日累计涨幅（窗口第一天开盘至今）
                    try:
                        open_first = stock_df.loc[effective_dates[0], '开盘']
                        close_today = stock_df.loc[d, '收盘']
                        cum_ret = (close_today - open_first) / open_first if open_first != 0 else 0
                    except Exception:
                        cum_ret = -1
                    # 量比（相对于前20日均量）
                    try:
                        day_vol = stock_df.loc[d, '成交量']
                        past_vol = stock_df.loc[:d].iloc[:-1]['成交量'].tail(20).mean()
                        if pd.isna(past_vol) or past_vol == 0:
                            past_vol = day_vol
                        vratio = day_vol / past_vol
                    except Exception:
                        vratio = 0
                    # 次日跳空高开（只在非当日且有第二日时生效）
                    gap = 0
                    if i == 1 and not only_today:
                        try:
                            prev_close = stock_df.loc[effective_dates[0], '收盘']
                            today_open = stock_df.loc[d, '开盘']
                            if prev_close != 0:
                                gap = (today_open - prev_close) / prev_close
                        except Exception:
                            pass
                    # 异动判断
                    if only_today:
                        # 当日窗口：只用当日涨幅≥3%或量比≥1
                        if cum_ret >= 0.03 or vratio >= 1:
                            first_abnormal_dt = d
                            cum_ret_final = cum_ret
                            max_vol_ratio_final = vratio
                            break
                    else:
                        if cum_ret >= 0.03 or vratio >= 1 or gap >= 0.01:
                            first_abnormal_dt = d
                            cum_ret_final = cum_ret
                            max_vol_ratio_final = max(max_vol_ratio_final, vratio)
                            break
                    cum_ret_final = cum_ret
                    max_vol_ratio_final = max(max_vol_ratio_final, vratio)

                if first_abnormal_dt is None:
                    continue  # 异动失败

                # ----- 资金承接确认（异动非今日时） -----
                abn_date_str = to_date_str(first_abnormal_dt)
                if abn_date_str != date_str:  # 异动不是今天，需要检查次日数据
                    next_day = pd.to_datetime(abn_date_str) + pd.Timedelta(days=1)
                    # 尝试获取下一交易日（跳过周末）
                    next_trade_dates = get_trade_dates(abn_date_str, 2)  # 返回字符串列表，包含今天和明天
                    if len(next_trade_dates) >= 2:
                        next_date_str = next_trade_dates[1]
                        next_dt = pd.to_datetime(next_date_str)
                        if next_dt <= pd.to_datetime(date_str) and next_dt in stock_df.index:
                            try:
                                tomorrow_close = stock_df.loc[next_dt, '收盘']
                                tomorrow_vol = stock_df.loc[next_dt, '成交量']
                                ab_close = stock_df.loc[first_abnormal_dt, '收盘']
                                ab_low = stock_df.loc[first_abnormal_dt, '最低']
                                ab_vol = stock_df.loc[first_abnormal_dt, '成交量']
                                # 收盘不跌破异动日最低，成交量未萎缩超50%
                                if tomorrow_close < ab_low or tomorrow_vol < ab_vol * 0.5:
                                    continue  # 资金承接失败，淘汰
                            except Exception:
                                pass

                # 记录异动日最低价
                try:
                    ab_low = stock_df.loc[first_abnormal_dt, '最低']
                except Exception:
                    ab_low = None
                # 计算交易日差距（用于轮动级别）
                leader_zt_str = leader_first_zt_date
                try:
                    diff = abs(trade_dates_all.index(abn_date_str) - trade_dates_all.index(leader_zt_str))
                except ValueError:
                    diff = 999
                # 轮动级别
                if diff <= 1:
                    rotation_level = 1  # 一线跟风
                elif 2 <= diff <= 3:
                    rotation_level = 2  # 二线跟风
                else:
                    continue  # 补涨尾声，放弃

                # ----- 失效退出3：跟风股跌破异动当天最低价 -----
                current_close = stock_df["收盘"].iloc[-1]
                if ab_low is not None and current_close < ab_low:
                    continue

                zt_day_change = cum_ret_final
                vol_ratio = max_vol_ratio_final

                # 计算异动日成交量 / 前5日均量（用于优先级排序）
                try:
                    ab_vol = stock_df.loc[first_abnormal_dt, '成交量']
                    dates_before = stock_df.index[stock_df.index < first_abnormal_dt].tolist()
                    if len(dates_before) >= 5:
                        avg_vol_5 = stock_df.loc[dates_before[-5:], '成交量'].mean()
                        ab_vol_ratio_5 = ab_vol / avg_vol_5 if avg_vol_5 > 0 else 0.0
                    else:
                        ab_vol_ratio_5 = 0.0
                except Exception:
                    ab_vol_ratio_5 = 0.0

            # 相对强度过滤：跟风股3日涨幅 / 龙头3日涨幅 应介于0.5~1.5之间
            if not use_sector_as_leader and leader_ret_3d is not None:
                follower_ret_3d = _read_stock_3d_ret(code, eval_date=date_str)
                if follower_ret_3d is None:
                    continue
                ratio = follower_ret_3d / leader_ret_3d if leader_ret_3d != 0 else 999
                if ratio < 0.5 or ratio > 1.5:
                    continue

            # 抗跌性检查：龙头首次回调>3%时，跟风股当天跌幅必须小于龙头跌幅
            if not use_sector_as_leader and leader_callback_date is not None:
                try:
                    follower_cb = stock_returns.loc[leader_callback_date]
                except Exception:
                    continue
                if follower_cb < leader_callback_pct:
                    continue

            recent_high = stock_df["最高"].tail(CONFIG.high_price_lookback).max()
            current_close = stock_df["收盘"].iloc[-1]
            high_distance = (recent_high - current_close) / recent_high
            # [近高点跌幅过滤已移除]
            # [流通市值过滤已移除]

            # ---- 筹码安全垫：龙头启动前20日累计涨幅≤20% ----
            pre_end = leader_first_zt_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            pre_start = pre_end - pd.Timedelta(days=CONFIG.pre_event_days + 5)  # 留缓冲
            pre_cut = stock_df.loc[(stock_df.index < pre_end) & (stock_df.index >= pre_start)]
            if len(pre_cut) >= 2:
                # 取实际最后20个交易日
                pre_cut = pre_cut.tail(CONFIG.pre_event_days)
                if len(pre_cut) >= 2:
                    pre_gain = (pre_cut["收盘"].iloc[-1] - pre_cut["收盘"].iloc[0]) / pre_cut["收盘"].iloc[0]
                    if pre_gain > CONFIG.pre_event_max_gain:
                        continue

            matched_count += 1
            stock_name = code  # 无名称数据，使用代码代替

            candidates.append({
                "代码": code,
                "名称": stock_name,
                "板块": board_name,
                "对应龙头": leader_name,
                "龙头代码": leader_code,
                "相关系数": round(corr, 4),
                "龙头涨停日涨幅": round(zt_day_change, 4),
                "龙头涨停日量比": round(vol_ratio, 2),
                "近高点距离": round(high_distance, 4),
                "当前价格": current_close,
                "日线数据": stock_df,
                "轮动级别": rotation_level,       # 1=一线跟风, 2=二线跟风
                "首选": rotation_level == 1,       # 一线跟风标记为首选
                "异动日天数差": diff if not use_sector_as_leader else 999,
                "异动日量比": round(ab_vol_ratio_5, 2) if not use_sector_as_leader else 0.0,
                "相对强度比值": round(ratio, 4) if not use_sector_as_leader else None,
            })

        print(f"    遍历成分股: {stock_count}, 匹配: {matched_count}")

    print(f"\n  候选跟风股总数: {len(candidates)}")
    return candidates


# ============================================================
# 第三步：技术信号共振确认
# ============================================================
def check_macd_golden_cross(df: pd.DataFrame) -> bool:
    """检查MACD金叉"""
    close = df["收盘"]
    volume = df["成交量"]

    diff, dea, macd_hist = calc_macd(close)

    if len(diff) < 2 or pd.isna(diff.iloc[-1]) or pd.isna(dea.iloc[-1]):
        return False

    today_diff = diff.iloc[-1]
    today_dea = dea.iloc[-1]
    yesterday_diff = diff.iloc[-2]
    yesterday_dea = dea.iloc[-2]

    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    today_vol = volume.iloc[-1]
    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 0

    golden_cross = (today_diff > today_dea) and (yesterday_diff <= yesterday_dea)
    diff_ok = today_diff >= CONFIG.macd_diff_threshold
    vol_ok = vol_ratio > 1.0

    return golden_cross and diff_ok and vol_ok


def check_bollinger_buy(df: pd.DataFrame) -> bool:
    close = df["收盘"]
    volume = df["成交量"]
    upper, mid, lower = calc_bbands(close)

    if len(mid) < 3:
        return False

    today_close = close.iloc[-1]
    today_mid = mid.iloc[-1]
    yesterday_mid = mid.iloc[-2]
    mid_slope = today_mid - yesterday_mid

    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    today_vol = volume.iloc[-1]
    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 0

    # 放宽：允许价格已在中轨上方（不强制昨日≤中轨），只需中轨斜率≥0且量能满足
    above_mid = today_close > today_mid
    slope_ok = mid_slope >= 0
    vol_ok = vol_ratio >= 0.9  # 甚至可以降到0.8
    return above_mid and slope_ok and vol_ok


def check_vegas_buy(df: pd.DataFrame) -> bool:
    close = df["收盘"]
    if len(close) < max(CONFIG.vegas_periods):
        return False
    emas = calc_vegas_emas(close)
    today_close = close.iloc[-1]
    above_count = 0
    for period in CONFIG.vegas_periods:
        ema = emas[period].iloc[-1]
        if pd.isna(ema): return False
        if today_close > ema:
            above_count += 1
    return above_count >= 2  # 只需站上其中两条EMA


def confirm_signals(candidates: List[Dict], eval_date: str) -> List[Dict]:
    """
    技术面买点确认：
    硬性条件（三个必须同时满足）：
      - MACD柱子连续3根在零轴上方且递增
      - 股价在EMA12上方，且EMA12向上
      - 布林中轨向上，价格在中轨线上
    同时计算 KDJ、RSI 供人工参考。
    """
    print("\n===== 第三步：技术面买点确认 =====")

    results = []

    for i, stock in enumerate(candidates):
        df = stock["日线数据"]
        code = stock["代码"]
        name = stock["名称"]

        # 必须站上20日均线
        if len(df) < 20:
            continue
        close = df["收盘"]
        ma20 = close.rolling(20).mean().iloc[-1]
        if close.iloc[-1] < ma20:
            continue

        if i % 10 == 0:
            print(f"  进度: {i + 1}/{len(candidates)}")

        macd_ok = check_macd_golden_cross(df)
        boll_ok = check_bollinger_buy(df)
        vegas_ok = check_vegas_buy(df)

        if macd_ok and (boll_ok or vegas_ok):
            results.append({
                "代码": code,
                "名称": name,
                "所属板块": stock["板块"],
                "对应龙头": stock["对应龙头"],
                "相关系数": stock["相关系数"],
                "MACD金叉": "是",
                "布林买点": "是" if boll_ok else "否",
                "维加斯买点": "是" if vegas_ok else "否",
                "当前价格": round(stock["当前价格"], 2),
                "识别日期": eval_date,
                # 以下字段用于优先级排序
                "异动涨幅": stock.get("龙头涨停日涨幅", 0),
                "异动日天数差": stock.get("异动日天数差", 999),
                "异动日量比": stock.get("异动日量比", 0),
                "相对强度比值": stock.get("相对强度比值", None),
            })

    print(f"\n  技术共振信号数量: {len(results)}")
    return results


# ============================================================
# 主执行流程
# ============================================================
def get_exit_signal(
    df: pd.DataFrame,
    entry_idx: int,
    is_leader: bool = False,
    leader_broken: bool = False,
    entry_price: float = None,
    highest_since_entry: float = None,
) -> Tuple[bool, str]:
    """
    增强型离场判断，支持：
    - 龙头/跟风股差异化均线出场
    - 龙头断板联动止损
    - 移动止盈（从最高点回撤过多离场）

    Parameters
    ----------
    df : 个股日线（含'收盘','成交量'列）
    entry_idx : 入场日在df中的索引
    is_leader : 是否为主板领涨龙头，影响均线周期
    leader_broken : 龙头是否发生了首次放量断板（仅当 is_leader=False 时生效）
    entry_price : 入场价，用于计算盈利（移动止盈）
    highest_since_entry : 入场后最高收盘价，用于移动止盈
    """
    close = df["收盘"]
    idx_now = len(df) - 1
    if idx_now <= entry_idx + 1:
        return False, ""

    # 0. 龙头断板联动止损（对跟风股有效）
    if CONFIG.leader_break_exit_enabled and leader_broken and not is_leader:
        return True, "龙头断板联动止损"

    # 1. 移动止盈：盈利超阈值，回撤过大立即止盈
    if entry_price is not None and highest_since_entry is not None:
        profit = (close.iloc[-1] - entry_price) / entry_price
        if profit > CONFIG.trailing_profit_threshold:
            drawdown = (highest_since_entry - close.iloc[-1]) / highest_since_entry
            if drawdown > CONFIG.trailing_drawdown_limit:
                return True, f"移动止盈（盈利{profit:.1%}, 回撤{drawdown:.1%})"

    # 2. MACD 死叉
    if CONFIG.exit_use_macd:
        diff, dea, _ = calc_macd(close)
        if idx_now >= 2 and not pd.isna(diff.iloc[-1]) and not pd.isna(dea.iloc[-1]):
            if diff.iloc[-1] < dea.iloc[-1] and diff.iloc[-2] >= dea.iloc[-2]:
                return True, "MACD死叉"

    # 3. 收盘跌破布林下轨
    if CONFIG.exit_use_boll_lower:
        _, _, lower = calc_bbands(close)
        if not pd.isna(lower.iloc[-1]) and close.iloc[-1] < lower.iloc[-1]:
            return True, "跌破布林下轨"

    # 4. 均线过滤：龙头用长周期，跟风股用短周期
    if CONFIG.exit_use_ma5:
        ma_period = CONFIG.leader_exit_ma_period if is_leader else CONFIG.follower_exit_ma_period
        ma = close.rolling(ma_period).mean()
        if not pd.isna(ma.iloc[-1]) and close.iloc[-1] < ma.iloc[-1]:
            label = f"跌破{ma_period}日均线"
            return True, label

    return False, ""


def daily_identify(eval_date: Optional[str] = None) -> List[Dict]:
    """
    每日识别主流程（增强版：含市场过滤、仓位建议、预留回测接口）

    Parameters
    ----------
    eval_date : str, optional
        评估日期，格式 "YYYY-MM-DD" 或 "YYYYMMDD"，默认今天

    Returns
    -------
    List[Dict] : 识别结果列表（包含建议仓位和出场条件说明）
    """
    if eval_date is None:
        eval_date = datetime.date.today().strftime("%Y-%m-%d")
    else:
        eval_date = to_dash_date(eval_date)

    # 非交易日自动前推
    trade_date = get_latest_trade_date(eval_date)
    if trade_date != eval_date:
        print(f"⚠ {eval_date} 非交易日，自动前推至最近交易日: {trade_date}")
        eval_date = trade_date

    print("=" * 80)
    print("策略名称: 板块龙头共振识别策略（增强版）")
    print(f"识别日期: {eval_date}")
    print("=" * 80)

    # ========================== 市场环境过滤 ==========================
    index_trend = check_index_trend(eval_date)
    if not index_trend:
        print("\n⚠ 市场环境偏空，建议空仓")

    # ========================== 主线板块（整合两条路径，去重保留最高优先级） ==========================
    leader_sectors = screen_main_sectors(eval_date)   # 龙头路径（含 S/A 级）
    trend_sectors = screen_trend_sectors(eval_date)   # 趋势路径（B 级）

    # 按板块名称去重，保留最高优先级（S > A > B）
    priority_order = {"S": 3, "A": 2, "B": 1}
    merged = {}
    for s in leader_sectors + trend_sectors:
        name = s["板块名称"]
        if name not in merged or priority_order.get(s.get("priority", "B"), 0) > priority_order.get(merged[name].get("priority", "B"), 0):
            merged[name] = s
    main_sectors = list(merged.values())

    if not main_sectors:
        print("\n⚠ 无主线板块，识别结束")
        return []

    print(f"\n✅ 合并后主线板块: {[s['板块名称'] for s in main_sectors]}")
    for s in main_sectors:
        print(f"    {s['板块名称']} 等级:{s.get('priority','B')}")

    candidates = screen_followers(main_sectors, eval_date)
    if not candidates:
        print("\n⚠ 无候选跟风股，识别结束")
        return []

    print(f"\n✅ 候选跟风股数量: {len(candidates)}")

    results = confirm_signals(candidates, eval_date)
    if not results:
        print("\n⚠ 无技术共振信号")
        return []

    # ========================== 仓位分配（分层） ==========================
    # 补齐日线数据（被 confirm_signals 丢弃）
    for res in results:
        code = res["代码"]
        stock_df = None
        for c in candidates:
            if c["代码"] == code:
                stock_df = c["日线数据"]
                break
        res["日线数据"] = stock_df
        # 标记所属板块等级
        for s in main_sectors:
            if s["板块名称"] == res["所属板块"]:
                res["板块等级"] = s.get("priority", "B")
                break

    results = allocate_position_hierarchical(results, main_sectors)

    # ========================== 仓位决策参考（基于KDJ_D和RSI） ==========================
    for r in results:
        d_val = r.get("KDJ_D")
        rsi_val = r.get("RSI")
        # 默认提示
        suggestion = ""
        if d_val is not None and rsi_val is not None:
            try:
                d_val = float(d_val)
                rsi_val = float(rsi_val)
            except (ValueError, TypeError):
                d_val = rsi_val = None

        if d_val is not None and rsi_val is not None:
            if d_val < 20 and rsi_val < 30:
                suggestion = "高性价比买点，正常仓位"
            elif 45 <= d_val <= 55 and 40 <= rsi_val <= 50:
                suggestion = "中等位置，轻仓试探"
            elif d_val > 75 or rsi_val > 65:
                suggestion = "偏高位置，等回调或只做超短线"
        r["仓位参考建议"] = suggestion

    # ========================== 优先级排序（只保留前5只） ==========================
    def calc_priority(sig):
        score = 0.0
        # 1. 异动强度：异动当天涨幅越大越好，直接使用百分比数值*100
        score += abs(sig.get("异动涨幅", 0)) * 100

        # 2. 反应速度：天数差越小得分越高
        diff = sig.get("异动日天数差", 99)
        if diff <= 1:
            score += 10
        elif diff == 2:
            score += 6
        elif diff == 3:
            score += 2
        else:
            score += 0

        # 3. 相对强度：比值越接近1.35分越高
        ratio = sig.get("相对强度比值")
        if ratio is not None and 0.5 <= ratio <= 1.5:
            dist = abs(ratio - 1.35)
            sub = max(0, 10 - (dist / 0.15) * 10)  # 距离0.15得0分
            score += sub

        # 4. 成交量配合：异动日量比≥1.5加10分
        if sig.get("异动日量比", 0) >= 1.5:
            score += 10

        return score

    results.sort(key=calc_priority, reverse=True)
    # 只保留前5只（若信号不足则全部保留）
    results = results[:5]

    # ========================== 输出结果 ==========================
    print("\n" + "=" * 80)
    print("===== 识别结果 （含出场规则定义） =====")
    print("=" * 80)
    print("出场规则：")
    print("  - 龙头断板联动止损")
    print(f"  - 盈利>{CONFIG.trailing_profit_threshold:.0%}后，从最高点回撤>{CONFIG.trailing_drawdown_limit:.0%}触发移动止盈")
    if CONFIG.exit_use_macd:
        print("  - MACD死叉（DIFF下穿DEA）")
    if CONFIG.exit_use_boll_lower:
        print("  - 收盘价跌破布林带下轨")
    if CONFIG.exit_use_ma5:
        print(f"  - 跟风股跌破{CONFIG.follower_exit_ma_period}日均线，龙头跌破{CONFIG.leader_exit_ma_period}日均线")
    print()

    for r in results:
        print(f"{r['代码']} {r['名称']} | 板块:{r['所属板块']}({r.get('板块等级','B')}级) | "
              f"龙头:{r['对应龙头']} | MACD柱递增:{r.get('MACD柱递增','?')} | "
              f"EMA12上方:{r.get('EMA12上方','?')} | 布林中轨上方:{r.get('布林中轨上方','?')} | "
              f"KDJ_K:{r.get('KDJ_K','')} D:{r.get('KDJ_D','')} J:{r.get('KDJ_J','')} RSI:{r.get('RSI','')} | "
              f"建议仓位:{r.get('建议仓位',0):.2%} | 参考:{r.get('仓位参考建议','')}")

    result_df = pd.DataFrame(results)
    # 省略日线数据列避免打印过长
    display_cols = [c for c in result_df.columns if c != "日线数据"]
    print(result_df[display_cols].to_string(index=False))

    # ========================== 轮动预警（连续3天主线板块集合不同） ==========================
    current_sector_names = set(s["板块名称"] for s in main_sectors)
    _sector_rotation_history.append(current_sector_names)
    # 只保留最近3次记录
    if len(_sector_rotation_history) > 3:
        _sector_rotation_history.pop(0)

    if len(_sector_rotation_history) == 3:
        if all(
            current != prev
            for prev, current in zip(_sector_rotation_history, _sector_rotation_history[1:])
        ):
            print("\n⚠ 轮动预警：连续3天主线板块不同，建议轻仓参与")

    print(f"\n===== 识别完毕，共 {len(results)} 只个股 =====")
    return results


# ============================================================
# 回测函数：2026年5月全部交易日，展示选股持有10天涨跌幅
# ============================================================
def backtest_may_2026():
    """
    回测2026年5月所有交易日：
    - 对每天运行 daily_identify 获取选股
    - 结果按天保存到 backtest_results/ 目录
    - 计算每只选出股自选出日（T日）起持有10个交易日的涨跌幅
    - 输出汇总表
    """
    # 获取2026年5月的所有交易日（以20260531为基准向前取31个日期，再过滤）
    all_trade_dates = get_trade_dates("20260531", 31)
    may_dates = [d for d in all_trade_dates if d.startswith("202605")]
    if not may_dates:
        print("❌ 未找到2026年5月的交易日")
        return

    print(f"📅 2026年5月共有 {len(may_dates)} 个交易日: {may_dates}")

    # 创建保存每日结果的目录
    os.makedirs("backtest_results", exist_ok=True)

    all_signals = []  # 用于最终汇总

    for eval_date_str in may_dates:
        eval_date_fmt = to_dash_date(eval_date_str)
        print(f"\n{'='*40}\n回测日期: {eval_date_fmt}\n{'='*40}")

        # ---- 当日选股 ----
        try:
            signals = daily_identify(eval_date_fmt)
        except Exception as e:
            print(f"  ❌ {eval_date_fmt} 选股异常: {e}")
            continue

        if not signals:
            print(f"  ℹ️ {eval_date_fmt} 无选股信号")
            continue

        # ---- 保存当日原始结果 ----
        day_df = pd.DataFrame(signals)
        # 过滤掉 DataFrame 中可能存在的非可序列化字段
        save_df = day_df.drop(columns=["日线数据"], errors="ignore")
        day_file = os.path.join("backtest_results", f"{eval_date_str}.csv")
        save_df.to_csv(day_file, index=False, encoding="utf_8_sig")
        print(f"  💾 当日信号已保存至 {day_file}")

        # ---- 计算每只选出股持有10个交易日的涨跌幅 ----
        # 获取从 eval_date 开始的未来交易日列表（取30个确保够用）
        future_end = (pd.to_datetime(eval_date_str) + pd.Timedelta(days=60)).strftime("%Y%m%d")
        future_all = get_trade_dates(future_end, 30)
        future_dates = [d for d in future_all if d >= eval_date_str]
        if len(future_dates) < 11:
            print(f"  ⚠ {eval_date_fmt} 未来交易日不足10天，跳过持有收益计算")
            continue

        t_plus_10 = future_dates[10]  # 第10个交易日（T+10）

        for s in signals:
            code = s["代码"]
            name = s["名称"]
            entry_price = s.get("当前价格")
            if entry_price is None:
                print(f"    {code} 缺少入场价，跳过")
                continue

            try:
                # 获取 T+10 日的数据
                t10_df = fetch_stock_daily(code, t_plus_10, t_plus_10)
                if t10_df.empty or "收盘" not in t10_df.columns:
                    print(f"    {code} 无法获取 {t_plus_10} 的收盘价")
                    continue
                exit_price = t10_df["收盘"].iloc[-1]
                pnl = (exit_price - entry_price) / entry_price
                s["T+10日期"] = to_dash_date(t_plus_10)
                s["持有10日涨跌幅"] = round(pnl, 4)
            except Exception as e:
                print(f"    {code} 收益计算失败: {e}")
                s["T+10日期"] = ""
                s["持有10日涨跌幅"] = None

        # 累加到汇总列表
        all_signals.extend(signals)

    # ---- 最终汇总展示 ----
    if all_signals:
        print("\n" + "=" * 80)
        print("📊 2026年5月回测汇总 - 持有10日涨跌幅")
        print("=" * 80)
        result_df = pd.DataFrame(all_signals)
        display_cols = [
            "识别日期", "代码", "名称", "所属板块", "对应龙头",
            "当前价格", "T+10日期", "持有10日涨跌幅", "建议仓位"
        ]
        # 只保留实际存在的列
        available_cols = [c for c in display_cols if c in result_df.columns]
        print(result_df[available_cols].to_string(index=False))

        # 也可保存汇总文件
        result_df[available_cols].to_csv("backtest_202605_summary.csv", index=False, encoding="utf_8_sig")
        print("\n汇总结果已保存至 backtest_202605_summary.csv")
    else:
        print("\n⚠ 整个5月无任何符合条件的信号")






# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # 填入你的 Tushare token（从 https://tushare.pro 个人中心获取）
    tushare_token = "b9dcf9759a297c0a8fef5cf8b73d7d09af50c31444c041c115ba66d7"
    if tushare_token:
        ok = init_tushare(tushare_token)
        if ok:
            print("✅ Tushare 已初始化")
        else:
            print("❌ Tushare 初始化失败，请检查 token 或安装 tushare")
    else:
        print("⚠ 未设置 Tushare token，将使用爬虫（速度慢）")
    result = daily_identify()
    # 首次写入：强制重建概念和行业缓存
    # print("\n===== 开始构建概念缓存 =====")
    # build_stock_concept_index(force_refresh=True)  # 强制重建
    # print("\n===== 开始构建行业缓存 =====")
    # build_industry_index_sw()
    # print("\n✅ 所有缓存写入完成")


