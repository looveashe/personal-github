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

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import akshare as ak

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
    exit_ma5_period: int = 5

    # 板块筛选
    board_return_top_pct: float = 0.05
    board_daily_limit_up_min: int = 3
    leader_consecutive_days: int = 3
    top_sectors_count: int = 2

    # 跟风股筛选
    corr_threshold: float = 0.7
    corr_lookback: int = 60
    follower_change_threshold: float = 0.05
    follower_volume_ratio: float = 1.5
    high_price_distance: float = 0.15
    high_price_lookback: int = 63
    market_cap_limit: float = 10000

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_diff_threshold: float = -0.5

    # 布林带
    bbands_period: int = 20
    bbands_nbdev: int = 2
    bbands_volume_ratio: float = 1.2

    # 维加斯通道
    vegas_periods: Tuple[int, int, int] = (12, 144, 169)

    # 数据
    data_lookback_days: int = 400
    request_delay: float = 3.0           # 改为3秒，大幅降低被断连概率
    max_retries: int = 5                 # 更多重试机会


CONFIG = StrategyConfig()

DAILY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "daily")


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
def _safe_ak_call(func, *args, description="", **kwargs):
    """
    带指数退避重试的 akshare 请求包装器。
    处理 ConnectionError、超时等网络异常。
    """
    last_error = None
    for attempt in range(1, CONFIG.max_retries + 1):
        try:
            time.sleep(CONFIG.request_delay + random.uniform(0, 0.2))
            result = func(*args, **kwargs)
            return result
        except (
            requests.exceptions.RequestException,  # 捕获所有 requests 异常，含 RemoteDisconnected
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
    """获取同花顺概念板块列表"""
    print("  获取概念板块列表...")
    df = _safe_ak_call(ak.stock_board_concept_name_ths, description="概念板块列表")
    if df is None:
        return pd.DataFrame()
    # 统一列名：兼容中英文两种列名格式
    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "code" and "概念代码" not in df.columns and "代码" not in df.columns:
            rename_map[col] = "概念代码"
        elif col_lower == "name" and "概念名称" not in df.columns:
            rename_map[col] = "概念名称"
    if rename_map:
        df = df.rename(columns=rename_map)
    print(f"  概念板块数量: {len(df)}, 列名: {list(df.columns)}")
    return df


def fetch_board_index(board_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取板块指数K线（同花顺）"""
    result = _safe_ak_call(
        ak.stock_board_concept_index_ths,
        symbol=board_code,
        start_date=start_date,
        end_date=end_date,
        description=f"板块指数 {board_code}",
    )
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        result["日期"] = pd.to_datetime(result["日期"])
        result = result.sort_values("日期").reset_index(drop=True)
    return result


def _parse_html_tables(text: str) -> List[pd.DataFrame]:
    """安全解析 HTML 表格，抑制 lxml/html5lib 的解析输出"""
    # 预检查：确认 HTML 中包含 <table 标签
    if "<table" not in text.lower():
        return []

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        return pd.read_html(io.StringIO(text))
    except ValueError:
        return []
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def fetch_board_constituents(board_code: str, board_name: str = "") -> pd.DataFrame:
    """获取板块成分股（解析板块主页HTML表格，不走AJAX接口）"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    all_records = []
    session = requests.Session()
    session.headers.update(headers)

    main_url = f"http://q.10jqka.com.cn/gn/detail/code/{board_code}/"

    for attempt in range(1, CONFIG.max_retries + 1):
        try:
            time.sleep(CONFIG.request_delay + random.uniform(0, 0.2))
            resp = session.get(main_url, timeout=15)
            resp.raise_for_status()

            # 主页用 utf-8，AJAX 分页用 gbk
            try:
                text = resp.content.decode("gbk")
            except Exception:
                text = resp.text

            # 从主页 HTML 中找成分股表格（使用安全解析函数）
            dfs = _parse_html_tables(text)
            if not dfs:
                return pd.DataFrame()

            # 找到成分股表格（通常第一个含代码列的表格）
            found = False
            for df in dfs:
                if df.empty or df.shape[1] < 3:
                    continue
                # 检查第2列是否像股票代码（6位数字）
                sample = str(df.iloc[:, 1].dropna().iloc[0]) if len(df) > 0 else ""
                if not (sample.replace(" ", "").isdigit() and 5 <= len(sample.replace(" ", "")) <= 6):
                    continue

                for _, row in df.iterrows():
                    code = str(row.iloc[1]).replace(" ", "").zfill(6)
                    name = str(row.iloc[2]).replace(" ", "")
                    if code.isdigit() and len(code) == 6:
                        all_records.append({"代码": code, "名称": name})
                found = True
                break

            if not found:
                return pd.DataFrame()

            # 如果主页只有第一页（通常50条），尝试翻页加载更多
            if len(all_records) >= 40:
                for page in range(2, 30):
                    ajax_url = f"http://q.10jqka.com.cn/gn/detail/order/desc/page/{page}/ajax/1/code/{board_code}/"
                    ajax_headers = {
                        "Referer": main_url,
                        "X-Requested-With": "XMLHttpRequest",
                        "Accept": "text/html, */*; q=0.01",
                    }
                    try:
                        time.sleep(CONFIG.request_delay + random.uniform(0, 0.2))
                        resp2 = session.get(ajax_url, headers=ajax_headers, timeout=15)
                        resp2.raise_for_status()
                        try:
                            page_text = resp2.content.decode("gbk")
                        except Exception:
                            page_text = resp2.text
                        if not page_text.strip() or "暂无数据" in page_text:
                            break
                        dfs2 = _parse_html_tables(page_text)
                        if not dfs2 or dfs2[0].empty:
                            break
                        df2 = dfs2[0]
                        page_added = 0
                        for _, row in df2.iterrows():
                            code = str(row.iloc[1]).replace(" ", "").zfill(6)
                            name = str(row.iloc[2]).replace(" ", "")
                            if code.isdigit() and len(code) == 6:
                                all_records.append({"代码": code, "名称": name})
                                page_added += 1
                        if page_added == 0 or len(df2) < 40:
                            break
                    except Exception:
                        break

            break
        except Exception as e:
            if attempt < CONFIG.max_retries:
                wait = CONFIG.request_delay * (2 ** attempt)
                time.sleep(wait)
            else:
                print(f"    ❌ 请求失败 (板块成分股 {board_code}): {e}")
                return pd.DataFrame()

    if not all_records:
        return pd.DataFrame()

    result = pd.DataFrame(all_records)
    result["板块代码"] = board_code
    return result


def fetch_limit_up_pool(trade_date: str) -> pd.DataFrame:
    """获取某日涨停板数据"""
    result = _safe_ak_call(
        ak.stock_zt_pool_em,
        date=trade_date,
        description=f"涨停板 {trade_date}",
    )
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        result["日期"] = trade_date
    return result


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

    # 2. 从新浪接口获取数据（仅此一个数据源）
    result = _safe_ak_call(
        ak.stock_zh_a_daily,
        symbol=f"{'sh' if code.startswith(('6','9')) else 'sz'}{code}",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
        description=f"个股日线(新浪) {code}",
    )

    if result is None or result.empty:
        return pd.DataFrame()

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


def fetch_index_daily(index_code: str = "000300", start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """获取大盘指数日K线（沪深300/上证等）"""
    symbol_map = {
        "000300": "sh000300",
        "000001": "sh000001",
        "399001": "sz399001",
    }
    symbol = symbol_map.get(index_code, index_code)
    if start_date is None:
        start_date = (pd.to_datetime("today") - pd.Timedelta(days=CONFIG.index_start_offset)).strftime("%Y%m%d")
    if end_date is None:
        end_date = pd.to_datetime("today").strftime("%Y%m%d")

    result = _safe_ak_call(
        ak.stock_zh_index_daily,
        symbol=symbol,
        description=f"大盘指数 {index_code}",
    )
    if result is None or result.empty:
        return pd.DataFrame()
    # 标准列名：date, open, high, low, close, volume
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)
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
    构建 stock_code → [{概念名称, 概念代码}, ...] 反向索引。
    首次/超30天自动全量下载板块成分股（约3-5分钟），之后秒查。
    """
    # ---- 检查缓存：30天内有效且非空 ----
    if not force_refresh and os.path.exists(STOCK_CONCEPT_CACHE):
        mtime = datetime.date.fromtimestamp(os.path.getmtime(STOCK_CONCEPT_CACHE))
        if (datetime.date.today() - mtime).days < 30:
            print(f"  ✓ 加载本地缓存 ({mtime}): data/stock_concept_index.json")
            with open(STOCK_CONCEPT_CACHE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                return cached
            print("  ⚠ 缓存为空，重新构建...")

    print("  ⏳ 构建股票→概念反向索引（首次需下载全部板块成分股，约3-5分钟，30天仅一次）...")

    boards = fetch_board_list()
    if boards.empty:
        return {}

    stock_concept_map = {}
    total = len(boards)

    for idx, (_, row) in enumerate(boards.iterrows()):
        board_name = row["概念名称"]
        raw_code = str(row.get("概念代码", row.get("code", "")))
        if "/" in raw_code:
            board_num_code = raw_code.rstrip("/").split("/")[-1]
        else:
            board_num_code = raw_code

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"    进度: {idx+1}/{total}")

        constituents = fetch_board_constituents(board_num_code, board_name)
        if constituents.empty:
            continue

        for _, stock_row in constituents.iterrows():
            code = stock_row["代码"]
            if code not in stock_concept_map:
                stock_concept_map[code] = []
            stock_concept_map[code].append({
                "概念名称": board_name,
                "概念代码": board_num_code,
            })

    # ---- 写入缓存 ----
    cache_dir = os.path.dirname(STOCK_CONCEPT_CACHE)
    os.makedirs(cache_dir, exist_ok=True)
    with open(STOCK_CONCEPT_CACHE, "w", encoding="utf-8") as f:
        json.dump(stock_concept_map, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 索引已缓存至: {STOCK_CONCEPT_CACHE}，覆盖 {len(stock_concept_map)} 只股票")

    return stock_concept_map


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

    print(f"  评估日期: {eval_date}")
    print(f"  近5个交易日: {recent_5_dates}")

    # ----------------------------------------------------------------
    # 第1步：找出潜在龙头（近5日涨停>=4天）
    # ----------------------------------------------------------------
    print("\n  [第1步] 找出潜在龙头（近5日涨停>=4天）...")

    all_limit_up_data = {}
    for d in recent_5_dates:
        lt_data = fetch_limit_up_pool(d)
        if lt_data is not None and not lt_data.empty:
            lt_data = lt_data.drop_duplicates(subset=["代码"])
        all_limit_up_data[d] = lt_data

    stock_zt_count = {}
    stock_zt_detail = {}

    for d in recent_5_dates:
        lt_data = all_limit_up_data.get(d)
        if lt_data is None or lt_data.empty:
            continue
        for _, row in lt_data.iterrows():
            code = row["代码"]
            stock_zt_count[code] = stock_zt_count.get(code, 0) + 1
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
    for code, count in stock_zt_count.items():
        if count >= 4:
            info = stock_zt_detail[code]
            info["涨停天数"] = count
            potential_leaders.append(info)

    if not potential_leaders:
        print("  ❌ 无潜在龙头（近5日涨停>=4天），识别结束")
        return []

    potential_leaders.sort(key=lambda x: (x["涨停天数"], x["连板数"]), reverse=True)
    leader_codes_set = {ldr["代码"] for ldr in potential_leaders}

    print(f"  潜在龙头数量: {len(potential_leaders)}")
    for ldr in potential_leaders[:10]:
        print(f"    {ldr['名称']}({ldr['代码']}) 涨停{ldr['涨停天数']}天 连板{ldr['连板数']}")

    # ----------------------------------------------------------------
    # 第2步：检索龙头所属的全部概念板块（从缓存反向索引秒查）
    # ----------------------------------------------------------------
    print("\n  [第2步] 检索龙头所属的全部概念板块...")

    stock_concept_index = build_stock_concept_index()
    if not stock_concept_index:
        print("  ❌ 概念索引为空")
        return []

    board_candidate_pool = {}

    for ldr in potential_leaders:
        code = ldr["代码"]
        concepts = stock_concept_index.get(code, [])
        if not concepts:
            print(f"    ⚠ {ldr['名称']}({code}) 无概念板块数据")
            continue

        for c in concepts:
            board_name = c["概念名称"]
            if board_name not in board_candidate_pool:
                board_candidate_pool[board_name] = {
                    "板块名称": board_name,
                    "板块代码": c["概念代码"],
                    "龙头列表": [],
                    "成分股代码集": set(),
                }
            board_candidate_pool[board_name]["龙头列表"].append(ldr)

    if not board_candidate_pool:
        print("  ❌ 无概念板块包含龙头")
        return []

    # 仅对候选板块（通常 10-30 个）拉取成分股
    for board_name, board_info in board_candidate_pool.items():
        constituents = fetch_board_constituents(board_info["板块代码"], board_name)
        if not constituents.empty:
            board_info["成分股代码集"] = set(constituents["代码"].tolist())

    print(f"  有龙头的概念板块数量: {len(board_candidate_pool)}")

    # ----------------------------------------------------------------
    # 第3步：计算板块近3日涨幅和日均涨停家数
    # ----------------------------------------------------------------
    print("\n  [第3步] 计算板块近3日涨幅和日均涨停家数...")

    start_d = (pd.to_datetime(eval_date) - pd.Timedelta(days=30)).strftime("%Y%m%d")

    for board_name, board_info in board_candidate_pool.items():
        # --- 板块近3日涨幅 ---
        board_idx = fetch_board_index(board_name, start_d, date_str)
        if board_idx.empty or len(board_idx) < 4:
            board_info["近3日涨幅"] = -999.0
        else:
            close_col = "收盘价" if "收盘价" in board_idx.columns else "收盘"
            closes = board_idx[close_col].values
            board_info["近3日涨幅"] = float((closes[-1] - closes[-4]) / closes[-4])

        # --- 近3日每日涨停家数 ---
        daily_limit_count = []
        constituent_codes = board_info["成分股代码集"]
        for d in recent_3_dates:
            lt_data = all_limit_up_data.get(d)
            if lt_data is not None and not lt_data.empty:
                lt_codes = set(lt_data["代码"].tolist())
                daily_limit_count.append(len(constituent_codes & lt_codes))
            else:
                daily_limit_count.append(0)

        board_info["涨停家数列表"] = daily_limit_count
        board_info["日均涨停家数"] = float(np.mean(daily_limit_count)) if daily_limit_count else 0.0

        # --- 领涨龙头：连板数最高 ---
        sorted_leaders = sorted(
            board_info["龙头列表"],
            key=lambda x: (x.get("连板数", 0), x.get("涨停天数", 0)),
            reverse=True,
        )
        board_info["领涨龙头"] = sorted_leaders[0] if sorted_leaders else {}

    # ----------------------------------------------------------------
    # 第4步：涨幅排名前5%
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

    # ----------------------------------------------------------------
    # 第5步：筛选同时满足条件的板块
    # ----------------------------------------------------------------
    qualified = []
    for board_name, info in board_candidate_pool.items():
        return_3d = info.get("近3日涨幅", -999)
        avg_limit = info.get("日均涨停家数", 0)
        leader = info.get("领涨龙头", {})

        if return_3d >= top_threshold and avg_limit >= CONFIG.board_daily_limit_up_min:
            qualified.append({
                "板块代码": info["板块代码"],
                "板块名称": board_name,
                "近3日涨幅": return_3d,
                "日均涨停家数": avg_limit,
                "领涨龙头": leader,
                "成分股代码集": info["成分股代码集"],
                "排序分": avg_limit * 10 + leader.get("连板数", 0),
            })

    qualified.sort(key=lambda x: x["排序分"], reverse=True)
    main_sectors = qualified[: CONFIG.top_sectors_count]

    print(f"\n  主线板块筛选结果 ({len(main_sectors)}个):")
    for s in main_sectors:
        ldr = s["领涨龙头"]
        print(f"    {s['板块名称']} | 涨幅:{s['近3日涨幅']:.2%} | "
              f"日均涨停:{s['日均涨停家数']:.1f} | "
              f"龙头:{ldr.get('名称','?')}({ldr.get('代码','?')}) "
              f"涨停{ldr.get('涨停天数',0)}天 连板{ldr.get('连板数',0)}")

    return main_sectors


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

    candidates = []

    for sector in main_sectors:
        leader = sector["领涨龙头"]
        leader_code = leader["代码"]
        leader_name = leader["名称"]
        board_name = sector["板块名称"]
        constituent_codes = sector.get("成分股代码集", set())

        print(f"\n  板块: {board_name}, 龙头: {leader_name}({leader_code})")
        print(f"  成分股数量: {len(constituent_codes)}")

        leader_df = fetch_stock_daily(leader_code, start_date, date_str)
        if leader_df.empty or len(leader_df) < 60:
            print(f"    ⚠ 龙头数据不足，跳过该板块")
            continue

        leader_df = leader_df.set_index("日期")
        leader_returns = leader_df["收盘"].pct_change().dropna()

        leader_first_zt_date = leader["首次涨停日期"]
        leader_lb = leader["连板数"]
        if leader_lb > 1:
            trade_dates_all = get_trade_dates(eval_date, 20)
            try:
                idx = trade_dates_all.index(leader_first_zt_date)
                first_idx = max(0, idx - leader_lb + 1)
                leader_first_zt_date = trade_dates_all[first_idx]
            except (ValueError, IndexError):
                pass

        leader_first_zt_dt = pd.to_datetime(leader_first_zt_date)

        stock_count = 0
        matched_count = 0
        for code in constituent_codes:
            if code == leader_code:
                continue
            stock_count += 1

            stock_df = fetch_stock_daily(code, start_date, date_str)
            if stock_df.empty or len(stock_df) < 60:
                continue

            stock_df = stock_df.set_index("日期")
            stock_returns = stock_df["收盘"].pct_change().dropna()

            common_idx = leader_returns.index.intersection(stock_returns.index)
            if len(common_idx) < 30:
                continue

            corr = leader_returns.loc[common_idx].corr(stock_returns.loc[common_idx])
            if pd.isna(corr) or corr < CONFIG.corr_threshold:
                continue

            if leader_first_zt_dt not in stock_df.index:
                continue

            zt_day_data = stock_df.loc[leader_first_zt_dt]
            if isinstance(zt_day_data, pd.DataFrame):
                zt_day_data = zt_day_data.iloc[0]

            zt_day_change = zt_day_data.get("涨跌幅", 0)
            if pd.isna(zt_day_change):
                zt_day_change = 0
            zt_day_change = zt_day_change / 100 if abs(zt_day_change) > 1 else zt_day_change

            zt_day_volume = zt_day_data.get("成交量", 0)
            avg_vol_20 = stock_df["成交量"].rolling(20).mean()
            if leader_first_zt_dt in avg_vol_20.index:
                vol_20 = avg_vol_20.loc[leader_first_zt_dt]
            else:
                vol_20 = stock_df["成交量"].tail(20).mean()
            vol_ratio = zt_day_volume / vol_20 if vol_20 > 0 else 0

            if not (zt_day_change >= CONFIG.follower_change_threshold or vol_ratio >= CONFIG.follower_volume_ratio):
                continue

            recent_high = stock_df["最高"].tail(CONFIG.high_price_lookback).max()
            current_close = stock_df["收盘"].iloc[-1]
            high_distance = (recent_high - current_close) / recent_high
            if high_distance >= CONFIG.high_price_distance:
                continue

            market_cap = 0
            try:
                if "流通市值" in stock_df.columns:
                    market_cap = stock_df["流通市值"].iloc[-1]
            except Exception:
                pass

            if market_cap > 0 and market_cap / 1e8 > CONFIG.market_cap_limit:
                continue

            matched_count += 1
            stock_name = ""
            try:
                stock_name = zt_day_data.get("名称", code) if isinstance(zt_day_data, pd.Series) else code
            except Exception:
                stock_name = code

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
    """检查布林通道买点（站上中轨）"""
    close = df["收盘"]
    volume = df["成交量"]

    upper, mid, lower = calc_bbands(close)

    if len(mid) < 2 or pd.isna(mid.iloc[-1]):
        return False

    today_close = close.iloc[-1]
    yesterday_close = close.iloc[-2]
    today_mid = mid.iloc[-1]
    yesterday_mid = mid.iloc[-2]

    mid_slope = today_mid - yesterday_mid
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    today_vol = volume.iloc[-1]
    vol_ratio = today_vol / avg_vol_20 if avg_vol_20 > 0 else 0

    above_mid = today_close > today_mid
    crossover = yesterday_close <= yesterday_mid
    slope_ok = mid_slope >= 0
    vol_ok = vol_ratio >= CONFIG.bbands_volume_ratio

    return above_mid and crossover and slope_ok and vol_ok


def check_vegas_buy(df: pd.DataFrame) -> bool:
    """检查维加斯通道买点（站上通道上沿）"""
    close = df["收盘"]

    if len(close) < CONFIG.vegas_periods[-1]:
        return False

    emas = calc_vegas_emas(close)
    today_close = close.iloc[-1]

    ema_vals = []
    for period in CONFIG.vegas_periods:
        ema = emas[period].iloc[-1]
        if pd.isna(ema):
            return False
        ema_vals.append(ema)

    vegas_upper = max(ema_vals)
    return today_close > vegas_upper


def confirm_signals(candidates: List[Dict], eval_date: str) -> List[Dict]:
    """
    技术信号共振确认：
    MACD金叉 + (布林买点 或 维加斯买点)
    """
    print("\n===== 第三步：技术信号共振确认 =====")

    results = []

    for i, stock in enumerate(candidates):
        df = stock["日线数据"]
        code = stock["代码"]
        name = stock["名称"]

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
            })

    print(f"\n  技术共振信号数量: {len(results)}")
    return results


# ============================================================
# 主执行流程
# ============================================================
def get_exit_signal(df: pd.DataFrame, entry_idx: int) -> Tuple[bool, str]:
    """
    判断给定日线数据是否触发离场条件。
    df: 个股日线（已按日期升序），包含 '收盘'、'成交量' 列
    entry_idx: 入场日在 df 中的索引位置
    返回 (是否离场, 离场原因)
    """
    close = df["收盘"]
    volume = df["成交量"]
    idx_now = len(df) - 1

    # 需有足够数据
    if idx_now <= entry_idx + 1:
        return False, ""

    # 1. MACD 死叉
    if CONFIG.exit_use_macd:
        diff, dea, _ = calc_macd(close)
        if idx_now >= 2 and not pd.isna(diff.iloc[-1]) and not pd.isna(dea.iloc[-1]):
            if diff.iloc[-1] < dea.iloc[-1] and diff.iloc[-2] >= dea.iloc[-2]:
                return True, "MACD死叉"

    # 2. 收盘跌破布林下轨
    if CONFIG.exit_use_boll_lower:
        _, _, lower = calc_bbands(close)
        if not pd.isna(lower.iloc[-1]) and close.iloc[-1] < lower.iloc[-1]:
            return True, "跌破布林下轨"

    # 3. 收盘跌破5日均线
    if CONFIG.exit_use_ma5:
        ma5 = close.rolling(CONFIG.exit_ma5_period).mean()
        if not pd.isna(ma5.iloc[-1]) and close.iloc[-1] < ma5.iloc[-1]:
            return True, "跌破5日均线"

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

    # ========================== 主线板块 ==========================
    main_sectors = screen_main_sectors(eval_date)
    if not main_sectors:
        print("\n⚠ 无主线板块，识别结束")
        return []

    print(f"\n✅ 主线板块: {[s['板块名称'] for s in main_sectors]}")

    candidates = screen_followers(main_sectors, eval_date)
    if not candidates:
        print("\n⚠ 无候选跟风股，识别结束")
        return []

    print(f"\n✅ 候选跟风股数量: {len(candidates)}")

    results = confirm_signals(candidates, eval_date)
    if not results:
        print("\n⚠ 无技术共振信号")
        return []

    # ========================== 仓位分配 ==========================
    # 注意：confirm_signals 返回的结果已包含 "日线数据"（由候选带入），此处追加仓位
    for i, res in enumerate(results):
        # 候选列表中顺序一致的，需要把日线数据传下去
        # confirm_signals 中我们曾从 stock 继承字段，需确认日线数据是否保留
        # 原 confirm_signals 没有显式传递日线数据，为支持仓位计算，需修改 confirm_signals 使其保留 df
        # 此处对 confirm_signals 进行微调：在原函数返回结果的字典中增加 "日线数据"
        pass

    # 因 confirm_signals 返回的字典中未包含日线数据，我们先简单调整：
    # 修改 confirm_signals 使其返回结果中包含 "日线数据"
    # 这里偷懒直接在 daily_identify 中重建映射（效率低但安全）
    for res in results:
        code = res["代码"]
        # 从原候选列表中找回日线数据
        stock_df = None
        for c in candidates:
            if c["代码"] == code:
                stock_df = c["日线数据"]
                break
        res["日线数据"] = stock_df

    results = allocate_position(results)

    # ========================== 输出结果 ==========================
    print("\n" + "=" * 80)
    print("===== 识别结果 （含出场规则定义） =====")
    print("=" * 80)
    print("出场规则：检测到下列任一条件即离场")
    if CONFIG.exit_use_macd:
        print("  - MACD死叉（DIFF下穿DEA）")
    if CONFIG.exit_use_boll_lower:
        print("  - 收盘价跌破布林带下轨")
    if CONFIG.exit_use_ma5:
        print(f"  - 收盘价跌破{CONFIG.exit_ma5_period}日均线")
    print()

    for r in results:
        print(f"{r['代码']} {r['名称']} | 板块:{r['所属板块']} | "
              f"龙头:{r['对应龙头']} | MACD金叉:{r.get('MACD金叉','?')} | "
              f"布林买点:{r.get('布林买点','?')} | 维加斯买点:{r.get('维加斯买点','?')} | "
              f"建议仓位:{r.get('建议仓位',0):.2%}")

    result_df = pd.DataFrame(results)
    # 省略日线数据列避免打印过长
    display_cols = [c for c in result_df.columns if c != "日线数据"]
    print(result_df[display_cols].to_string(index=False))

    print(f"\n===== 识别完毕，共 {len(results)} 只个股 =====")
    return results


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 日常信号识别
    results = daily_identify()

    # ---------------------------------------------------------------
    # 如需对历史信号进行回测，可编写如下函数（需额外数据准备）
    # def run_backtest(start_date, end_date):
    #     signals = daily_identify(end_date)
    #     for signal in signals:
    #         df = signal["日线数据"]
    #         entry_idx = len(df) - 1  # 当天为入场
    #         for i in range(entry_idx+1, len(df)):
    #             exited, reason = get_exit_signal(df.iloc[:i+1], entry_idx)
    #             if exited:
    #                 print(f"Signal {signal['代码']} exited on {df.index[i]} due to {reason}")
    #                 break
    # 注意：实际回测需考虑逐日交易和持仓状态，此处仅提供接口示例
    # ---------------------------------------------------------------
