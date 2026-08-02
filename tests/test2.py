import jqdatasdk as jq
from jqdatasdk import *

# ========== 使用前必须登录聚宽账号 ==========
jq.auth('15606865536', 'Liuqinzhong192')

def get_stock_concepts(stock_code, date=None):
    """
    获取某只股票所属的全部概念板块（概念代码列表）
    通过遍历所有概念并检查成分股来实现（比直接API慢，但可靠）
    :param stock_code: 股票代码，聚宽格式（如 '000001.XSHE'）
    :param date: 查询日期，默认为最新
    :return: list[str] 概念代码列表
    """
    if not jq.is_auth():
        raise RuntimeError("请先调用 auth() 登录聚宽账号")

    # 1. 获取所有概念代码及其名称
    concept_df = jq.get_concepts()
    concept_names = dict(zip(concept_df.index, concept_df.get('name', [])))
    all_concepts = concept_df.index.tolist()

    # 2. 筛选出包含该股票的概念，并附带中文名
    matching = []
    for concept in all_concepts:
        try:
            stocks = jq.get_concept_stocks(concept, date=date)
            if stock_code in stocks:
                matching.append({
                    'code': concept,
                    'name': concept_names.get(concept, '')
                })
        except Exception:
            # 某些概念可能出现访问异常，忽略即可
            continue
    return matching

def get_concept_stocks(concept_codes, date=None):
    """
    获取一个或多个概念板块的成分股代码
    :param concept_codes: 概念代码（字符串）或概念代码列表
    :param date: 查询日期，默认为最新
    :return: 若传入单个代码，返回 list[str]；若传入列表，返回 dict{概念代码: [股票代码列表]}
    """
    if not jq.is_auth():
        raise RuntimeError("请先调用 auth() 登录聚宽账号")

    if isinstance(concept_codes, str):
        return jq.get_concept_stocks(concept_codes, date=date)
    else:
        result = {}
        for code in concept_codes:
            result[code] = jq.get_concept_stocks(code, date=date)
        return result


if __name__ == "__main__":
    # result = get_stock_concepts('003032.XSHE', date='2025-04-24')
    # print(result) 'SC0002' 'SC0013' 'SC0110' 'SC0385' 'SC0418'
    result2 = get_concept_stocks(['SC0002', 'SC0013', 'SC0110', 'SC0385', 'SC0418'], date='2025-04-24')
    print(result2)