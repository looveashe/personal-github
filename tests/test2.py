import akshare as ak

board_df = ak.sw_index_third_info()
# 看前几行行业代码
print(board_df.head()[['行业代码', '行业名称']])
# 取第一个行业代码
code = board_df.iloc[0]['行业代码']
df = ak.sw_index_third_cons(symbol=code)
print(df.shape, df.columns)