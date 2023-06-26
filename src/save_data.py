#%%
import os
import akshare as ak
from datetime import datetime
import pandas as pd

def save_stock_history(stock_id,period,adjust,start_date):
    """
    save the stock price history
    up to today
    from start_date

    Args:
        stock_id (str): the stock id
        period (str):"daily","weekly", "monthly" for prices per that session.
        adjust (str): "hfq" for 后复权, "qfq" for 前复权.
        start_date: the beginning date from which the price history would be downloaded.
    Returns:
        None, but save the prices into a csv file.
    """
    today = datetime.now()
    ak_get_price = ak.stock_zh_a_hist
    end_date = today.strftime('%Y%m%d')
    stock_price = ak_get_price(stock_id,period,start_date,end_date,adjust)
    datapath = f"./data/stocks/{stock_id}"
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    stock_price.to_csv(datapath + f"/{period}_{start_date}_to_{end_date}.csv")
#%%

if __name__ == "__main__":
    stock_list = ["600900","600519","601318","601111"]
    period_list = ["monthly"]
    adjust = "hfq"
    start_date = "20091201"
    for stock_id in stock_list:
        for period in period_list:
            save_stock_history(stock_id, period, adjust,start_date)