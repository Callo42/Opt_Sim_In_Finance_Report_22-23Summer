#%%
import pandas as pd
import numpy as np
import cvxpy as cp
import os

class portfolio(object):
    def __init__(self,asset_list,annual_expect_return,data_path):
        self._asset_list = asset_list
        self._asset_num = len(asset_list)
        self._w, self._mu = {}, {} 
        self._annual_expect_return = annual_expect_return
        self._data_path = data_path
        self._filename = "monthly_20091201_to_20230625.csv"

    def print(self):
        print(f"The asset list of the portfolio is {self._asset_list}\n"
              f"Including #{self._asset_num} assets\n"
              f"Expected annual return rate is {self._annual_expect_return}%\n"
              f"Data are stored in `{self._data_path}`\n"
              f"w initialized as {self._w} and\n mu initialized as {self._mu}")

    def load_data(self,stock_id):
        path = self._data_path + stock_id + "/" + self._filename
        if os.path.exists(path):
            print(f"Loading data from {path}... finished.")
        else:
            raise Warning(f"Try to find data from '{path}'\n"
                          f"But failed, please check does the file really exists.")
        return pd.read_csv(path,usecols=["日期","收盘"])

    def yearly_return_init_w_mu(self):
        '''
        caculate yearly return from the original monthly data
        and initialize w
        and calculate average yearly return mu
        for each stock
        '''
        yearly_return = []
        print(f"Calculating yearly return per stock and initializing w and mu")
        for stock_id in self._asset_list:
            print(f"Dealing with stock id {stock_id}")
            #initialize w 
            self._w[stock_id] = 0
            #load data
            df = self.load_data(stock_id=stock_id)
            yearly_df = df[df["日期"].str.match("^20[0-9]{2}-12-[0-9]{2}$")==True]
            yearly_close = yearly_df["收盘"].to_numpy()
            # print(f"yearly close for {stock_id} is {yearly_close}")

            #calculate yearly return (in unit of %)
            year_return_list = ((yearly_close[1::]-yearly_close[:-1:])/yearly_close[:-1:])*100
            # print(f"shape of year_return_list = {year_return_list.shape}")
            print(f"yearly_return_list = {year_return_list}")
            yearly_return.append(year_return_list)

            #calculate mu
            self._mu[stock_id] = year_return_list.mean()
            print(f"GeoMean of annual return of {stock_id} is {self._mu[stock_id]}")
        yearly_return = np.array(yearly_return) #shape (asset_num,year_num) in which year_num==len(yearly_close)
        return yearly_return
    
    def solve_w(self):
        '''
        solve the problem and return the fraction w=(w1,w2,...,wn)
        for the Markowtiz Mean-Variance model
        Minimize risk while achiving expected return
        '''
        print("=================================================================\n"
            f"Creating quadratic programming problem with cvxpy\n"
            "=================================================================\n")
        yearly_return = self.yearly_return_init_w_mu()
        #covariance matrix
        Sigma = np.cov(yearly_return)
        print(f"Computing covariance matrix... done. \n Sigma={Sigma}")

        #reforming mu into a nparray
        mu = np.array([self._mu[stock_id] for stock_id in self._asset_list])
        print(f"mu = {mu}")

        #setting problem and solve the problem using cvx
        n = self._asset_num
        w = cp.Variable(n)

        constraints = [w.T @ mu >= self._annual_expect_return,
                       np.ones(n).T @ w == 1,
                       w >= 0]
        
        obj = cp.Minimize(cp.quad_form(w,Sigma))

        prob = cp.Problem(obj,constraints)
        prob.solve()
        print("=================================================================\n"
            "Problem solved. Fractions w were updated.\n"
            "=================================================================\n")
        for i in range(n):
            stock_id = self._asset_list[i]
            self._w[stock_id] = round(w.value[i],4)
        return prob


if __name__ == "__main__":
    asset_list = ["600900","600519","601318","601111"]
    annual_expect_return = 20 #in percentage, 20 for 20%
    data_path = "./data/stocks/"
    portfolio_1 = portfolio(asset_list=asset_list, annual_expect_return=annual_expect_return,data_path=data_path)
    prob = portfolio_1.solve_w()

    print("\nThe optimal value is", prob.value)
    print("A solution w is")
    print(f"{portfolio_1._w}")
# %%
