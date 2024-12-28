import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class BackTester:
    def __init__(self, start_date, end_date, trade_date_path, factor_path, ret_path, ud_path, group):
        self.start_date = start_date
        self.end_date = end_date
        self.trade_date_path = trade_date_path        # 交易日历
        self.factor_path = factor_path                # 因子数据
        self.ret_path = ret_path                      # 收益率数据
        self.ud_path = ud_path                        # 涨跌停牌数据
        self.group = group                            # 分组数 
        self.pnl = {}
        self.pro = {}
        self.ic = {}
        self.result = pd.DataFrame()

    def prepare_data(self):
        trade_date = pickle.load(open(self.trade_date_path, 'rb'))
        factor_files = sorted(os.listdir(self.factor_path))
        ret_files = sorted(os.listdir(self.ret_path))

        for file in tqdm(factor_files):
            # get current date and next date
            date = file[:-4]
            next_date = trade_date[trade_date.index(date) + 1]
            
            if file in ret_files and date >= self.start_date and date <= self.end_date:
                # at "date", all the factors are computed
                # so at "next_date", based on the factor data we allocate the portfolio
                # hence we need the return data in "next_date"
                factor_df = pd.read_csv(f'{self.factor_path}/{date}.csv', header=0)
                ret_df = pd.read_csv(f'{self.ret_path}/{next_date}.csv', header=0)
                ud_df = pd.read_csv(f'{self.ud_path}/{next_date}.csv', header=0)

                factor_df.set_index('code', inplace=True)
                ret_df.set_index('code', inplace=True)
                ud_df.set_index('code', inplace=True)
                ret_ud_df = pd.concat([ret_df, ud_df[['zt', 'dt', 'paused']]], axis=1)

                # only consider the codes that have both factor and return data
                codes_inter = factor_df.index.intersection(ret_ud_df.index)


                final = pd.concat([factor_df.loc[codes_inter], ret_ud_df.loc[codes_inter]], axis=1)
                # exclude stocks that are in the limit
                cond1 = final['zt'] == 0
                cond2 = final['dt'] == 0
                cond3 = final['paused'] == 0
                final = final.loc[cond1 & cond2 & cond3, :].replace([np.inf, -np.inf], 0)
                self.calculate_return(final, date, factor_df)

    def calculate_return(self, final, cur_date, factor_df):
        for col in factor_df.columns.tolist():
            if col not in self.pnl:
                self.pnl[col] = pd.DataFrame()
                self.pro[col] = pd.DataFrame()
                self.ic[col] = []

            # too many repeated values in factor, skip
            if len(final[col].dropna().quantile(self.group).unique()) < 11:
                continue

            final[col+"_group"] = pd.qcut(final[col], len(self.group) - 1, labels=False, duplicates='drop') + 1
            
            # calculate excess return and pnl
            final['return_pro'] = final['1vwap_pct'] - final['1vwap_pct'].mean()
            final['return_pnl'] = final['1vwap_pct']

            res_agg = final.pivot_table(index=col+"_group", values=['return_pro', 'return_pnl'], aggfunc=np.mean)
            
            pnl_df = res_agg[['return_pnl']].T
            pnl_df['date'] = cur_date
            pnl_df.set_index('date', inplace=True)
            pnl_df.columns.name = None
            pnl_df.index.name = None
            self.pnl[col] = pd.concat([self.pnl[col], pnl_df], axis=0)

            pro_df = res_agg[['return_pro']].T
            pro_df['date'] = cur_date
            pro_df.set_index('date', inplace=True)
            pro_df.columns.name = None
            pro_df.index.name = None
            self.pro[col] = pd.concat([self.pro[col], pro_df], axis=0)

            ic_df = final[col].corr(final['1vwap_pct'])
            self.ic[col].append(ic_df)

    def calculate_effectiveness(self):
        for factor in self.ic:
            ic = np.mean(self.ic[factor]).round(3)
            if (ic > 0):
                self.pro[factor].columns = self.pro[factor].columns[::-1]
                self.pnl[factor].columns = self.pnl[factor].columns[::-1]

            # self.return_plot(factor, 'pro')
            self.pnl[factor]['hedge'] = self.pnl[factor][1] - self.pnl[factor][10]
            self.return_plot(factor, 'pnl')
            self.calculate_metrics(factor, ic)
    
    def return_plot(self, factor, return_type):
        if return_type == 'pro':
            df = self.pro[factor]
        if return_type == 'pnl':
            df = self.pnl[factor]
        cumulative_return = (1+df).cumprod()
        dates = df.index
        plt.figure(figsize=(12, 6))
        for group in cumulative_return.columns:
            plt.plot(dates, cumulative_return[group], label=f'group_{group}')
        
        plt.title(f'{factor}_{return_type}')
        plt.xlabel('date')
        plt.ylabel(return_type)
        plt.legend()
        x_ticks_inteval = 30
        plt.xticks(dates[::x_ticks_inteval], dates[::x_ticks_inteval], rotation=45)
        plt.show()

    def calculate_metrics(self, factor, ic):
        icir = (np.mean(self.ic[factor])) / np.std(self.ic[factor]).round(3)
        long_ret = (np.mean(self.pnl[factor][1]) * 252).round(3)
        long_std = (np.std(self.pnl[factor][1]) * np.sqrt(252)).round(3)
        long_sr = (np.mean(self.pnl[factor][1]) * 252 / (np.std(self.pnl[factor][1]) * np.sqrt(252))).round(3)

        cumulative_return = (1 + self.pnl[factor]).cumprod()
        long_drawdown = ((cumulative_return[1].cummax() - cumulative_return[1]) / cumulative_return[1].cummax()).round(3)
        long_maxdd = long_drawdown[long_drawdown.argmax()] * 100

        hedge_ret = (np.mean(self.pnl[factor]['hedge']) * 252).round(3)
        hedge_std = (np.std(self.pnl[factor]['hedge']) * np.sqrt(252)).round(3)
        hedge_sr = (np.mean(self.pnl[factor]['hedge']) * 252 / (np.std(self.pnl[factor]['hedge']) * np.sqrt(252))).round(3)
        hedge_drawdown = ((cumulative_return['hedge'].cummax() - cumulative_return['hedge']) / cumulative_return['hedge'].cummax()).round(3)
        hedge_maxdd = hedge_drawdown[hedge_drawdown.argmax()] * 100
        new_row = {'factor_name': factor, 'ic:': ic, 'icir:': icir, '多头收益率:': long_ret, '多头波动率:': long_std, '多头sr:': long_sr, '多头最大回撤:': long_maxdd, 
                       '对冲收益率:': hedge_ret, '对冲波动率:': hedge_std, '对冲sr:': hedge_sr, '对冲最大回撤:': hedge_maxdd}
        self.result = pd.concat([self.result, pd.DataFrame([new_row])], ignore_index=True)

