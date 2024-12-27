import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import pickle
import os
import warnings
import ray
import time
warnings.filterwarnings('ignore')

@ray.remote
class NeutralizationProcessor:
    def __init__(self, barra_path, factor_path, barra_col, output_folder, mode):
        self.barra_path = barra_path            # path of barra data
        self.factor_path = factor_path          # path of factor data
        self.output_folder = output_folder      # output folder
        self.mode = mode                        # mode of neutralization
        self.barra_col = barra_col              # name of barra column

    def neutralize(self, factor_df, factor_col, barra_col, mode):
        """
        factor_df: pandas dataframe of factor data
        factor_col: name of factor column
        barra_col: name of barra column
        """

        ans = pd.DataFrame()
        for col in factor_col:
            """
            clip extreme values to avoid extreme values in neutralization
            the extreme values are defined as 3 standard deviations away from the mean
            """
            edge_up = factor_df[col].mean() + 3 * factor_df[col].std()
            edge_down = factor_df[col].mean() - 3 * factor_df[col].std()
            factor_df[col] = factor_df[col].clip(edge_down, edge_up)

            """
            normalize factor data
            mode 1: min-max normalization
            mode 2: standardization
            mode 3: log normalization
            """
            if mode == 1:
                factor_df[col] = (factor_df[col] - factor_df[col].min()) / (factor_df[col].max() - factor_df[col].min())
            elif mode == 2:
                factor_df[col] = (factor_df[col] - factor_df[col].mean()) / factor_df[col].std()
            elif mode == 3:
                factor_df[col] = factor_df[col] / 10**np.ceil(np.log10(factor_df[col].abs().max()))

            # return residuals as the neutralized factor
            results = sm.OLS(factor_df[col], factor_df[barra_col]).fit()
            ans[col] = results.resid
        return ans
    
    def process_file(self, file):
        """Process a single file"""
        
        factor_file = sorted(os.listdir(self.factor_path))
        if file not in factor_file:
            return

        date = file[:-4]     # date form is "YYYY-MM-DD.csv"
        factor_df = pd.read_csv(f'{self.factor_path}/{file}', index_col=0, header=0)
        barra_df = pd.read_csv(f'{self.barra_path}/{file}', index_col=0, header=0)

        # make sure the barra data and factor data have the same stock codes
        codes_inter = barra_df.index.intersection(factor_df.index)

        barra_df = barra_df.loc[codes_inter, :]

        final = pd.concat([factor_df, barra_df], axis=1)
        final.replace([np.inf, -np.inf], np.nan, inplace=True)
        final.fillna(0, inplace=True)

        factor_col = factor_df.columns.tolist()
        data = self.neutralize(final, factor_col, self.barra_col, self.mode)
        data.to_csv(f'{self.output_folder}/{file}')

    def process_data(self):
        barra_file = sorted(os.listdir(self.barra_path))

        # Parallelize using ray
        results = [self.process_file(file) for file in barra_file]
