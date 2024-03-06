import pandas as pd
import numpy as np
import wrds
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME
START_DATE = config.START_DATE
END_DATE = config.END_DATE
OUTPUT_DIR = Path(config.OUTPUT_DIR)

from load_CRSP_fund import load_CRSP_combined_file
from load_mflink import load_mflink1
from sklearn.linear_model import LinearRegression

def monthly_mutual_fund():
    path = Path(OUTPUT_DIR) / "main_sample.parquet"
    df_combo = pd.read_parquet(path)

    df_crsp = load_CRSP_combined_file()
    df_mflink1 = load_mflink1()
    
    df_crsp = df_crsp.merge(df_mflink1, how="inner", on="crsp_fundno").reset_index(drop=True)

    df_crsp = df_crsp.sort_values(["caldt", "wficn"])
    df_crsp['mret'] = df_crsp['mret'].fillna(0)
    df_crsp['lipper_class_name'] = df_crsp['lipper_class_name'].fillna('None')

    df_crsp = df_crsp[~df_crsp['lipper_class_name'].astype(str).str.contains('International|Fixed Income|Precious Metal', case=False, regex=True)]
    ret = df_crsp.groupby(["caldt", "wficn", 'lipper_class_name'])["mret"].mean().reset_index().rename(columns={"mret": "crsp_ret"})
    tna = df_crsp.groupby(["caldt", "wficn", 'lipper_class_name'])["mtna"].sum().reset_index().rename(columns={"mtna": "crsp_tna"})
    df_crsp = pd.merge(pd.merge(ret, tna, on=["caldt", "wficn", 'lipper_class_name'], how="inner"), 
                       df_crsp[["caldt", "wficn", 'lipper_class_name', 'index_fund_flag']], on=["caldt", "wficn", 'lipper_class_name'], how="inner").sort_values(["caldt", "wficn"])
    df_crsp = df_crsp.drop_duplicates()
    df_crsp = df_crsp.rename(columns={"caldt": "date"})

    df_crsp['year'] = df_crsp['date'].dt.year.astype('int')
    df_crsp = pd.merge(df_crsp, df_combo[['year', 'wficn']], on=["year", "wficn"], how="inner")

    df_crsp['date'] = df_crsp['date'].dt.strftime('%Y%m').astype('int')
    return df_crsp


def fama_french_factors():
    df_ff = pd.read_csv(Path(config.DATA_DIR)/'manual'/'F-F_Research_Data_5_Factors_2x3.csv').drop(['RF'], axis=1)
    df_mom = pd.read_csv(Path(config.DATA_DIR)/'manual'/'F-F_Momentum_Factor.csv')
    df_ff = df_ff.merge(df_mom, how='inner', on=['date'])
    df_ff = df_ff[(df_ff['date'] >= 198001) & (df_ff['date'] <= 201912)]
    return df_ff


def regression_df(df_crsp, df_ff):
    df_reg = pd.merge(df_crsp[df_crsp['date'] <= 201912], df_ff, on=['date'], how="outer").sort_values(["date"])
    flow = df_reg.groupby('wficn').apply(lambda d: d['crsp_tna']/(d['crsp_tna'].shift(1)) - (1+d['crsp_ret'])).reset_index().rename(columns={'level_1': 'index', 0: "flow"})
    flow.set_index('index', inplace=True)
    df_reg = pd.merge(df_reg, flow[['flow']], left_index=True, right_index=True).sort_values(['wficn', 'date'])
    df_reg[['crsp_ret', 'flow']] *= 100
    df_reg.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_reg= df_reg.fillna(0)
    return df_reg


def regression(df):
    beta = pd.DataFrame(columns = ['Mkt-RF', 'SMB', 'HML', 'MOM', 'CMA', 'RMW', 'flow'])

    for fund, data in df.groupby('wficn'):
        if len(data) >= 60: 
            for month in range(len(data)-59):
                sample = data.iloc[month:month+60, :]
                for rw in range(len(sample)-23):
                    rolling_window = sample.iloc[rw:rw+24, :]
                    X = rolling_window[['Mkt-RF', 'SMB', 'HML', 'MOM', 'CMA', 'RMW', 'flow']]
                    y = rolling_window[['crsp_ret']]
                    model = LinearRegression().fit(X, y)
                    coef = pd.DataFrame((model.coef_).reshape(-1, 7), columns = ['Mkt-RF', 'SMB', 'HML', 'MOM', 'CMA', 'RMW', 'flow'])
                    beta = pd.concat([beta, coef], axis=0)
    return beta


def calc_penal_A(df_reg):
    all_funds = regression(df_reg)
    panelA = all_funds.describe().loc[['mean', 'std']].append(all_funds.quantile(0.05)).append(all_funds.describe().loc[['25%', '50%', '75%']]).append(all_funds.quantile(0.95))
    panelA = panelA.rename(index={0.05: 'P5', '25%': 'P25', '50%': 'P50', '75%': 'P75', 0.95: 'P95'})
    return panelA


def calc_penal_B(df_reg):
    df_growth = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Growth', case=False, regex=True)]
    df_value = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Value', case=False, regex=True)]
    df_base = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Base', case=False, regex=True)]
    df_large_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Large-Cap', case=False, regex=True)]
    df_mid_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Mid-Cap', case=False, regex=True)]
    df_small_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Small-Cap', case=False, regex=True)]
    
    growth = regression(df_growth)
    value = regression(df_value)
    base = regression(df_base)
    large_cap = regression(df_large_cap)
    mid_cap = regression(df_mid_cap)
    small_cap = regression(df_small_cap)
    panelB = pd.DataFrame({'All': all_funds.mean(), 'Growth': growth.mean(), 'Value': value.mean(), 
                  'Large cap': large_cap.mean(), 'Medium cap': mid_cap.mean(), 'Small cap': small_cap.mean()}).T
    return panelB


def calc_penal_C(df_reg):
    df_index = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('D|B|E', case=False, regex=True)]
    df_enhanced = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('E', case=False, regex=True)]
    df_base = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('B', case=False, regex=True)]
    df_pure = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('D', case=False, regex=True)]
    df_non_index = df_reg[~df_reg['index_fund_flag'].astype(str).str.contains('D|B|E', case=False, regex=True)]

    index = regression(df_index)
    enhanced = regression(df_enhanced)
    base = regression(df_base)
    pure = regression(df_pure)
    non_index = regression(df_non_index)
    panelC = pd.DataFrame({'All index funds': index.mean(), 'Enhanced': enhanced.mean(), 'Base': base.mean(), 
                  'Pure': pure.mean(), 'All non-index funds': non_index.mean()}).T
    return panelC

