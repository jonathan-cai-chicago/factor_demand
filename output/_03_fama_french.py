#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# - This notebook intends to walk through the steps of replicating Table 2.
# - `df_combo` is the main sample as shown in Table 1.

# In[ ]:


from pathlib import Path

import numpy as np
import pandas as pd
import wrds

from load_CRSP_fund import load_CRSP_combined_file
from load_mflink import load_mflink1

import config
OUTPUT_DIR = Path(config.OUTPUT_DIR)
path = Path(OUTPUT_DIR) / "main_sample.parquet" 
df_combo = pd.read_parquet(path)

df_crsp = load_CRSP_combined_file()
df_mflink1 = load_mflink1()


# # CRSP Mutual Fund Data
# 
# - CRSP data and mflink1 are merged based on `crsp_fundno`to obtain the appropriate `wficn`.
# - Calculate `mret` and `mtna` for each `wficn`.
# - The new CRSP data is then merged with `df_combo` by `year` and `wficn` to get the main sample's monthly returns. 

# In[ ]:


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
df_crsp


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
float_format_func = lambda x: '{:.2f}'.format(x)
df_crsp = df_crsp.rename(columns={'lipper_class_name': 'lipper class','crsp_ret': '$crsp_{ret}$', 'crsp_tna':'$crsp_{TNA}$', 'index_fund_flag':'index fund flag'}) 

latexTS_crsp_t2 = df_crsp.tail(10).to_latex(float_format = float_format_func)

path_to_save = f'../output/table_crsp_t2.tex'

with open(path_to_save, 'w') as f: 
    f.write(latexTS_crsp_t2)
    
df_crsp = df_crsp.rename(columns={'lipper class name':'lipper_class_name','$crsp_{ret}$': 'crsp_ret', '$crsp_{TNA}$':'crsp_tna', 'index fund flag':'index_fund_flag'}) 


# # Fama French Factors
# 
# - Factor returns, `df_ff`, are pulled from Kenneth R. French's website.

# In[ ]:


df_ff = pd.read_csv(Path(config.DATA_DIR)/'manual'/'F-F_Research_Data_5_Factors_2x3.csv').drop(['RF'], axis=1)
df_mom = pd.read_csv(Path(config.DATA_DIR)/'manual'/'F-F_Momentum_Factor.csv')
df_ff = df_ff.merge(df_mom, how='inner', on=['date'])
df_ff = df_ff[(df_ff['date'] >= 198001) & (df_ff['date'] <= 201912)]
df_ff


# - CRSP data and factor returns are merged, and for each fund i in month t, $flow_{i,t}$ is calculated using the formula:
# 
# $$
# \text{flow}_{i,t} = \frac{\text{TNA}_{i,t}}{\text{TNA}_{i,t-1}} \times (1 + \text{ret}_{i,t})
# $$

# In[ ]:


df_reg = pd.merge(df_crsp[df_crsp['date'] <= 201912], df_ff, on=['date'], how="outer").sort_values(["date"])
flow = df_reg.groupby('wficn').apply(lambda d: d['crsp_tna']/(d['crsp_tna'].shift(1)) - (1+d['crsp_ret'])).reset_index().rename(columns={'level_1': 'index', 0: "flow"})
flow.set_index('index', inplace=True)
df_reg = pd.merge(df_reg, flow[['flow']], left_index=True, right_index=True).sort_values(['wficn', 'date'])
df_reg[['crsp_ret', 'flow']] *= 100
df_reg.replace([np.inf, -np.inf], np.nan, inplace=True)
df_reg= df_reg.fillna(0)
df_reg



# In[ ]:


latexTS_df_reg = df_reg.tail(10).to_latex(float_format = float_format_func)

path_to_save = f'../output/table_fama_french_t2.tex'

with open(path_to_save, 'w') as f: 
   f.write(latexTS_df_reg) 


# # Reporting the mean, std, and percentiles of factor betas across all funds
# 
# - To replicate Panel A of Table 2, for each fund i in month t, we run the following rolling time-series regression:
# $$
# \text{ret}_{i,t+1-k} = \alpha_{i,t} + \beta_{\text{MKT} i,t} \times \text{MKT}_{t+1-k} + \beta_{\text{HML} i,t} \times \text{HML}_{t+1-k} + \beta_{\text{SMB} i,t} \times \text{SMB}_{t+1-k} + \beta_{\text{MOM} i,t} \times \text{MOM}_{t+1-k} + \beta_{\text{CMA} i,t} \times \text{CMA}_{t+1-k} + \beta_{\text{RMW} i,t} \times \text{RMW}_{t+1-k} + \beta_{\text{flow} i,t} \times \text{flow}_{i,t+1-k} + \epsilon_{i,t,t+1-k}
# $$
# where k = 1,2,...,60.
# 
# - We require a fund should have 60 months of returns data and each rolling window contains 24 monthly observationswe need to run regression

# In[ ]:


from sklearn.linear_model import LinearRegression

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


# In[ ]:


df_reg = df_reg.sample(frac=0.3, random_state=42)
all_funds = regression(df_reg)
panelA = all_funds.describe().loc[['mean', 'std']].append(all_funds.quantile(0.05)).append(all_funds.describe().loc[['25%', '50%', '75%']]).append(all_funds.quantile(0.95))
panelA = panelA.rename(index={0.05: 'P5', '25%': 'P25', '50%': 'P50', '75%': 'P75', 0.95: 'P95'})
panelA


# In[ ]:


latexTS_panelA = panelA.to_latex(float_format = float_format_func)

path_to_save = f'../output/table_panelA.tex'

with open(path_to_save, 'w') as f: 
    f.write(latexTS_panelA) 


# # Reporting the mean factor betas by Lipper mutual fund classifications
# 
# - To replicate Panel B of Table 2, we classify funds according to `lipper_class_name`, and then run the regressions again. 

# In[ ]:


df_growth = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Growth', case=False, regex=True)]
df_value = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Value', case=False, regex=True)]
df_base = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Base', case=False, regex=True)]
df_large_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Large-Cap', case=False, regex=True)]
df_mid_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Mid-Cap', case=False, regex=True)]
df_small_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Small-Cap', case=False, regex=True)]


# In[ ]:


growth = regression(df_growth)
value = regression(df_value)
base = regression(df_base)
large_cap = regression(df_large_cap)
mid_cap = regression(df_mid_cap)
small_cap = regression(df_small_cap)
panelB = pd.DataFrame({'All': all_funds.mean(), 'Growth': growth.mean(), 'Value': value.mean(), 
              'Large cap': large_cap.mean(), 'Medium cap': mid_cap.mean(), 'Small cap': small_cap.mean()}).T
panelB


# In[ ]:


latexTS_panelB = panelB.to_latex(float_format = float_format_func)

path_to_save = f'../output/table_panelB.tex'

with open(path_to_save, 'w') as f: 
    f.write(latexTS_panelB) 


# # Reporting the mean factor betas by index fund status
# - To replicate Panel C of Table 2, we classify funds according to index fund status.
# - `index_fund_flag` identifies if a fund is an index fund:
# - B = index-based fund
# - D = pure index fund
# - E = index fund enhanced

# In[ ]:


df_index = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('D|B|E', case=False, regex=True)]
df_enhanced = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('E', case=False, regex=True)]
df_base = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('B', case=False, regex=True)]
df_pure = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('D', case=False, regex=True)]
df_non_index = df_reg[~df_reg['index_fund_flag'].astype(str).str.contains('D|B|E', case=False, regex=True)]


# In[ ]:


index = regression(df_index)
enhanced = regression(df_enhanced)
base = regression(df_base)
pure = regression(df_pure)
non_index = regression(df_non_index)
panelC = pd.DataFrame({'All index funds': index.mean(), 'Enhanced': enhanced.mean(), 'Base': base.mean(), 
              'Pure': pure.mean(), 'All non-index funds': non_index.mean()}).T
panelC


# In[ ]:


latexTS_panelC = panelC.to_latex(float_format = float_format_func)

path_to_save = f'../output/table_panelC.tex'

with open(path_to_save, 'w') as f: 
    f.write(latexTS_panelC) 


# # Recalculation using data up until the Present

# In[ ]:


df_ff = pd.read_csv(Path(config.DATA_DIR)/'manual'/'F-F_Research_Data_5_Factors_2x3.csv').drop(['RF'], axis=1)
df_mom = pd.read_csv(Path(config.DATA_DIR)/'manual'/'F-F_Momentum_Factor.csv')
df_ff = df_ff.merge(df_mom, how='inner', on=['date'])
df_ff = df_ff[df_ff['date'] >= 202001]

df_reg = pd.merge(df_crsp[df_crsp['date'] >= 202001], df_ff, on=['date'], how="outer").sort_values(["date"])
flow = df_reg.groupby('wficn').apply(lambda d: d['crsp_tna']/(d['crsp_tna'].shift(1)) - (1+d['crsp_ret'])).reset_index().rename(columns={'level_1': 'index', 0: "flow"})
flow.set_index('index', inplace=True)
df_reg = pd.merge(df_reg, flow[['flow']], left_index=True, right_index=True).sort_values(['wficn', 'date'])
df_reg[['crsp_ret', 'flow']] *= 100
df_reg.replace([np.inf, -np.inf], np.nan, inplace=True)
df_reg= df_reg.fillna(0)
df_reg


# In[ ]:


all_funds_pre = regression(df_reg)
panelA_pre = all_funds_pre.describe().loc[['mean', 'std']].append(all_funds_pre.quantile(0.05)).append(all_funds_pre.describe().loc[['25%', '50%', '75%']]).append(all_funds_pre.quantile(0.95))
panelA_pre = panelA_pre.rename(index={0.05: 'P5', '25%': 'P25', '50%': 'P50', '75%': 'P75', 0.95: 'P95'})
panelA_pre


# In[ ]:


df_growth = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Growth', case=False, regex=True)]
df_value = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Value', case=False, regex=True)]
df_base = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Base', case=False, regex=True)]
df_large_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Large-Cap', case=False, regex=True)]
df_mid_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Mid-Cap', case=False, regex=True)]
df_small_cap = df_reg[df_reg['lipper_class_name'].astype(str).str.contains('Small-Cap', case=False, regex=True)]

growth_pre = regression(df_growth)
value_pre = regression(df_value)
base_pre = regression(df_base)
large_cap_pre = regression(df_large_cap)
mid_cap_pre = regression(df_mid_cap)
small_cap_pre = regression(df_small_cap)
panelB_pre = pd.DataFrame({'All': all_funds_pre.mean(), 'Growth': growth_pre.mean(), 'Value': value_pre.mean(), 
              'Large cap': large_cap_pre.mean(), 'Medium cap': mid_cap_pre.mean(), 'Small cap': small_cap_pre.mean()}).T
panelB_pre


# In[ ]:


df_index = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('D|B|E', case=False, regex=True)]
df_enhanced = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('E', case=False, regex=True)]
df_base = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('B', case=False, regex=True)]
df_pure = df_reg[df_reg['index_fund_flag'].astype(str).str.contains('D', case=False, regex=True)]
df_non_index = df_reg[~df_reg['index_fund_flag'].astype(str).str.contains('D|B|E', case=False, regex=True)]

index_pre = regression(df_index)
enhanced_pre = regression(df_enhanced)
base_pre = regression(df_base)
pure_pre = regression(df_pure)
non_index_pre = regression(df_non_index)
panelC_pre = pd.DataFrame({'All index funds': index_pre.mean(), 'Enhanced': enhanced_pre.mean(), 'Base': base_pre.mean(), 
              'Pure': pure_pre.mean(), 'All non-index funds': non_index_pre.mean()}).T
panelC_pre


# In[ ]:




