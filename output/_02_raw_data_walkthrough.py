#!/usr/bin/env python
# coding: utf-8

# # Overview
# - This notebook intends to walk through the raw data saved in `data/pull`, outlining potential issues and pitfalls. 
# - After running `doit`, your local `data/pull` directory should have four files:
#     1. `CRSP_fund_combined.parquet`: CRSP monthly mutual funds data.
#     2. `s12.parquet`: S12 quarterly mutual fund holdings data. 
#     3. `mflink1.parquet`: To link `crsp_fundno` with `wficn`. 
#     4. `mflink2.parquet`: To link S12's `fundno` with `wficn`. 

# In[ ]:


from pathlib import Path

import numpy as np
import pandas as pd
import wrds

from load_CRSP_fund import load_CRSP_combined_file
from load_s12 import load_s12_file
from load_mflink import load_mflink1, load_mflink2


import config
WRDS_USERNAME = config.WRDS_USERNAME

df_crsp = load_CRSP_combined_file()
df_s12 = load_s12_file()
df_mflink1 = load_mflink1()
df_mflink2 = load_mflink2()


# # CRSP Mutual Fund Data
# 
# ## Tables and Filters
# - The main data is pulled from `crsp.monthly_tna_ret_nav`. (https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_q_mutualfunds/monthly_tna_ret_nav/)
# - The paper specifies that it utilized only "US domestic equity" funds, so we need to identify this information. 
# - Through various trials and errors, I found out that the best way to achieve this filter is through the `crsp.fund_style` table's `crsp_obj_cd` column. (https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_q_mutualfunds/fund_style/)
# - I left join these two tables above. 
# - Based on the CRSP manual, I require the first two characters of this code to be "ED", representing "Equity" and "Domestic". 
# 
# ## Obtaining `wficn`
# - I discovered that **each mutual fund can have multiple `crsp_fund_no`**, representing different _share classes_. 
# - It is of critical important for us to obtain the `wficn`, which is a fund-level identifier, and then aggregate the results. 
# - The author describes the algorithm to perform this aggregation in **footnote 4**. 
# 
# ## Multiple `wficn` for each `crsp_fundno`
# - As explained above, it's very common for one `wficn` to match with multiple `crsp_fundno`, because the latter represents a specific share class of a fund. 
# - However, I discovered rare occurrances where one `crsp_fundno` matches with multiple `wficn`. This is unexpected, since each `wficn` should conceptually represents one "institution" or "fund". 
# - I could not figure out the underlying reasons, but suspect that it could have something to do with delisting / merging of funds. For instance, one fund could be fully aqcuired by another fund, and thus assumed two fund identifiers. 

# In[ ]:


df_mflink1.groupby("crsp_fundno").size().value_counts()


# - It is also important to point out that certain `crsp_fundno` cannot be matched with any `wficn`. 
# - Based on the descriptions in the paper, I decide to drop these samples. 
# - Next, let us merge CRSP data and `mflink1` to obtain the appropriate `wficn`. 

# In[ ]:


print(f"Before merging, df_crsp has {df_crsp.shape[0]} rows")
df_crsp = df_crsp.merge(df_mflink1, how="inner", on="crsp_fundno").reset_index(
    drop=True
)
print(f"After merging, df_crsp has {df_crsp.shape[0]} rows")


# In[ ]:


df_crsp = df_crsp.sort_values(["caldt", "wficn"])
df_crsp['year'] = df_crsp['caldt'].dt.year.astype('int')
df_crsp['month'] = df_crsp['caldt'].dt.month.astype('int')
df_crsp = df_crsp[df_crsp['wficn'].notnull()]
df_crsp['wficn'] = df_crsp['wficn'].astype('int')
df_crsp['mret'] = df_crsp['mret'].fillna(0)
df_crsp.head()


# ## Computing Yearly Returns
# - To replicate Table 1, we need to compute yearly returns. 
# - To do that, we first need to compute each fund's monthly returns. 
# - **Intention**: We would like to follow footnote 4's approach of using `mtna` as weight. 
# - **Issue**: Not all `mtna` are available, most likely because the mutual funds did not report this number. This is especially severe for 1990 and earlier. 
# - **Solution**: The paper does not specify the method of resolving this issue. We could pull in TNA values elsewhere. Here, I will simply use **simple average** instead. This is reasonable, because it's most likely that different share classes of the same mutual fund should have very close returns. 

# In[ ]:


df_crsp['mtna'].isnull().mean()


# In[ ]:


df_ret = (
    df_crsp.groupby(
        [
            "wficn",
            "year",
            "month",
        ]
    )["mret"]
    .mean()
    .reset_index()
)
df_ret["mult"] = 1 + df_ret["mret"]
df_ret["cumret"] = (
    df_ret.sort_values(["year", "month"]).groupby(["wficn", "year"])["mult"].cumprod()
)

# only care about yearly return
df_ret = df_ret.query("month==12")
df_ret['yret'] = df_ret['cumret'] - 1


# In[ ]:


df_ret.head()


# ## Year-end TNA
# - Table 1 reports TNA, but it does not specify whether it is average of max or year-end TNA. 
# - For this project, I will use year-end TNA only. 
# - We then merge the TNA and yearly return information. 

# In[ ]:


df_tna = df_crsp.query("month==12").groupby(["wficn", "year"])["mtna"].sum().reset_index().rename(columns={"mtna": "crsp_tna"})
df_crsp_clean = pd.merge(df_tna, df_ret)[['wficn', 'year', 'crsp_tna', 'yret']]


# In[ ]:


df_crsp_clean.head()


# # S12 Data
# - The S12 database link: https://wrds-www.wharton.upenn.edu/data-dictionary/tr_mutualfunds/s12/
# 
# ## Missing TNA Values
# - The author specified in the paper that "we require that the TNAs reported in the Thomson Reuters database and in the CRSP database do not di￿er by more than a factor of two."
# - Looking at s12 table, it's clear that `assets` represents the TNA values. 
# - **Issues**: The s12 table has a lot of missing assets fields for 2010 and 2013, especially 2011 and 2012. See below for a demonstration.  
# - **Solution**: I cannot think of any obvious solution. One possibility might be to assume the TNA is merely the sum of all holdings' values provided by the s12 table.

# In[ ]:


query = """
SELECT
  EXTRACT(YEAR FROM fdate) AS year,
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN assets IS NULL THEN 1 END) AS missing_assets,
  (COUNT(CASE WHEN assets IS NULL THEN 1 END) * 100.0 / COUNT(*)) AS missing_percentage
FROM
  tfn.s12
WHERE
  fdate >= '2007-01-01' and fdate <= '2016-12-31'
GROUP BY
  EXTRACT(YEAR FROM fdate)
ORDER BY
  year;
"""

db = wrds.Connection(wrds_username=WRDS_USERNAME)
temp = db.raw_sql(query)
db.close()


# In[ ]:


temp


# ## Merging `s12` and `mflink2`
# - Unlike `mflink1`, the mapping for s12 to wficn has date information. 
# - If I simply use the tuple of (fdate, fundno) to merge, there will be a lot of missing matches. 
# - To circumvent this issue, I decided to **obtain the last valid record of wficn for each (year, fundno)**. 

# In[ ]:


df_mflink2.head()


# In[ ]:


df_mflink2["year"] = df_mflink2["fdate"].dt.year.astype("int")
eoy_link2 = (
    df_mflink2[df_mflink2.wficn.notnull()]
    .sort_values("fdate")
    .groupby(["fundno", "year"])["wficn"]
    .last()
    .reset_index()
)
eoy_link2['wficn'] = eoy_link2['wficn'].astype('int')


# - We observe a huge reduction in sample size after the merge, probably because s12 contain a lot of **non domestic funds** which are not covered WRDS's MFLINK

# In[ ]:


# merge with s12 data
df_s12["year"] = df_s12["fdate"].dt.year.astype("int")
print(f"Before merging, df_s12 has {df_s12.shape[0]} rows")
df_s12 = df_s12.merge(eoy_link2, how="inner", on=["fundno", "year"]).reset_index(
    drop=True
)
print(f"After merging, df_s12 has {df_s12.shape[0]} rows")


# In[ ]:


df_s12.head()


# ## Domestic Equity?
# - S12 data has a "country" to identify countries of the stocks, and "stkcdesc" to identify classes of the stocks. 
# - However, these data are not missing before 2000.
# - Since we've already filtered on domestic equity funds in `df_crsp_clean`, I decided to just **group together** all holdings and assume they are all US equities. 

# In[ ]:


# temporarily fillna with 0 to avoid missing records
df_s12['assets'] = df_s12['assets'].fillna(0)
df_eq = df_s12.groupby(['year', 'fdate', 'wficn', 'assets', ])['useq_tna_k'].sum().reset_index()


# - As explained before, I am not aware of any clear solution for replacing `assets`, and I will simply keep it as NaN for now. 

# In[ ]:


df_eq['assets'] = np.where(df_eq['assets'] == 0, np.nan, df_eq['assets'])


# - `s12` is updated quarterly, but the paper does not specify how to aggregate on a yearly basis. 
# - For simplicity, I will simply get the last record for each year for now. 

# In[ ]:


df_eq.head()


# In[ ]:


df_eq = df_eq.groupby(['wficn', 'year'])[['assets', 'useq_tna_k']].last().reset_index()


# # Merging CRSP and S12 Data
# - It is finally time to merge. 

# In[ ]:


df_eq.head()


# In[ ]:


df_crsp_clean.head()    


# In[ ]:


df_combo = pd.merge(df_crsp_clean, df_eq, on=["wficn", "year"], how="inner").sort_values("year")
df_combo.head()


# # Applying Filters To Identify Universe
# - Let us first see the universe before applying any filters.
# - It is interesting to observe that without applying any filters, this number of funds data match closely for certain years, but mismatch greatly for others. 
# - For 1980 and 1993, for instance, the numbers are identical. 

# In[ ]:


df_combo.groupby('year').size().reset_index().rename(columns={0: 'count'})


# - Next, I will apply some filters as specified in the paper. 

# In[ ]:


# TNA above 1 million at year end
df_combo = df_combo.query("crsp_tna > 1")
df_combo.groupby("year").size().reset_index().rename(columns={0: "count"})


# - Since we have a lot of missing "assets" for certain years, for simplicity, I assume the TNA ratio would be 1 in that case. 

# In[ ]:


df_combo["tna_ratio"] = np.where(
    df_combo["assets"].isnull(),
    1,
    df_combo["crsp_tna"] * 1e6 / df_combo["assets"] / 1e4,
)
df_combo = df_combo.query("tna_ratio > 0.5 and tna_ratio < 2")
df_combo.groupby("year").size().reset_index().rename(columns={0: "count"})


# - Finally we compute the equity ratio. 
# - The paper doesn't specify which TNA to use. 
# - I will use both and require at least one of them to fall between 0.8 and 1.05.

# In[ ]:


df_combo["eq_ratio_1"] = df_combo["useq_tna_k"] * 1e3 / (df_combo["crsp_tna"] * 1e6)
df_combo["eq_ratio_2"] = np.where(
    df_combo["assets"].isnull(),
    1,
    df_combo["useq_tna_k"] * 1e3 / (df_combo["assets"] * 1e4),
)
df_combo.head()


# In[ ]:


df_combo = df_combo[
    (df_combo["eq_ratio_1"].between(0.8, 1.05))
    | (df_combo["eq_ratio_2"].between(0.8, 1.05))
]
df_combo.groupby("year").size().reset_index().rename(columns={0: "count"})


# - The above filter appears to remove too many samples for most years. 

# ## Returns and TNA
# - It's not easy to match these numbers, especially the return numbers. 

# In[ ]:


df_combo.groupby("year")[["crsp_tna", "yret"]].agg(["mean", "median"]).reset_index().round(2)


# In[ ]:




