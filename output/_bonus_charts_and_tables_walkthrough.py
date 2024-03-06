#!/usr/bin/env python
# coding: utf-8

# # Additional Charts and Tables (Jean-Sebastien Gaultier)
# 
# This notebook is used as a step-by-step for the creation of intial plots and graphs that can be used to grasp a first understanding of the data.
# 
# More specifically we will work with understanding the returns of the funds.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# 
import config
from pathlib import Path
OUTPUT_DIR = Path(config.OUTPUT_DIR)


# In[ ]:


import numpy as np
from matplotlib import pyplot as plt


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


# In[ ]:


df_mflink1.groupby("crsp_fundno").size().value_counts()
df_mflink1


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


# In[ ]:


df_crsp["crsp_obj_cd"].value_counts()


# ## Funds grouped by their years
# 
# The idea with this is that each fund will have all the information about that specific year. 
# Additionally, we do not know exactly what the monthly return means. (for example if monthly return in january was 10% and in February it was 10% does that mean the monthly return over the year was 20%, or just 10%) Therefore we take both of those into account.

# In[ ]:


df_crsp['year'] = pd.to_numeric(df_crsp['year'], errors='coerce')


# In[ ]:


# Here we get the monthly return but as a sum for each fund
grouped_data = df_crsp.groupby(['crsp_fundno', 'year'])
result_df = grouped_data['mret'].sum().reset_index()
result_df


# In[ ]:


grouped_data = df_crsp.groupby(['crsp_fundno', 'year'])
result_df = grouped_data['mret'].mean().reset_index()
result_df


# In[ ]:


grouped_yearly_data = result_df.groupby('year')
yearly_stats_df = grouped_yearly_data.agg(
    num_funds=('crsp_fundno', 'size'),
    mret_mean=('mret', 'mean'),
    mret_median=('mret', 'median')
).reset_index()
yearly_stats_df


# When we run this, we notice that we do not have the same number of funds per year than the table 1 has in the report. Therefore that might explain why the returns are different as well. We will try to get rid of some funds.

# # Yearly average return plot
# 
# This plot can be used to see which years the funds were good to invest in and the years they were not as much

# In[ ]:


plt.figure(figsize=(10, 6))
plt.bar(yearly_stats_df['year'], yearly_stats_df['mret_mean'], color='skyblue')
plt.xlabel('Year')
plt.ylabel('Average Monthly Return')
plt.title('Distribution of Average Monthly Returns by Year')
plt.grid(axis='y')
plt.show()
plt.savefig('histogram.png')


# This plot is wonderful because we clearly see the crisis in 2008 had a big impact and all the funds were down on average 40% which is huge.

# ## Plotting Returns by year and fund group type
# 
# The important thing to understand is that funds are divided into different groups
# 
# Some of those groups are (we are assuming this is what the codes mean):
# 
# - EDYG:  "Equity Domestic Growth" funds, which typically invest in stocks with a focus on dividends and growth potential.
# 
# - EDYB:  "Equity Domestic Balanced" funds, which may invest in both stocks and bonds, with an emphasis on dividends.
# 
# - EDCS: "Equity Consumer Staples" funds, which invest primarily in companies that produce or distribute consumer staples such as food, beverages, and household goods.

# In[ ]:


grouped_data = df_crsp.groupby(['year', 'crsp_obj_cd'])
yearly_objcode_stats_df = grouped_data['mret'].mean().reset_index()
print(yearly_objcode_stats_df)


# In[ ]:


grouped_data = yearly_objcode_stats_df.groupby('crsp_obj_cd')
objcode_mean_mret_df = grouped_data['mret'].mean().reset_index()
objcode_mean_mret_df


# In[ ]:


# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(objcode_mean_mret_df['crsp_obj_cd'], objcode_mean_mret_df['mret'], color='skyblue')
plt.xlabel('CRSP Object Code')
plt.ylabel('Average Monthly Return')
plt.title('Average Monthly Return for Each Object Code (40-year Span)')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()
plt.savefig('histogram_mret_per_code.png')


# This graph is pretty good as well because it shows how each type of fund acted over the duration of the data. We do notice that most people will have made money aside from the people that had their money in the EDYS funds

# In[ ]:


object_code_counts = yearly_objcode_stats_df['crsp_obj_cd'].value_counts().reset_index()
object_code_counts.columns = ['crsp_obj_cd', 'count']

# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(object_code_counts['crsp_obj_cd'], object_code_counts['count'], color='skyblue')
plt.xlabel('CRSP Object Code')
plt.ylabel('Count')
plt.title('Count of Occurrences for Each Object Code')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()
plt.savefig('histogram_code_count.png')


# For this graph we notice that not all of the fund types existed throughout all of the years.

# ## Count of the number of funds per object code

# In[ ]:


object_code_counts = df_crsp['crsp_obj_cd'].value_counts().reset_index()
object_code_counts.columns = ['crsp_obj_cd', 'count']

# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(object_code_counts['crsp_obj_cd'], object_code_counts['count'], color='skyblue')
plt.xlabel('CRSP Object Code')
plt.ylabel('Count')
plt.title('Count of Occurrences for Each Object Code (Overall)')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()
plt.savefig('histogram_code_count_overall.png')


# This graph concludes our analysis of the object codes. We notice that EDYG funds have existed for the longest and are the most popular. Their return seems to follow the market at an average of almost 10% per year.
# 
# For example investing $100,000 in 1980 into an EDYG would now run you over 
# # $3.14 million

# # Final Charts
# 
# In the data/manual, there are various histograms that have been included that describe the data and the returns.
# - histogram.png: Distribution of Average Monthly Returns by Year
# - histogram_mret_per_code.png: Average Monthly Return for Each Object Code (40-year Span)
# - histogram_code_count.png: Count of Occurrences for Each Object Code
# - histogram_code_count_overall.png: Count of Occurrences for Each Object Code (Overall)
# 
# Additionally there are some interesting dataframes:
# - objcode_mean_mret_df: which gives the average monthly return per year per object code
# - yearly_stats_df: which gives the average monthly return per year
# 

# ## Unit test

# In[ ]:


import unittest

data = {
    'year': [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987],
    'mean': [0.09, 0.08, 0.21, -0.01, 0.01, 0.17, 0.04, -0.22],
    'median': [0.10, 0.08, 0.23, -0.01, 0.01, 0.16, 0.04, -0.21]
}
expected_df = pd.DataFrame(data)
class TestYearlyStats(unittest.TestCase):
    
    def test_head_matches(self):
        self.assertTrue(yearly_stats_df.head().equals(expected_df.head()))

# Run the unit tests
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# 
