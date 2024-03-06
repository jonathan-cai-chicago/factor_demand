import numpy as np
import wrds
import config
from pathlib import Path

DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME
START_DATE = config.START_DATE
END_DATE = config.END_DATE
OUTPUT_DIR = Path(config.OUTPUT_DIR)

from sklearn.linear_model import LinearRegression
from load_CRSP_fund import load_CRSP_combined_file
from load_mflink import load_mflink1
from factor_betas_calculation import monthly_mutual_fund, fama_french_factors, regression_df, regression, calc_penal_A, calc_penal_B, calc_penal_C


df_crsp = monthly_mutual_fund()
df_ff = fama_french_factors()
df_reg = regression_df(df_crsp, df_ff)


def test_panel_A():
    panelA = calc_penal_A(df_reg)
    data = {'Mkt-RF': [0.98, 0.22, 0.64, 0.90, 0.99, 1.07, 1.28],
            'SMB': [0.16, 0.36, -0.31, -0.1, 0.07, 0.38, 0.83],
            'HML': [-0.02, 0.34, -0.52, -0.20, -0.01, 0.16, 0.47],
            'MOM': [0.00, 0.18, -0.27, -0.08, 0.00, 0.07, 0.28],
            'CMA': [-0.08, 0.41, -0.68, -0.26, -0.06, 0.11, 0.46],
            'RMW': [-0.04, 0.33, -0.55, -0.17, -0.01, 0.13, 0.38],
            'flow': [0.01, 0.18, -0.17, -0.03, 0.00, 0.04, 0.23]}

    panelA_test = pd.DataFrame(data, index=['mean', 'std', 'P5', 'P25', 'P50', 'P75', 'P95'])
    assert (((panelA-0.2) <= panelA_test) & (panelA_test <= (panelA+0.2))).all().all()

    
def test_panel_B():
    panelB = calc_penal_B(df_reg)
    data = {'Mkt-RF': [0.98, 1.04, 1.00, 0.98, 1.03, 1.02],
            'SMB': [0.16, 0.29, 0.2, -0.08, 0.38, 0.73],
            'HML': [-0.02,-0.19, 0.23, -0.02, -0.03, 0.07],
            'MOM': [0.00, 0.09, -0.07, 0.01, 0.03, 0.03],
            'CMA': [-0.08, -0.23, 0.06, -0.07, -0.10, -0.10],
            'RMW': [-0.04, -0.16, 0.10, 0.00, -0.05, 0.00],
            'flow': [0.01, 0.00, -0.01, -0.01, 0.00, 0.00]}

    panelB_test = pd.DataFrame(data, index=['All', 'Growth', 'Value', 'Large cap', 'Medium cap', 'Small cap'])
    assert (((panelB-0.2) <= panelB_test) & (panelB_test <= (panelB+0.2))).all().all()

    
def test_panel_C():
    panelC = calc_penal_C(df_reg)
    data = {'Mkt-RF': [1.02, 0.63, 0.93, 1.01, 1.00],
            'SMB': [0.09, 0.08, 0.07, 0.09, 0.25],
            'HML': [-0.02,-0.40, 0.01, -0.02, -0.01],
            'MOM': [-0.06, -0.01, -0.04, -0.06, 0.02],
            'CMA': [-0.08, 0.40, -0.05, -0.09, -0.09],
            'RMW': [0.01, -0.30, 0.05, 0.00, -0.03],
            'flow': [0.00, -0.01, -0.04, -0.01, 0.00]}

    panelC_test = pd.DataFrame(data, index=['All index funds', 'Enhanced', 'Base', 'Pure', 'All non-index funds'])

    assert (((panelC-0.2) <= panelC_test) & (panelC_test <= (panelC+0.2))).all().all()
