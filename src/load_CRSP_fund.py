"""
Functions to pull and load CRSP mutual fund data

- CRSP updates mutual fund data quarterly. 
- List of all tables: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_q_mutualfunds/
- We use `monthly_tna_ret_nav` to pull TNA and monthly returns.
- We use `fund_style` table to identify US Equity funds.

Author: Jonathan Cai [mcai@uchicago.edu]
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

import config

DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME
START_DATE = config.START_DATE
END_DATE = config.END_DATE


def pull_CRSP_combined_file(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    wrds_username: str = WRDS_USERNAME,
) -> pd.DataFrame:
    """
    Pull CRSP mutual fund TNA and style data from WRDS

    Args:
    - start_date: str, start date in "YYYY-MM-DD" format
    - end_date: str, end date in "YYYY-MM-DD" format
    - wrds_username: str, WRDS username

    Returns:
    - df: pd.DataFrame, CRSP mutual fund TNA and style data
    """
    # Connect to WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    query = f"""
    SELECT 
        a.*, b.lipper_asset_cd, b.lipper_class_name, b.policy
    FROM 
        crsp.monthly_tna_ret_nav a
    LEFT JOIN 
        crsp.fund_style b
    ON 
        a.crsp_fundno = b.crsp_fundno
    WHERE 
        a.caldt BETWEEN '{start_date}' AND '{end_date}' AND
        a.caldt BETWEEN b.begdt AND b.enddt;
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()

    return df


def load_CRSP_combined_file(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load CRSP mutual fund TNA and style data from WRDS

    Args:
    - data_dir: Path, root data directory.

    Returns:
    - df: pd.DataFrame, CRSP mutual fund TNA and style data
    """
    path = data_dir / "pulled" / "CRSP_fund_combined.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pull_CRSP_combined_file()
        df.to_parquet(path)
    return df


########################################################################################
# Old functions to pull data separately
########################################################################################

def pull_CRSP_TNA_file(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    wrds_username: str = WRDS_USERNAME,
) -> pd.DataFrame:
    """
    Pull CRSP mutual fund TNA and return data from WRDS

    Args:
    - start_date: str, start date in "YYYY-MM-DD" format
    - end_date: str, end date in "YYYY-MM-DD" format
    - wrds_username: str, WRDS username

    Returns:
    - df: pd.DataFrame, CRSP mutual fund TNA and return data
    """
    # Connect to WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    query = f"""
    SELECT * 
    FROM crsp.monthly_tna_ret_nav
    WHERE 
        caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()

    return df


def pull_CRSP_fund_style_file(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    wrds_username: str = WRDS_USERNAME,
) -> pd.DataFrame:
    """
    Pull CRSP mutual fund style data from WRDS

    Args:
    - start_date: str, start date in "YYYY-MM-DD" format
    - end_date: str, end date in "YYYY-MM-DD" format
    - wrds_username: str, WRDS username

    Returns:
    - df: pd.DataFrame, CRSP mutual fund style data
    """
    # Connect to WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    query = f"""
    SELECT 
        crsp_fundno, begdt, enddt, lipper_asset_cd, lipper_class_name, policy
    FROM crsp.fund_style
    WHERE 
        enddt >= '{start_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["begdt", "enddt"])
    db.close()

    return df


def load_CRSP_TNA_file(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load CRSP mutual fund TNA and return data from WRDS

    Args:
    - data_dir: Path, root data directory.

    Returns:
    - df: pd.DataFrame, CRSP mutual fund TNA and return data
    """
    path = data_dir / "pulled" / "CRSP_fund_tna.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pull_CRSP_TNA_file()
        df.to_parquet(path)
    return df


def load_CRSP_fund_style_file(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load CRSP mutual fund style data from WRDS

    Args:
    - data_dir: Path, root data directory.

    Returns:
    - df: pd.DataFrame, CRSP mutual fund style data
    """
    path = data_dir / "pulled" / "CRSP_fund_style.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pull_CRSP_fund_style_file()
        df.to_parquet(path)
    return df

if __name__ == "__main__":

    # df_tna = pull_CRSP_TNA_file(start_date=START_DATE, end_date=END_DATE)
    # path = Path(DATA_DIR) / "pulled" / "CRSP_fund_tna.parquet"
    # df_tna.to_parquet(path)

    # df_style = pull_CRSP_fund_style_file(start_date=START_DATE, end_date=END_DATE)
    # path = Path(DATA_DIR) / "pulled" / "CRSP_fund_style.parquet"
    # df_style.to_parquet(path)

    df_combined = pull_CRSP_combined_file(start_date=START_DATE, end_date=END_DATE)
    path = Path(DATA_DIR) / "pulled" / "CRSP_fund_combined.parquet"
    df_combined.to_parquet(path)
