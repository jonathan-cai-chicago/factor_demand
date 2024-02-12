"""
Functions to pull and load CRSP mutual fund data

- Link to table: https://wrds-www.wharton.upenn.edu/data-dictionary/tr_mutualfunds/s12/

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

def pull_s12(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    wrds_username: str = WRDS_USERNAME,
) -> pd.DataFrame:
    """
    Pull S12 data from WRDS.

    Args:
    - start_date: str, start date in "YYYY-MM-DD" format
    - end_date: str, end date in "YYYY-MM-DD" format
    - wrds_username: str, WRDS username

    Returns:
    - df: pd.DataFrame, S12 data
    """
    # Connect to WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    
    query = f"""
    SELECT
    fdate,
    fundno,
    rdate,
    assets,
    stkcdesc,
    CASE WHEN country = 'UNITED STATES' THEN 1 ELSE 0 END AS us,
    SUM(shares * prc / 1000.0) AS useq_tna_k
    FROM
    tfn.s12
    WHERE 
        fdate >= '{start_date}' and fdate <= '{end_date}' AND
        prc > 0 AND shares > 0
    GROUP BY
    fdate,
    fundno,
    rdate,
    assets,
    stkcdesc,
    us
    """

    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["fdate", "rdate"])
    db.close()
    return df

def load_s12_file(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load S12 data from WRDS.

    Args:
    - data_dir: Path, path to data directory

    Returns:
    - df: pd.DataFrame, S12 data
    """
    path = data_dir / "pulled" / "s12.parquet"
    df = pd.read_parquet(path)
    return df


if __name__ == "__main__":
    df_s12 = pull_s12(START_DATE, END_DATE, WRDS_USERNAME)
    path = Path(DATA_DIR) / "pulled" / "s12.parquet"
    df_s12.to_parquet(path)
