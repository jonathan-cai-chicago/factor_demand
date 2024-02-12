"""
Functions to pull MFLink data

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


def pull_mflink1(
    wrds_username: str = WRDS_USERNAME,
) -> pd.DataFrame:
    """
    Pull the database with crsp_fundno to wficn mapping
    """
    # Connect to WRDS
    query = "SELECT * from mfl.mflink1"
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query)
    db.close()
    return df


def load_mflink1(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load crsp_fundno to wficn mapping
    """
    path = data_dir / "pulled" / "mflink1.parquet"
    df = pd.read_parquet(path)
    return df


def pull_mflink2(
    wrds_username: str = WRDS_USERNAME,
) -> pd.DataFrame:
    """
    Pull the database with s12 fundno to wficn mapping
    """
    # Connect to WRDS
    query = "SELECT * from mfl.mflink2"
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query)
    db.close()
    return df


def load_mflink2(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load s12 fundno to wficn mapping
    """
    path = data_dir / "pulled" / "mflink2.parquet"
    df = pd.read_parquet(path)
    return df


if __name__ == "__main__":
    df_link1 = pull_mflink1()
    path = Path(DATA_DIR) / "pulled" / "mflink1.parquet"
    df_link1.to_parquet(path)

    df_link2 = pull_mflink2()
    path = Path(DATA_DIR) / "pulled" / "mflink2.parquet"
    df_link2.to_parquet(path)
