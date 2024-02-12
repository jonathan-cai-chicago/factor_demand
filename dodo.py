import sys
sys.path.insert(1, './src/')


import config
from pathlib import Path
from doit.tools import run_once


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)

def task_pull_CRSP():
    """
    Pull CRSP mutual fund data from WRDS
    """
    file_dep = [
        "./src/config.py",
        "./src/load_CRSP_fund.py",
    ]
    targets = [
        Path(DATA_DIR) / "pulled" / file for file in [
            "CRSP_TNA.parquet", "CRSP_style.parquet"
            ]
    ]

    return {
        "actions": [
            "python ./src/config.py",
            "python ./src/load_CRSP_fund.py",
        ],
        "file_dep": file_dep,
        "targets": targets,
        "clean": True,
        "verbosity": 2,
    }