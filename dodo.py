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
            "CRSP_fund_combined.parquet",
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


def task_pull_s12():
    """
    Pull s12 mutual fund data from WRDS
    """
    file_dep = [
        "./src/config.py",
        "./src/load_s12.py",
    ]
    targets = [
        Path(DATA_DIR) / "pulled" / file for file in [
            "s12.parquet",
            ]
    ]

    return {
        "actions": [
            "python ./src/config.py",
            "python ./src/load_s12.py",
        ],
        "file_dep": file_dep,
        "targets": targets,
        "clean": True,
        "verbosity": 2,
    }


def task_mflink():
    """
    Pull Linking data from WRDS
    """
    file_dep = [
        "./src/config.py",
        "./src/load_mflink.py",
    ]
    targets = [
        Path(DATA_DIR) / "pulled" / file for file in [
            "mflink1.parquet",
            "mflink2.parquet",
            ]
    ]

    return {
        "actions": [
            "python ./src/config.py",
            "python ./src/load_mflink.py",
        ],
        "file_dep": file_dep,
        "targets": targets,
        "clean": True,
        "verbosity": 2,
    }



## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"


def task_convert_notebooks_to_scripts():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    build_dir = Path(OUTPUT_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        "01_example_notebook.ipynb",
        "02_raw_data_walkthrough.ipynb",
    ]
    file_dep = [Path("./src") / file for file in notebooks]
    stems = [notebook.split(".")[0] for notebook in notebooks]
    targets = [build_dir / f"_{stem}.py" for stem in stems]

    actions = [
        # *[jupyter_execute_notebook(notebook) for notebook in notebooks_to_run],
        # *[jupyter_to_html(notebook) for notebook in notebooks_to_run],
        *[jupyter_clear_output(notebook) for notebook in stems],
        *[jupyter_to_python(notebook, build_dir) for notebook in stems],
    ]
    return {
        "actions": actions,
        "targets": targets,
        "task_dep": [],
        "file_dep": file_dep,
        "clean": True,
    }


def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    notebooks = [
        "01_example_notebook.ipynb",
        "02_raw_data_walkthrough.ipynb",
    ]
    stems = [notebook.split(".")[0] for notebook in notebooks]

    file_dep = [
        # 'load_other_data.py',
        *[Path(OUTPUT_DIR) / f"_{stem}.py" for stem in stems],
    ]

    targets = [
        ## 01_example_notebook.ipynb output
        OUTPUT_DIR / "sine_graph.png",
        ## Notebooks converted to HTML
        *[OUTPUT_DIR / f"{stem}.html" for stem in stems],
    ]

    actions = [
        *[jupyter_execute_notebook(notebook) for notebook in stems],
        *[jupyter_to_html(notebook) for notebook in stems],
        *[jupyter_clear_output(notebook) for notebook in stems],
        # *[jupyter_to_python(notebook, build_dir) for notebook in notebooks_to_run],
    ]
    return {
        "actions": actions,
        "targets": targets,
        "task_dep": [],
        "file_dep": file_dep,
        "clean": True,
    }


def task_compile_latex_docs():

    file_dep = [
        "./reports/project_writeup.tex", 
        "./src/02_raw_data_walkthrough.py", 
    ]
    file_output = [
        "./reports/project_writeup.pdf", 
    ]
    
    targets = [file for file in file_output]

    return {
        "actions": [
            "latexmk -xelatex -cd ./reports/project_writeup.tex",  # Compile
            "latexmk -xelatex -c -cd ./reports/project_writeup.tex",  # Clean 
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


