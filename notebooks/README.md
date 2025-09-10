# Demo Notebooks

Get the [ Folder of demo datafiles ](https://drive.google.com/drive/folders/1vWzhtpDkqN7JgHN07r5WvIWPdUy0aZWT?usp=sharing) and store it in `~/DATA/physion_Demo-Datasets`

## Usage 

Notebooks are stored as lightweight python script.
They can be executed using two methods:

1. (*best*) -- Using the [Jupyter extension](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) in the [Visual Studio Code editor](https://code.visualstudio.com/)

2. -- Using the `jupytext` module in `python`. You can get `jupytext` with `pip install jupytext`.
    Then you can transform the scripts to jupyter notebooks with:
    ```
    jupytext --to ipynb *.py
    ```