# Data Visualization

This 


## Interactive viewer for NWB files

This interactive data visualization module relies on the excellent [PyQtGraph module](http://pyqtgraph.org/).


## Producing figures showing raw data

Demo in the notebook [Visualize-Raw-Data.py](../../../notebooks/Visualize-Raw-Data.py)

## Snapshot generation

- To visualize from the demo dataset (assumes you have the [demo data](https://drive.google.com/drive/folders/1vWzhtpDkqN7JgHN07r5WvIWPdUy0aZWT?usp=sharing) in `~/DATA/physion_Demo-Datasets`):
```
python -m physion.dataviz.snapshot 
```

- To see the plot layout:
```
python -m physion.dataviz.snapshot show-layout
```

- To generate a template file
```
python -m physion.dataviz.snapshot generate-template
```

## Movie generation

```
python -m physion.dataviz.movie 
```