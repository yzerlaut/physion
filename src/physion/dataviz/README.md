# Data Visualization

The data visualization module consists in cutom code relying on the excellent [PyQtGraph module](http://pyqtgraph.org/).

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

