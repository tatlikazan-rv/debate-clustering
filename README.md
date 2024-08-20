# Setup
DEPRECATED DEPENDENCIES

# Argument Clustering in Debate Format: Game of Thrones

## Folder structure
```
.
├── argument_clustering/
│   ├── Argument_clustering_notebook.ipynb
│   └── argument_clustering.py
├── data/
│   └── Game_of_Thrones_Script.csv
├── models
│   └──BERTopic_vis.ipynb
│   └──bigartm.py
│   └──dbi_vis.ipynb
│   └──pipline_test.ipynb
│   └──pipeline_test2.ipynb
│   └──sc_vis.ipynb
│   └──tomotopy_models.ipynb
│   └──topic_modelling_utils.py
│   └──topic_processing.ipynb
├── topic_clustering/
│   ├── Topic_clustering_notebook.ipynb
│   ├── linkage.npy
│   └── topic_clustering.py
├── Argument_Clustering_in_Debate_Format__Game_of_Thrones.pdf
├── README.md
├── pipeline.ipynb
├── pre_processing.py
├── requirements.txt
└── utils.py
```

## Files description

### argument_clustering
```Argument_clustering_notebook.ipynb```
Notebook for working and training with GloVe, Word2Vec and Doc2Vec.


```argument_clustering.py```
Functions used for argument clustering and visualisation

### data
```Game_of_Thrones_Script.csv```
Source dataset from https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons/code?resource=download


### models
folder containing the models of all the topic modelling methods we tried.

```BERTopic_vis.ipynb```
Notebook for loading and visualizing the BERTopic model

```bigartm.py```
a separate file for the bigartm topic modelling method since it only works on linux devices

```dbi_vis.ipynb & sc_vis.ipynb```
Notebooks displaying the DBI and SC for each of the topic modelling methods except BERTopic

```pipline_test.ipynb & pipline_test2.ipynb```
Our initial attempt at creating a pipeline for topic modelling. They were also used to create the BERTopic models

```tomotopy_models.ipynb```
Notebook used for generating and saving all the topic modelling models that 
are available in the tomotopy library

```topic_modelling_utils.py```
Various functions that are used in the different topic modelling notebooks

```topic_processing.ipynb```
A notebook to show off different types of information that can be extracted from the assigned topics


### topic_clustering
Topic clustering was deprecated later in our work, so is also missing from the report.

```Topic_clustering_notebook.ipynb```
Notebook for training the clustering based on tfidf vectors and visualising the clusters with different parameters. 

```topic_clustering.py```
Functions used for topic clustering and visualisation


### main

```Argument_Clustering_in_Debate_Format__Game_of_Thrones.pdf```
Final project report

```pipeline.ipynb```
The skeleton of our project - contains all steps of our pipeline: preprocessing, topic modelling and argument clustering.

```requirements.txt```
List of the packages used in the project

```utils.py```
Basic utility functions used thorughout the project
