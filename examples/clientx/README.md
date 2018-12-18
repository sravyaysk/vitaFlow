
# vitaFlow - VideoImageTextAudioFlow
 ![](../../vitaflow-icon.png)


## Introduction
This is the second experiment being performed with the help of vitaFlow. 
We will be creating a new experiment from scratch which would provide a better insight in dealing with new datasets. 
We will be solving the Named Entity Extraction problem on a dataset that is different than CONLL2003.
  
## Problem Statement
The objective is to build a model with a dataset that can detect and tag different types of named entities in a given document.

**TODO**
Put live example, may be a picture or some tagged sentences
	
## Setup
### Getting Started
In order to setup and install dependencies kindly refer the [this](examples/conll2003/README.md),

### Folder Structure
The focus on folder named examples. Create  a new folder called **clientx** and place config.py. 
This file can be a copy of conll2003 config.py but make sure to change the necessary
values like entity column and word column. We have changed the experiment and dataset name to match 
out clientx affix

```
~\vitaFlow
	\examples
		\clientx
			config.py
			__init__.py
```

This folder houses the configurations for the whole experiment which is passed to the running engine.
The config.py contains the location of  dataset directories, parameters and hyper-parameters. 
You can read more on this at [architecture]()

## Code Modifications

- Create a folder called as clientx under 
```bash
~/vitaFlow/vitaflow/data/text
```
-In this folder create a dataset file called as clientx_dataset.py. This file
will handle all the dataset related operations. For reference look at _vitaflow/data/text/conll/conll_2003_dataset.py_

- Next we need to register the newly created dataset so change the line in 
_vitaflow/run/factory/dataset.py_ as follows.

```python
dataset_path = {
        # file_name : package
        "conll_2003_dataset": "vitaflow.data.text.conll.conll_2003_dataset",
    }

datasets = {
        # file_name : class_name
        "conll_2003_dataset": "CoNLL2003Dataset"
    }
```

```python
dataset_path = {
        # file_name : package
        "conll_2003_dataset": "vitaflow.data.text.conll.conll_2003_dataset",
        "clientx_dataset": "vitaflow.data.text.clientx.client_dataset",
    }

datasets = {
        # file_name : class_name
        "conll_2003_dataset": "CoNLL2003Dataset",
        "clinetx_dataset": "CLIENTXDataset"
    }
```
- Add an entry of dataset type in _vitaflow/data/text/iterators/internal/dataset_types.py_

#TODO dataset creation from the intial files
