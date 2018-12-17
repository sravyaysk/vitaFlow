
# vitaFlow - VideoImageTextAudioFlow
 ![](../../vitaflow-icon.png)


## Introduction
In order to showcase the working the VitaFlow, let us solve the NER(named entity recognition) problem on CONLL 2003 dataset. By end of this readme, you will be able to use vitaFlow for training a model from scratch and run prediction on it. 
## Problem Statement
The objective is to build a model that can detect and tag different types of named entities in a given document.

**TODO**
Put live example, may be a picture or some tagged sentences
	
## Setup
### Getting Started
Clone the repository 

``
git clone https://github.com/Imaginea/vitaFlow
cd vitaFlow
``
### Folder Structure
The folder to be focussed in this experiment is placed in examples
```
~\vitaFlow
	\examples
		\conll2003
			config.py
```
This folder houses the configurations for the whole experiment which is passed to the running engine.
The config.py contains the location of  dataset directories, parameters and hyper-parameters. 
You can read more on this at [architecture]()

### Requirement
The recommended python interpreter is python3 and  conda  as dependency and environment manager
```bash 
# One time setup
conda create -n vitaflow-gpu python3 pip 
source activate vitaflow-gpu
python --version
```
Install the prerequisites

```bash
pip install -r requirements.txt
```
Incase you don't have a GPU please install tensorflow CPU version
```
conda install tensorflow
```
## How to run
Like every machine learning project, we have train and predict mode which needs to be passed as an argument while running the experiment.
The entry point in the system is *run.py*. The arguments passed to it are the config file and mode in which it has to run. By default it supports three modes (train, retrain and predict).
### 1. Train
```bash
python vitaflow/run/run.py --config_python_file=examples/conll2003/config.py --mode=train
``` 
We start the training by passing **--mode = train** .
run.py will pickup the config file provided to it  and look for the dataset. 


If the dataset is not present on the drive, the conll_2003_dataset.py will download, preprocess and keep the data ready. There will be 3 different files
```
~/vitaflow
	/data
		 /text
			 /conll
				 /conll_2003_dataset.py
```
### 2. Retrain
```bash
python vitaflow/run/run.py --config_python_file=examples/conll2003/config.py --mode=retrain
``` 
### 3. Predict
- Place all the files to be predicted in the /home/$USER/vitaFlow/conll_2003_dataset/preprocessed_data/test/ folder 
```bash
python vitaflow/run/run.py --config_python_file=examples/conll2003/config.py --mode=predict
```
- The result of the output will be generated in the folder
- The predictions will be placed in the folder ~/{experiment_root_directory}/{experiment_name}/{iterator_name}/predictions.

```bash
home/$USER/vitaFlow/conll_2003_dataset/conll_data_iterator/predictions
```