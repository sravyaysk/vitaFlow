[![Documentation Status](https://readthedocs.org/projects/vitaflow/badge/?version=latest)](https://vitaflow.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/imaginea/vitaflow/blob/master/LICENSE)
 

# VitaFlow - VideoImageTextAudioFlow
 ![](vitaflow-icon.png)

# Table of content

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [Development](#development)
5. [Architecture](#architecture)
6. [VitaFlow in Action](#vitaflow-in-action)
7. [License](#license)
8. [Contributions](#contributions) 
 
# Introduction

Deep Learning projects are increasing exponentially in recent times and the complexity of handling such projects alike across the domains. Be it business environment or in any online competitions, the main challenge is how we can reuse the code base what we have developed for one requirement/problem statement and use it to a slightly different new requirement/problem statement. 

We are incubating an ideation to solve this typical software design problem for Deep Learning projects, which would simplify the whole process of data preparation, modelling and serving in a easy configurable plug and play framework.

# Problem Statement

To come up with a framework that enables a fast prototyping of Deep Learning models that handles Video/Image/Text/Audio and provide an seamless way of serving them in different end points.

- __A Navie Developer__: Can I get an experimentation play ground, with some set of open datasets, data iterators and models? to learn by doing?
- __A Data Scientist__: Can I build a model with an available open dataset and later switch to production dataset when the Data Engineering team bring in the production data?
- __A Data Engineer__: Can I use any available model and run through my data during my data preparation cycle?
- __An Online Competitor__: Can I reuse the models and pieces of modules that I have developed for my last competition in my current competition? 
- __Business Deadlines__ : We had spend few months of effort while addressing a clients proposals and build a prototype.Can we showcase the prototype developed to upcoming project proposals as quick as possible? 

In our Deep Learning exploration/projects we wanted to build a framework that is modular as much as possible, plug ang play architecture for main modules and reuse our experience (hours of development and testing) from one project to another, while improving on existing capabilities without breaking it.  

# Proposed Solution

> __Data Science wisdom comes only through failed experimentation - Damian Mingle__

The thought process is to come up with following modular components which then can be then glued through
configuration:

 - Data Collection and Cleansing
 - Dataset modules with pre-processing modules
 - DataIterator modules (backed by [TF Data](https://www.tensorflow.org/guide/datasets))
 - Tensorflow Models (backed by [TF Estimators](https://www.tensorflow.org/guide/estimators))
 - An Engine to run the models
 - Tensorflow model serving using [TFLite](https://www.tensorflow.org/lite/)
    - Web app
    - Mobile

# Development

VitaFlow is continuously extended by solving some real work problem. To check some of these solved work please check [examples](https://github.com/Imaginea/vitaFlow/tree/master/examples) folder in repo.

_TODO: Update VitaFlow Features here_
<!--
    1. VitaFlow Features - what has been done/Available Features
    2. Current Implementation - what is being prepared 
    3. Future Implementations Plans - what is next 
-->

### Receipt OCR

Receipt OCR is related to [Information Extraction](https://en.wikipedia.org/wiki/Information_extraction) Domain. Using a open source Receipts dataset, we working on building a simple pipeline for valuable information extration.

For more updates on development & progress, please check [Receipt OCR - Readme](https://github.com/Imaginea/vitaFlow/tree/master/examples/receipt_ocr)

# Architecture

![](docs/images/vitaflow_stack.png)

# VitaFlow in Action

CoNLL 2003 data set is considered since the data size small and easy to test:

Run a experiment pointing to this [config](examples/conll2003/config.py),
which uses this [dataset](https://imaginea.github.io/vitaFlow/build/html/api/data/text/conll/conll_2003_dataset.html),
and this [data iterator](https://imaginea.github.io/vitaFlow/build/html/api/data/text/iterators/conll_csv_in_memory.html),
and this [model](https://imaginea.github.io/vitaFlow/build/html/api/models/text/seq2seq/bilstm_crf.html).

``` bash
cd /path/to/vitaflow/
python vitaflow/run/run.py --config_python_file=examples/conll2003/config.py
```

- The whole experimentation is configured from once place, based on the string names and corresponding config params 
- The `experiment run engine` takes care of setting up the modules which includes but not limited to, downloading the dataset, 
pre-processing the dataset, iterator initialization and finally running the model with check pointing the modle data as configured.

The big difference comes from the way the whole experiment can be configured. Say, if the model needs to replaced based 
on a latest paper published, it is simple as writing a model (that adheres some rules) and plugin into the vitaFlow, 
you are good to go with new model with old tested dataset and data iterator modules.

This level of configuration and project design will allows us to iterate the Deep Learning ideas as fast as possible to 
meet the business demands.



# License

The VitaFlow is licensed under the terms of the Apache License - Version 2.0.

# Contributions

* For readme - please follow
    * http://tom.preston-werner.com/2010/08/23/readme-driven-development.html
    * https://changelog.com/posts/top-ten-reasons-why-i-wont-use-your-open-source-project
