[![Build Status](https://travis-ci.com/dhiraa/shabda.svg?branch=master)](https://travis-ci.com/dhiraa/shabda/)
[![Documentation Status](https://readthedocs.org/projects/shabda/badge/?version=latest)](https://shabda.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/imaginea/vitaflow/blob/master/LICENSE)
 

# vitaFlow - VideoImageTextAudioFlow
 
## Introduction

Deep Learning projects are increasing exponentially in recent times and the complexity 
of handling such projects alike. We are incubating a ideation which would simplify the whole
process of data preparation, modelling and serving in a easy configurable plug and play framework.

## Problem Statement

To come up with a framework that enables a fast prototyping of Deep Learning 
models that handles Video/Image/Text/Audio and provide an seamless way serving them
in different end points.

## Proposed Solution

Come up with following modular components which can be then used as plug and play components:
 - Dataset modules with preprocessing modules
 - DataIterator modules
 - Tensorflow Models (Estimators)
 - Engine to run the models
 - Tensorflow model serving using TFLite
    - Web app
    - Mobile

## Architecture

![](docs/images/vitaflow_stack.png)

## A Simple Demo
```
cd /path/to/vitaflow/
python vitaflow/run/run.py --config_python_file=examples/conll2003/config.py
```