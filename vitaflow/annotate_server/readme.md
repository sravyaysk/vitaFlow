

# VF Image Annotation Tool


Initial goal of this Annotation tool was to provide a simple annotation tool that can do the following

* A Image Annotation - Web tool (can run in local or over web for a group collaboration)
* Generation of XML-annotation file for each image file

As the growth continued, we added following image features to it.

* Fix rotation images
* Crop the image to user required size
* Automated Binarisation of images

Other pages

* A Summary page: To review how well the cropping & binarisation are working
* A Stats page: To know the statistics of on annotation completed

# History

Today, Image(document) Annotation has become of the important tool & job required for machine learning models to learn and try to predict. But as of 2019, we find only a limited number of tools that fit into out category that would help quickly annotated the image.

We have inspired from the works of `Image Annotation Programme` and wished to take forward, meeting our needs and increasing it capability by assisting users in their work.



# Project Development Page.

To know more the works in development, please check the following Project Page

[https://github.com/Imaginea/vitaFlow/projects/10](https://github.com/Imaginea/vitaFlow/projects/10)

## Experiment Feature

* TF Records Generation
* Support for HDFS files
* Support for Spark for image process/text extraction

# User Guide

## How to start Annotation Server

Please check for `requirement.txt` for installing required packages for Annotation Server.

```
$ pwd
vitaFlow/vitaflow/annotate_server

$ ~/anaconda3/bin/python vitaFlow/vitaflow/annotate_server/run.py
...
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
...
```

## How to Start Annotation

`#TODO`

## How to Cropper Image

`#TODO`

## Functionality - How the folder structure is

`#TODO`


## Where to collect images & annotation xml files

`#TODO`

## How to select models

`#TODO`

## How to use existing ML/DL Models to auto tag images

`#TODO`

## Workflow

Previous Plan of Work

         Image >> Image Processing  >> DL Model for Image Annotations >> Extract Annotated Images >> Text >> Receipt csv

New Plan of Action

         Image >> Image Processing  >> Line to Line Text Extraction >> Annotation Model >> Receipt csv

In Previous Plan, we are expecting following from DL Model

1. Identify the regions of text with in a large Image and then 
2. Classify these text into categories (Merchant/Line Items/Total)

In our new plan of action, as we are doing identification of text line, we are only expecting the model to only 2 step in above.    

### Lifecycle for Images

##### Step1: (Optional)For Automatic Receipt Localisation
1. East Folder - All the images along with east-text files are placed here.
2. Images Folder - using `receipt_localisation.py` generated the images here.

##### Step2: Annotation Server
1. Images Folder - Store all the images required to be annoated here.
2. Using Annotation Server - correct & convert the images for binarisation
3. Post binarisation - convert image to text files using `image_to_text.py`

##### Step3: Annotation Server 
1. Pass these text files to `Annotation Model` for Annotation(__WIP__)

