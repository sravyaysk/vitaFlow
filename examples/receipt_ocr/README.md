# Receipt OCR

Given a receipt, the aim is to extract the domain specific fields by using deep-learning techinques.

__General Receipt Tags__

- merchant
- date
- line_item
- total
- tax

# Table of content

1. [Introduction](#introduction)
2. [Goal](#goal)
3. [Development Plan](#development-plan)
4. [Development Progress](#development-progress)
   1. [Dataset](dataset)
   2. [Object Detection Model](object_detection_model)
   3. [Image Annotation Tool](image_annotation_tool)
   4. [Text Extraction](text_extraction)
5. [Explorations](explorations)
6. [Acknowledgments](acknowledgments)

# Introduction

```
In this example, we show solution for extracting information from the given documents.
```

```
Scanned PDFs/Images  --> Image Preprocessing 
                        ---> Resizing
                        ---> Deskewing
                        ---> Rotation
                        ---> Sharpening 
                        ---> Background noise removal
                        ---> Binarization 
                        ---> Stiching etc.,
                     --> OCR (Tesseract/DL Models)
                     --> Doc/Text Classifiers
                        ---> Identifying the receipts etc.,
                     --> Annotators (Optional)
                     --> ML/DL Model (Optional)
                     --> Data Insights 
                        ---> Format templates
                        ---> Regex
                     ---> Text PostProcessing
                        ---> Spell checks
                        ---> Word combiners
```

<!-- _# AIM : Given a receipt, the aim is to extract the domain specific fields by using deep-learning techinques. -->

# Goal

- Provide a end-to-end solution for information extraction from images.
- Achieve decent accuracy with very few annotated images.

# Development Plan

![](./docs/images/block-diagram.jpg)

The above diagram depicts the various 

- __Dataset__:

  - We use the dataset present on the [website](http://expressexpense.com/view-receipts.php).
    These are the receipts collected from various sources and are of various types.

- __Data Preparation__:

  - Web scrapping: scrapy 
  - We use the python library called scrapy in order to crawl and download all the images
  - we have around ~2k receipt images.
    ![](./docs/images/ReceiptSwiss.jpg)

- __Data Annotation__:

  - We annotate around 125 images using image annotation [tool](https://github.com/frederictost/images_annotation_programme). 
  - The information we are interested from the receipts are 
    - Merchant - name of the store
    - Receipt number - bill id
    - Date-  receipt issue date
    - Line items and their value - items in cart and their value
    - Total - final total value paid
    - Tax - tax calculated on line items
    - Mode of payment - card/cash
      ![](./docs/images/image-anno.png)

- __Object Detection__:

  - We leverage the existing pre-trained models for the detecting the above points of interest by using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

- __Image Segmentation and Cropping__:

  - Following folder structure is adopted to output the images by croping the receipt image for predicted co-ordinates and labels

    ```
    image_name_1\
        label_1.jpg
        label_2.jpg
        label_3.jpg
     ...
    ```

- __OCR__: Tesseract. Converts the cropped images to text

  ```
  image_name_1\
      label_1.txt
      label_2.txt
      label_3.txt
  ...  
  ```

- __Text Postprocessing__: Domain specific 

  ```
  image_name_1.csv
  ...
  ```

# Development Progress

Please check the current progress at [Project Page](https://github.com/Imaginea/vitaFlow/projects/10) and [OCR Project](https://github.com/Imaginea/vitaFlow/projects/8).

## Dataset

- [x] Annotated 100-150 Receipt Images with Tags(merchant, date, line_items, total, tax,..)
- [x] Image Synthesizer for generation of dummy receipt images
- [ ] Annotated 500-600 Receipt Images

## Object Detection Model 

using

- [ ] mask_rcnn_inception_v2_coco_2018_01_28 for detecting receipts(instance localization) with in images.
- [x] faster_rcnn for identifying specific text objects within Receipt Image

## Image Annotation Tool

 A Flask Application for image processing, tagging and predition of receipts images.

- [x] Page - Text Tagging & Generation of Annotated XML
- [x] Page - Cropping & Rotation
- [ ] Page - Binarization 
- [ ] Page - Re-size all image to a standard size(free from prixalation & image loss)(Check https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for)
- [ ] Page - Summary of Annotated Images
- [ ] Page - Training data with models(drop-down list) and automatically annoating data
- [ ] Support for tagging images sources(Amazon bucket or Google Drive)

## Text Extraction

- [x] OCR with Tessaract (with parallel processing feature)
- [ ] OCR with ocropus

# Explorations

**Python Libs:**

- https://pypi.org/project/pytesseract/
- https://gitlab.gnome.org/World/OpenPaperwork/pyocr
- https://github.com/deanmalmgren/textract

**Paper**

- https://hal.archives-ouvertes.fr/hal-01654191/document 
- https://nlp.fi.muni.cz/raslan/2017/paper06-Ha.pdf
- https://arxiv.org/pdf/1708.07403.pdf
- http://cs229.stanford.edu/proj2016/report/LiuWanZhang-UnstructuredDocumentRecognitionOnBusinessInvoice-report.pdf

**Dataset**

- http://www.cs.cmu.edu/~aharley/rvl-cdip/
- https://storage.googleapis.com/openimages/web/index.html
- http://machinelearning.inginf.units.it/data-and-tools/ghega-dataset
- http://expressexpense.com/view-receipts.php

**Git References:**

- [https://github.com/invoice-x/invoice2data](https://github.com/invoice-x/invoice2data)
- [https://github.com/mre/receipt-parser](https://github.com/mre/receipt-parser) Works only for a single format
- [https://github.com/VTurturika/receipt-recognition](https://github.com/VTurturika/receipt-recognition)

**Other**

- https://uu.diva-portal.org/smash/get/diva2:967971/FULLTEXT01.pdf
- https://eng.infoscout.co/receipt_store_detection/
- https://ryubidragonfire.github.io/blogs/2017/06/06/Automating-Receipt-Processing.html

# Acknowledgments

**Annotation Tool**

- https://github.com/frederictost/images_annotation_programme

- # https://github.com/fengyuanchen/cropper

# Round Two

## Introduction

In the previous experiments we found that the results were not up to the mark. So in this run we will try to improve on the accuracy of the system by tweaking individuals blocks of the pipeline.

## Dataset

Consists of 2k scrapped images images. We have the following statistics about the annotated images.

- 200 images tagged for receipt localization.
- 125 receipts tagged for 7 classes mentioned above.
- Script to generate dataset for region detection like merchant+logo, line items, tax. 
  - This scripts needs to be enriched with lot of augmentations so that we can add things like 
    - rotations
    - cropped regions
    - basically use ImgAug for getting a dataset generation as close to real world receipt image, failing to this would cause the trained model to perform badly on real world data.

## Steps Proposed

1. Assume r, a receipt which belongs to a set S.
2. for receipt in S:
   - perform receipt identification/localisation which detects exact position of receipt in the given image.
   - Detect the skewness/rotation. correct the skewness.
   - Detect/perform if perspective transformation is required.
   - Model to detect the various regions **Merchant+logo, Line items, Tax** .
   - For the above detected regions, model to hierarchically detect the items.
     - **Merchant+logo**
       - Merchant - name of the store.
       - Receipt number - bill id.
       - Date-  receipt issue date.
     - **Line items**,
       - Line items and their value - items in cart and their value.
     - **Tax** 
       - Total - final total value paid.
       - Tax - tax calculated on line items.
       - Mode of payment - card/cash.
   - Perform OCR.
   - Combine Output into CSV.
   - END: TO BE DECIDED.

## Things to make note of

- Try reducing the image size (preprocessed images to network like binarized or edge detected images).

- Mix the image from real world with generated by the scripts.

- CNN are better at detecting noise which is gaussian in nature, but will fail for image which are crumpled or having dog-ear.

- For detecting the receipt in a image, we can use the text detectors and form regions.

- Also thinking of using networks trained on tesseract features for extracting text/character from images.

## Receipt Image - Limitations

- Recept size width is atleast 350.
- Expected the images size are having no corner trademarks or Crop Image while annotation
