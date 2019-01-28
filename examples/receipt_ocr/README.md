# Receipt OCR

```
Scanned PDFs/Images  ---> Image Preprocessing 
                        ---> Resizing
                        ---> Deskewing
                        ---> Rotation
                        ---> Sharpening 
                        ---> Background noise removal
                        ---> Binarization 
                        ---> Stiching etc.,
                     ---> OCR (Tesseract/DL Models)
                     ---> Doc/Text Classifiers
                        ---> Identifying the receipts etc.,
                     ---> Annotators (Optional)
                     ---> ML/DL Model (Optional)
                     ---> Data Insights 
                        ---> Format templates
                        ---> Regex
                     ---> Text PostProcessing
                        ---> Spell checks
                        ---> Word combiners
``` 

## receipt OCR

- Dataset
    - http://expressexpense.com/view-receipts.php 
- Data Preparation
    - Web scrabbing : scrapy
- Data Annotation
    - https://github.com/frederictost/images_annotation_programme
- Object Detection
    - https://github.com/tensorflow/models/tree/master/research/object_detection
- Image Segmentation and Cropping
- OCR : Tesseract
- Text Postprocessing

## Explorations

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