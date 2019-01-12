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


## Paper
- https://arxiv.org/pdf/1708.07403.pdf
- http://cs229.stanford.edu/proj2016/report/LiuWanZhang-UnstructuredDocumentRecognitionOnBusinessInvoice-report.pdf


## Git References:
- [https://github.com/invoice-x/invoice2data](https://github.com/invoice-x/invoice2data)
- [https://github.com/mre/receipt-parser](https://github.com/mre/receipt-parser) Works only for a single format
- [https://github.com/VTurturika/receipt-recognition](https://github.com/VTurturika/receipt-recognition)


## Other
- https://uu.diva-portal.org/smash/get/diva2:967971/FULLTEXT01.pdf
- https://eng.infoscout.co/receipt_store_detection/
- https://ryubidragonfire.github.io/blogs/2017/06/06/Automating-Receipt-Processing.html