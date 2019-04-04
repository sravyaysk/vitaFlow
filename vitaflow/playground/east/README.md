# EAST: An Efficient and Accurate Scene Text Detector

**Refernce Git** : https://github.com/argman/EAST
**Paper** : https://arxiv.org/abs/1704.03155v2
**Dataset ICDAR 2015** : http://rrc.cvc.uab.es/?ch=4&com=introduction
**Imaginea Folks** :

- Use Google Drive Link: https://drive.google.com/drive/folders/1CmEkiDHWQB-miGQRSU3pK7qoAfoifb-R?usp=sharing
- Goto 2015 folder and download following files to /opt/data/icdar/2015/:
  - ch4_training_images.zip
  - ch4_training_localization_transcription_gt.zip
  - ch4_test_images.zip
  - Challenge4_Test_Task1_GT.zip

## Data Setup

```sh
cd /opt/data/icdar/2015/
unzip ch4_training_images.zip -d train/
unzip ch4_training_localization_transcription_gt.zip -d train/

unzip ch4_test_images.zip -d test_images/
cd test_images
mkdir test
mkdir val
mv `ls  | head -400` ./test/
mv *.jpg ./val/
cd ..

unzip Challenge4_Test_Task1_GT.zip -d test_txts/
cd  test_txts
mkdir test
mkdir val
mv `ls  | head -400` ./test/
mv *.txt ./val/
cd ..

mkdir test
mkdir val
mv ./test_images/test/* ./test/
mv ./test_images/val/* ./val/
mv ./test_txts/test/* ./test/
mv ./test_txts/val/* ./val/
rm -rf test_images
rm -rf test_txts
ls
```

## Requirements

```sh
pip install tensorflow-gpu
pip install gin-config
pip install numpy
pip install opencv-python
pip install shapely
pip install tqdm
pip install matplotlib
pip install scipy
```

## Configuration

Check this [file](config.gin)!

## Running

```sh
#train
python main.py
#predict
python main --predict=true
```

### PS

- As compared to original EAST repo, we have used Tensorflow high level APIs tf.data and tf.Estimators
- This comes in handy when we move to big dataset or if we wanted to experiment with different models/data