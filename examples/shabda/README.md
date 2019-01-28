
### [DeWave](https://github.com/chaodengusc/DeWave)

We have considered DeWave as a starting point to our exploration in Audio and/or Video sepration task.

DeWave is simplest of all which uses Deep Clustering to seperate two mixture of audio signals.

We have provided a [download.sh](download.sh) script that downloads data of size ~20GB and unzips the data to default location.

Please refer below on how to run the experiments.

DeWave repo is sucked into three python classes that resides as [tedlium_dataset.py](tedlium_dataset.py), 
[tedlium_iterator.py](tedlium_iterator_basic.py) and [deep_clustering.py](deep_clustering.py) respectively.


### Experiment

```
 find . -name "*.pyc" -exec rm -f {} \;
```
**Cache the Preprocessed Data Manually**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/run_experiments.py --mode=run_iterator -config_python_file=examples/shabda/config.py
```
**Train**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/run_experiments.py --mode=train -config_python_file=examples/shabda/config.py
```
**Predict on single file**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/bin/run_experiments.py \
--mode=predict_instance \
--test_file_path=mix.wav \
-config_python_file=examples/shabda/config.py
```


### Reference: 

* https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html
* https://github.com/zhr1201/Multi-channel-speech-extraction-using-DNN/blob/master/Manuscript-InterNoise2017-ZhouHaoran_0525.pdf

Look to Listen/Cocktail Party:
* https://github.com/chaodengusc/DeWave
* https://github.com/TotallyFine/deep-clustering
* https://github.com/avivga/audio-visual-speech-enhancement
* https://github.com/andrewowens/multisensory
* https://github.com/crystal-method/Looking-to-Listen
* https://github.com/Kajiyu/LLLNet
* https://github.com/vishwajeet97/Cocktail-Party-Problem
* https://github.com/marl/audiosetdl 
* Dataset (https://research.google.com/audioset/)



### Dataset:
* https://projets-lium.univ-lemans.fr/ted-lium/
* [TPS](http://www-mmsp.ece.mcgill.ca/Documents/Data/)

### Audio Basics
- [Decibel](https://www.rapidtables.com/electric/decibel.html)
- http://www.cs.princeton.edu/~fiebrink/314/2009/week12/FFT_handout.pdf

### Maths Refresher:

**Imaginary Numbers**
    - https://www.youtube.com/watch?v=T647CGsuOVU 
    - [imaginary_numbers_are_real_rev2_for_screen.pdf](https://static1.squarespace.com/static/54b90461e4b0ad6fb5e05581/t/5a6e7bd341920260ccd693cf/1517190204747/imaginary_numbers_are_real_rev2_for_screen.pdf)
    - https://www.electronics-tutorials.ws/accircuits/complex-numbers.html
    - Polar form vs Rectangular form
**Euler's Formula**
    - https://www.youtube.com/watch?v=m2MIpDrF7Es&t=23s
    - https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/
**FFT**
    - https://www.youtube.com/watch?v=spUNpyF58BY
    - https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/

