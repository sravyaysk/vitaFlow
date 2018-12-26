
### [DeWave](https://github.com/chaodengusc/DeWave)

We have considered DeWave as a starting poin to our exploration in Audio and/or Video sepration task.

DeWave is simplest of all which uses Deep Clustering to seperate two mixture of audio signals.

We have provided a [download.sh](download.sh) script that downloads data of size ~20GB and unzips the data to default location.

Please refer below on how to run the experiments.

DeWave repo is sucked into three python classes that resides as [tedlium_dataset.py](tedlium_dataset.py), 
[tedlium_iterator.py](tedlium_iterator.py) and [deep_clustering.py](deep_clustering.py) respectively.


### Experiment

**Train**
```
CUDA_VISIBLE_DEVICES=0 python vitaflow/run/run.py --mode=train -config_python_file=examples/shabda/config.py
```


### Reference: 

https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html
Look to Listen/Cocktail Party:
https://github.com/avivga/audio-visual-speech-enhancement
https://github.com/andrewowens/multisensory
https://github.com/chaodengusc/DeWave
https://github.com/crystal-method/Looking-to-Listen
https://github.com/Kajiyu/LLLNet
https://github.com/vishwajeet97/Cocktail-Party-Problem
https://github.com/marl/audiosetdl 
DAtaset (https://research.google.com/audioset/)



Dataset:
- https://projets-lium.univ-lemans.fr/ted-lium/
