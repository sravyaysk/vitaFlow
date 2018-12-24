import glob
import librosa
import os
import numpy as np
from tqdm import tqdm

from vitaflow.core import IPreprocessor, HParams
from vitaflow.helpers.print_helper import print_error, print_warn, print_info


class TEDLiumDataset(IPreprocessor):
    """

    .. code-block:: text

        ~/vitaflow/
            TEDLiumDataset/
                raw_data/
                    train/
                        sph/
                    dev/
                        sph/
                    val/
                        sph/
                    processed_data/
                        train/
                        val/

    """
    def __init__(self, hparams=None):
        IPreprocessor.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams, self.default_hparams())

        self.extract_speech_clips()

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
                "experiment_name" : "experiment_name",
                "preprocessed_data_path" : "preprocessed_data",
                "train_data_path" : "train",
                "validation_data_path" : "val",
                "test_data_path" : "test"
            }

        Here:

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "experiment_name" : str
            Name for the current experiment

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored


        :return: A dictionary of hyperparameters with default values
        """
        hparams = IPreprocessor.default_hparams()

        hparams.update({
            "experiment_name": "TEDLiumDataset",
            "num_clips" : 128,
            "start_time" : 0,
            "end_time" : 600,
            "sampling_rate" : 16000,
            "duration": 5
        })

        return hparams


    def audio_clip(self,
                   data_dir,
                   num_clips,
                   start_time,
                   end_time,
                   duration,
                   output_dir,
                   sampling_rate):
        """

        :param data_dir:input data folder
        :param num_clips: The number of clips for each speaker
        :param start_time: starting time of the audio from which the clip is sampled
        :param end_time: ending time of the audio from which the clip is sampled
        :param duration: The duration of each clip
        :param output_dir: the output directory
        :param sampling_rate: Sampling rate
        :return:
        """
        speakers = glob.glob(os.path.join(data_dir, "*.sph"))
        speakers.extend(glob.glob(os.path.join(data_dir, "*.wav")))

        for i in tqdm(range(len(speakers)), desc="speakers"):
            file_path = speakers[i]
            speaker = file_path.split("/")[-1].split(".")[0] #each speaker can have multiple talks
            speaker_dir = os.path.join(output_dir, speaker)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)

                y, _ = librosa.load(file_path, sr=sampling_rate)
                for j in range(num_clips):
                    wav_file = os.path.join(speaker_dir, str(j)) + ".wav"
                    k = int(np.random.randint(start_time, end_time, size=1))
                    librosa.output.write_wav(wav_file,
                                             y[k*sampling_rate : (k+duration)*sampling_rate],
                                             sampling_rate)
            else:
                print_info("While processing {}".format(file_path))
                print_warn("Found existing data @ {}".format(speaker_dir))
                print_info("Skipping!!!")


    def extract_speech_clips(self):
        self.audio_clip(data_dir=os.path.join(self.TRAIN_IN_PATH, "sph"),
                        num_clips=self._hparams.num_clips,
                        start_time=self._hparams.start_time,
                        end_time=self._hparams.end_time,
                        duration=self._hparams.duration,
                        output_dir=self.TRAIN_OUT_PATH,
                        sampling_rate=self._hparams.sampling_rate)

        self.audio_clip(data_dir=os.path.join(self.VAL_IN_PATH, "sph"),
                        num_clips=self._hparams.num_clips,
                        start_time=self._hparams.start_time,
                        end_time=self._hparams.end_time,
                        duration=self._hparams.duration,
                        output_dir=self.VAL_OUT_PATH,
                        sampling_rate=self._hparams.sampling_rate)

        self.audio_clip(data_dir=os.path.join(self.TEST_IN_PATH, "sph"),
                        num_clips=self._hparams.num_clips,
                        start_time=self._hparams.start_time,
                        end_time=self._hparams.end_time,
                        duration=self._hparams.duration,
                        output_dir=self.TEST_OUT_PATH,
                        sampling_rate=self._hparams.sampling_rate)





