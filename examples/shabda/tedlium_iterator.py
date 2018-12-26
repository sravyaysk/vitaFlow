import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import TensorShape, Dimension
from tqdm import tqdm

from examples.shabda.core.feature_types.shabda_wav_pair_feature import ShabdaWavPairFeature
from examples.shabda.tools import boost_yield_samples
from vitaflow.core import HParams, IIteratorBase
from vitaflow.helpers.print_helper import print_info


class TEDLiumIterator(IIteratorBase, ShabdaWavPairFeature):
    def __init__(self, hparams=None, dataset=None):
        IIteratorBase.__init__(self, hparams=hparams)
        ShabdaWavPairFeature.__init__(self)
        self._hparams = HParams(hparams=hparams, default_hparams=self.default_hparams())
        self._dataset = dataset
        print(dataset)

        self.TRAIN_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.VAL_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.TEST_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)

        if self._hparams.reinit_file_pair:
            self.TRAIN_WAV_PAIR = self.generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self.generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self.generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

        else:
            self.TRAIN_WAV_PAIR = self.read_pickle("train_wav_pair.p")
            if self.TRAIN_WAV_PAIR is None:
                self.TRAIN_WAV_PAIR = self.generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self.read_pickle("val_wav_pair.p")
            if self.VAL_WAV_PAIR is None:
                self.VAL_WAV_PAIR = self.generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self.read_pickle("test_wav_pair.p")
            if self.TEST_WAV_PAIR is None:
                self.TEST_WAV_PAIR = self.generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

    # TODO: All property methods to be here after __init__
    # TODO: All underscore methods to be here after __init__ & property methods

    @staticmethod
    def default_hparams():

        params = IIteratorBase.default_hparams()
        params.update({
            "sampling_rate": 16000,
            "frame_size": 256,
            "neff": 129,
            "min_amp": 10000,
            "amp_fac": 10000,
            "threshold": 40,
            # prams for pre-whitening
            "global_mean": 44,
            "global_std": 15.5,
            "frames_per_sample": 100,
            "reinit_file_pair": False,
            "prefetch_size": 32
        }
        )
        return params

    @property
    def num_labels(self):
        return -1

    @property
    def num_train_samples(self):
        count = 0
        for speaker in self.TRAIN_SPEAKER_WAV_FILES_DICT:
            files = self.TRAIN_SPEAKER_WAV_FILES_DICT[speaker]
            count += len(files)
        return count

    @property
    def num_val_samples(self):
        return 128

    @property
    def num_test_samples(self):
        return 96

    def get_speaker_files(self, data_dir):
        """

        :param data_dir: dir containing the training data (root_dir + speaker_dir + wavfiles)
        :returns:  speaker_wav_files (dict) : {speaker : [files]}
        """
        # get dirs for each speaker
        speakers_dirs = [os.path.join(data_dir, speaker) for speaker in os.listdir(data_dir) \
                         if os.path.isdir(os.path.join(data_dir, speaker))]

        speaker_wav_files_dict = {}

        # get the files in each speakers dir
        # TODO: Convert below to dict-comprehension using collections.defaultdict
        for speaker_dir in speakers_dirs:
            speaker = speaker_dir.split("/")[-1]
            wav_files = [os.path.join(speaker_dir, file) for file in os.listdir(speaker_dir) if file.endswith("wav")]
            for wav_file in wav_files:
                if speaker not in speaker_wav_files_dict:
                    speaker_wav_files_dict[speaker] = []
                speaker_wav_files_dict[speaker].append(wav_file)
        return speaker_wav_files_dict

    def get_shape(self, wav_file):
        # sig, _ = librosa.core.load(wav_file, sr=self._hparams.sampling_rate)
        # return sig.shape[0]
        st = os.stat(wav_file)
        return st.st_size

    def generate_match_dict(self, speaker_wav_files_dict):
        wav_file_pair = {}
        # generate match dict
        for speaker_1 in tqdm(speaker_wav_files_dict, desc="pair_dict"):
            for wav_file_1 in tqdm(speaker_wav_files_dict[speaker_1], desc="pair_dict_inner"):
                size_1 = self.get_shape(wav_file_1)
                if size_1 < 1000:
                    continue
                speaker_2 = random.choice(list(speaker_wav_files_dict))

                while (speaker_1 == speaker_2):
                    speaker_2 = random.choice(list(speaker_wav_files_dict))
                wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))

                wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                size_2 = self.get_shape(wav_file_2)

                while size_2 < 1000:
                    wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))
                    wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                    size_2 = self.get_shape(wav_file_2)

                wav_file_pair[wav_file_1] = wav_file_2

        return wav_file_pair

    def _yield_samples(self, speaker_file_match):
        '''Init the training data using the wav files'''
        sampling_rate = self._hparams.sampling_rate
        frame_size = self._hparams.frame_size
        amp_fac = self._hparams.amp_fac
        neff = self._hparams.neff
        min_amp = self._hparams.min_amp
        threshold = self._hparams.threshold
        global_mean = self._hparams.global_mean
        global_std = self._hparams.global_std
        frames_per_sample = self._hparams.frames_per_sample

        for (sample_mix, VAD, Y) in boost_yield_samples(speaker_file_match,
                                                        sampling_rate,
                                                        frame_size,
                                                        amp_fac,
                                                        neff,
                                                        min_amp,
                                                        threshold,
                                                        global_mean,
                                                        global_std,
                                                        frames_per_sample):
            yield sample_mix, VAD, Y

    def _yield_train_samples(self):
        return self._yield_samples(self.TRAIN_WAV_PAIR)

    def _yield_val_samples(self):
        return self._yield_samples(self.VAL_WAV_PAIR)

    def _yield_test_samples(self):
        return self._yield_samples(self.TEST_WAV_PAIR)

    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """

        dataset = tf.data.Dataset.from_generator(self._yield_train_samples,
                                                 (tf.float32, tf.bool, tf.bool),
                                                 output_shapes=(TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape(
                                                                    [Dimension(100), Dimension(129), Dimension(2)])))

        dataset = dataset.map(lambda x, y, z: ({self.FEATURE_1_NAME: x,
                                                self.FEATURE_2_NAME: y}, z))

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """

        dataset = tf.data.Dataset.from_generator(self._yield_val_samples,
                                                 (tf.float32, tf.bool, tf.bool),
                                                 output_shapes=(TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape(
                                                                    [Dimension(100), Dimension(129), Dimension(2)])))

        dataset = dataset.map(lambda x, y, z: ({self.FEATURE_1_NAME: x,
                                                self.FEATURE_2_NAME: y}, z))

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "val_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    def _get_test_input_function(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        dataset = tf.data.Dataset.from_generator(self._yield_test_samples,
                                                 (tf.float32, tf.bool, tf.bool),
                                                 output_shapes=(TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape(
                                                                    [Dimension(100), Dimension(129), Dimension(2)])))
        dataset = dataset.map(lambda x, y, z: ({self.FEATURE_1_NAME: x,
                                                self.FEATURE_2_NAME: y}, z))

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "test_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    # TODO: Remove dead code
    # def load_data(self, data_dir):
    #     '''
    #     Load in the audio file and transform the signal into
    #     the formats required by the model'''
    #     # loading and transformation
    #     speech_mix, _ = librosa.load(data_dir, self._hparams.sampling_rate)
    #     # fix the issue at the begining
    #     speech_mix = np.concatenate((speech_mix, speech_mix, speech_mix), axis=0)
    #     speech_mix_spec0 = stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
    #     speech_mix_spec = np.abs(speech_mix_spec0)
    #     speech_phase = speech_mix_spec0 / speech_mix_spec
    #     speech_mix_spec = np.maximum(
    #         speech_mix_spec, np.max(speech_mix_spec) / self._hparams.min_amp)
    #     speech_mix_spec = 20. * np.log10(speech_mix_spec * self._hparams.amp_fac)
    #     max_mag = np.max(speech_mix_spec)
    #     speech_VAD = (speech_mix_spec > (max_mag - self._hparams.threshold)).astype(int)
    #     speech_mix_spec = (speech_mix_spec - self._hparams.global_mean) / self._hparams.global_std
    #     len_spec = speech_mix_spec.shape[0]
    #     k = 0
    #     self.ind = 0
    #     self.samples = []
    #     # feed the transformed data into a sample list
    #     while(k + self._hparams.frames_per_sample < len_spec):
    #         phase = speech_phase[k: k + self._hparams.frames_per_sample, :]
    #         sample_mix = speech_mix_spec[k:k + self._hparams.frames_per_sample, :]
    #         VAD = speech_VAD[k:k + self._hparams.frames_per_sample, :]
    #         sample_dict = {'Sample': sample_mix,
    #                        'VAD': VAD,
    #                        'Phase': phase}
    #         self.samples.append(sample_dict)
    #         k = k + self._hparams.frames_per_sample
    #     # import ipdb; ipdb.set_trace()
    #     n_left = self._hparams.frames_per_sample - len_spec + k
    #     # store phase for waveform reconstruction
    #     phase = np.concatenate((speech_phase[k:, :], np.zeros([n_left, self._hparams.neff])))
    #     sample_mix = np.concatenate(
    #         (speech_mix_spec[k:, :], np.zeros([n_left, self._hparams.neff])))
    #     VAD = np.concatenate((speech_VAD[k:, :], np.zeros([n_left, self._hparams.neff])))
    #     sample_dict = {'Sample': sample_mix,
    #                    'VAD': VAD,
    #                    'Phase': phase}
    #     self.samples.append(sample_dict)
    #     self.tot_samp = len(self.samples)
