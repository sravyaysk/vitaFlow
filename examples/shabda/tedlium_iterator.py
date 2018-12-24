import numpy as np
import librosa
from numpy.lib import stride_tricks
import os
import numpy as np
import librosa
import pickle
from numpy.lib import stride_tricks
import os
import argparse
import glob

from vitaflow.core import HParams, IIteratorBase
from vitaflow.helpers.print_helper import print_error


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


class TEDLiumIterator(IIteratorBase):
    def __init__(self, hparams=None, dataset=None):
        IIteratorBase.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams=hparams, default_hparams=self.default_hparams())
        self._dataset = dataset
        print(dataset)

        self.TRAIN_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.VAL_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.TEST_SPEAKER_WAV_FILES_DICT = self.get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)


    @staticmethod
    def default_hparams():

        params = IIteratorBase.default_hparams()

        params.update( {
            "sampling_rate" : 16000,
            "frame_size" : 256,
            "neff" : 129,
            "min_amp" : 10000,
            "amp_fac" : 10000,
            "threshold" : 40,
            # prams for pre-whitening
            "global_mean" : 44,
            "global_std" : 15.5,
            "frames_per_sample" : 100}
        )
        return params

    @property
    def num_labels(self):
        return -1

    @property
    def num_train_samples(self):
        return 256

    @property
    def num_val_samples(self):
        return 128

    @property
    def num_test_samples(self):
        return 96

    def get_speaker_files(self, data_dir):
        """

        :param data_dir:
        :returns:  speaker_wav_files (dict) : {speaker : [files]}
        """
        """
        preprocess the training data
        data_dir: dir containing the training data
        format:root_dir + speaker_dir + wavfiles
        """
        # get dirs for each speaker
        print_error(data_dir)
        speakers_dirs = [os.path.join(data_dir, speaker) for speaker in os.listdir(data_dir) \
                             if os.path.isdir(os.path.join(data_dir, speaker))]
        print_error(speakers_dirs)

        speaker_wav_files = {}

        # get the files in each speakers dir
        for speaker_dir in speakers_dirs:
            speaker = speaker_dir.split("/")[-1]
            wav_files = [os.path.join(speaker_dir, file) \
                         for file in os.listdir(speaker_dir) if file[-3:]=="wav"]
            for wav_file in wav_files:
                if speaker not in speaker_wav_files:
                    speaker_wav_files[speaker] = []
                speaker_wav_files[speaker].append(wav_file)
        return speaker_wav_files


    def generate_match_dict(self, speaker): #TODO pickle
        # generate match dict


    def generate_samples(self):
        '''Init the training data using the wav files'''
        self.speaker_file_match = {}
        ## training datasets
        self.samples = []
        ## the begining index of a batch
        self.ind = 0

        # for each file pair, generate their mixture and reference samples
        for i in self.speaker_file_match:
            j = self.speaker_file_match[i]
            speech_1, _ = librosa.core.load(i, sr=self._hparams.sampling_rate)
            # amp factor between -3 dB - 3 dB
            fac = np.random.rand(1)[0] * 6 - 3
            speech_1 = 10. ** (fac / 20) * speech_1
            speech_2, _ = librosa.core.load(j, sr=self._hparams.sampling_rate)
            fac = np.random.rand(1)[0] * 6 - 3
            speech_2 = 10. ** (fac / 20) * speech_2
            # mix
            length = min(len(speech_1), len(speech_2))
            speech_1 = speech_1[:length]
            speech_2 = speech_2[:length]
            speech_mix = speech_1 + speech_2
            # compute log spectrum for 1st speaker
            speech_1_spec = np.abs(stft(speech_1, self._hparams.frame_size)[:, :self._hparams.neff])
            speech_1_spec = np.maximum(
                speech_1_spec, np.max(speech_1_spec) / self._hparams.min_amp)
            speech_1_spec = 20. * np.log10(speech_1_spec * self._hparams.amp_fac)
            # same for the 2nd speaker
            speech_2_spec = np.abs(stft(speech_2, self._hparams.frame_size)[:, :self._hparams.neff])
            speech_2_spec = np.maximum(
                speech_2_spec, np.max(speech_2_spec) / self._hparams.min_amp)
            speech_2_spec = 20. * np.log10(speech_2_spec * self._hparams.amp_fac)
            # same for the mixture
            speech_mix_spec0 = stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
            speech_mix_spec = np.abs(speech_mix_spec0)
            # speech_phase = speech_mix_spec0 / speech_mix_spec
            speech_mix_spec = np.maximum(
                speech_mix_spec, np.max(speech_mix_spec) / self._hparams.min_amp)
            speech_mix_spec = 20. * np.log10(speech_mix_spec * self._hparams.amp_fac)
            max_mag = np.max(speech_mix_spec)
            # if np.isnan(max_mag):
            # import ipdb; ipdb.set_trace()
            speech_VAD = (speech_mix_spec > (max_mag - self._hparams.threshold)).astype(int)
            # print 'mean:' + str(np.mean(speech_mix_spec)) + '\n'
            # print 'std:' + str(np.std(speech_mix_spec)) + '\n'
            speech_mix_spec = (speech_mix_spec - self._hparams.global_mean) / self._hparams.global_std

            len_spec = speech_1_spec.shape[0]
            k = 0
            while(k + self._hparams.frame_per_sample < len_spec):
                sample_1 = speech_1_spec[k:k + self._hparams.frame_per_sample, :]
                sample_2 = speech_2_spec[k:k + self._hparams.frame_per_sample, :]
                # phase = speech_phase[k: k + self._hparams.frame_per_sample, :]
                sample_mix = speech_mix_spec[k:k + self._hparams.frame_per_sample, :] \
                    .astype('float16')
                # Y: indicator of the belongings of the TF bin
                # 1st speaker or second speaker
                Y = np.array(
                    [sample_1 > sample_2, sample_1 < sample_2]).astype('bool')
                Y = np.transpose(Y, [1, 2, 0])
                VAD = speech_VAD[k:k + self._hparams.frame_per_sample, :].astype('bool')
                sample_dict = {'Sample': sample_mix,
                               'VAD': VAD,
                               'Target': Y}
                self.samples.append(sample_dict)
                k = k + self._hparams.frame_per_sample
        # dump the generated sample list
        pickle.dump(self.samples, open(self.output, 'wb'))
        self.tot_samp = len(self.samples)
        np.random.shuffle(self.samples)


    def load_data(self, data_dir):
        '''
        Load in the audio file and transform the signal into
        the formats required by the model'''
        # loading and transformation
        speech_mix, _ = librosa.load(data_dir, self._hparams.sampling_rate)
        # fix the issue at the begining
        speech_mix = np.concatenate((speech_mix, speech_mix, speech_mix), axis=0)
        speech_mix_spec0 = stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
        speech_mix_spec = np.abs(speech_mix_spec0)
        speech_phase = speech_mix_spec0 / speech_mix_spec
        speech_mix_spec = np.maximum(
            speech_mix_spec, np.max(speech_mix_spec) / self._hparams.min_amp)
        speech_mix_spec = 20. * np.log10(speech_mix_spec * self._hparams.amp_fac)
        max_mag = np.max(speech_mix_spec)
        speech_VAD = (speech_mix_spec > (max_mag - self._hparams.threshold)).astype(int)
        speech_mix_spec = (speech_mix_spec - self._hparams.global_mean) / self._hparams.global_std
        len_spec = speech_mix_spec.shape[0]
        k = 0
        self.ind = 0
        self.samples = []
        # feed the transformed data into a sample list
        while(k + self._hparams.frame_per_sample < len_spec):
            phase = speech_phase[k: k + self._hparams.frame_per_sample, :]
            sample_mix = speech_mix_spec[k:k + self._hparams.frame_per_sample, :]
            VAD = speech_VAD[k:k + self._hparams.frame_per_sample, :]
            sample_dict = {'Sample': sample_mix,
                           'VAD': VAD,
                           'Phase': phase}
            self.samples.append(sample_dict)
            k = k + self._hparams.frame_per_sample
        # import ipdb; ipdb.set_trace()
        n_left = self._hparams.frame_per_sample - len_spec + k
        # store phase for waveform reconstruction
        phase = np.concatenate((speech_phase[k:, :], np.zeros([n_left, self._hparams.neff])))
        sample_mix = np.concatenate(
            (speech_mix_spec[k:, :], np.zeros([n_left, self._hparams.neff])))
        VAD = np.concatenate((speech_VAD[k:, :], np.zeros([n_left, self._hparams.neff])))
        sample_dict = {'Sample': sample_mix,
                       'VAD': VAD,
                       'Phase': phase}
        self.samples.append(sample_dict)
        self.tot_samp = len(self.samples)