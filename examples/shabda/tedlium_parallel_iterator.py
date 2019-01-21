import os
import random
import itertools

import numpy as np
import tensorflow as tf
from overrides import overrides
from tensorflow import TensorShape, Dimension
from tqdm import tqdm
import librosa
from numpy.lib import stride_tricks
from multiprocessing import Pool

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from examples.shabda.core.feature_types.shabda_wav_pair_feature import ShabdaWavPairFeature
# from examples.shabda.tools import _yield_samples_multicore
from vitaflow.core import HParams, IIteratorBase
from vitaflow.helpers.print_helper import print_info
from vitaflow.run import Executor
from vitaflow.helpers.print_helper import print_error


#=================================================================================================================

enable_multiprocessing = True
enable_file_pickling = False


def _get_speech_data(wav_file, sampling_rate):
    """

    :param wav_file:
    :param sampling_rate:
    :return:
    """
    speech, _ = librosa.core.load(wav_file, sr=sampling_rate)
    # amp factor between -3 dB - 3 dB
    fac = np.random.rand(1)[0] * 6 - 3
    speech = 10. ** (fac / 20) * speech
    return speech

def _get_log_spectrum_features(speech, frame_size, neff, amp_fac, min_amp):
    """

    :param speech:
    :param frame_size:
    :param neff:
    :param amp_fac:
    :param min_amp:
    :return:
    """
    speech_features = np.abs(_stft(speech, frame_size)[:, :neff])
    speech_features = np.maximum(speech_features, np.max(speech_features) / min_amp)
    speech_features = 20. * np.log10(speech_features * amp_fac)
    return speech_features

def _stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """
    short time fourier transform of audio signal
    :param sig:
    :param frameSize:
    :param overlapFac:
    :param window:
    :return:
    """
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


def _get_speech_samples(speech_1_features,
                        speech_2_features,
                        frames_per_sample,
                        len_spec,
                        speech_mix_features,
                        speech_VAD):
    """
    Processes one audio pair
    :param speech_1_features: Spectral audio features
    :param speech_2_features: Spectral audio features
    :param frames_per_sample:
    :param len_spec: Total number of frames (to say?) in the given signal
    :param speech_mix_features:
    :param speech_VAD:
    :return: Returns an array of shape ((self._hparams.frames_per_sample, self._hparams.neff),(self._hparams.frames_per_sample, self._hparams.neff),(self._hparams.frames_per_sample, self._hparams.neff, 2))
    """
    #

    current_frame = 0
    bag = []

    while current_frame + frames_per_sample < len_spec:
        #simple array slicing on the first dimension (time/freq axis)
        sample_1 = speech_1_features[current_frame:current_frame + frames_per_sample, :]
        sample_2 = speech_2_features[current_frame:current_frame + frames_per_sample, :]
        sample_mix = speech_mix_features[current_frame:current_frame + frames_per_sample, :].astype('float32')
        VAD = speech_VAD[current_frame:current_frame + frames_per_sample, :].astype('bool')
        # phase = speech_phase[k: k + frames_per_sample, :]

        # Y: indicator of the belongings of the TF bin
        # 1st speaker or second speaker
        Y = np.array([sample_1 > sample_2,
                      sample_1 < sample_2]
                     ).astype('bool')
        # Y now will have 2-axis one for each signal comparision with other
        # returned Y will be [2, x, y], hence the transpose [x, y, 2]
        Y = np.transpose(Y, [1, 2, 0]).astype('bool')
        bag.append((sample_mix, VAD, Y))
        current_frame = current_frame + frames_per_sample
    return bag


def _get_speech_features(args):
    """

    :param args:
    :return:
    """
    (wav_file_1, wav_file_2, sampling_rate, frame_size, neff, amp_fac, min_amp,
     threshold, global_mean, global_std, frames_per_sample) = args

    try:
        # TODO: experimental though - is multi-processing required here? to reduce IO
        speech_1 = _get_speech_data(wav_file_1, sampling_rate)
        speech_2 = _get_speech_data(wav_file_2, sampling_rate)

        # print(wav_file_1, wav_file_2)

        # find the minimum length of two speeches
        length = min(len(speech_1), len(speech_2))
        #trim both the speeches to the minimum length
        speech_1 = speech_1[:length]
        speech_2 = speech_2[:length]
        # mix the signals
        speech_mix = speech_1 + speech_2

        #get the spectral features
        speech_1_features = _get_log_spectrum_features(speech_1, frame_size, neff, amp_fac, min_amp)
        speech_2_features = _get_log_spectrum_features(speech_2, frame_size, neff, amp_fac, min_amp)

        # find short time fourier transform for the mixture and trim it to given NEFF
        speech_mix_spec0 = _stft(speech_mix, frame_size)[:, :neff]
        speech_mix_features = np.abs(speech_mix_spec0)

        # speech_phase = speech_mix_spec0 / speech_mix_spec
        # find the maximum amplitude between speech mix features and minimum amplititude w.r.t to its max value
        speech_mix_features = np.maximum(speech_mix_features, np.max(speech_mix_features) / min_amp)
        # https://www.cds.caltech.edu/~murray/wiki/Why_is_dB_defined_as_20log_10%3F
        # http://www.sengpielaudio.com/calculator-FactorRatioLevelDecibel.htm
        speech_mix_features = 20. * np.log10(speech_mix_features * amp_fac)

        max_mag = np.max(speech_mix_features)
        # apply threshold to the feature signal
        speech_VAD = (speech_mix_features > (max_mag - threshold)).astype(int)
        #normalize the signal values with given global mean and std
        speech_mix_features = (speech_mix_features - global_mean) / global_std

        len_spec = speech_1_features.shape[0]

        new_data = _get_speech_samples(speech_1_features,
                                       speech_2_features,
                                       frames_per_sample,
                                       len_spec,
                                       speech_mix_features,
                                       speech_VAD)
    except Exception as err:
        # t = '-------------------------\n'
        # print_error(err)
        # print('{} Failed to Run: _get_speech_features {} {} \n {}'.format(t, wav_file_1, wav_file_2, t))
        # print('---' * 5)
        new_data = []

    return new_data


def _yield_samples_multicore(speaker_file_match,
                             sampling_rate,
                             frame_size,
                             amp_fac,
                             neff,
                             min_amp,
                             threshold,
                             global_mean,
                             global_std,
                             frames_per_sample,
                             num_threads,
                             tqdm_desc):
    """

    :param speaker_file_match: Pair of audio files thats needs to be mixed
    :param sampling_rate:
    :param frame_size:
    :param self._hparams.amp_fac:
    :param neff:
    :param min_amp:
    :param threshold:
    :param global_mean:
    :param global_std:
    :param frames_per_sample:
    :param num_threads:
    :param tqdm_desc:
    :return:
    """

    # for each file pair, generate their mixture and reference samples

    p = Pool(num_threads)

    input_params = [(wav_file_1, wav_file_2, sampling_rate, frame_size, neff, amp_fac, min_amp,
                     threshold, global_mean, global_std, frames_per_sample) for (wav_file_1, wav_file_2) in
                    speaker_file_match.items()]

    with tqdm(total=len(speaker_file_match.items())) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(_get_speech_features, input_params)), desc=tqdm_desc):
            pbar.update()
            for data in res:
                yield data


#=================================================================================================================

class TEDLiumIterator(IIteratorBase, ShabdaWavPairFeature):
    def __init__(self, hparams=None, dataset=None):
        """
        Uses the TEDLium preprocessed (speaker wise folders) data and generates the mixture signals
        of two speakers
        :param hparams:
        :param dataset:
        """
        IIteratorBase.__init__(self, hparams=hparams)
        ShabdaWavPairFeature.__init__(self)
        self._hparams = HParams(hparams=hparams, default_hparams=self.default_hparams())
        self._dataset = dataset
        print(dataset)

        # get dict of {speaker : [files], ...}
        self.TRAIN_SPEAKER_WAV_FILES_DICT = self._get_speaker_files(data_dir=dataset.TRAIN_OUT_PATH)
        self.VAL_SPEAKER_WAV_FILES_DICT = self._get_speaker_files(data_dir=dataset.VAL_OUT_PATH)
        self.TEST_SPEAKER_WAV_FILES_DICT = self._get_speaker_files(data_dir=dataset.TEST_OUT_PATH)

        if self._hparams.reinit_file_pair:
            # Generate speaker pair
            self.TRAIN_WAV_PAIR = self._generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self._generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self._generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
            self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

        else:
            self.TRAIN_WAV_PAIR = self.read_pickle("train_wav_pair.p")
            if self.TRAIN_WAV_PAIR is None:
                self.TRAIN_WAV_PAIR = self._generate_match_dict(self.TRAIN_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TRAIN_WAV_PAIR, "train_wav_pair.p")

            self.VAL_WAV_PAIR = self.read_pickle("val_wav_pair.p")
            if self.VAL_WAV_PAIR is None:
                self.VAL_WAV_PAIR = self._generate_match_dict(self.VAL_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.VAL_WAV_PAIR, "val_wav_pair.p")

            self.TEST_WAV_PAIR = self.read_pickle("test_wav_pair.p")
            if self.TEST_WAV_PAIR is None:
                self.TEST_WAV_PAIR = self._generate_match_dict(self.TEST_SPEAKER_WAV_FILES_DICT)
                self.store_as_pickle(self.TEST_WAV_PAIR, "test_wav_pair.p")

    @property
    def num_labels(self):
        """
        No labels associated with this data iterator, hence None
        :return: None
        """
        return None

    @property
    def num_train_samples(self):
        return len(self.TRAIN_WAV_PAIR)

    @property
    def num_val_samples(self):
        return len(self.VAL_WAV_PAIR)

    @property
    def num_test_samples(self):
        return len(self.TEST_WAV_PAIR)

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
            "prefetch_size": 32,
            "num_threads" : 8
        }
        )
        return params

    def _get_speaker_files(self, data_dir):
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

    def _get_size(self, wav_file):
        """
        Finds the size of the given file
        :param wav_file: Wav file
        :return: int Size in bytes
        """
        st = os.stat(wav_file)
        return st.st_size

    def _generate_match_dict(self, speaker_wav_files_dict):
        """
        Generates pair of files from given dict of speakers and their speeches
        :param speaker_wav_files_dict: {speaker : [files]}
        :return: dict of files to files
        """
        wav_file_pair = {}

        # generate match dict
        for speaker_1 in tqdm(speaker_wav_files_dict, desc="pair_dict"):
            for wav_file_1 in speaker_wav_files_dict[speaker_1]:
                size_1 = self._get_size(wav_file_1)
                if size_1 < 1000: #ignore files < 1KB
                    continue
                speaker_2 = random.choice(list(speaker_wav_files_dict))

                while (speaker_1 == speaker_2): #make sure we are not using the same speakers for mixture
                    speaker_2 = random.choice(list(speaker_wav_files_dict))

                wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))
                wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                size_2 = self._get_size(wav_file_2)

                while size_2 < 1000: #ignore files < 1KB
                    wav_file_2_pos = np.random.randint(len(speaker_wav_files_dict[speaker_2]))
                    wav_file_2 = speaker_wav_files_dict[speaker_2][wav_file_2_pos]
                    size_2 = self._get_size(wav_file_2)

                wav_file_pair[wav_file_1] = wav_file_2

        return wav_file_pair


    def _yield_samples(self, speaker_file_match, tqdm_desc):
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

        for (sample_mix, VAD, Y) in _yield_samples_multicore(speaker_file_match,
                                                             sampling_rate,
                                                             frame_size,
                                                             amp_fac,
                                                             neff,
                                                             min_amp,
                                                             threshold,
                                                             global_mean,
                                                             global_std,
                                                             frames_per_sample,
                                                             num_threads=self._hparams.num_threads,
                                                             tqdm_desc=tqdm_desc):
            # print_info((sample_mix, VAD, Y))
            yield sample_mix, VAD, Y

    # To make easy integration with TF Data generator APIs, following methods are added
    def _yield_train_samples(self):
        return self._yield_samples(self.TRAIN_WAV_PAIR, tqdm_desc="Train: ")

    def _yield_val_samples(self):
        return self._yield_samples(self.VAL_WAV_PAIR, tqdm_desc="Val: ")

    def _yield_test_samples(self):
        return self._yield_samples(self.TEST_WAV_PAIR, tqdm_desc="Test: ")

    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: dataset
        """

        # TF dataset APIs
        dataset = tf.data.Dataset.from_generator(self._yield_train_samples,
                                                 (tf.float32, tf.bool, tf.bool),
                                                 output_shapes=(TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape(
                                                                    [Dimension(100), Dimension(129), Dimension(2)])))
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(lambda x, y, z: ({self.FEATURE_1_NAME: x,
                                                self.FEATURE_2_NAME: y}, z))

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        # TF dataset APIs
        dataset = tf.data.Dataset.from_generator(self._yield_val_samples,
                                                 (tf.float32, tf.bool, tf.bool),
                                                 output_shapes=(TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape([Dimension(100), Dimension(129)]),
                                                                TensorShape(
                                                                    [Dimension(100), Dimension(129), Dimension(2)])))
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(lambda x, y, z: ({self.FEATURE_1_NAME: x,
                                                self.FEATURE_2_NAME: y}, z))

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "val_data_cache")) TODO: cache file grows in size
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
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(lambda x, y, z: ({self.FEATURE_1_NAME: x,
                                                self.FEATURE_2_NAME: y}, z))

        dataset = dataset.batch(batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "test_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset


    def _get_predict_samples(self, file_path):
        """

        :param file_path:
        :return:
        """
        speech_mix, _ = librosa.load(file_path, self._hparams.sampling_rate)
        # replicate the speech signal thrice,1: original audio,  2:speaker one, 3: speaker two
        speech_mix = np.concatenate((speech_mix, speech_mix, speech_mix), axis=0)

        # feature enginerring like before
        speech_mix_spec0 = _stft(speech_mix, self._hparams.frame_size)[:, :self._hparams.neff]
        speech_mix_spec = np.abs(speech_mix_spec0)
        speech_phase = speech_mix_spec0 / speech_mix_spec
        speech_mix_spec = np.maximum(speech_mix_spec, np.max(speech_mix_spec) / self._hparams.min_amp)
        speech_mix_spec = 20. * np.log10(speech_mix_spec * self._hparams.amp_fac)
        max_mag = np.max(speech_mix_spec)
        speech_VAD = (speech_mix_spec > (max_mag - self._hparams.threshold)).astype(int)
        speech_mix_spec = (speech_mix_spec - self._hparams.global_mean) / self._hparams.global_std
        len_spec = speech_mix_spec.shape[0] # number of frames

        current_frame = 0
        self.ind = 0
        self.freq_features = []
        self.VAD_features = []
        self.phase_features = []

        # feed the transformed data into a sample list
        while (current_frame + self._hparams.frames_per_sample < len_spec):
            phase = speech_phase[current_frame: current_frame + self._hparams.frames_per_sample, :]
            sample_mix = speech_mix_spec[current_frame:current_frame + self._hparams.frames_per_sample, :]
            VAD = speech_VAD[current_frame:current_frame + self._hparams.frames_per_sample, :]
            self.freq_features.append(sample_mix.astype('float32'))
            self.VAD_features.append(VAD.astype('bool'))
            self.phase_features.append(phase)

            current_frame = current_frame + self._hparams.frames_per_sample
            # import ipdb; ipdb.set_trace()

        # Corner case for the frames that was left
        remaining_frames = self._hparams.frames_per_sample - len_spec + current_frame
        print(remaining_frames)

        # append zeros for missing frames from last position
        # store phase for waveform reconstruction
        phase = np.concatenate((speech_phase[current_frame:, :], np.zeros([remaining_frames, self._hparams.neff])))
        sample_mix = np.concatenate((speech_mix_spec[current_frame:, :], np.zeros([remaining_frames, self._hparams.neff])))
        VAD = np.concatenate((speech_VAD[current_frame:, :], np.zeros([remaining_frames, self._hparams.neff])))

        self.freq_features.append(sample_mix.astype('float32'))
        self.VAD_features.append(VAD.astype('bool'))
        self.phase_features.append(phase)
        return self.freq_features,  self.VAD_features, self.phase_features

    def predict_on_instance(self, executor, file_path):
        """

        :param executor:
        :param test_file_path:
        :return:
        """

        estimator = executor.estimator

        in_data_features, VAD_data_features, phase_features = self._get_predict_samples(file_path=file_path)

        in_data_features = np.asarray(in_data_features)
        VAD_data_features = np.asarray(VAD_data_features)

        N_frames = in_data_features.shape[0]
        hop_size = self._hparams.frame_size // 4

        # oracle flag to decide if a frame need to be seperated
        sep_flag = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 10000
        # oracle permutation to concatenate the chuncks of output frames
        oracal_p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 10000

        out_audio1 = np.zeros([(N_frames * self._hparams.frames_per_sample - 1) *
                               hop_size + self._hparams.frame_size])
        out_audio2 = np.zeros([(N_frames *self._hparams.frames_per_sample - 1) *
                               hop_size + self._hparams.frame_size])
        mix = np.zeros([(N_frames *self._hparams.frames_per_sample - 1) *
                        hop_size + self._hparams.frame_size])

        def get_dataset():
            dataset = tf.data.Dataset.from_tensor_slices((
                {self.FEATURE_1_NAME: in_data_features,
                 self.FEATURE_2_NAME: VAD_data_features},
                np.ones_like(in_data_features)
            ))
            dataset = dataset.batch(batch_size=1)
            print_info(dataset.output_shapes)
            return dataset

        predict_fn = estimator.predict(input_fn=lambda: get_dataset())

        print_info("Shape of in data: {}".format(in_data_features.shape))
        print_info("Number of frames for given file: {}".format(N_frames))

        embeddings = []
        i = 0
        for predicted_value in predict_fn:
            # print("i = {}".format(i))
            """
            TODO:
            strange behaviour!
            For each sample the embeddings is of shape [batch_size * frames_per_sample, NEFF, embd_dim].
            For prediction batch size is made 1.
            Hence the embeddings colapse to [frames_per_sample, NEFF, embd_dim]
            1 sample predictions will have `frames_per_sample` outputs
            Eg: If input audio file has 75 frames, the prediction will have [7500, NEFF, embd_dim]
            """
            embeddings.append(predicted_value)
            i += 1

        print_info("Number of embeddings predicted for given file: {}".format(len(embeddings)))
        print_error(np.asarray(embeddings).shape)

        N_assign = 0
        step = 0

        for frame_i in tqdm(range(N_frames)):

            # expand the dimesion to be inline with TF batch size
            in_data_np = np.expand_dims(in_data_features[frame_i], axis=0)
            in_phase_np = np.expand_dims(phase_features[frame_i], axis=0)
            VAD_data_np = np.expand_dims(VAD_data_features[frame_i], axis=0)
            embedding_np = np.asarray(embeddings[frame_i: frame_i+self._hparams.frames_per_sample])

            # ----------------------------------------------

            embedding_ac = []
            for i, j in itertools.product(range(self._hparams.frames_per_sample), range(self._hparams.neff)):
                if VAD_data_np[0, i, j] == 1:
                    embedding_ac.append(embedding_np[i, j, :])

            if(sep_flag[step] == 1):
                # if the frame need to be seperated
                # cluster the embeddings
                # import ipdb; ipdb.set_trace()
                if embedding_ac == []:
                    break
                kmean = KMeans(n_clusters=2, random_state=0).fit(embedding_ac)
            else:
                # if the frame don't need to be seperated
                # don't split the embeddings
                kmean = KMeans(n_clusters=1, random_state=0).fit(embedding_ac)

            # ----------------------------------------------

            mask = np.zeros([self._hparams.frames_per_sample, self._hparams.neff, 2])
            ind = 0

            if N_assign == 0:
                # if their is no existing speaker in previous frame
                center = kmean.cluster_centers_
                N_assign = center.shape[0]
            elif N_assign == 1:
                # if their is one speaker in previous frame
                center_new = kmean.cluster_centers_
                # assign the embedding for a speaker to the speaker with the
                # closest centroid in previous frames
                if center_new.shape[0] == 1:
                    # update and smooth the centroid for 1 speaker
                    center = 0.7 * center + 0.3 * center_new
                else:
                    # update and smooth the centroid for 2 speakers
                    N_assign = 2
                    # compute their relative affinity
                    cor = np.matmul(center_new, np.transpose(center))
                    # ipdb.set_trace()
                    if(cor[1] > cor[0]):
                        # rearrange their sequence if not consistant with
                        # previous frames
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    center = kmean.cluster_centers_
            else:
                # two speakers have appeared
                center_new = kmean.cluster_centers_
                cor = np.matmul(center_new[0, :], np.transpose(center))
                # rearrange their sequence if not consistant with previous
                # frames
                if(cor[1] > cor[0]):
                    if(sep_flag[step] == 1):
                        kmean.cluster_centers_ = np.array(
                            [kmean.cluster_centers_[1],
                             kmean.cluster_centers_[0]])
                        kmean.labels_ = (kmean.labels_ == 0).astype('int')
                    else:
                        kmean.labels_ = (kmean.labels_ == 1).astype('int')
                # need permutation of their order(Oracle)
                if(oracal_p[step]):
                    kmean.cluster_centers_ = np.array(
                        [kmean.cluster_centers_[1],
                         kmean.cluster_centers_[0]])
                    kmean.labels_ = (kmean.labels_ == 0).astype('int')
                else:
                    kmean.labels_ = (~kmean.labels_).astype('int')
                center = center * 0.7 + 0.3 * kmean.cluster_centers_

            # ----------------------------------------------

            # transform the clustering result and VAD info. into masks
            for i in range(self._hparams.frames_per_sample):
                for j in range(self._hparams.neff):
                    if VAD_data_np[0, i, j] == 1:
                        mask[i, j, kmean.labels_[ind]] = 1
                        ind += 1

            for i in range(self._hparams.frames_per_sample):
                # apply the mask and reconstruct the waveform
                tot_ind = step * self._hparams.frames_per_sample + i
                # ipdb.set_trace()
                # amp = (in_data_np[0, i, :] *
                #        data_batch[0]['Std']) + data_batch[0]['Mean']
                amp = in_data_np[0, i, :] * self._hparams.global_std + self._hparams.global_mean
                out_data1 = (mask[i, :, 0] * amp * VAD_data_np[0, i, :])
                out_data2 = (mask[i, :, 1] * amp * VAD_data_np[0, i, :])
                out_mix = amp
                out_data1_l = 10 ** (out_data1 / 20) / self._hparams.amp_fac
                out_data2_l = 10 ** (out_data2 / 20) / self._hparams.amp_fac
                out_mix_l = 10 ** (out_mix / 20) / self._hparams.amp_fac

                out_stft1 = out_data1_l * in_phase_np[0, i, :]
                out_stft2 = out_data2_l * in_phase_np[0, i, :]
                out_stft_mix = out_mix_l * in_phase_np[0, i, :]

                con_data1 = out_stft1[-2:0:-1].conjugate()
                con_data2 = out_stft2[-2:0:-1].conjugate()
                con_mix = out_stft_mix[-2:0:-1].conjugate()

                out1 = np.concatenate((out_stft1, con_data1))
                out2 = np.concatenate((out_stft2, con_data2))
                out_mix = np.concatenate((out_stft_mix, con_mix))
                frame_out1 = np.fft.ifft(out1).astype(np.float64)
                frame_out2 = np.fft.ifft(out2).astype(np.float64)
                frame_mix = np.fft.ifft(out_mix).astype(np.float64)

                start = tot_ind * hop_size
                out_audio1[start:(start + len(frame_out1))] += frame_out1 * 0.5016
                out_audio2[start:(start + len(frame_out2))] += frame_out2 * 0.5016
                mix[start:(start + len(frame_mix))] += frame_mix * 0.5016

            step += 1

        ## the audio has been padded 3 times in AudioReader
        ## restore the original audio
        len1 = len(out_audio1) // 3
        len2 = len(out_audio2) // 3
        source1 = out_audio1[len1:2*len1]
        source2 = out_audio2[len2:2*len2]

        print_info("Writing file {}".format(os.path.splitext(file_path)[0] + "_source1.wav"))
        print_info("Writing file {}".format(os.path.splitext(file_path)[0] + "_source2.wav"))

        librosa.output.write_wav(os.path.splitext(file_path)[0] + "_source1.wav", source1, self._hparams.sampling_rate)
        librosa.output.write_wav(os.path.splitext(file_path)[0] + "_source2.wav", source2, self._hparams.sampling_rate)
        return [(source1, self._hparams.sampling_rate), (source2, self._hparams.sampling_rate)]