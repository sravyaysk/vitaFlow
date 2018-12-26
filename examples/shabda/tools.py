import librosa
import numpy as np
from numpy.lib import stride_tricks
from tqdm import tqdm


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


def _helper1(speech_1_features, speech_2_features, frames_per_sample, len_spec,
             speech_mix_features, speech_VAD):
    # Processes one audio pair
    # TODO: rename this function
    # TODO: Add docs
    k = 0
    bag = []
    while k + frames_per_sample < len_spec:
        sample_mix = speech_mix_features[k:k + frames_per_sample, :].astype('float32')
        VAD = speech_VAD[k:k + frames_per_sample, :].astype('bool')

        sample_1 = speech_1_features[k:k + frames_per_sample, :]
        sample_2 = speech_2_features[k:k + frames_per_sample, :]
        # phase = speech_phase[k: k + frames_per_sample, :]
        # Y: indicator of the belongings of the TF bin
        # 1st speaker or second speaker
        Y = np.array([sample_1 > sample_2,
                      sample_1 < sample_2]
                     ).astype('bool')
        Y = np.transpose(Y, [1, 2, 0]).astype('bool')
        bag.append((sample_mix, VAD, Y))
        k = k + frames_per_sample
    return bag


def _get_log_spectrum_features(speech, frame_size, neff, amp_fac, min_amp):
    # TODO: rename this function
    # TODO: Add docs
    speech_features = np.abs(stft(speech, frame_size)[:, :neff])
    speech_features = np.maximum(speech_features, np.max(speech_features) / min_amp)
    speech_features = 20. * np.log10(speech_features * amp_fac)
    return speech_features


def _helper2(wav_file_1, wav_file_2,  sampling_rate, frame_size, neff, amp_fac, min_amp,
             threshold, global_mean, global_std, frames_per_sample):
    # TODO: rename this function
    # TODO: Add docs
    speech_1, _ = librosa.core.load(wav_file_1, sr=sampling_rate)
    # amp factor between -3 dB - 3 dB
    fac = np.random.rand(1)[0] * 6 - 3
    speech_1 = 10. ** (fac / 20) * speech_1

    speech_2, _ = librosa.core.load(wav_file_2, sr=sampling_rate)
    fac = np.random.rand(1)[0] * 6 - 3
    speech_2 = 10. ** (fac / 20) * speech_2
    #
    # print_warn(speech_1.shape)
    # print_warn(speech_2.shape)

    # mix
    length = min(len(speech_1), len(speech_2))
    speech_1 = speech_1[:length]
    speech_2 = speech_2[:length]
    speech_mix = speech_1 + speech_2

    # compute log spectrum for 1st speaker
    # speech_1_features = np.abs(stft(speech_1, frame_size)[:, :neff])
    # speech_1_features = np.maximum(speech_1_features, np.max(speech_1_features) / min_amp)
    # speech_1_features = 20. * np.log10(speech_1_features * amp_fac)
    speech_1_features = _get_log_spectrum_features(speech_1, frame_size, neff, amp_fac, min_amp)

    # same for the 2nd speaker
    # speech_2_features = np.abs(stft(speech_2, frame_size)[:, :neff])
    # speech_2_features = np.maximum(speech_2_features, np.max(speech_2_features) / min_amp)
    # speech_2_features = 20. * np.log10(speech_2_features * amp_fac)
    speech_2_features = _get_log_spectrum_features(speech_2, frame_size, neff, amp_fac, min_amp)

    # same for the mixture
    speech_mix_spec0 = stft(speech_mix, frame_size)[:, :neff]
    speech_mix_features = np.abs(speech_mix_spec0)

    # speech_phase = speech_mix_spec0 / speech_mix_spec
    speech_mix_features = np.maximum(speech_mix_features, np.max(speech_mix_features) / min_amp)
    speech_mix_features = 20. * np.log10(speech_mix_features * amp_fac)
    max_mag = np.max(speech_mix_features)

    speech_VAD = (speech_mix_features > (max_mag - threshold)).astype(int)
    speech_mix_features = (speech_mix_features - global_mean) / global_std

    len_spec = speech_1_features.shape[0]
    # print_error("len_spec {}".format(len_spec))

    new_data = _helper1(speech_1_features, speech_2_features, frames_per_sample, len_spec,
                        speech_mix_features, speech_VAD)
    return new_data


def boost_yield_samples(speaker_file_match, sampling_rate, frame_size, amp_fac, neff,
                        min_amp, threshold, global_mean, global_std, frames_per_sample):
    # self.speaker_file_match = {}
    ## training datasets
    # self.samples = []
    ## the begining index of a batch
    # self.ind = 0

    # for each file pair, generate their mixture and reference samples
    bag = []
    for wav_file_1 in tqdm(speaker_file_match, desc="speaker_file_match"):
        wav_file_2 = speaker_file_match[wav_file_1]
        new_data = _helper2(wav_file_1, wav_file_2,  sampling_rate, frame_size, neff, amp_fac, min_amp,
             threshold, global_mean, global_std, frames_per_sample)
        bag += new_data
    return bag
