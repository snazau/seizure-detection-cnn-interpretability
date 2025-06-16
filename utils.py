import os

import mne
import numpy as np
from scipy.signal import butter, sosfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # data.shape = (C, T)

    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=-1)
    return y


def generate_raw_samples(raw_eeg, sample_start_times, sample_duration):
    # if mask is not None:
    #     my_annot = mne.Annotations(
    #         onset=[sample_start_time for sample_start_time in sample_start_times],
    #         duration=[sample_duration for _ in sample_start_times],
    #         description=[f'seizure' if is_seizure_sample else f'normal' for idx, is_seizure_sample in enumerate(mask)]
    #     )
    #     raw_eeg.set_annotations(my_annot)
    #     raw_eeg.plot()
    #     import matplotlib.pyplot as plt
    #     plt.show()

    # events, event_id = mne.events_from_annotations(self.raw)
    # epochs = mne.Epochs(self.raw, events, tmin=0, tmax=self.sample_duration, baseline=None, event_repeated='drop')
    # self.raw_samples = epochs.get_data()
    # self.raw_samples = self.raw_samples[:, :, :self.sample_duration * 128]

    raw_data = raw_eeg.get_data()

    samples_num = len(sample_start_times)
    frequency = raw_eeg.info['sfreq']
    sample_len_in_idxs = int(sample_duration * frequency)
    channels_num = len(raw_eeg.info['ch_names'])

    samples = np.zeros((samples_num, channels_num, sample_len_in_idxs))
    time_idxs_start = np.zeros((samples_num, ))
    time_idxs_end = np.zeros((samples_num, ))
    for sample_idx, sample_start_time in enumerate(sample_start_times):
        start_idx = int(frequency * sample_start_time)
        end_idx = start_idx + sample_len_in_idxs
        samples[sample_idx] = raw_data[:, start_idx:end_idx]
        time_idxs_start[sample_idx] = start_idx
        time_idxs_end[sample_idx] = end_idx

    return samples, time_idxs_start, time_idxs_end


def get_baseline_stats(
        raw_data,
        baseline_length_in_seconds=500,
        sfreq=128,
        freqs=np.arange(1, 40.01, 0.1),
        return_baseline_spectrum=False,
):
    min_time, max_time = raw_data.times.min(), raw_data.times.max()

    sample_start_times = np.arange(min_time, max_time - baseline_length_in_seconds, max(baseline_length_in_seconds, max_time // baseline_length_in_seconds))
    samples, _, _ = generate_raw_samples(
        raw_data,
        sample_start_times,
        baseline_length_in_seconds
    )

    samples_std = np.std(samples, axis=2)  # std over time
    samples_avg_std = np.mean(samples_std, axis=1)  # mean over channels
    baseline_idx = np.argmin(samples_avg_std)
    baseline_segment = samples[baseline_idx:baseline_idx + 1]

    baseline_power_spectrum = mne.time_frequency.tfr_array_morlet(
        baseline_segment,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=freqs,
        output='power',
        n_jobs=-1
    )
    baseline_mean = np.mean(baseline_power_spectrum[0], axis=2, keepdims=True)
    baseline_std = np.std(baseline_power_spectrum[0], axis=2, keepdims=True)

    if return_baseline_spectrum:
        return baseline_mean, baseline_std, baseline_power_spectrum[0]
    else:
        return baseline_mean, baseline_std


def drop_unused_channels(eeg_file_path, raw_file):
    # drop unnecessary channels
    if 'data1' in eeg_file_path:
        channels_to_drop = ['EEG ECG', 'EEG MKR+ MKR-', 'EEG Fpz', 'EEG EMG']
    elif 'data2' in eeg_file_path:
        channels_to_drop = ['EEG ECG', 'Value MKR+', 'EEG Fpz', 'EEG EMG']
    else:
        raise NotImplementedError

    channels_num = len(raw_file.info['ch_names'])
    channels_to_drop = channels_to_drop[:2 + (channels_num - 27)]
    raw_file.drop_channels(channels_to_drop)


class EEGReader:
    supported_extensions = ['.dat', '.edf', '.fif']

    @staticmethod
    def read_eeg(file_path, preload=False):
        file_name, file_ext = os.path.splitext(file_path)

        assert os.path.exists(file_path), file_path
        assert file_ext in EEGReader.supported_extensions

        if file_ext == '.dat':
            mne_raw = EEGReader.read_dat(file_path)
        elif file_ext == '.edf':
            mne_raw = EEGReader.read_edf(file_path, preload)
        elif file_ext == '.fif':
            mne_raw = EEGReader.read_fif(file_path, preload)
        else:
            raise NotImplementedError
        return mne_raw

    @staticmethod
    def read_dat(file_path):
        raw_data = np.loadtxt(file_path)
        raw_data = raw_data.T * 1e-6

        channel_num = raw_data.shape[0]
        channel_names = [
            'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5',
            'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'F9', 'T9', 'ECG', 'P9', 'F10', 'T10', 'P10', 'MKR+ MKR-',
        ]
        channel_names = [f'EEG {channel_name}' for channel_name in channel_names]

        if channel_num == 27:
            del channel_names[1]
        elif channel_num == 28:
            pass
        else:
            raise NotImplementedError

        info = mne.create_info(ch_names=channel_names, ch_types='eeg', sfreq=128)
        mne_raw = mne.io.RawArray(raw_data, info)
        return mne_raw

    @staticmethod
    def read_edf(file_path, preload=False):
        mne_raw = mne.io.read_raw_edf(file_path, preload=preload)
        return mne_raw

    @staticmethod
    def read_fif(file_path, preload=False):
        mne_raw = mne.io.read_raw_fif(file_path, preload=preload)
        return mne_raw
