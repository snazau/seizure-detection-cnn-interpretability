import copy
import gc
import json
import os

import mne
import numpy as np
import torch

from resnet import EEGResNet18Spectrum
import utils
import visualization


def get_gradcam(
        power_spectrum,
        model_params,
        checkpoint_path,
        train_segment_duration_in_sec,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
        sfreq,
        device,
):
    # power_spectrum.shape = (1, C, F, T)
    # T should be the multiple of (sfreq * train_segment_duration_in_sec)

    assert power_spectrum.shape[0] == 1

    # reshape power_spectrum into the 10sec segments as model input
    _, channels, freq_dim, time_dim = power_spectrum.shape[:4]
    # assert time_dim % (sfreq * train_segment_duration_in_sec) == 0

    batch_input = torch.split(power_spectrum, train_segment_duration_in_sec * sfreq, dim=3)  # list of (1, C, F, 1280)
    batch_input = torch.cat(batch_input, dim=0)  # (B, C, F, 1280)
    print(f'batch_input[0] = {batch_input[0].shape} len = {len(batch_input)}')

    # prepare input
    log = model_params['preprocessing_params']['log']
    baseline_correction = model_params['preprocessing_params']['baseline_correction']
    if log:
        batch_input = np.log(batch_input)
    elif baseline_correction:
        raise NotImplementedError

    normalization = model_params['preprocessing_params']['normalization']
    if normalization == 'minmax':
        batch_input = (batch_input - batch_input.min()) / (batch_input.max() - batch_input.min())
    elif normalization == 'meanstd':
        batch_input = (batch_input - batch_input.mean()) / batch_input.std()
    elif normalization == 'cwt_meanstd':
        raise NotImplementedError

    # load model
    model_class_name = model_params.get('class_name', 'EEGResNet18Spectrum')
    model_kwargs = model_params.get('kwargs', dict())
    model = globals()[model_class_name](**model_kwargs)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['model']['state_dict']
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        new_state_dict = {f'model.{key}': value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # inference
    samples_num = batch_input.shape[0]
    batch_input = batch_input.to(device)

    with torch.no_grad():
        batch_outputs = torch.squeeze(model(batch_input))  # (B, )
        batch_probs = torch.sigmoid(batch_outputs)  # (B, )

    # calc Grad-CAM
    heatmaps, fmaps, grads = list(), list(), list()
    for sample_idx in range(samples_num):
        sample_heatmap, sample_fmaps, sample_grads = model.interpret(
            batch_input[sample_idx:sample_idx + 1], return_fm=True
        )  # (1, 1, F, 1280), (B, 512, F / 32, 40), (B, 512, F / 32, 40)
        heatmaps.append(sample_heatmap)
        fmaps.append(sample_fmaps)
        grads.append(sample_grads)
    batch_heatmap = torch.cat(heatmaps, dim=0)  # (B, 1, F, 1280)
    batch_fmap = torch.cat(fmaps, dim=0)  # (B, 512, F / 32, 40)
    batch_grad = torch.cat(grads, dim=0)  # (B, 512, F / 32, 40)

    # calc channel importance by occlusion
    power_spectrum_batched = torch.cat(torch.split(power_spectrum, train_segment_duration_in_sec * sfreq, dim=3), dim=0)  # (B, C, F, 1280)
    occlusion_idx = torch.argmin(
        torch.sum(power_spectrum_batched[max(segment_of_int_idx_start - 6, 0):segment_of_int_idx_start], dim=(1, 2, 3))
    )
    occlusion_idx = occlusion_idx + max(segment_of_int_idx_start - 6, 0)
    occlusion = batch_input[occlusion_idx:occlusion_idx + 1]  # (1, C, F, 1280)

    channel_importance_occluded = torch.zeros((batch_input.shape[0], batch_input.shape[1]), dtype=torch.float32)
    with torch.no_grad():
        for channel_idx in range(batch_input.shape[1]):
            batch_input_occluded = batch_input[segment_of_int_idx_start:segment_of_int_idx_end + 1]  # (N_10_soi, C, F, 1280)
            batch_input_occluded[:, channel_idx] = occlusion[:, channel_idx]

            batch_outputs_occluded = torch.squeeze(model(batch_input_occluded))  # (N_10_soi, )
            batch_probs_occluded = torch.sigmoid(batch_outputs_occluded)  # (N_10_soi, )

            channel_importance_occluded[segment_of_int_idx_start:segment_of_int_idx_end + 1, channel_idx] = torch.abs(
                batch_probs[segment_of_int_idx_start:segment_of_int_idx_end + 1] - batch_probs_occluded,
            )

    # calc freq_importance - avg over time of Grad-CAM
    freq_importance = torch.mean(batch_heatmap, dim=3, keepdim=True)  # (B, 1, F, 1)
    freq_importance = torch.squeeze(freq_importance)  # (B, F)

    # concat Grad-CAM for each 10sec segment along time dim
    batch_heatmap = torch.cat(
        [
            batch_heatmap[sample_idx:sample_idx + 1]
            for sample_idx in range(batch_heatmap.shape[0])
        ],
        dim=3,
    )  # (1, 1, F, B * 1280)
    print(f'batch_heatmap = {batch_heatmap.shape}')

    batch_fmap = torch.cat(
        [
            batch_fmap[sample_idx:sample_idx + 1]
            for sample_idx in range(batch_fmap.shape[0])
        ],
        dim=3,
    )  # (1, 512, F, B * 40)
    print(f'batch_fmap = {batch_fmap.shape}')

    batch_grad = torch.cat(
        [
            batch_grad[sample_idx:sample_idx + 1]
            for sample_idx in range(batch_grad.shape[0])
        ],
        dim=3,
    )  # (1, 512, F, B * 40)
    print(f'batch_grad = {batch_grad.shape}')

    return (
        batch_probs.cpu().numpy(),
        batch_heatmap.cpu().numpy(),
        channel_importance_occluded.cpu().numpy(),
        freq_importance.cpu().numpy(),
        batch_fmap.cpu().numpy(),
        batch_grad.cpu().numpy(),
    )


def select_most_important_channels_v3(channel_importance, important_num, channel_names):
    # channel_importance.shape = (N_10, 25)

    # calc mean importance_score for each EEG channel over time of interest
    channel_importance_score = np.mean(channel_importance, axis=0)

    # sorted idxs from min importance_score to max importance_score
    channel_importance_score_sorted_idxs = np.argsort(channel_importance_score)

    # we are selecting channels with highest importance_score
    important_channel_idxs = channel_importance_score_sorted_idxs[-important_num:]

    channels_to_show = [
        channel_names[channel_idx].replace('EEG ', '')
        for channel_idx in important_channel_idxs
    ]
    channels_to_show = sorted(channels_to_show)
    return channels_to_show


def visualize_samples(
        samples,
        prediction_number,
        time_starts,
        channel_names,
        sfreq,
        baseline_mean,
        set_name,
        subject_key,
        visualization_dir,
        checkpoint_path=None,
        model_params=None,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
):
    # samples.shape = (1, C, T)
    freqs = np.arange(1, 40.01, 0.1)

    print('channel_names', channel_names)
    channels_to_show = ['F3', 'F4', 'T5', 'T6', 'P3', 'P4', 'Cz']

    freq_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 30),
        'gamma': (30, 40),
    }
    channel_groups = {
        'frontal': {
            'channel_names': ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['Fp1', 'Fp2', 'F9', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F10']])
            ],
        },
        'central': {
            'channel_names': ['C3', 'Cz', 'C4'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['C3', 'Cz', 'C4']])
            ],
        },
        'perietal-occipital': {
            'channel_names': ['P3', 'Pz', 'P4', 'O1', 'O2'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['P3', 'Pz', 'P4', 'O1', 'O2']])
            ],
        },
        'temporal-left': {
            'channel_names': ['T9', 'T3', 'P9', 'T5'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T9', 'T3', 'P9', 'T5']])
            ],
        },
        'temporal-right': {
            'channel_names': ['T10', 'T4', 'P10', 'T6'],
            'channel_idxs': [
                channel_idx
                for channel_idx, channel_name in enumerate(channel_names)
                if any([c in channel_name for c in ['T10', 'T4', 'P10', 'T6']])
            ],
        },
    }

    # power_spectrum.shape = (1, C, F, T)
    power_spectrum = mne.time_frequency.tfr_array_morlet(
        samples,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=freqs,
        output='power',
        n_jobs=-1
    )

    segments_num = int(samples.shape[-1] / sfreq / 10)
    if set_name == 'fn':
        # idxs of segments with GT seizures
        segment_of_int_idx_start = int(seizure_times_list[0]['start'] / 10)
        segment_of_int_idx_end = int(seizure_times_list[0]['end'] / 10) - 1
    else:
        # idxs of segments with PRED seizures
        segment_of_int_idx_start = int(seizure_times_list[2]['start'] / 10)
        segment_of_int_idx_end = int(seizure_times_list[2]['end'] / 10) - 1
    segment_of_int_idx_end = min(segments_num - 1, segment_of_int_idx_end)

    heatmap = None
    channel_importance, channel_importance_occluded, freq_importance = None, None, None
    if model_params is not None and checkpoint_path is not None and os.path.exists(checkpoint_path):
        pred_prob, heatmap, channel_importance_occluded, freq_importance, fmap, grad = get_gradcam(
            torch.from_numpy(power_spectrum).float(),
            model_params,
            checkpoint_path,
            train_segment_duration_in_sec=10,
            segment_of_int_idx_start=segment_of_int_idx_start,
            segment_of_int_idx_end=segment_of_int_idx_end,
            sfreq=128,
            device=torch.device('cpu'),
        )
        # N_10 = T // 1280
        # pred_prob.shape = (N_10, )
        # heatmap.shape = (1, 1, F, T)
        # channel_importance_occluded.shape = (N_10, 25)
        # freq_importance.shape = (N_10, F)
        # fmap.shape = (1, 512, F / 32, T / 32)
        # grad.shape = (1, 512, F / 32, T / 32)

        channels_to_show = select_most_important_channels_v3(
            channel_importance_occluded,
            important_num=7,
            channel_names=channel_names,
        )

        # heatmap = None
        channel_names.append('__heatmap__')
        channels_to_show.append('__heatmap__')

    from utils import butter_bandpass_filter
    raw_signal_filtered = butter_bandpass_filter(samples[0], lowcut=1, highcut=40, fs=128, order=5)
    power_spectrum_filtered = mne.time_frequency.tfr_array_morlet(
        raw_signal_filtered[np.newaxis, ...],
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=freqs,
        output='power',
        n_jobs=-1
    )
    print(f'raw_signal_filtered = {raw_signal_filtered.shape}')
    print(f'power_spectrum_filtered = {power_spectrum_filtered.shape}')
    print(f'samples[0] = {samples[0].shape}')
    print(f'power_spectrum[0] = {power_spectrum[0].shape}')

    set_dir = os.path.join(visualization_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)
    for idx in range(power_spectrum.shape[0]):
        power_spectrum_filtered_corrected = (power_spectrum_filtered[idx] - baseline_mean) / baseline_mean

        visualization.visualize_raw_with_spectrum_data_v7(
            freq_ranges,
            channel_groups,
            power_spectrum_filtered_corrected,
            raw_signal_filtered,
            heatmap[idx] if heatmap is not None else None,
            channel_importance_occluded,
            channel_names=channel_names,
            channels_to_show=channels_to_show,
            segment_of_int_idx_start=segment_of_int_idx_start,
            segment_of_int_idx_end=segment_of_int_idx_end,
            save_path=os.path.join(visualization_dir, set_name, f'{subject_key.replace("/", "_")}_seizure{int(prediction_number[idx])}_V7.png'),
            sfreq=128,
            time_shift=time_starts[idx],
            seizure_times_list=seizure_times_list,
            seizure_times_colors=seizure_times_colors,
            seizure_times_ls=seizure_times_ls,
            max_spectrum_value=20,
            min_importance_value=0.75,
            min_importance_matrix_value=None,
            max_importance_matrix_value=None,
            localization='en',
            spectrum_cmap='viridis',
            other_cmap='Blues',
        )


def visualize_predicted_segment_v2(checkpoint_path, model_params, subject_eeg_path, subject_key, prediction_data, set_name, save_dir):
    raw = utils.EEGReader.read_eeg(subject_eeg_path, preload=True)
    utils.drop_unused_channels(subject_eeg_path, raw)
    channel_names = raw.info['ch_names']
    recording_duration = raw.times.max() - raw.times.min()

    # baseline stats
    import numpy as np
    freqs = np.arange(1, 40.01, 0.1)
    baseline_mean, baseline_std = utils.get_baseline_stats(
        raw,
        baseline_length_in_seconds=500,
        sfreq=raw.info['sfreq'],
        freqs=freqs,
        return_baseline_spectrum=False,
    )  # (C, F, 1), (C, F, 1), [(C, F, T)]

    # padding_sec = 20
    padding_sec = 0

    # tp
    true_segment = prediction_data['true']
    true_segment_dilated = prediction_data['true_dilated']
    pred_segments = prediction_data['preds']

    # segment limits rounded to closest 10
    segment_min_start_time = round(true_segment_dilated['start'] - padding_sec, -1)
    segment_max_end_time = round(true_segment_dilated['end'] + padding_sec, -1)

    start_time = max(0, segment_min_start_time - padding_sec)
    duration = segment_max_end_time - segment_min_start_time + padding_sec * 2
    if (start_time + duration) > recording_duration:
        duration = duration - (start_time + duration - recording_duration) - 1

    print(f'{os.path.basename(subject_eeg_path)} duration = {duration} segment_min_start_time = {segment_min_start_time} segment_max_end_time = {segment_max_end_time}')

    seizure_sample, _, _ = utils.generate_raw_samples(
        raw,
        sample_start_times=[start_time],
        sample_duration=duration,
    )

    segments_to_visualize = [
        {
            'start': true_segment['start'] - segment_min_start_time + padding_sec,
            'end': true_segment['end'] - segment_min_start_time + padding_sec,
        },
        {
            'start': true_segment_dilated['start'] - segment_min_start_time + padding_sec,
            'end': true_segment_dilated['end'] - segment_min_start_time + padding_sec,
        },
    ]
    segments_times_colors = ['#F3A9A5', 'orange']
    # segments_times_colors = ['green', 'orange']
    segments_times_ls = ['solid', 'solid']

    possible_ls = ('dotted', 'dashed', 'dashdot')
    for pred_idx, pred_segment in enumerate(pred_segments):
        segments_to_visualize.append(
            {
                'start': pred_segment['start'] - segment_min_start_time + padding_sec,
                'end': pred_segment['end'] - segment_min_start_time + padding_sec,
            }
        )

        # color = 'red'
        color = 'blue'
        ls = possible_ls[pred_idx % len(possible_ls)]
        segments_times_colors.append(color)
        segments_times_ls.append(ls)

    fname_segment_idx = [0]
    visualize_samples(
        seizure_sample,
        fname_segment_idx,
        [start_time],
        copy.deepcopy(channel_names),
        sfreq=128,
        baseline_mean=baseline_mean,
        set_name=set_name,
        subject_key=subject_key,
        visualization_dir=save_dir,
        checkpoint_path=checkpoint_path,
        model_params=model_params,
        seizure_times_list=segments_to_visualize,
        seizure_times_colors=segments_times_colors,
        seizure_times_ls=segments_times_ls,
    )
    del seizure_sample, segments_to_visualize
    gc.collect()
    torch.cuda.empty_cache()

    del raw, baseline_mean, baseline_std
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    subject_key = 'example_01_seizure_01'
    subject_eeg_path = './assets/eeg/data1/example_01.dat'
    checkpoint_path = './assets/checkpoints/two_stage/best.pth.tar'
    prediction_data_path = './assets/prediction_examples/example_01.json'
    prediction_set_name = 'tp'
    save_dir = './assets/visualizations'

    model_params = {
        'class_name': 'EEGResNet18Spectrum',
        'kwargs': dict(),
        'preprocessing_params': {
            'normalization': 'meanstd',
            'transform': None,
            'data_type': 'power_spectrum',
            'baseline_correction': False,
            'log': True,
        }
    }

    with open(prediction_data_path) as f:
        prediction_data = json.load(f)

    visualize_predicted_segment_v2(
        checkpoint_path,
        model_params,
        subject_eeg_path,
        subject_key,
        prediction_data,
        prediction_set_name,
        save_dir,
    )
