import gc
import os
import traceback

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
import numpy as np
import pandas as pd
import torch


def get_importance_matrix(freq_importance, freq_ranges, channel_importance, channel_groups, freqs):
    # freq_importance.shape = (F, )
    # channel_importance.shape = (C, )
    # freqs.shape = (F, )

    # cartesian product of avg GCAM and Channel Importance (GCAM x CI)
    freq_range_importances = np.zeros((len(freq_ranges), ), dtype=np.float32)
    channel_group_importances = np.zeros((len(channel_groups), ), dtype=np.float32)
    importance_matrix = np.zeros((len(freq_ranges), len(channel_groups)), dtype=np.float32)
    for freq_range_idx, freq_range_name in enumerate(freq_ranges.keys()):
        freq_range_min, freq_range_max = freq_ranges[freq_range_name]
        freq_range_start_idx = np.where(freqs == freq_range_min)[0][0]
        freq_range_end_idx = np.where(freqs == freq_range_max)[0][0]
        freq_range_importance = np.mean(freq_importance[freq_range_start_idx:freq_range_end_idx])
        freq_range_importances[freq_range_idx] = freq_range_importance

        for channel_group_idx, channel_group_name in enumerate(channel_groups.keys()):
            channel_group_idxs = channel_groups[channel_group_name]['channel_idxs']
            channel_group_importance = np.mean(channel_importance[channel_group_idxs])
            channel_group_importances[channel_group_idx] = channel_group_importance

            importance_matrix[freq_range_idx, channel_group_idx] = freq_range_importance * channel_group_importance

    # # Long tick names
    # freq_range_names = list(freq_ranges.keys())
    # channel_group_names = [
    #     f'{channel_group_name}\n({",".join(channel_groups[channel_group_name]["channel_names"])})'
    #     for channel_group_name in channel_groups.keys()
    # ]

    # Short tick names
    freq_range_names = [fr'$\{freq_range_name }$' for freq_range_name in freq_ranges.keys()]
    channel_group_names = [
        f'{"".join([name[0] for name in channel_group_name.split("-")])}'.upper()
        for channel_group_name in channel_groups.keys()
    ]

    # flip freq axes (make lower freqs at the bottom)
    freq_range_importances = np.flip(freq_range_importances, axis=0)
    importance_matrix = np.flip(importance_matrix, axis=0)
    freq_range_names.reverse()

    # print(freq_range_names)
    # print(freq_range_importances)
    # print(channel_group_names)
    # print(channel_group_importances)
    # print()

    return importance_matrix, freq_range_names, channel_group_names


def get_importance_matrices(heatmap, freq_ranges, channel_importances, channel_groups, segment_len_in_sec=10, sfreq=128):
    # heatmap.shape = (1, F, T)
    # channel_importances.shape = (N_10, C)

    segment_num = channel_importances.shape[0]

    heatmap = torch.from_numpy(heatmap)
    heatmap_segments = torch.split(heatmap, segment_len_in_sec * sfreq, dim=2)  # list of (1, F, 1280)
    heatmap_segments = torch.stack(heatmap_segments, dim=0)  # (N_10, 1, F, 1280)
    heatmap_segments_avg_freq = torch.mean(heatmap_segments, dim=3, keepdim=True)  # (N_10, 1, F, 1)

    freqs = np.arange(1, 40.01, 0.1)
    freqs = np.round(freqs, decimals=1)

    importance_matrices = list()
    for segment_idx in range(segment_num):
        heatmap_avg = heatmap_segments_avg_freq[segment_idx].squeeze().detach().cpu().numpy()  # (F, )
        importance_matrix, freq_range_names, channel_group_names = get_importance_matrix(
            heatmap_avg,
            freq_ranges,
            channel_importances[segment_idx],
            channel_groups,
            freqs,
        )
        importance_matrices.append(importance_matrix)

    importance_matrices = np.array(importance_matrices)  # (N_10, 5, 5)

    return importance_matrices, freq_range_names, channel_group_names


def visualize_importance_matrix(importance_matrix, freq_range_names, channel_group_names, axis, vmin, vmax, visualize_text=True, fontsize=matplotlib.rcParams["font.size"], cmap='Reds'):
    # importance_matrix, freq_range_names, channel_group_names = get_importance_matrix(
    #     freq_importance,
    #     freq_ranges,
    #     channel_importance,
    #     channel_groups,
    #     freqs,
    # )

    im = axis.imshow(importance_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(channel_group_names)), labels=channel_group_names)
    axis.set_yticks(np.arange(len(freq_range_names)), labels=freq_range_names)

    if visualize_text:
        for i in range(len(freq_range_names)):
            for j in range(len(channel_group_names)):
                text = axis.text(j, i, f'{importance_matrix[i, j]:.2f}', ha="center", va="center", color="black", fontsize=fontsize)

    return im


def visualize_channel_importance_at_time(channel_importance, start_time_sec, channel_names, axes, time_step_sec=10, vmin=None, vmax=None, cmap='Reds', colorbar=False, topk=5):
    # channel_importance.shape = (C, )
    channel_importance = channel_importance[np.newaxis]  # (1, C)

    # create df from channel_importance
    df_importance_columns = [channel_name.replace('EEG ', '') for channel_name in channel_names]
    df_importance = pd.DataFrame(channel_importance, columns=df_importance_columns)
    df_importance = df_importance.rename(columns={'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'})
    df_importance = df_importance * 1e-6

    # add missing channels to df
    available_channel_names_for_montage = [
        channel_name.replace('EEG ', '').replace('T3', 'T7').replace('T4', 'T8').replace('T5', 'P7').replace('T6', 'P8')
        for channel_name in df_importance.columns
    ]
    montage = mne.channels.make_standard_montage('standard_1020')

    missing_channels = list(set(montage.ch_names) - set(available_channel_names_for_montage))
    df_importance[missing_channels] = 0
    df_importance = df_importance.reindex(columns=montage.ch_names)

    # create info object
    fake_info = mne.create_info(ch_names=montage.ch_names, sfreq=1.0 / time_step_sec, ch_types='eeg')
    evoked = mne.EvokedArray(df_importance.to_numpy().T, fake_info)
    evoked.set_montage(montage)
    evoked = evoked.drop_channels(missing_channels)

    # create mask for top-k important channels
    mask_params = dict(markersize=20, markerfacecolor="y")
    mask = np.zeros_like(df_importance.to_numpy().T)  # (94, N_10)

    topk_channel_idxs = torch.topk(torch.from_numpy(channel_importance), k=topk, dim=1).indices.numpy()  # (N_10, k)
    channel_names_old_idx_to_new_idx = {
        old_idx: list(df_importance.columns).index(channel_name)
        for old_idx, (channel_name) in enumerate(available_channel_names_for_montage)
    }
    for segment_idx in range(topk_channel_idxs.shape[0]):
        for channel_idx_old in topk_channel_idxs[segment_idx]:
            channel_idx_new = channel_names_old_idx_to_new_idx[channel_idx_old]
            mask[channel_idx_new, segment_idx] = 1

    channel_idxs_to_delete_from_mask = [
        list(df_importance.columns).index(channel_name)
        for channel_name in missing_channels
    ]
    channel_idxs_to_delete_from_mask = np.array(channel_idxs_to_delete_from_mask)
    mask = np.delete(mask, channel_idxs_to_delete_from_mask, axis=0)

    # plot
    evoked_fig = evoked.plot_topomap(
        evoked.times,
        mask=mask,
        mask_params=mask_params,
        units='Importance',
        nrows='auto',
        ncols='auto',
        ch_type='eeg',
        time_format=f'{int(start_time_sec):d}-{int(start_time_sec) + 10:d} sec',
        show_names=True,
        # show_names=False,
        axes=axes,
        colorbar=False,
        vlim=(vmin, vmax),
        cmap=cmap,
    )


def visualize_raw_with_spectrum_data_v7(
        freq_ranges,
        channel_groups,
        power_spectrum,
        raw_signal,
        heatmap,
        channel_importances,
        channel_names,
        channels_to_show,
        segment_of_int_idx_start,
        segment_of_int_idx_end,
        save_path=None,
        sfreq=128,
        time_shift=0,
        seizure_times_list=None,
        seizure_times_colors=('red', 'green', 'blue', 'yellow', 'cyan'),
        seizure_times_ls=('-', '--', ':'),
        max_spectrum_value=None,
        min_importance_value=None,
        min_importance_matrix_value=None,
        max_importance_matrix_value=None,
        localization='ru',
        spectrum_cmap='viridis',
        other_cmap='Reds',
):
    # power_spectrum.shape = (C, F, T)
    # raw_signal.shape = (C, T)
    # heatmap.shape = (1, F, T)
    # channel_importances.shape = (N_10, C)

    assert localization in ['ru', 'en']

    import matplotlib
    fontsize = 42
    matplotlib.rcParams.update({'font.size': fontsize, 'legend.fontsize': fontsize, 'lines.markersize': fontsize})

    if channel_names[-1] == '__heatmap__':
        channel_names = channel_names[:-1]

    if channels_to_show is None:
        channels_to_show = channel_names

    channel_dim, freq_dim, time_dim = power_spectrum.shape[:3]
    segment_num = int(time_dim / sfreq / 10)

    time_step = 10
    time_ticks_10sec = [time_tick for time_tick in range(int(time_shift), int(time_shift) + int(time_dim / sfreq) + 1, time_step)]

    x_ticks = [0 + i * time_dim / 10 for i in range(0, 10)]
    x_ticks_labels = [f'{time_shift + i * time_dim / 10 / sfreq}' for i in range(0, 10)]
    y_ticks = [10 * freq - 10 for freq in [1, 4, 8, 14, 30, 40]]
    y_ticks_labels = [f'{freq:d}Гц' if localization == 'ru' else f'{freq:d}Hz' for freq in [1, 4, 8, 14, 30, 40]]

    # fig_height = 7 * 5
    fig_height = 12 * 5
    fig_width = min(80 * 10, int(30 * time_dim / sfreq / 120))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    segment_of_int_num = segment_of_int_idx_end - segment_of_int_idx_start + 1
    gs = GridSpec(6, segment_of_int_num, figure=fig)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_spectrum = fig.add_subplot(gs[1, :])
    ax_heatmap = fig.add_subplot(gs[5, :])

    channel_idx = [channel_idx for channel_idx, channel_name in enumerate(channel_names) if channels_to_show[0] in channel_name]
    assert len(channel_idx) == 1
    channel_idx = channel_idx[0]

    raw_signal_channel = raw_signal[channel_idx]  # (T, )
    power_spectrum_channel = power_spectrum[channel_idx]  # (F, T)
    channel_name = channel_names[channel_idx]

    # plot 10 sec lines for raw signal
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        ax_raw.axvline(x=time_tick_10sec, color='#000000', ls='--')

    # plot raw signal
    time_values = np.linspace(time_shift, time_shift + time_dim / sfreq, raw_signal.shape[1])
    ax_raw.plot(time_values, raw_signal_channel)
    ax_raw.set_title(channel_name, fontsize=fontsize)
    ax_raw.set_xlim([time_shift, time_shift + time_dim / sfreq])

    # plot power spectrum
    vmin = power_spectrum_channel.min()
    vmax = power_spectrum_channel.max() if max_spectrum_value is None else max_spectrum_value
    im = ax_spectrum.imshow(power_spectrum_channel, cmap=spectrum_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax_spectrum.invert_yaxis()
    ax_spectrum.set_xticks(x_ticks)
    ax_spectrum.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_yticks(y_ticks)
    ax_spectrum.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_spectrum.set_xlabel('Время, с' if localization == 'ru' else 'Time, sec')
    ax_spectrum.set_ylabel('Частота, Гц' if localization == 'ru' else 'Frequency, Hz')

    # add relative time ticks for spectrum
    ax_spectrum_twin = ax_spectrum.twiny()
    ax_spectrum_twin.set_xlim(ax_spectrum.get_xlim())
    ax_spectrum_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    ax_spectrum_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # plot heatmap
    vmin = heatmap[0].min()
    vmax = heatmap[0].max()
    im = ax_heatmap.imshow(heatmap[0], cmap=spectrum_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax_heatmap.invert_yaxis()
    ax_heatmap.set_xticks(x_ticks)
    ax_heatmap.set_xticklabels(x_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(y_ticks_labels, fontsize=fontsize)
    ax_heatmap.set_xlabel('Время, с' if localization == 'ru' else 'Time, sec')
    ax_heatmap.set_ylabel('Частота, Гц' if localization == 'ru' else 'Frequency, Hz')

    # add relative time ticks for spectrum
    heatmap_twin = ax_heatmap.twiny()
    heatmap_twin.set_xlim(ax_heatmap.get_xlim())
    heatmap_twin.set_xticks([i * sfreq * 10 for i in range(0, segment_num + 1)])
    heatmap_twin.set_xticklabels([f'{i * 10}' for i in range(0, segment_num + 1)], fontsize=fontsize)

    # add vertical lines
    if seizure_times_list is not None:
        for seizure_idx, seizure_time in enumerate(seizure_times_list):
            seizure_line_color = seizure_times_colors[seizure_idx % len(seizure_times_list)]
            seizure_line_style = '-'
            seizure_line_width = plt.rcParams['lines.linewidth'] * 4

            if time_shift <= (seizure_time['start'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['start']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

                x = seizure_time['start'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} start')

            if time_shift <= (seizure_time['end'] + time_shift) <= (time_shift + time_dim / sfreq):
                x = time_shift + seizure_time['end']
                ax_raw.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_spectrum.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

                x = seizure_time['end'] * sfreq
                ax_heatmap.axvline(x=x, color=seizure_line_color, ls=seizure_line_style, lw=seizure_line_width, label=f'Seizure {seizure_idx:02} end')

    # plot 10 sec lines for spectrum
    for time_tick_10sec in time_ticks_10sec[1:-1]:
        if time_shift <= time_tick_10sec <= (time_shift + time_dim / sfreq):
            x = (time_tick_10sec - time_shift) * sfreq
            ax_spectrum.axvline(x=x, color='#FFFFFF', ls='--')
            ax_heatmap.axvline(x=x, color='#FFFFFF', ls='--')

    # plot band lines for spectrum
    for tick in y_ticks:
        ax_spectrum.axhline(y=tick, color='#FFFFFF', ls='--')
        ax_heatmap.axhline(y=tick, color='#FFFFFF', ls='--')

    # plot topogram with importances (occluded)
    if min_importance_value is not None:
        channel_importances_clipped = np.clip(channel_importances, a_min=min_importance_value, a_max=None)
        channel_importances_clipped = (channel_importances_clipped - channel_importances_clipped.min()) / (channel_importances_clipped.max() - channel_importances_clipped.min())
    else:
        channel_importances_clipped = channel_importances.copy()

    for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1):
        ax_segment = fig.add_subplot(gs[4, segment_idx - segment_of_int_idx_start])
        visualize_channel_importance_at_time(
            channel_importances_clipped[segment_idx],
            start_time_sec=segment_idx * 10,
            channel_names=channel_names,
            axes=ax_segment,
            time_step_sec=10,
            vmin=0,
            vmax=1,
            cmap=other_cmap,
            colorbar=True,
            topk=5,
        )
        ax_segment.tick_params(axis='both', which='major', labelsize=fontsize)
        if (segment_idx - segment_of_int_idx_start) == 0:
            ax_segment.set_ylabel('СI', fontsize=fontsize)

    # plot importance matrices
    importance_matrices, freq_range_names, channel_group_names = get_importance_matrices(
        heatmap, freq_ranges, channel_importances, channel_groups, segment_len_in_sec=10, sfreq=sfreq,
    )  # (N_10, 5, 5), list with row names, list with col names

    if min_importance_matrix_value == 'min':
        min_importance_matrix_value = importance_matrices.min()

    if max_importance_matrix_value == 'max':
        max_importance_matrix_value = importance_matrices.max()

    for segment_idx in range(segment_of_int_idx_start, segment_of_int_idx_end + 1, 2):
        ax_segment = fig.add_subplot(gs[2:4, segment_idx - segment_of_int_idx_start:segment_idx - segment_of_int_idx_start + 2])
        ax_segment.set_title(f'{segment_idx * time_step}-{(segment_idx + 1) * time_step} ' + ('c' if localization == 'ru' else 'sec'), fontsize=fontsize)
        if (segment_idx - segment_of_int_idx_start) == 0:
            ax_segment.set_ylabel(r'$FRI_{[f_{0},f_{1}]}^{R}$', fontsize=fontsize)

        im = visualize_importance_matrix(
            importance_matrices[segment_idx],
            freq_range_names,
            channel_group_names,
            ax_segment,
            vmin=min_importance_matrix_value,
            vmax=max_importance_matrix_value,
            visualize_text=False,
            fontsize=fontsize,
            cmap=other_cmap,
        )

    # save figure tp disk
    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=1)
        except Exception as e:
            print(f'Unable to save {save_path}')
            print(f'{traceback.format_exc()}')
    fig.clear()
    plt.close(fig)

    del fig, gs, ax_raw, ax_spectrum, time_values, im, raw_signal_channel, power_spectrum_channel

    gc.collect()
