#####################      CSV         ###########################################

import csv


def data_to_csv(device):
    with open('./csv/accelerometer.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(device.labels.split(','))
        for str in device.values:
            writer.writerow(str.split(','))


#####################      MATHS         ###########################################
import numpy as np


def spectral_centroid_spread(fft_magnitude, sampling_rate):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(fft_magnitude) + 1)) * \
          (sampling_rate / (2.0 * len(fft_magnitude)))
    eps = 0.00000001
    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    centroid = (NUM / DEN)

    # Spread:
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)

    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)
    spread = spread / (sampling_rate / 2.0)
    return centroid


def energy_entropy(frame, n_short_blocks):
    """Computes entropy of energy"""
    # total frame energy
    eps = 0.00000001
    for i in frame:
        frame_energy = np.sum(i ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


def energy(frame):
    return np.sum(frame ** 2) / np.float64(len(frame))


#####################      GROUP VALUES         ###########################################
import datetime


def get_time_value(x):
    t = datetime.datetime.fromtimestamp(x / 1000.0)
    t = t.replace(microsecond=0)
    return t.timestamp()


def get_time_ignore_second(x):
    t = datetime.datetime.fromtimestamp(x / 1000.0)
    t = t.replace(microsecond=0)
    t = t.replace(second=int(t.second / 10))
    return t.timestamp()


def get_window_10(x):
    t = datetime.datetime.fromtimestamp(x)
    t = t.replace(second=int(t.second / 10))
    return t.timestamp()

