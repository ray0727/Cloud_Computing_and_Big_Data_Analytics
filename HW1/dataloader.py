import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math, random
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import pandas as pd
import numpy as np
import os

def getData(mode):
    if mode == 'train':
        data = pd.read_csv('./music-regression/train.csv')
        audio_name = data.track
        score = data.score
        return np.squeeze(audio_name.values), np.squeeze(score.values)
    else:
        data = pd.read_csv('./music-regression/test.csv')
        audio_name = data.track
        return np.squeeze(audio_name.values)

class AudioUtil():
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)

  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

class SoundDataset(Dataset):
    def __init__(self, audio_path, mode):
        self.audio_path = audio_path
        self.mode = mode
        if self.mode == "train":
            self.audio_name, self.score = getData(mode)
        else:
            self.audio_name = getData(mode)

        self.duration = 1000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.2
        
    def __getitem__(self, index):
        if self.mode == "train":
            single_audio_name = os.path.join(self.audio_path, self.audio_name[index])
            aud = AudioUtil.open(single_audio_name)
            reaud = AudioUtil.resample(aud, self.sr)
            rechan = AudioUtil.rechannel(reaud, self.channel)

            # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
            # shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
            sgram = AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

            score = self.score[index]
            return aug_sgram, score
        else: 
            single_audio_name = os.path.join(self.audio_path, self.audio_name[index])
            aud = AudioUtil.open(single_audio_name)
            reaud = AudioUtil.resample(aud, self.sr)
            rechan = AudioUtil.rechannel(reaud, self.channel)

            # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
            # shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
            sgram = AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
            return aug_sgram, single_audio_name[32:]

    def __len__(self):
        return len(self.audio_name)
