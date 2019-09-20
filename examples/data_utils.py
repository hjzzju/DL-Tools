import json
import os
import random
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

import layers
from datasets import audio
from utils import apply_mv_norm, get_mask_from_lengths, add_delta_deltas
from wavenet_vocoder.util import assert_ready_for_upsampling, ensure_divisible


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, type, hparams):
        ds_name = hparams.ds_name
        self.data = None
        data = np.load(ds_name + '.npz', allow_pickle=True)
        self.audio_and_text_keys = list(sorted(data.keys(), key=lambda x: int(x)))
        self.hparams = hparams
        self.type = type
        if hparams.full_song:
            if type == 'val':
                self.audio_and_text_keys = self.audio_and_text_keys[0:200:6]
            else:
                random.seed(1234)
                random.shuffle(self.audio_and_text_keys)
        elif hparams.speech:
            if type == 'val':
                self.audio_and_text_keys = self.audio_and_text_keys[:50]
            else:
                self.audio_and_text_keys = self.audio_and_text_keys[10:]
                random.seed(1234)
                random.shuffle(self.audio_and_text_keys)
        else:
            if type == 'val':
                if not hparams.reverse:
                    # add spk2 data for val
                    self.audio_and_text_keys = self.audio_and_text_keys[0:240:6] + \
                                               self.audio_and_text_keys[18001:18300:7]
                    # self.audio_and_text_keys[18000:19000:7]
                else:
                    self.audio_and_text_keys = self.audio_and_text_keys[0:480:6] + \
                                               self.audio_and_text_keys[17670:17670 + 300:6]  # add spk2 data for val
            else:
                if self.hparams.leave_val == 1:
                    if hparams.multispeaker:
                        self.audio_and_text_keys = self.audio_and_text_keys[480:17670] + \
                                                   self.audio_and_text_keys[18300:]  # add spk2 training data
                    else:
                        self.audio_and_text_keys = self.audio_and_text_keys[480:17670]

                random.seed(1234)
                random.shuffle(self.audio_and_text_keys)
        print(type, "load #data:", len(self.audio_and_text_keys), "from ", ds_name)

    def __getitem__(self, indexs):
        if self.data is None:
            self.data = np.load(self.hparams.ds_name + '.npz', allow_pickle=True)

        ret = []
        if not isinstance(indexs, list):
            _indexs = [indexs]
        else:
            _indexs = indexs
        for index in _indexs:
            data = self.data[self.audio_and_text_keys[index]].item()
            text = np.array(data['text'], np.int)

            if self.hparams.multispeaker:
                spk_id = data.get('spk_id', 0)
            else:
                spk_id = 0

            if self.hparams.add_sil == 2:
                text = text.reshape(-1, 3 if not self.hparams.use_pinyin else 2)
                text = np.concatenate([text, 2 * np.ones([text.shape[0], 1], np.int)], -1)  # [L, 4]
                text = text.reshape(-1)  # [L + L//3]
            elif self.hparams.add_sil == 3:
                text = np.stack([text, 128 + text, 256 + text], -1)  # [L, 3]
                text = text.reshape(-1, 9 if not self.hparams.use_pinyin else 2)  # [L/3, 9]
                text = np.concatenate([text, 2 * np.ones([text.shape[0], 1], np.int)], -1)  # [L/3, 10]
                text = text.reshape(-1)  # [10L/3]

            text = torch.from_numpy(text)
            mel = torch.from_numpy(np.array(data['mels']).reshape(-1, 80).T)
            if self.hparams.use_linear or self.hparams.linear_directly:
                linear = torch.from_numpy(np.array(data['linear']).reshape(-1, self.hparams.num_freq).T)
            else:
                linear = None

            if self.hparams.speech and self.type != 'val':
                mel = mel[:, :1550]
                text = text[:350]
                if linear:
                    linear = linear[:1550]

            pitch = None
            if self.hparams.use_pitch:
                pitch_key = 'pitches' if not self.hparams.use_smooth_pitch else 'smooth_pitches'
                pitch = torch.from_numpy(np.array(data[pitch_key], np.int))

            utt_ids = torch.from_numpy(np.array(data['utt_id'], np.int))
            if self.hparams.prefix_len > 0:
                text_len = int(self.hparams.prefix_len * text.shape[0] / mel.shape[1])
                text = text[:text_len]
                mel = mel[:, :self.hparams.prefix_len]
                pitch = pitch[:self.hparams.prefix_len]

            attn = None
            if self.hparams.use_ali or self.hparams.use_ali_mask:
                attn = np.zeros((mel.shape[1], text.shape[0]))
                if self.hparams.use_phoneme_align:
                    mel_splits = [int(x * self.hparams.audio_sample_rate / self.hparams.hop_size)
                                  for x in data['splits']]
                    last = 0
                    for t_idx, s in enumerate(mel_splits):
                        attn[last:s, t_idx] = 1
                else:
                    splits_begin = np.clip(np.array(data['splits'], np.int), 0, mel.shape[1] - 1)
                    splits_end = np.clip(np.array(data['splits_end'], np.int), 0, mel.shape[1] - 1)
                    splits_begin = [0] + list(splits_begin)
                    splits_end = [0] + list(splits_end)
                    if not self.hparams.use_ali_mask2:  # TODO: PINYIN?
                        if self.hparams.use_pinyin:
                            for i in range(text.shape[0] // 3):
                                splits_begin_step = (splits_begin[i + 1] - splits_begin[i] - 3) / 2
                                if self.hparams.attn_step_clip10:
                                    splits_begin_step = np.clip(splits_begin_step, 0, 10)
                                attn[int(splits_begin[i]):
                                     int(splits_begin[i] + splits_begin_step), i * 3] += 1
                                attn[int(splits_begin[i] + splits_begin_step):
                                     int(splits_begin[i] + splits_begin_step * 2), i * 3 + 1] += 1
                                attn[int(splits_begin[i + 1]) - 3:
                                     int(splits_begin[i + 1]), i * 3 + 2] += 1
                        else:
                            if self.hparams.add_sil == 0:
                                for i in range(text.shape[0] // 3):
                                    splits_begin_step = (splits_begin[i + 1] - splits_begin[i]) / 3
                                    splits_end_step = (splits_end[i + 1] - splits_end[i]) / 3
                                    if self.hparams.attn_step_clip10:
                                        splits_begin_step = np.clip(splits_begin_step, 0, 10)
                                        splits_end_step = np.clip(splits_end_step, 0, 10)
                                    attn[int(splits_begin[i]):
                                         int(splits_begin[i] + splits_begin_step), i * 3] += 0.5

                                    attn[int(splits_begin[i] + splits_begin_step):
                                         int(splits_begin[i] + splits_begin_step * 2), i * 3 + 1] += 0.5

                                    attn[int(splits_begin[i] + splits_begin_step * 2):
                                         int(splits_begin[i + 1]), i * 3 + 2] += 0.5

                                    attn[int(splits_end[i]):
                                         int(splits_end[i] + splits_end_step), i * 3] += 0.5

                                    attn[int(splits_end[i] + splits_end_step):
                                         int(splits_end[i] + splits_end_step * 2), i * 3 + 1] += 0.5

                                    attn[int(splits_end[i] + splits_end_step * 2):
                                         int(splits_end[i + 1]), i * 3 + 2] += 0.5
                            elif self.hparams.add_sil == 2:
                                for i in range(text.shape[0] // 4):
                                    splits_begin_step = (splits_begin[i + 1] - splits_begin[i] - 3) / 3
                                    splits_end_step = (splits_end[i + 1] - splits_end[i] - 3) / 3
                                    if self.hparams.attn_step_clip10:
                                        splits_begin_step = np.clip(splits_begin_step, 0, 10)
                                        splits_end_step = np.clip(splits_end_step, 0, 10)

                                    attn[int(splits_begin[i]):
                                         int(splits_begin[i] + splits_begin_step), i * 4] += 0.5
                                    attn[int(splits_begin[i] + splits_begin_step):
                                         int(splits_begin[i] + splits_begin_step * 2), i * 4 + 1] += 0.5
                                    attn[int(splits_begin[i] + splits_begin_step * 2):
                                         int(splits_begin[i + 1]) - 3, i * 4 + 2] += 0.5
                                    attn[int(splits_begin[i + 1]) - 3:
                                         int(splits_begin[i + 1]), i * 4 + 3] += 0.5

                                    attn[int(splits_end[i]):
                                         int(splits_end[i] + splits_end_step), i * 4] += 0.5
                                    attn[int(splits_end[i] + splits_end_step):
                                         int(splits_end[i] + splits_end_step * 2), i * 4 + 1] += 0.5
                                    attn[int(splits_end[i] + splits_end_step * 2):
                                         int(splits_end[i + 1]) - 3, i * 4 + 2] += 0.5
                                    attn[int(splits_end[i + 1]) - 3:
                                         int(splits_end[i + 1]), i * 4 + 3] += 0.5
                    else:
                        for i in range(text.shape[0] // 3):
                            attn[int(splits_begin[i]): int(splits_begin[i + 1]), i * 3:(i + 1) * 3] = 1
                            attn[int(splits_end[i]): int(splits_end[i + 1]), i * 3:(i + 1) * 3] = 1
                attn = torch.from_numpy(attn)

            if self.hparams.use_wavenet:
                wav = torch.from_numpy(np.array(data['raw_wav']))
                max_time_steps = self.hparams.wavenet_max_time
                if wav.shape[0] > max_time_steps:
                    max_time_frames = max_time_steps // audio.get_hop_size(self.hparams)
                    start_cond_idx = torch.randint(mel.shape[1] - max_time_frames, [])
                else:
                    start_cond_idx = 0
            else:
                wav = None
                start_cond_idx = None
            if self.hparams.linear_directly:
                mel = linear
            ret.append([text, mel, pitch, utt_ids, attn, linear, spk_id, wav, start_cond_idx])
        if not isinstance(indexs, list):
            return ret[0]
        else:
            return ret


    def __len__(self):
        return len(self.audio_and_text_keys)

    def get_lengths(self):
        data = np.load(self.hparams.ds_name + '.npz', allow_pickle=True)
        length_filename = self.hparams.ds_name + '_lengths.npy'
        if os.path.exists(length_filename):
            all_lengths = np.load(length_filename, allow_pickle=True).item()
            for i in range(min(10, len(self.audio_and_text_keys))):
                k = self.audio_and_text_keys[i]
                assert len(data[k].item()['mels']) // 80 == all_lengths[k], (
                    len(data[k].item()['mels']), all_lengths[k])
            return [all_lengths[k] for k in self.audio_and_text_keys]
        else:
            print("Load Training set length..")
            data = np.load(self.hparams.ds_name + '.npz', allow_pickle=True)
            return [len(data[i].item()['pitches']) for i in tqdm(self.audio_and_text_keys)]


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step, hparams):
        self.hparams = hparams
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        if not isinstance(batch[0][0], torch.Tensor):
            batch = batch[0]

        if self.hparams.reverse or self.hparams.attn_at_first:
            _, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([x[1].shape[1] for x in batch]),
                dim=0, descending=True)
        else:
            _, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x[0]) for x in batch]),
                dim=0, descending=True)

        max_input_len = max([x[0].size(0) for x in batch])
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        input_lengths = torch.LongTensor(len(batch))

        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        num_mels = batch[0][1].size(0)
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        utt_ids = torch.LongTensor(len(batch))
        utt_ids.fill_(-1)
        spk_ids = torch.LongTensor(len(batch))
        spk_ids.fill_(0)


        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            input_lengths[i] = text.size(0)
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            utt_ids[i] = batch[ids_sorted_decreasing[i]][3][0]
            spk_ids[i] = batch[ids_sorted_decreasing[i]][6]

        pitch_padded = None
        if self.hparams.use_pitch:
            pitch_padded = torch.LongTensor(len(batch), max_target_len)
            pitch_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                pitch = batch[ids_sorted_decreasing[i]][2]
                pitch_padded[i, :pitch.size(0)] = pitch

        attns = None
        if self.hparams.use_ali:
            attns = torch.FloatTensor(len(batch), max_target_len, max_input_len)
            attns.fill_(0)
            for i in range(len(ids_sorted_decreasing)):
                attns[i, :output_lengths[i], :input_lengths[i]] = batch[ids_sorted_decreasing[i]][4]

        linear_padded = None
        for i in range(len(ids_sorted_decreasing)):
            linear_padded = torch.FloatTensor(len(batch), self.hparams.num_freq, max_target_len)
            linear_padded.zero_()
            if self.hparams.use_linear:
                linear = batch[ids_sorted_decreasing[i]][5]
                linear_padded[i, :, :linear.shape[1]] = linear

        wavs = None
        start_cond_idxs = None
        if self.hparams.use_wavenet:
            max_wav_len = max([x[7].size(0) for x in batch])
            wavs = torch.FloatTensor(len(batch), max_wav_len)
            wavs.fill_(0)
            start_cond_idxs = torch.LongTensor(len(batch))
            start_cond_idxs.fill_(0)
            for i in range(len(ids_sorted_decreasing)):
                wav = batch[ids_sorted_decreasing[i]][7]
                wavs[i, :wav.size(0)] = wav
                start_cond_idxs[i] = batch[ids_sorted_decreasing[i]][8]

        if self.hparams.reverse:
            attn_ks = input_lengths.float() / ((output_lengths - 1) // 4 + 1).float()
        else:
            attn_ks = input_lengths.float() / output_lengths.float()

        return text_padded, input_lengths, mel_padded, output_lengths, pitch_padded, \
               attn_ks, spk_ids, utt_ids, attns, linear_padded, \
               wavs, start_cond_idxs


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    def __init__(self, lengths, batch_size=15000):
        self.lengths = lengths
        _, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        self.fixed_batches = []
        self.indices_group_by_lengths = [[] for _ in range(21)]
        self.length_boundary = [x * 50 for x in range(1, 21)] + [100000]  # [50, ... 1000, 100000]

        now_bin_id = 0
        for i in self.sorted_indices:
            while lengths[i] > self.length_boundary[now_bin_id]:
                now_bin_id += 1
            self.indices_group_by_lengths[now_bin_id].append(i.item())

    def __iter__(self):
        all_indices = []
        for i in range(len(self.indices_group_by_lengths)):
            random.shuffle(self.indices_group_by_lengths[i])
            all_indices += self.indices_group_by_lengths[i]

        batches = []
        total_lengths = 0
        new_batches = []
        for i in all_indices:
            if self.lengths[i] + total_lengths > self.batch_size:
                new_batches.append(batches)
                batches = []
                total_lengths = 0
            batches.append(i)
            total_lengths += self.lengths[i]

        if len(batches) >= 1:
            new_batches.append(batches)

        random.shuffle(new_batches)
        return iter(new_batches)

    def __len__(self):
        return len(self.sorted_indices)
