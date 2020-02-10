from torch.utils.data import Dataset
import torch
import os, sys
import random
import numpy as np
from dsp import *
import re

bits = 16
seq_len = hop_length * 5

class Paths:
    def __init__(self, name, data_dir, checkpoint_dir="model_checkpoints", output_dir="model_outputs"):
        self.name = name
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

    def model_path(self):
        return f'{self.checkpoint_dir}/{self.name}.pyt'

    def model_hist_path(self, step):
        return f'{self.checkpoint_dir}/{self.name}_{step}.pyt'

    def step_path(self):
        return f'{self.checkpoint_dir}/{self.name}_step.npy'

    def gen_path(self):
        return f'{self.output_dir}/{self.name}/'

    def logfile_path(self):
        return f'log/{self.name}'


class UnknownSpeakerDataset(Dataset):

    def __init__(self, path):
        print("path: %s" % path)
        self.path = path
        self.search_mels = [x for x in sorted(os.listdir(path + '/search_mel/'))]
        self.search_waves = [x for x in sorted(os.listdir(path + '/search_wav/'))]
        self.search_discretes = [x for x in sorted(os.listdir(path + '/search_discrete/'))]
        self.query_mfccs = [x for x in sorted(os.listdir(path + '/query_mfcc/'))]
        self.label_file = [x for x in os.listdir(path) if x.endswith("groundtruth.txt")][0]
        print("label file: ", self.label_file)
        self.labels = [[0 for _ in range(len(self.query_mfccs))] for _ in range(len(self.search_waves))]
        self.search_files = {self.search_waves[i].split('.')[0]: i for i in range(0, len(self.search_waves))}
        # self.search_discretes = {self.search_discretes[i].split('.')[0]: i for i in range(0, len(self.search_discretes))}
        self.query_files = {self.query_mfccs[i].split('.')[0]: i for i in range(0, len(self.query_mfccs))}
        with open(path + '/' + self.label_file, "r") as f:
            lines = f.readlines()
        c = 0
        for line in lines:
            l = line.rstrip().split('\t')
            if l[2] == '1':
                c += 1
            self.labels[self.search_files[l[0]]][self.query_files[l[1]]] = int(l[2])
        self.present = True
        print("Num positive labels: ", c)
        self.all_zeros = False

    def __getitem__(self, index):
        search_file_name = self.search_waves[index].split('.')[0]
        # to do equi class sampling
        if self.present:
            i = 0
            query_file_name = self.query_mfccs[i].split('.')[0]
            label = self.labels[self.search_files[search_file_name]][self.query_files[query_file_name]]
            while (label != 1) and i < len(self.query_mfccs)-1:
                i += 1
                query_file_name = self.query_mfccs[i].split('.')[0]
                label = int(self.labels[self.search_files[search_file_name]][self.query_files[query_file_name]])
                if label == 1:
                    self.present = False
                    break
            else:
                random_index = random.randint(0, len(self.query_mfccs) - 1)
                query_file_name = self.query_mfccs[random_index].split('.')[0]
                label = int(self.labels[self.search_files[search_file_name]][self.query_files[query_file_name]])
        else:
            random_index = random.randint(0, len(self.query_mfccs) - 1)
            query_file_name = self.query_mfccs[random_index].split('.')[0]
            label = int(self.labels[self.search_files[search_file_name]][self.query_files[query_file_name]])
            self.present = True

        # search_mel = np.load(f'{self.path}/search_mel/{search_file_name}.npy')
        search_wav = np.load(f'{self.path}/search_wav/{search_file_name}.npy')
        query_mfcc = np.load(f'{self.path}/query_mfcc/{query_file_name}.npy')

        return search_wav, query_mfcc.T, label

    def __len__(self):
        return len(self.search_waves)


def collate_unknownspeaker_samples(batch):
    search_wav = [x[0] for x in batch]
    #search_discrete = [x[1] for x in batch]
    query_mfcc = [x[1] for x in batch]
    label = [x[2] for x in batch]
    #search_filename = [x[4] for x in batch]
    #query_filename = [x[5] for x in batch]
    max_len_wav = max(i.shape[0] for i in search_wav)
    #max_len_discrete = max(i.shape[0] for i in search_discrete)
    max_len_mfcc = max(i.shape[0] for i in query_mfcc)
    search_wav16 = []
    # print("search wav length: ", len(search_wav))
    search_wav16.append([(x[0] + 0.5) / (2 ** 15 - 0.5) for x in search_wav])
    search_wav16 = np.array([_pad(wav, max_len_wav) for wav in search_wav], dtype=np.float32)
    #search_discrete16 = np.array([_pad_2d(discrete, max_len_discrete)
    query_mfcc16 = np.array([_pad_2d(mfcc, max_len_mfcc) for mfcc in query_mfcc], dtype=np.float32)
    return torch.FloatTensor(search_wav16), torch.FloatTensor(query_mfcc16), torch.LongTensor(label)
    # return torch.FloatTensor(search_wav16), torch.FloatTensor(search_discrete16), \
    #        torch.FloatTensor(query_mfcc16), torch.LongTensor(label), search_filename, query_filename


def _pad(seq, max_len):
    if len(seq) == max_len:
        return seq
    return np.pad(seq, (0, max_len - len(seq)), mode='constant')


def _pad_2d(seq, max_len):
    seq = np.pad(seq, [(0, max_len - len(seq)), (0, 0)], mode="constant")
    return seq


def default_paths(name, data_dir):
    return Paths(name, data_dir, checkpoint_dir="model_checkpoints", output_dir="model_outputs")


def remove_consecutive_duplicates(your_list):
    your_list = your_list.tolist()
    t = [v for i, v in enumerate(your_list) if i == 0 or v != your_list[i-1]]
    return np.array(t)


# def collate_samples(left_pad, window, right_pad, batch):
#     index_to_remove = []
#
#     print(f'collate: window={window}')
#     search_waves = [x[0] for x in batch]
#     shapes = [x[0].shape for x in search_waves]
#     print(f'shape[0] search waves={shapes}')
#     #sys.exit()
#
#     max_offsets = [x.shape[0] - window for x in search_waves]
#     print(f'max_offset={max_offsets}')
#     offsets = [np.random.randint(0, offset) for offset in max_offsets if offset>0]
#     search_wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[
#               offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(search_waves)]


# def collate(left_pad, mel_win, right_pad, batch):
#     max_offsets = [x[0].shape[-1] - mel_win for x in batch]
#     mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
#     sig_offsets = [offset * hop_length for offset in mel_offsets]
#     mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
#
#     wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x[1], np.zeros(right_pad, dtype=np.int16)])[
#               sig_offsets[i]:sig_offsets[i] + left_pad + 64 * mel_win + right_pad] for i, x in enumerate(batch)]
#
#     mels = np.stack(mels).astype(np.float32)
#     wave16 = np.stack(wave16).astype(np.int64) + 2 ** 15
#     coarse = wave16 // 256
#     fine = wave16 % 256
#
#     mels = torch.FloatTensor(mels)
#     coarse = torch.LongTensor(coarse)
#     fine = torch.LongTensor(fine)
#
#     coarse_f = coarse.float() / 127.5 - 1.
#     fine_f = fine.float() / 127.5 - 1.
#
#     return mels, coarse, fine, coarse_f, fine_f


def restore(path, model):
    model.load_state_dict(torch.load(path))
    match = re.search(r'_([0-9]+)\.pyt', path)
    if match:
        return int(match.group(1))
    step_path = re.sub(r'\.pyt', '_step.npy', path)
    return np.load(step_path)


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader
    DATA_PATH = 'vctk'
    with open(f'{DATA_PATH}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    dataset = UnknownSpeakerDataset(index, DATA_PATH)
    loader = DataLoader(dataset, batch_size=1)
    for x in loader:
        speaker_onehot, audio = x
