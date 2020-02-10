import glob
import numpy as np
import sys
from dsp import *
import os

bits = 16
notebook_name = 'nb4'

SEARCH_INPUT_PATH = sys.argv[1]
QUERY_INPUT_PATH = sys.argv[2]
GROUNDTRUTH_LABEL_INPUT_PATH = sys.argv[3]
DATA_PATH = sys.argv[4]
os.makedirs(DATA_PATH, exist_ok=True)

def get_files(path) :
    filenames = []
    for filename in os.listdir(path):
        filenames += [path + filename]
    return filenames

search_wav_files = get_files(SEARCH_INPUT_PATH)
query_mfcc_files = get_files(QUERY_INPUT_PATH)

def convert_file(path) :
    wav = load_wav(path, encode=False)
    mel = melspectrogram(wav)
    return wav.astype(np.float32), mel.astype(np.int16)

SEARCH_WAVE_PATH = DATA_PATH + '/search_wav/'
SEARCH_MEL_PATH = DATA_PATH + '/search_mel/'
QUERY_MFCC_PATH = DATA_PATH + '/query_mfcc/'

os.makedirs(SEARCH_WAVE_PATH, exist_ok=True)
os.makedirs(SEARCH_MEL_PATH, exist_ok=True)
os.makedirs(QUERY_MFCC_PATH, exist_ok=True)

for i, s in enumerate(search_wav_files):
    wave, mel = convert_file(s)
    # save wav
    id = s.split('/')[-1].split('.')[0]
    np.save(SEARCH_WAVE_PATH + id, wave)
    np.save(SEARCH_MEL_PATH + id, mel)
    print("\r%i/%i", (i + 1, len(search_wav_files)), end='')

for j, q in enumerate(query_mfcc_files):
    mfcc = load_mfcc(q, encode=False).astype(np.float32)
    id = q.split('/')[-1].split('.')[0]
    np.save(QUERY_MFCC_PATH + id, mfcc)
    print("\r%i/%i", (j + 1, len(query_mfcc_files)), end='')