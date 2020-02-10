from __future__ import print_function

import numpy as np
import librosa
import sys
import os


def main():
    directory_path = sys.argv[1]
    audio_files = []
    i = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            print(file_path)
            audio_files.append(file_path)
            i += 1
            if i == 120:
                break
        else:
            continue
    mfcc_features = []
    print(audio_files)
    hop_length = 512
    for file in audio_files:
        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        output_file = (file.split('/')[-1]).split('.')[0] + '.mfcc'
        with open(output_file, 'w') as f:
            np.savetxt(f, mfcc)
        mfcc_features.append(mfcc)
        print(mfcc_features)

if __name__ == "__main__":
    main()
