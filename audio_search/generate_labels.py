import sys
import os

def main():
    query_dir = sys.argv[1]
    groundtruth_file = sys.argv[2]
    with open(groundtruth_file, "a") as f:
        for filename in os.listdir(query_dir):
            if filename.endswith(".wav"):
                wav_1, wav_2, query = (filename.split(".")[0]).split("_")
                wav = wav_1 + "_" + wav_2
                f.write(wav + "\t" + filename.split(".")[0] + "\t" + "1" + "\n")


if __name__ == "__main__":
    main()
