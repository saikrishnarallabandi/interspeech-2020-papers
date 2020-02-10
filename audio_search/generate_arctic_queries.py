import sys
import os
import subprocess


def main():
    dur_dir = sys.argv[1]
    wav_dir = sys.argv[2]
    query_dir = sys.argv[3]
    for filename in os.listdir(dur_dir):
        if filename.endswith(".dur"):
            dur_file_path = os.path.join(dur_dir, filename)
            wav_file_name = filename.split(".")[0] + ".wav"
            wav_file_path = os.path.join(wav_dir, wav_file_name)
            with open(dur_file_path, "r") as f:
                for line in f.readlines():
                    word, start, duration = line.rstrip().split(" ")
                    query_filename = filename.split(".")[0] + "_" + word + ".wav"
                    query_file_path = os.path.join(query_dir, query_filename)
                    s = ["sox", wav_file_path, query_file_path, "trim", start, duration]
                    out = subprocess.check_output(s)


if __name__ == "__main__":
    main()
