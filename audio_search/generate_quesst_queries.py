import sys
import os
import subprocess
import random
import ast


def main():
    wav_dir = sys.argv[1]
    query_dir = sys.argv[2]
    for filename in os.listdir(wav_dir):
        if filename.endswith(".wav"):
            wav_file_name = filename.split(".")[0] + ".wav"
            wav_file_path = os.path.join(wav_dir, wav_file_name)
            s = ['soxi', '-D', wav_file_path]
            duration = subprocess.check_output(s)
            duration = duration.decode('ascii')
            duration = int(ast.literal_eval(duration))
            query_filename = filename.split(".")[0] + "_query.wav"
            query_file_path = os.path.join(query_dir, query_filename)
            possible_duration = duration // 2
            start = random.randint(0, duration - possible_duration)
            end = random.randint(start + possible_duration, duration)
            s = ["sox", wav_file_path, query_file_path, "trim", str(start), str(end - start)]
            subprocess.check_output(s)

if __name__ == "__main__":
    main()
