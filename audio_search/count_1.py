import sys


def main():
    file = sys.argv[1]
    count = 0
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            label = line.rstrip().split('\t')[2]
            if int(label) == 1:
                count += 1
    print(count)


if __name__ == "__main__":
    main()
