import numpy as np
from PIL import Image
import os
import sys


def score(f1, f2):
    im1 = np.array(Image.open(f1))
    im2 = np.array(Image.open(f2))
    return np.mean(im1 == im2)


def main():
    labels = R"D:\students\dnn\assignment2\training\labels_plain"
    if len(sys.argv) != 2:
        print("usage: {} <dir>".format(sys.argv[0]))
        exit(1)
    d = sys.argv[1]
    files = os.listdir(d)
    scores = []
    files_total = len(files)
    for i, f in enumerate(files):
        print("{} / {} \r".format(i, files_total), end='')
        f1 = os.path.join(labels, f)
        f2 = os.path.join(d, f)
        sc = score(f1, f2)
        scores.append(sc)
    print("Mean of {} labels: {} ".format(len(scores), np.mean(scores)))


if __name__ == "__main__":
    main()
