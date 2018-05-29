import numpy as np
from scipy import stats
from PIL import Image
import sys
import json
import glob
import os
import multiprocessing


PROCESSES_CNT = 10


def main():
    if len(sys.argv) != 2:
        print("Usage: {} <json>".format(sys.argv[0]))
        exit(1)
    with open(sys.argv[1], "r") as f:
        conf = json.loads(f.read())
    pref = conf["prefix"]
    outdir = os.path.join(pref, conf["output_dir"])
    inputs = []
    for dglob in conf["inputs"]:
        dirs = glob.glob(os.path.join(pref, dglob))
        inputs.append([])
        for d in dirs:
            inputs[-1].extend(glob.glob(os.path.join(d, '*')))

    inputs_cnt = len(inputs)
    img_cnt = len(inputs[0])
    print("Found {} input directories.".format(inputs_cnt))
    print("Selected {} to ensemble.".format(img_cnt))
    print("Saving to {}.".format(outdir))
    os.makedirs(outdir, exist_ok=True)
    print("Using {} processes.".format(PROCESSES_CNT))
    pool = multiprocessing.Pool(PROCESSES_CNT)

    args = [(i, img_cnt, inputs_cnt, inputs, outdir) for i in range(img_cnt)]
    hists = pool.map(process_image, args)
    # hists = []
    # for i in range(img_cnt):
    #     print("{} / {} \r".format(i+1, img_cnt), end='')
    #     basename = os.path.basename(inputs[0][i])
    #     imgs = []
    #     for inp in inputs:
    #         found = list(filter(lambda x: x.endswith(basename), inp))
    #         if not found:
    #             print("\nLacking images for {}! Skipping.".format(basename))
    #             break
    #         if len(found) > 1:
    #             print("\nWarning: found multiple images {}. Choosing first one.".format(found))
    #         imgs.append(np.array(Image.open(found[0])))
    #     if len(imgs) != inputs_cnt:
    #         continue
    #     out_img_arr, h = stats.mode(imgs, axis=0)
    #     hists.append(h)
    #     out_path = os.path.join(outdir, basename)
    #     Image.fromarray(out_img_arr[0], mode="P").save(out_path)
    print("Done.")


def process_image(args):
    i, img_cnt, inputs_cnt, inputs, outdir = args
    print("{} / {} \r".format(i+1, img_cnt), end='')
    basename = os.path.basename(inputs[0][i])
    out_path = os.path.join(outdir, basename)
    if os.path.exists(out_path):
        print("\nPrediction exist {}. Skipping.".format(out_path))
        return
    imgs = []
    for inp in inputs:
        found = list(filter(lambda x: x.endswith(basename), inp))
        if not found:
            print("\nLacking images for {}! Skipping.".format(basename))
            break
        if len(found) > 1:
            print("\nWarning: found multiple images {}. Choosing first one.".format(found))
        imgs.append(np.array(Image.open(found[0])))
    if len(imgs) != inputs_cnt:
        return
    out_img_arr, h = stats.mode(imgs, axis=0)
    Image.fromarray(out_img_arr[0], mode="P").save(out_path)
    return h


if __name__ == "__main__":
    main()
