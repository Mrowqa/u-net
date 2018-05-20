import os
import random
import json
import itertools
import glob


def create_split_from_directory(data_dir, parts_cnt, small_valid_size=0):
    imgs_dir = os.path.join(data_dir, 'images')
    imgs = list(map(lambda fname: fname[:-4], os.listdir(imgs_dir)))
    assert len(imgs) > 0
    random.shuffle(imgs)
    dataset = {
        "data_dir": data_dir,
        "cnt": len(imgs),
        "small_valid": [],
        "parts": [],
    }
    small_valid_size = int(small_valid_size * len(imgs))
    dataset["small_valid"] = imgs[:small_valid_size]
    imgs = imgs[small_valid_size:]

    step = int((1.0 / parts_cnt) * len(imgs))
    for _ in range(parts_cnt - 1):
        dataset["parts"].append(imgs[:step])
        imgs = imgs[step:]
    dataset["parts"].append(imgs)
    return dataset


def get_evaluation_set(data_dir):
    paths = glob.glob(os.path.join(data_dir, '*'))
    return list([(p, os.path.split(p)[-1]) for p in paths])


def select_part_for_training(dataset, part_no):
    assert "parts" in dataset
    assert part_no < len(dataset["parts"])

    parts = dataset["parts"]
    valid = parts[part_no]
    train = list(itertools.chain(*(parts[:part_no] + parts[part_no+1:])))

    return {
        "data_dir": dataset["data_dir"],
        "cnt": dataset["cnt"],
        "small_valid": dataset["small_valid"],
        "train": train,
        "valid": valid,
    }


def build_full_paths(dataset, subset):
    def map_to_img_lbl(fname):
        return (os.path.join(dataset["data_dir"], "images", fname + ".jpg"),
                os.path.join(dataset["data_dir"], "labels_plain", fname + ".png"))
    return list([map_to_img_lbl(fname) for fname in dataset[subset]])


def dump_to_file(dataset, filename):
    data = json.dumps(dataset)
    with open(filename, 'w') as f:
        f.write(data)


def load_from_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
        return json.loads(data)
