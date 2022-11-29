import re
import os
import configparser
import numpy as np
from PIL import Image
from encasm.env import PetriDish
import matplotlib.pyplot as plt


terrain_types = {
    "F": "food",
    "L": "life",
    "P": "poison",
    "W": "water",
    "S": "sink",
}


def get_env_config(folder):
    # Parses the first .config file contained in the given folder,
    # returns a tuple of the environment shape and the config object
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.name.endswith(b".config"):
            config = configparser.ConfigParser()
            config.read(entry.path.decode())
            #(int(config["Environment"]["width"]), int(config["Environment"]["height"])),
            return config


def img_to_grid(img):
    # Converts an image to a grid of 0s and 1s
    img = np.asarray(Image.open(img))
    if len(img.shape) > 2:
        img = img[:, :, -1]
    mask = img != 0
    a = np.zeros(mask.shape)
    a[mask] = 1
    return a


def gen_env_dict(folder, config):
    # Loads folder into a dictonary of environments of the given shape

    # Pattern matches files like F_0-T_1.png where 01 would be the environment key, T would be the channel type of the image
    # These are returned in groups
    pattern = re.compile(r"F_(\d+)-([A-Z])_(\d+).png")
    envs = {}
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.name.endswith(b".png"):
            match = pattern.match(entry.name.decode())
            if match:
                env_key = match.group(1) + "_" + match.group(3)
                terrain_type = match.group(2)
                if env_key not in envs:
                    # Create a new environment with id env_key
                    envs[env_key] = PetriDish(env_key, config["Environment"])
                envs[env_key].set_channel(
                    terrain_types[terrain_type], img_to_grid(entry.path.decode()))
    return envs


def load_tests(folder, flat=False):
    # Walks through the subfolders, gathers their metadata, and adds their environments
    # to a dictionary whose key is the metadata title
    tests = {}
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.is_dir():
            config = get_env_config(entry.path.decode())
            tests[entry.name.decode()] = gen_env_dict(
                entry.path.decode(), config)

    if flat:
        # Flatten the dictionary
        flat_tests = {}
        for test in tests:
            for env in tests[test]:
                flat_tests[test + "_" + env] = tests[test][env]
        return flat_tests
    return tests
