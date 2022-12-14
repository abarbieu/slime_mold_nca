import re
import os
import configparser
import numpy as np
from PIL import Image
from encasm.env import PetriDish
import matplotlib.pyplot as plt


def get_env_shape(folder):
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
    img = Image.open(img)
    img = np.asarray(img)
    mask = img[..., -1] != 0
    a = np.zeros(mask.shape)
    a[mask] = 1
    return a


def gen_env_dict(folder, config):
    # Loads folder into a dictonary of environments of the given shape

    patt = "F_(\d+)(-L_(\d+))?.png"
    food_maps = {}  # food map index -> food map filename
    life_maps = {}  # food map index -> (life map index -> filename)

    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.name.endswith(b".png"):
            mtch = re.search(patt, str(entry.name))
            if mtch is not None:
                grps = mtch.groups()
                # If food map index is not in the lifemap dictionary, add it
                if grps[0] not in life_maps:
                    life_maps[grps[0]] = {}
                # If the life map index is not None, add its contents to the life map dictionary
                if grps[2] is not None:
                    life_maps[grps[0]][grps[2]] = entry.path.decode()
                # If the food map index is not None, add its contents to the food map dictionary
                elif grps[0] is not None:
                    if grps[0] not in food_maps:
                        food_maps[grps[0]] = entry.path.decode()
    envs = {}
    for fmap_k in food_maps.keys():
        envs[fmap_k] = {}
        for lmap_k in life_maps[fmap_k].keys():
            env = PetriDish(
                id=f"test_F_{fmap_k}-L_{lmap_k}", config=config["Environment"])
            env.food = img_to_grid(food_maps[fmap_k])
            env.life = img_to_grid(life_maps[fmap_k][lmap_k])
            envs[fmap_k][lmap_k] = env
    return envs


def load_tests(folder):
    # Walks through the subfolders, gathers their metadata, and adds their environments
    # to a dictionary whose key is the metadata title
    tests = {}
    for entry in os.scandir(path=os.fsencode(folder)):
        if entry.is_dir():
            config = get_env_shape(entry.path.decode())

            tests[entry.name.decode()] = gen_env_dict(
                entry.path.decode(), config)
    return tests
