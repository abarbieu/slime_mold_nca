'''
Evaluates the efficacy of an environment for transporting nutrients
Outline:
    - Pick a random patch that has life on it
    - 
'''
import re
import os
import ca_environment as caenv


def load_tests():
    mold_tests = os.fsencode("./mold-test-imgs/")
    patt = "F_(\d+)(-L_(\d+))?.png"
    food_maps = {}  # food map index -> food map filename
    life_maps = {}  # food map index -> (life map index -> filename)

    for entry in os.scandir(path=mold_tests):
        if entry.name.endswith(b".png"):
            mtch = re.search(patt, str(entry.name))
            if mtch is not None:
                grps = mtch.groups()
                if grps[0] not in life_maps:
                    life_maps[grps[0]] = {}
                if grps[2] is not None:
                    life_maps[grps[0]][grps[2]] = entry.path.decode()
                elif grps[0] is not None:
                    if grps[0] not in food_maps:
                        food_maps[grps[0]] = entry.path.decode()
    envs = {}
    for fmap_k in food_maps.keys():
        envs[fmap_k] = {}
        for lmap_i in life_maps[fmap_k].keys():
            env = caenv.CAEnvironment()
            env.update_shape((6, 32, 32))
            env.set_channel(env.food_i, food_maps[fmap_k])
            env.set_channel(env.life_i, life_maps[fmap_k][lmap_i])
            envs[fmap_k][lmap_i] = env
            env.display()
    return envs
