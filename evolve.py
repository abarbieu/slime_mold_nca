import ca_environment as caenv
import ca_agent as caag
import nevopy as ne
import time
import numpy as np


def fitness_function(genome, env=None, log=False, vid_speed=10):
    env = caenv.CAEnvironment()
    env.update_shape((3 + 1, 128, 128), n_hidden_chs=3)
    chs = np.loadtxt(
        f"./test_envs/{int((time.time()*2)%100)}.txt", dtype=float)
    env.channels = chs.reshape(env.eshape)

    agent = caag.CAAgent(kernel="moore")

    def walk_func(chunk, foveal, env):
        return genome.process(np.append(chunk.flatten(), foveal.flatten()))

    agent.set_walk_func(walk_func)
    agent.foveal_size = 4
    agent.n_spatial_chs = 3
    return agent.apply_walk_to_env(env, max_steps=1000)


fake = caag.CAAgent(kernel="moore")
fake.foveal_size = 4
fake.n_spatial_chs = 3

population = ne.neat.NeatPopulation(
    size=100,                 # number of genomes in the population
    num_inputs=fake.n_walk_inputs(),  # number of input nodes in the genomes
    num_outputs=fake.n_walk_outputs(),   # number of output nodes in the genomes
)

history = population.evolve(generations=10,
                            fitness_function=fitness_function)

print(fitness_function(population.fittest()))
