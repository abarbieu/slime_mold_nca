import numpy as np
import ca_environment as caenv
import math
from bcolors import bcolors


class CAAgent:

    # For slime mold/traditional agenst
    apply_rules = None

    # For random walk agents
    apply_walk = None
    foveal_size = 4  # arbitrary for now
    n_spatial_chs = 3  # arbitrary for now

    # apply_rules must be in the form f()

    def __init__(self, kernel="von_n"):
        # ----- Neighborhood/Channel Parameters -----
        moore = (np.array([1,  1,  1,  0,  0,  -1, -1, -1]),
                 np.array([-1,  0,  1, -1,  1,  -1,  0,  1]))  # Moore neigh indices

        von_n = (np.array([-1,  0,  0, 1]),
                 np.array([0, -1,  1,  0]))  # Von Neumann neighborhood indices

        moore_f = (np.array([1,  1,  1,  0,  0,  0,  -1, -1, -1]),
                   np.array([-1,  0,  1, -1,  0,  1,  -1,  0,  1]))  # Includes center

        von_n_f = (np.array([-1,  0,  0,  0, 1]),
                   np.array([0, -1,  0,  1,  0]))  # Includes center

        if kernel == "moore":
            self.kernel = moore
            self.kernel_full = moore_f  # Incliudes center
        else:
            self.kernel = von_n
            self.kernel_full = von_n_f  # Incliudes center

        self.n_neighs = len(self.kernel[0])

    def set_rule_func(self, func):
        self.apply_rules = func

    def set_walk_func(self, func):
        self.apply_walk = func

    def n_walk_inputs(self):
        return len(self.kernel_full[0]) * (self.n_spatial_chs + 1) + self.foveal_size

    def n_walk_outputs(self):
        return 2 + self.n_spatial_chs + self.foveal_size

    # Stochastically applies agent to every alive cell
    # !! Rules must be independent, ie. application order doesn't matter
    def apply_to_env(self, env: caenv.CAEnvironment, log=False, vid_speed=10, dropout=0.5):
        if self.apply_rules is None:
            print(bcolors.WARNING +
                  "ca_agent.py:apply_to_env: Must set rule function before applying to an environment" + bcolors.ENDC)
            return

        if dropout <= 1:
            inds = np.random.choice(
                (env.cutsize)*(env.cutsize), (int)((env.cutsize)*(env.cutsize)*dropout))
        else:
            inds = np.arange(0, env.cutsize*env.cutsize)
        coords = np.unravel_index(inds, (env.cutsize, env.cutsize))

        total_steps = 0
        for i, j in zip(coords[0]+1, coords[1]+1):
            input = env.channels[:, self.kernel_full[0] + i,
                                 self.kernel_full[1]+j]
            if input[env.life_i].sum() > 0:
                desires = self.apply_rules(input.flatten(), env)
                env.update_chunk(i, j, desires)

                if log and vid_speed < 10 and total_steps % (math.pow(2, vid_speed)) == 0:
                    env.add_state_to_video()

                total_steps += 1
        if log:
            env.add_state_to_video()

    def apply_walk_to_env(self, env: caenv.CAEnvironment, max_steps=None, max_dist=None, log=False, vid_speed=10):
        '''
        A walk takes as input:
            A "foveal" set of memory channels (output of the NN)
            The current neighborhood, including any food cells and hidden channels stored by the agent
        And produces as ouptut:
            A horizontal vector for where to go next
            A vertical vector for where to go next
            The new foveal memory, passed to the next iteration
            A spatial memory to be stored in the center of the neighborhood
        The quality of a walk is judged by how much food it gathers in a set number of steps (or a set total distance travelled)
        '''
        if self.n_spatial_chs > env.n_hidden:
            print(bcolors.FAIL + "ca_agent.py:apply_walk_to_env: Must apply to an environment with n_hidden channels == agents n_spatial channels" + bcolors.ENDC)
        if self.apply_walk is None:
            print(bcolors.FAIL +
                  "ca_agent.py:apply_walk_to_env: Must set walk function before applying to an environment" + bcolors.ENDC)
            return
        if (max_steps is not None and max_dist is not None) or (max_steps is None and max_dist is None):
            print(bcolors.FAIL +
                  "ca_agent.py:apply_walk_to_env: Must specify either n_steps or tot_distance " + bcolors.ENDC)
            return

        coords = [np.random.randint(
            env.pad, env.cutsize), np.random.randint(env.pad, env.cutsize)]
        tot_steps = 0
        tot_dist = 0
        running = True
        foveal_mem = np.random.random(self.foveal_size)
        coords_hist = np.array([coords])
        food_count = 0
        while running:
            food_count += env.channels[env.food_i, coords[0], coords[1]]
            env.channels[env.food_i, coords[0], coords[1]] = 0

            input = env.channels[:, self.kernel_full[0] + coords[0],
                                 self.kernel_full[1] + coords[1]]

            output = self.apply_walk(input.flatten(), foveal_mem, env)

            env.channels[1:, coords[0], coords[1]
                         ] = output[2+self.foveal_size:]

            newx = int(min(env.eshape[1]-env.pad-2,
                           max(coords[0] + output[0], env.pad)))
            dx = newx - coords[0]

            newy = int(min(env.eshape[2]-env.pad-2,
                           max(coords[1] + output[1], env.pad)))
            dy = newy - coords[1]

            coords[0] += dx
            coords[1] += dy

            foveal_mem = np.array(output[2:self.foveal_size+2])
            if log:
                coords_hist = np.append(coords_hist, [coords], axis=0)

            if log and vid_speed < 10 and tot_steps % (math.pow(2, vid_speed)) == 0:
                env.add_state_to_video()

            tot_steps += 1
            if max_dist is not None:
                running = tot_dist < max_dist
            elif max_steps is not None:
                running = tot_steps < max_steps

        if log:
            env.add_state_to_video()
            return food_count, coords_hist

        return food_count

    # Vid speed of 10 generates a frame for every full application of an agent
    # def apply_agent_n(agent, env, n=10, apply_until_dead=False, gen_frames=False, vid_speed=10, constraints=constraints):
    #     frames = None
    #     i = 0
    #     fitness = 0
    #     running = True
    #     # tot_delta=0
    #     while running:
    #         temp_frames = apply_agent(
    #             agent, env, gen_frames, vid_speed, constraints=constraints)
    #         if gen_frames:
    #         frames = add_frames(frames, temp_frames)

    #         if apply_until_dead:
    #         running = env[LIFE].sum() > 0
    #         else:
    #         running = i < n
    #         i += 1
    #     return frames
