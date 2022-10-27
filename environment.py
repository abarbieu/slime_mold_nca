import numpy as np

from scipy.stats import uniform
from scipy.stats import levy_stable


class environment:
    # ----- Neighborhood/Channel Parameters -----
    moore_f = (np.array([1,  1,  1,  0,  0,  -1, -1, -1]),
               np.array([-1,  0,  1, -1,  1,  -1,  0,  1]))  # Moore neigh indices

    von_n = (np.array([-1,  0,  0, 1]),
             np.array([0, -1,  1,  0]))  # Von Neumann neighborhood indices

    moore_f = (np.array([1,  1,  1,  0,  0,  0,  -1, -1, -1]),
               np.array([-1,  0,  1, -1,  0,  1,  -1,  0,  1]))  # Includes center

    von_n_f = (np.array([-1,  0,  0,  0, 1]),
               np.array([0, -1,  0,  1,  0]))  # Includes center

    def __init__(self):
        self.life_i = 0  # Channel index of life
        self.food_i = 1  # Channel index of food
        self.resv_i = 2  # Channel index of resevoir
        self.hidden_i = 3  # Index of first hidden channel

        self.kernel = self.von_n
        self.kernel_full = self.von_n_f  # Incliudes center
        self.n_neighs = len(self.kernel[0])
        self.n_hidden = 4
        self.n_channels = 3 + self.n_hidden

        self.esize = 32  # Width/Height of environment
        self.eshape = (self.n_channels, self.esize, self.esize)

        self.alpha = 1  # Alpha for levy food distribution
        self.beta = 1  # Beta for levy food distribution
        self.food_amt = 16  # Number of food points created
        self.max_food = 4  # Densest food source -- necessary?

        # TODO: explore larger perceptive fields? sobel filters?
        # Center + neighs, all channels
        self.input_shape = (1 + self.n_neighs) * self.n_channels
        # Draw from neighs, hidden channels for center
        self.output_shape = self.n_neighs + self.n_hidden

        self.life_cost = 0.3  # How much life is depleted per time step
        self.min_life = 0.1  # Life below this is treated as dead
        self.life_transfer_rate = 0.2  # Amount of food transferrable by 1 unit of life
        self.food_transfer_rate = 1  # Amount of life transferrable by 1 unit of food
        self.max_life = 1

        # put x in range -w/2 to w/2

        def norm_center(x, w):
            x -= x.min()
            x *= w/x.max()
            return x

        def get_levy_dust(shape: tuple, points: int, alpha: float, beta: float) -> np.array:
            # uniformly distributed angles
            angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi)

            # Levy distributed step length
            r = abs(levy_stable.rvs(alpha, beta, size=points))

            x = norm_center(np.cumsum(r * np.cos(angle)), shape[0]-1)
            y = norm_center(np.cumsum(r * np.sin(angle)), shape[1]-1)

            return np.array([x, y])

        def gen_padded_env(pad=1) -> None:
            shape = (self.eshape[1], self.eshape[2])
            env = np.zeros(self.eshape)

            dust = get_levy_dust(
                (shape[0]-pad*2, shape[1]-pad*2), self.food_amt, self.alpha, self.beta).T
            dust = np.array(dust, dtype=np.int64)
            dust, density = np.unique(dust, axis=0, return_counts=True)

            env[self.food_i, dust[:, 0]+1, dust[:, 1]+1] = density
            return env

        # Innoculate the cell with the most food - this is likely to be surrounded
        # by more food given def of levy dust

        def innoculate_env(env):
            maxfood = np.unravel_index(
                np.argmax(env[self.food_i]), env[self.food_i].shape)
            env[self.life_i][maxfood[0], maxfood[1]] = 1
            return env
