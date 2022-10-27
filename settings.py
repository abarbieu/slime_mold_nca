import numpy as np

class settings:
    # ----- Neighborhood/Channel Parameters ----- 
    moore_f = (np.array([ 1,  1,  1,  0,  0,  -1, -1, -1]),
               np.array([-1,  0,  1, -1,  1,  -1,  0,  1])) # Moore neigh indices

    von_n = (np.array([-1,  0,  0, 1]),
             np.array([ 0, -1,  1,  0])) # Von Neumann neighborhood indices

    moore_f = (np.array([ 1,  1,  1,  0,  0,  0,  -1, -1, -1]),
               np.array([-1,  0,  1, -1,  0,  1,  -1,  0,  1])) # Includes center

    von_n_f = (np.array([-1,  0,  0,  0, 1]),
               np.array([ 0, -1,  0,  1,  0])) # Includes center

    def __init__(self):

        self.life_i = 0 # Channel index of life
        self.food_i = 1 # Channel index of food
        self.resv_i = 2 # Channel index of resevoir
        self.hidden_i = 3 # Index of first hidden channel

        self.kernel = von_n
        self.kernel_full = von_n_f # Incliudes center
        self.n_neighs = len(kernel[0])
        self.n_hidden = 4
        self.n_channels = 3 + n_hidden

        self.esize = 32 # Width/Height of environment
        self.eshape = (self.n_channels, self.esize, self.esize)
        
        self.alpha = 1 # Alpha for levy food distribution
        self.beta = 1 # Beta for levy food distribution
        self.food_amt = 16 # Number of food points created
        self.max_food = 4 # Densest food source -- necessary?

        self.input_shape = (1 + self.n_neighs) * self.n_channels # Center + neighs, all channels 
        self.output_shape = self.n_neighs + self.n_hidden # Draw from neighs, hidden channels for center

        self.life_cost = 0.3 # How much life is depleted per time step
        self.min_life = 0.1 # Life below this is treated as dead
        self.life_transfer_rate = 0.2 # Amount of food transferrable by 1 unit of life
        self.food_transfer_rate = 1 # Amount of life transferrable by 1 unit of food
        self.max_life = 1
        