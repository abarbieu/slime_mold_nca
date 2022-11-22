import numpy as np
from configparser import SafeConfigParser


class PetriDish:
    """A Petri dish is a multi channeled container for populations of cells. 

    Attributes:
        esize (int): The size of each channel of the environment (W x H)
        channels (np.array): A 3d array (W x H x n_hidden + 3) of the following channels:
            0: food, the nutrient concentration of each position in the environment
            1: life, how alive each cell is (sparse)
            2: resv, the nutrients passing through each live cell's resevoir (resv ⊆ life)
            3+: hidden, any other channels that may be used to store info
        food (np.array): A reference to the food channel
        life (np.array): A reference to the life channel
        resv (np.array): A reference to the resv channel
        hidden (np.array): A reference to a list of hidden channels
        n_hidden (int): The number of hidden channels
        alpha (float): Parameter for levy dust distribution
        beta (float): Parameter for levy dust distribution
        food_amt (int): Total amount of food in the environment
        max_food (int): Maximum amount of food in a single cell

    """

    def __init__(self, id, config_file):
        """Initializes a new Petri dish from a config file.

        Args:
            config_file (str): The path to the config file

        """
        self.id = id
        self.config_file = config_file
        parser = SafeConfigParser()
        parser.read(config_file)
        self.config = parser["Environment"]

    def __init__(self, new_id, other_env):
        """Initializes a new Petri dish from an existing environment.

        Args:
            other_env (PetriDish): The environment to copy

        """
        self.id = new_id
        self.config_file = other_env.config_file
        self.channels = np.copy(other_env.channels)
        self.config = other_env.config
