import numpy as np
import matplotlib.pyplot as plt
from configparser import SafeConfigParser
import encasm.utils as utils


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
    def __init__(self, id: str, config: SafeConfigParser):
        """Initializes a new Petri dish from a SafeConfigParser object.

        Parameters:
            config (SafeConfigParser): The SafeConfigParser object
        """
        self.id = id
        self.config = config

        self.init_channels()
    
    @classmethod
    def from_config_file(cls, id: str, config_file: str):
        """Creates a new Petri dish from a config file.

        Parameters:
            config_file (str): The path to the config file
        """
        parser = SafeConfigParser()
        parser.read(config_file)
        return cls(id, parser["Environment"])
    
    @classmethod
    def from_env(cls, new_id: str, other_env: 'PetriDish'):
        """Creates a new Petri dish from an existing environment.

        Parameters:
            env (PetriDish): The environment to copy
        """
        env = cls(new_id, other_env.config)
        env.food = np.copy(other_env.food)
        env.life = np.copy(other_env.life)
        env.resv = np.copy(other_env.resv)
        return env


    def init_channels(self):
        """Initializes the channels of the environment from the config dictionary."""
        self.width = self.config.getint("width", 32)
        self.height = self.config.getint("height", 32)
        self.n_hidden = self.config.getint("n_hidden", 4)
        self.n_channels = self.n_hidden + 3
        self.food = np.zeros((self.width, self.height))
        self.life = np.zeros((self.width, self.height))
        self.resv = np.zeros((self.width, self.height))
        self.hidden = np.zeros((self.n_hidden, self.width, self.height))

    def generate_food(self):
        """Generates a Lévy dust distribution of food in the environment
            as specified by the config dictionary.
        """
        dust = utils.levy_dust(
            (self.width, self.height),
            self.config.getint("food_amt", 16),
            self.config.getfloat("alpha", 1),
            self.config.getfloat("beta", 1),
            self.config.getint("pad", 1)
        )
        self.food += utils.discretize_levy_dust(
            dust, (self.width, self.height), self.config.getint("pad", 1))

    def display(self, chs: list = ["food", "life", "resv"], cmaps: list = ["copper", "gray", "hot"], cols: int = 3):
        """Displays the specified channels of the environment.

        Parameters:
            chs (list): The list of channels to display
            cmaps (list): The list of colormaps to use for each channel
        """
        # Ensures each channel has a color, repeats the last color if necessary
        if len(cmaps) < len(chs):
            cmaps += [cmaps[-1]] * (len(chs) - len(cmaps))

        rows = int(np.ceil(len(chs) / cols))
        # Displays a grid of matshow subplots for each channel, with the specified colormap
        # each plot with a title of the channel name and a colorbar
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        for i, ch in enumerate(chs):
            # adds padding between plots to account for colorbars
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            ax = axs[i]
            ax.matshow(getattr(self, ch), cmap=cmaps[i])
            ax.set_title(self.id + ": " + ch)
            # Creates a colorbar for each subplot that is the same size as the subplot with padding
            fig.colorbar(ax.images[0], ax=ax, fraction=0.045)
            # fig.colorbar(ax.images[0], ax=ax, pad=0.01)



# rows = int(np.ceil(len(chs) / cols))
# fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
# for i, ch in enumerate(chs):
#     ax = axs[i // cols, i % cols]
#     ax.matshow(getattr(self, ch), cmap=cmaps[i])
#     ax.set_title(ch + " channel, " + self.id)
#     ax.colorbar()
        