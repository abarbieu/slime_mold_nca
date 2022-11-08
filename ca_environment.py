from operator import ge
from re import A
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import uniform
from scipy.stats import levy_stable
from bcolors import bcolors

from graphics import grid_image, gen_vid


class CAEnvironment:
    food_i = 0  # Channel index of food
    life_i = 1  # Channel index of life
    hidden_i = 2  # Index of first hidden channel

    n_hidden = 4
    n_channels = 2 + n_hidden

    esize = 32  # Width/Height of environment
    pad = 1
    cutsize = esize-2*pad
    eshape = (n_channels, esize, esize)
    channels = np.zeros(eshape)

    alpha = 1  # Alpha for levy food distribution
    beta = 1  # Beta for levy food distribution
    food_amt = 16  # Number of food points created
    max_food = 4  # Densest food source -- necessary?

    # TODO: explore larger perceptive fields? sobel filters?
    # Center + neighs, all channels
    # input_shape = (1 + n_neighs) * n_channels
    # Draw from neighs, hidden channels for center
    # output_shape = n_neighs + n_hidden

    # life_cost = 0.3  # How much life is depleted per time step
    # min_life = 0.1  # Life below this is treated as dead
    # life_transfer_rate = 0.2  # Amount of food transferrable by 1 unit of life
    # food_transfer_rate = 1  # Amount of life transferrable by 1 unit of food
    # max_life = 1

    def __init__(self, id):
        self.id = id

    vid_channels = (food_i, life_i)
    vid_cmaps = [None] * 2

    frames = None

    def update_shape(self, new_shape, n_hidden_chs=4):
        '''
        Shape is in form (n_channels, width, height)
        Resets environment
        '''
        if new_shape[1] != new_shape[2]:
            print(bcolors.WARNING +
                  "ca_environment.py:update_shape: Unknown behavior for non-square envs" + bcolors.ENDC)
        self.n_channels = new_shape[0]
        self.n_hidden = n_hidden_chs
        self.hidden_i = new_shape[0] - n_hidden_chs
        if self.hidden_i < 2:
            self.life_i = None
            if self.hidden_i == 0:
                self.food_i = None
            else:
                self.food_i = 0

        self.esize = new_shape[1]
        self.cutsize = self.esize-2*self.pad
        self.eshape = new_shape
        self.channels = np.zeros(self.eshape)

    def norm_center(self, x, w):
        x -= x.min()
        x *= w/x.max()
        return x

    def get_levy_dust(self, shape: tuple, points: int, alpha: float, beta: float) -> np.array:
        # uniformly distributed angles
        angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi)

        # Levy distributed step length
        r = abs(levy_stable.rvs(alpha, beta, size=points))
        # inds = r.argsort()
        # if abs(r[inds[-2]] - r[inds[-1]])/abs(r[inds[-2]] - r[inds[-3]]) > 10:
        #     r[inds[-1]] = r[inds[-2]] * 2

        x = np.cumsum(r * np.cos(angle)) % (shape[0]-1)
        y = np.cumsum(r * np.sin(angle)) % (shape[1]-1)

        return np.array([x, y])

    def gen_padded_food(self) -> np.array:
        dust = self.get_levy_dust(
            (self.esize-self.pad*2, self.esize-self.pad*2), self.food_amt, self.alpha, self.beta).T
        dust = np.array(dust, dtype=np.int64)
        dust, density = np.unique(dust, axis=0, return_counts=True)

        self.channels[self.food_i, dust[:, 0]+1, dust[:, 1]+1] = density
        return self.channels

    def img_to_grid(self, image):
        img = Image.open(image)
        img = np.asarray(img)
        mask = img[..., -1] != 0
        a = np.zeros(mask.shape)
        a[mask] = 1
        return a

    def set_channel(self, i, grid):
        if isinstance(grid, str):
            grid = self.img_to_grid(grid)
        self.channels[i] = grid

    # Innoculate the cell with the most food - this is likely to be surrounded
    # by more food given def of levy dust

    def innoculate(self):
        maxfood = np.unravel_index(
            np.argmax(self.channels[self.food_i]), self.channels[self.food_i].shape)

        self.channels[self.life_i][maxfood[0], maxfood[1]] = 1

    def update_chunk(self, i, j, update):
        self.channels[:, i, j] = update
    #     if neighs[LIFE].sum() > 0 or center[LIFE] > 0:
    #         center[HIDDEN:] = desires[N_NEIGHS:]
    #         center[LIFE] += con.food_transfer_rate * \
    #             center[FOOD]  # Life appears on nearby food
    #         center[LIFE] = min(center[LIFE], con.max_life)

    #         if center[LIFE] > con.min_life or sum(desires[:N_NEIGHS]) > 0.1:
    #             # Neighbor's life will change by at most what they can transfer
    #         delta_neigh = np.minimum(
    #             con.life_transfer_rate * neighs[LIFE], desires[:N_NEIGHS])
    #         delta_center = sum(delta_neigh)
    #         # A cell can only hold 1 unit of life
    #         adj = min(delta_center, (con.max_life-center[LIFE]))/delta_center
    #         neighs[LIFE] -= delta_neigh * adj
    #         center[LIFE] += delta_center * adj

    #         center[LIFE] -= center[LIFE] * con.life_cost
    #         center[LIFE] = min(center[LIFE], con.max_life)
    #         if center[LIFE] <= con.min_life:
    #         center[LIFE] = 0
        # apply_physics(env[:, i, j], env[:, KERNEL[0]+i,
        #           KERNEL[1]+j], desires, constraints=constraints)

    def start_new_video(self, channels=None, cmaps=None):
        if channels is not None:
            if len(channels) > 4 or len(channels) < 1:
                print(
                    bcolors.FAIL + "ca_environment.py:start_new_video: Must provide 1-4 or None channels, default is (food_i, life_i)" + bcolors.ENDC)
                return
            self.vid_channels = channels

        if cmaps is not None:
            if len(cmaps) != len(self.vid_channels):
                print(
                    bcolors.FAIL + "ca_environment.py:start_new_video: cmaps length must be same as vid_channels: " + self.vid_channels + "" + bcolors.ENDC)
                return
            self.vid_cmaps = cmaps
        else:
            self.vid_cmaps = [None] * len(self.vid_channels)
        nvc = len(self.vid_channels)
        self.frames = np.zeros(
            (1, self.esize * ((nvc > 2) + 1), self.esize * ((nvc > 1) + 1), 4))

    def add_state_to_video(self):
        ltframe = ltframe = grid_image(
            self.channels[self.vid_channels[0]], cmap=self.vid_cmaps[0])
        if len(self.vid_channels) > 1:
            rtframe = grid_image(
                self.channels[self.vid_channels[1]], cmap=self.vid_cmaps[1])

        if len(self.vid_channels) > 2:
            lbframe = grid_image(
                self.channels[self.vid_channels[2]], cmap=self.vid_cmaps[2])
            ltframe = np.append(ltframe, lbframe, axis=0)

            if len(self.vid_channels) == 4:
                rbframe = grid_image(
                    self.channels[self.vid_channels[3]], cmap=self.vid_cmaps[3])
            else:
                rbframe = grid_image(
                    np.zeros((self.esize, self.esize)), cmap=None)

            rtframe = np.append(rtframe, rbframe, axis=0)

        if len(self.vid_channels) > 1:
            frame = np.append(ltframe, rtframe, axis=1)
        else:
            frame = ltframe

        if self.frames is None:
            self.frames = np.array([frame])
        else:
            self.frames = np.append(self.frames, [frame], axis=0)

    def display(self, channels=None, cmaps=None):
        if channels is None:
            channels = (0, 1)
        if cmaps is None:
            cmaps = (cm.copper, cm.gray)

        for i in range(len(channels)):
            fig, axs = plt.subplots(ncols=1, figsize=(12, 6))
            print(len(cmaps))
            if i >= len(cmaps):
                cmap = cm.gray
            else:
                cmap = cmaps[i]
            im = axs.matshow(self.channels[channels[i]], cmap=cmap)
            fig.colorbar(im, fraction=0.045, ax=axs)
            axs.set_title(self.id)

            # fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

            # ax1, ax2 = axes
            # food_im = ax1.matshow(self.channels[channels[0]], cmap=cmaps[0])
            # life_im = ax2.matshow(self.channels[channels[1]], cmap=cmaps[1])

            # ax1.set_title("Food")
            # ax2.set_title("Life")
            # fig.colorbar(food_im, fraction=0.045, ax=ax1)
            # fig.colorbar(life_im, fraction=0.045, ax=ax2)

    def gen_video(self, speed=1, scale=None, fname="test.mp4"):
        if scale is None:
            scale = 512/self.esize
        # for i in range(len(self.frames))
        return gen_vid(self.frames, scale, fname=fname)

        # fig, axs = plt.subplots(2, figsize=(6, 4))
        # axs[0].set_title("Food Channel")
        # axs[1].set_title("Life Channel")

        # axs[0].imshow(self.env[self.food_i], cmap=cm.copper,
        #               interpolation='nearest')
        # axs[0].colorbar()
        # axs[1].imshow(self.env[self.life_i], cmap=cm.gray,
        #               interpolation='nearest')
        # axs[1].colorbar()
        # plt.show()

        # def add_frame(frames, env):
        #     fframe = grid_image(self.env[], cmap=cm.copper)
        #     lframe = ch_image(env,LIFE, cmap=cm.gray)
        #     frame = np.append(fframe,lframe, axis=0)
        #     if frames is None:
        #         frames = np.array([frame])
        #     else:
        #         frames = np.append(frames, [frame], axis=0)
        #     return frames

        # def add_frames(frames, new_frames):
        #     if new_frames is not None and len(new_frames) > 0:
        #         if frames is None:
        #             frames = new_frames
        #         else:
        #             frames = np.append(frames, new_frames,axis=0)
        #     return frames

        # '''
        #     - Life increases from food first (anywhere in neigh?)
        #     - Increase ^ is proportional to food val
        #     - Life drains from neigh life otherwise
        #     - Life decreases by DEATH_RATE every time step
        #     - Only TRANSFER_RATE life can be drained from any one neighbor
        #     -
        # '''
        # def apply_update(coords, desires):

        #     if neighs[LIFE].sum() > 0 or center[LIFE] > 0:
        #         center[HIDDEN:] = desires[N_NEIGHS:]
        #         center[LIFE] += con.food_transfer_rate * \
        #             center[FOOD]  # Life appears on nearby food
        #         center[LIFE] = min(center[LIFE], con.max_life)

        #         if center[LIFE] > con.min_life or sum(desires[:N_NEIGHS]) > 0.1:
        #             # Neighbor's life will change by at most what they can transfer
        #         delta_neigh = np.minimum(
        #             con.life_transfer_rate * neighs[LIFE], desires[:N_NEIGHS])
        #         delta_center = sum(delta_neigh)
        #         # A cell can only hold 1 unit of life
        #         adj = min(delta_center, (con.max_life-center[LIFE]))/delta_center
        #         neighs[LIFE] -= delta_neigh * adj
        #         center[LIFE] += delta_center * adj

        #         center[LIFE] -= center[LIFE] * con.life_cost
        #         center[LIFE] = min(center[LIFE], con.max_life)
        #         if center[LIFE] <= con.min_life:
        #         center[LIFE] = 0
