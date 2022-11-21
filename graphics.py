import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import tensorflow as tf

import os


class VideoWriter:
    def __init__(self, filename=None, scale=None, fps=30.0, **kw):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.scale = scale
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)
        self.frames = np.array([])

    def add_img(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    # Creates a heat map image from a 2d numpy array and adds it to the video
    def add_grid(self, grid, scale=None, cmap="viridis"):
        if self.scale is None:
            if scale is None:
                # 512 is the default size of the video, grids smaller than this will be upscaled
                self.scale = 512/grid.shape[1]
            else:
                self.scale = scale
        norm = mpl.colors.Normalize(grid.min(), grid.max())
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        img = m.to_rgba(grid)
        self.add_img(self.to_rgb(self.zoom(np.array(img), scale)))

    def add_concat_grids(self, grids, scale=None, cols=3, cmaps=None):
        if cmaps is None:
            cmaps = ["viridis"]*len(grids)

        rows = (len(grids)-1)//cols+1
        h, w = grids[0].shape[:2]
        grid = np.zeros((h*rows, w*cols, 4))
        for i, (g, cmap) in enumerate(zip(grids, cmaps)):
            norm = mpl.colors.Normalize(g.min(), g.max())
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            grid[i//cols*h:(i//cols+1)*h, i %
                 cols*w:(i % cols+1)*w] = m.to_rgba(g)
            # self.to_alpha(
            #     self.zoom(self.to_rgb(m.to_rgba(g, cmap)), self.scale))
        if scale is None:
            # 512 is the default size of the video, grids smaller than this will be upscaled
            self.scale = 512/grid.shape[1]
        else:
            self.scale = scale
        self.add_img(self.to_rgb(self.zoom(grid, self.scale)))

    def to_alpha(self, x):
        return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

    def to_rgb(self, x):
        # assume rgb premultiplied by alpha
        rgb, a = x[..., :3], self.to_alpha(x)
        return 1.0-a+rgb

    def zoom(self, img, scale=4):
        img = np.repeat(img, scale, 0)
        img = np.repeat(img, scale, 1)
        return img

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


# def grid_plot(grid, cmap=None):
#     plt.imshow(grid, cmap=cmap, interpolation='nearest')
#     plt.colorbar()
#     plt.show()


# def grid_plimage(grid, cmap=cm.hot):
#     plt.clf()
#     fig = plt.gcf()
#     plt.imshow(grid, cmap=cmap, interpolation='nearest')
#     plt.colorbar()
#     fig = plt.gcf()
#     fig.canvas.draw()
#     # Now we can save it to a numpy array.
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     # buf = io.BytesIO()
#     # plt.savefig(buf)
#     # buf.seek(0)
#     return data


# def rprint(text):
#     sys.stdout.write("\r"+text)
#     sys.stdout.flush()

# def np2pil(a):
#   if a.dtype in [np.float32, np.float64]:
#     a = np.uint8(np.clip(a, 0, 1)*255)
#   return PIL.Image.fromarray(a)

# def imwrite(f, a, fmt=None):
#   a = np.asarray(a)
#   if isinstance(f, str):
#     fmt = f.rsplit('.',s 1)[-1].lower()
#     if fmt == 'jpg':
#       fmt = 'jpeg'
#     f = open(f, 'wb')
#   np2pil(a).save(f, fmt, quality=95)

# def imencode(a, fmt='jpeg'):
#   a = np.asarray(a)
#   if len(a.shape) == 3 and a.shape[-1] == 4:
#     fmt = 'png'
#   f = io.BytesIO()
#   imwrite(f, a, fmt)
#   return f.getvalue()

# def im2url(a, fmt='jpeg'):
#   encoded = imencode(a, fmt)
#   base64_byte_string = base64.b64encode(encoded).decode('ascii')
#   return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

# def imshow(a, fmt='jpeg'):
#   display(Image(data=imencode(a, fmt)))

# def tile2d(a, w=None):
#   a = np.asarray(a)
#   if w is None:
#     w = int(np.ceil(np.sqrt(len(a))))
#   th, tw = a.shape[1:3]
#   pad = (w-len(a))%w
#   a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
#   h = len(a)//w
#   a = a.reshape([h, w]+list(a.shape[1:]))
#   a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
#   return a
