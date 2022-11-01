import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
# import PIL.Image
# import PIL.ImageDraw
import numpy as np
# from IPython.display import Video, Image, HTML, clear_output
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import tensorflow as tf

from bcolors import bcolors


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


class VideoWriter:
    def __init__(self, filename=None, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0-a+rgb


# Creates a heat map image from a 2d numpy array
def grid_image(grid, cmap=None):
    norm = mpl.colors.Normalize(grid.min(), grid.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(grid)


def grid_plot(grid, cmap=None):
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.show()


def grid_plimage(grid, cmap=cm.hot):
    plt.clf()
    fig = plt.gcf()
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    fig = plt.gcf()
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # buf = io.BytesIO()
    # plt.savefig(buf)
    # buf.seek(0)
    return data


def gen_vid(frames, scale=None, fname="test.mp4"):
    if frames is None:
        print(bcolors.FAIL + "graphics.py: No Frames Found for Video" + bcolors.ENDC)
        return
    with VideoWriter(fname) as vid:
        for f in frames:
            vid.add(to_rgb(zoom(np.array(f), scale)))
    return fname


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
