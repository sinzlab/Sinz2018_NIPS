import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, zip_longest
import seaborn as sns


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def rename(rel, prefix='new_', exclude=[]):
    attrs = list(rel.heading.attributes.keys())
    original = [x for x in attrs if x not in exclude]
    keys = [k for k in exclude if k in attrs]
    name_map = {prefix+x: x for x in original}
    return rel.proj(*keys, **name_map)

def plot_images(df, prefixes, names=None, brain_area='V1', n_rows=15, order_by='pearson',
                panels=('normed_rf', 'normed_mei'), panel_names=('RF', 'MEI'), cmaps=('coolwarm', 'gray'),
                y_infos=('{prefix}test_corr', 'pearson'), save_path=None):
    if names is None:
        names = prefixes

    f = (df['brain_area'] == brain_area)
    area_data = df[f]
    area_data = area_data.sort_values(order_by, ascending=False)

    n_rows = min(n_rows, len(area_data))
    n_panels = len(panels)
    cols = len(prefixes) * n_panels;

    with sns.axes_style('white'):
        fig, axs = plt.subplots(n_rows, cols, figsize=(4 * cols, round(2 * n_cells)))
        st = fig.suptitle('MEIs on Shuffled {} dataset: {}'.format(brain_area, ', '.join(names)))
        [ax.set_xticks([]) for ax in axs.ravel()]
        [ax.set_yticks([]) for ax in axs.ravel()]

    for ax_row, (_, data_row), row_index in zip(axs, area_data.iterrows(), count()):
        for ax_group, prefix, name in zip(grouper(n_panels, ax_row), prefixes, names):
            for ax, panel, panel_name, y_info, cm in zip(ax_group, panels, panel_names, y_infos, cmaps):
                if row_index == 0:
                    ax.set_title('{}: {}'.format(panel_name, name))
                ax.imshow(data_row[prefix + panel].squeeze(), cmap=cm)
                if y_info is not None:
                    ax.set_ylabel('{:0.2f}%'.format(data_row[y_info.format(prefix=prefix)] * 100))

    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.98)
    st.set_fontsize(20)
    fig.subplots_adjust(top=0.95)
    if path is not None:
        fig.savefig(save_path)


def gen_gif(images, output_path, duration=5, scale=1, adj_single=False):
    h, w = images[0].shape
    imgsize = (w * scale, h * scale)
    images = np.stack([cv2.resize(img, imgsize) for img in images])

    axis = (1, 2) if adj_single else None
    images = images - images.min(axis=axis, keepdims=True)
    images = images / images.max(axis=axis, keepdims=True) * 255
    images = images.astype('uint8')

    single_duration = duration / len(images)
    if not output_path.endswith('.gif'):
        output_path += '.gif'
    imageio.mimsave(output_path, images, duration=single_duration)


def rescale_images(images, low=0, high=1, together=True):
    axis = None if together else (1, 2)
    images = images - images.min(axis=axis, keepdims=True)
    images = images / images.max(axis=axis, keepdims=True) * (high - low) + low
    return images

def scale_imagesize(images, scale=(2, 2)):
    h, w = images[0].shape
    imgsize = (w * scale[1], h * scale[0])
    return np.stack([cv2.resize(img, imgsize) for img in images])


def tile_images(images, rows, cols, vpad=0, hpad=0, normalize=False, base=0):
    n_images = len(images)
    assert rows * cols >= n_images
    h, w = images[0].shape

    total_image = np.zeros((h + (h + vpad) * (rows - 1), w + (w + hpad) * (cols - 1))) + base
    loc = product(range(rows), range(cols))
    for img, (i, j) in zip(images, loc):
        if normalize:
            img = rescale_images(img)
        voffset, hoffset = (h + vpad) * i, (w + hpad) * j
        total_image[voffset:voffset + h, hoffset:hoffset + w] = img
    return total_image

def repeat_frame(images, frame_pos=0, rep=4):
    parts = []
    if frame_pos < 0:
        frame_pos = len(images) + frame_pos

    if frame_pos > 0:
        parts.append(images[:frame_pos])
    parts.append(np.tile(images[frame_pos], (rep, 1, 1)))
    if frame_pos < len(images) - 1:
        parts.append(images[frame_pos+1:])
    return np.concatenate(parts)


def add_text(image, text, pos, fontsize=1, color=(0, 0, 0)):
    image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, text, pos, font, fontsize, color, 1, cv2.LINE_8)
    return image