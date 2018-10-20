import numpy as np
from tqdm import tqdm

from scipy.optimize import minimize_scalar
from scipy.stats import linregress, f_oneway


def reverse_correlate(X, y):
    """
    Computes receptive fields via reverse correlation.

    Inputs are:
    X : float32 numpy array N x w x h x 1
    y : float32 numpy array N x n

    where N is the number of images. (w,h) is the dimension of each image and n is the number of neurons.
    """
    print('Subtracting stimulus mean', X.mean(), flush=True)
    X = X - X.mean()
    rf = np.zeros(X.shape[1:3] + (y.shape[1],))

    n = X.shape[0]
    print('Computing statistics (n=%i)' % n, flush=True)
    for xx, yy in tqdm(zip(X, y)):
        tmp = xx * yy[None, None, ...]
        rf += tmp / n

    return rf


def matisse_frames(rows, cols, upscale_factor, orientations, frames_per_orientation, coherence):
    """
    Generates a matrix frames x width x height x 1 of oriented noise sample (Matisse stimulus).

    Args:
        rows:           number of rows of the image frame
        cols:           number of columns of the image frame
        upscale_factor:  upscale factor (int)
        orientations:   orientations in degrees
        frames_per_orientation: number of frames per orientation
        coherence:      pi/ori_coherence = bandwidth of orientations

    Returns:

    """
    print('Generating Matisse with coherence', coherence, flush=True)
    ori = np.kron(orientations, np.ones(frames_per_orientation))

    M = np.stack([make_matisse(rows, cols, upscale_factor=upscale_factor, ori=o,
                               coherence=coherence) for o in ori], axis=0).astype(np.float32)
    M -= M.mean()
    M /= M.std()
    return M[..., None], ori


def make_matisse(rows, cols, upscale_factor=1, ori=0, coherence=1):
    """
    Generates a sample of oriented noise

    Args:
        rows:               number of rows of the resulting image
        cols:               number of cols of the resulting image
        upscale_factor=1:   scale the image by this integer
        ori=0:              orientation of oriention bias in degrees
        coherence=1:        1 means unoriented noise. pi/ori_coherence is bandwidth of orientations

    Returns:
        sample from matisse stimulus

    """
    img = np.random.randn(rows, cols)
    return upscale_and_orientation_bias(img, upscale_factor, ori, coherence)


def hann(q):
    """
    Circuar hanning mask with symmetric opposite lobes

    Args:
        q:      angles in radians

    Returns:
        hanning window
    """
    return (0.5 + 0.5 * np.cos(q)) * (np.abs(q) < np.pi)


def upscale_and_orientation_bias(img, factor, ori, coherence):
    """
    Performs fast resizing of the image by the given integer factor with
    gaussian interpolation.

    Args:
        img:        noise image
        factor:     scaling factor
        ori:        orientation of the orientation bias
        coherence:  1 means unoriented noise. pi/ori_coherence is bandwidth of orientations

    Returns:
        upscaled image with oriented noise
    """

    ori_mix = coherence > 1  # how much of orientation to mix in
    kernel_sigma = factor
    shift = int(np.round(factor / 2))
    img = upsample(img, factor, shift) * factor
    sz = img.shape
    fx, fy = np.meshgrid(np.arange(-sz[1] / 2, sz[1] / 2) * 2 * np.pi / sz[1],
                         np.arange(-sz[0] / 2, sz[0] / 2) * 2 * np.pi / sz[0])

    # interpolate using gaussian kernel with DC gain = 1
    fmask = np.exp(-(fy ** 2 + fx ** 2) * kernel_sigma ** 2 / 2)

    # apply orientation selectivity
    theta = np.mod(np.arctan2(fx, fy) + ori * np.pi / 180 + np.pi / 2, np.pi) - np.pi / 2
    fmask = np.fft.ifftshift(fmask * (1 - ori_mix + ori_mix * hann(theta * coherence)))

    img = np.fft.ifft2(fmask * np.fft.fft2(img)).real

    # contrast compensation for the effect of orientation selectivity
    return img * (1 + ori_mix * (np.sqrt(coherence) - 1))


def upsample(img, n, phase):
    """
    Upsample an image.

    Args:
        img: image
        n:   two element tuple or list that defines the upscaling factors
        phase: two element tuple or list that defines the phase at which the image is inserted into the bigger image

    Returns:

    """
    if not type(n) in (tuple, list):
        n = 2 * (n,)
    if not type(phase) in (tuple, list):
        phase = 2 * (phase,)

    rows, cols = img.shape
    ret = np.zeros((n[0] * rows, n[1] * cols))
    ret[phase[0]::n[0], phase[1]::n[1]] = img
    return ret


def shuffle_iter(data, n):
    for _ in range(n):
        yield np.random.permutation(data.ravel()).reshape(data.shape)


class VonMises:
    """
    Von Mises tuning curve class.

    """

    def __init__(self, preferred=1, amplitude=1, offset=1, kappa=1, **kwargs):
        w = np.array([offset, amplitude, kappa, preferred])
        self.w = w
        self.pvalue = None

    @staticmethod
    def _von_mises(phi, w, affine=True):
        f = np.exp(w[2] * (np.cos(2 * (phi - w[3])) - 1))
        if not affine:
            return f
        else:
            return w[0] + w[1] * f

    def __call__(self, phi, affine=True, degree=False):
        to_radians = 2 * np.pi / 360 if degree else 1
        return self._von_mises(phi * to_radians, self.w, affine=affine)

    def to_dict(self):
        """

        Returns: parameters as a dictionary

        """
        ret = {name: val for name, val in zip(['offset', 'amplitude', 'kappa', 'preferred'], self.w)}
        ret['width_at_half_max'] = self.width_at_half_max
        if self.pvalue is not None:
            ret['pvalue'] = self.pvalue
        return ret

    @staticmethod
    def _fit(phi, data):
        von_mises = VonMises._von_mises
        mu = data.mean(axis=0)

        def obj_phi0(phi0):
            w[3] = phi0
            # return ((data.mean(axis=0) - von_mises(phi, w, affine=True)) ** 2).mean()
            return ((data - von_mises(phi, w, affine=True)) ** 2).mean()

        def obj_kappa(kappa):
            w[2] = kappa
            return ((data - von_mises(phi, w, affine=True)) ** 2).mean()

        w = np.array([mu.min(), mu.max() - mu.min(), 1, phi[np.argmax(mu)]])
        x = np.tile(phi, (data.shape[0],))

        for _ in range(5):
            w[2] = minimize_scalar(obj_kappa, bounds=(0, None), method='golden').x
            w[3] = minimize_scalar(obj_phi0, bounds=(0, None), method='golden').x
            w[1], w[0], r_value, p_value, std_err = linregress(von_mises(x, w, affine=False), data.ravel())

        return w, ((data - von_mises(phi, w, affine=True)) ** 2).mean()

    @staticmethod
    def _map_fit(t):
        return VonMises._fit(*t)

    @staticmethod
    def significance(phi, data, bootstrap):
        v = np.exp(2j * phi) / np.sqrt(data.shape[1])
        q = np.abs(data.mean(axis=0) @ v)
        qdistr = []

        for dat in shuffle_iter(data, bootstrap):
            qdistr.append(np.abs(dat.mean(axis=0) @ v))
        return (np.array(qdistr) > q).mean()

    def fit(self, phi, data, bootstrap=10000):
        """
        Fits the tuning curve to data.

        Args:
            phi: angles in radians
            data: responses

        Returns:

        """
        w, err = self._fit(phi, data)

        if bootstrap > 0:
            self.pvalue = self.significance(phi, data, bootstrap)

        self.w = w
        return self

    @property
    def width_at_half_max(self):
        """
        Computes width at half the maximum from the current parameters.
        """
        kappa = self.w[2]
        return np.arccos(np.log((1 + np.exp(-2 * kappa)) / 2) / kappa + 1)
