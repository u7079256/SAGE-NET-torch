import cv2
import os.path as path
import numpy as np


def resize_tensor(tensor, new_size):
    """
    Resize a numeric input 3D tensor with opencv. Each channel is resized independently from the others.

    Parameters
    ----------
    tensor: ndarray
        Numeric 3D tensor of shape (channels, h, w)
    new_size: tuple
        Tuple (new_h, new_w)
    Returns
    -------
    new_tensor: ndarray
        Resized tensor having size (channels, new_h, new_w)
    """
    channels = tensor.shape[0]
    new_tensor = np.zeros(shape=(channels,) + new_size)
    for i in range(0, channels):
        new_tensor[i] = cv2.resize(tensor[i], dsize=new_size[::-1])

    return new_tensor


def crop_tensor(tensor, indexes):
    """
    Crop a numeric 3D input tensor.

    Parameters
    ----------
    tensor: ndarray
        Numeric 3D tensor of shape (channels, h, w)
    indexes: tuple
        Crop indexes following convention (h1, h2, w1, w2)
    Returns
    -------
    new_tensor: ndarray
        Cropped tensor having size (channels, h2-h1, w2-w1)
    """
    h1, h2, w1, w2 = indexes
    new_tensor = tensor[:, h1:h2, w1:w2].copy()

    return new_tensor


def read_image(img_path, channels_first, color=True, color_mode='BGR', dtype=np.float32, resize_dim=None):
    """
    Reads and returns an image as a numpy array
    Parameters
    ----------
    img_path : string
        Path of the input image
    channels_first: bool
        If True, channel dimension is moved in first position
    color: bool, optional
        If True, image is loaded in color: grayscale otherwise
    color_mode: "RGB", "BGR", optional
        Whether to load the color image in RGB or BGR format
    dtype: dtype, optional
        Array is casted to this data type before being returned
    resize_dim: tuple, optional
        Resize size following convention (new_h, new_w) - interpolation is linear
    Returns
    -------
    image : np.array
        Loaded Image as numpy array of type dtype
    """

    if not path.exists(img_path):
        raise ValueError('Provided path "{}" does NOT exist.'.format(img_path))

    image = cv2.imread(img_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)

    if color and color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize_dim is not None:
        image = cv2.resize(image, dsize=resize_dim[::-1], interpolation=cv2.INTER_LINEAR)

    if color and channels_first:
        image = np.transpose(image, (2, 0, 1))

    return image.astype(dtype)
