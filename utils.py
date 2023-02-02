import os.path

import cv2
import os.path as path
import numpy as np

import config
import io_helper


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



import numpy as np

# cityscapes dataset palette
palette = np.array([[128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [70, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32]], dtype='uint8')


def seg_to_colormap(seg, channels_first):
    """
    Function to turn segmentation PREDICTION (not probabilities) to colormap.
    :param seg: the prediction image, having shape (h,w)
    :param channels_first: if true, returns (c,h,w) rather than (h,w,c)
    :return: the colormap image, having shape (h,w,3)
    """
    h, w = seg.shape
    color_image = palette[seg.ravel()].reshape(h, w, 3)

    if channels_first:
        color_image = color_image.transpose(2, 0, 1)

    return color_image


def read_lines_from_file(filename):
    """
    Function to read lines from file
    :param filename: The text file to be read.
    :return: content: A list of strings
    """
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content


def get_branch_from_experiment_id(experiment_id):
    """
    Function to return model branch name given experiment_id.
    :param experiment_id: experiment id
    :return: a string among ['all','image','optical_flow','semseg']
    """

    assert isinstance(experiment_id, basestring), "Experiment ID must be a string."

    branch = None
    if experiment_id.lower().startswith('dreyeve'):
        branch = "all"
    elif experiment_id.lower().startswith('color'):
        branch = "image"
    elif experiment_id.lower().startswith('flow'):
        branch = "optical_flow"
    elif experiment_id.lower().startswith('segm'):
        branch = "semseg"

    return branch

def dreye_mean_frame(dir=config.dreyeve_dir):
    """
    We have mean frames corresponding to individual sequences. Therefore, just add them up and get the average.
    :param dir: Dataset dir
    """
    sub_name_list = list(range(1,75))
    w = 1920
    h = 1080
    sum_img = np.zeros((3,1080,1920))
    sub_dir_list = [os.path.join(os.path.join(dir,str(ele).zfill(2)),'mean_frame.png') for ele in sub_name_list]
    for img_dir in sub_dir_list:
        img = io_helper.read_image(img_dir,channels_first=True)
        assert img.shape == (3,1080,1920) , img.shape
        sum_img += img
    sum_img = sum_img.astype(np.float32)
    sum_img = sum_img/74
    io_helper.write_image("E:\DREYEVE_DATA\dreyeve_mean_frame.png",sum_img,channels_first=True)



if __name__ == '__main__':
    dreye_mean_frame()


