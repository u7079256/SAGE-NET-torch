import numpy as np
import cv2
import torch
import torch.nn as nn
import os
from os.path import join
from model import SaliencyBranch
import cv2
import numpy as np
from utils import read_image,resize_tensor


def blend_map(img, map, factor, colormap=cv2.COLORMAP_JET):
    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1 - factor),
                            gamma=0)

    return blend


def normalize_map(s_map):
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
    return 255.0 * norm_s_map


def load_dreyeve_sample(sequence_dir, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448):
    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    I_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    I_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    for fr in range(frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr

        x = read_image(join(sequence_dir, 'frames', '{}.jpg'.format(offset)),
                       channels_first=True, resize_dim=(h, w)) - mean_dreyeve_image

        I_s[0, :, fr, :, :] = resize_tensor(x, new_size=(h_s, w_s))

    I_ff[0, :, 0, :, :] = x

    return [I_ff, I_s, I_c]


if __name__ == '__main__':
    frames_per_seq, h, w = 16, 448, 448
    verbose = True

    model_path_bdda = "../pretrained_models/dreyeve/bdda_image_branch.pth"
    model_path_sage = "../pretrained_models/dreyeve/sage_image_branch.pth"

    dreyeve_dir = 'demo_images/'

    # load mean dreyeve image
    mean_dreyeve_image = read_image(join(dreyeve_dir, 'dreyeve_mean_frame.png'),
                                    channels_first=True, resize_dim=(h, w))

    image_branch_bdda = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=False, branch='image')
    #image_branch_bdda.load_state_dict(torch.load(model_path_bdda))  # load weights

    image_branch_sage = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=False, branch='image')
    #image_branch_sage.load_state_dict(torch.load(model_path_sage))  # load weights

    X = load_dreyeve_sample(sequence_dir=dreyeve_dir, sample=16, mean_dreyeve_image=mean_dreyeve_image,
                            frames_per_seq=frames_per_seq, h=h, w=w)
    tensor_X = []
    for ele in X:
        print(ele.shape)
        tensor_X.append(torch.tensor(ele,dtype=torch.float32))

    X = tensor_X #torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        Y_image_bdda = image_branch_bdda(X[0],X[1],X[2])  # predict on image
        Y_image_sage = image_branch_sage(X[0],X[1],X[2])  # predict on image
    #print('pred total',Y_image_bdda.shape)
    for ele in Y_image_bdda:
        print('ele of Y bdda',ele.shape)
    im = read_image(join(dreyeve_dir, 'demo_img.jpg'),
                                    channels_first=True)

    h, w = im.shape[1], im.shape[2]

    bdda_pred = Y_image_bdda[0].numpy()
    sage_pred = Y_image_sage[0].numpy()
    print('pred', bdda_pred.shape)
    print('pred0',bdda_pred[0][0].shape)
    bdda_pred = np.expand_dims(cv2.resize(bdda_pred[0][0], dsize=(w, h)), axis=0)
    sage_pred = np.expand_dims(cv2.resize(sage_pred[0][0], dsize=(w, h)), axis=0)

    im = normalize_map(im)
    bdda_pred = normalize_map(bdda_pred)
    sage_pred = normalize_map(sage_pred)

    im = im.astype(np.uint8)
    bdda_pred = bdda_pred.astype(np.uint8)
    sage_pred = sage_pred.astype(np.uint8)

    im = np.transpose(im, (1, 2, 0))
    bdda_pred = np.transpose(bdda_pred, (1, 2, 0))
    sage_pred = np.transpose(sage_pred, (1, 2, 0))

    heatmap_bdda = blend_map(im, bdda_pred, factor=0.5)
    heatmap_sage = blend_map(im, sage_pred, factor=0.5)

    cv2.imwrite(join(dreyeve_dir, 'heatmap_bdda.jpg'), heatmap_bdda)
    cv2.imwrite(join(dreyeve_dir, 'heatmap_sage.jpg'), heatmap_sage)



