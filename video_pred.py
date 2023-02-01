from os.path import join
import torch
import torch.nn as nn
from model import SaliencyBranch
import argparse
import cv2
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from utils import resize_tensor, read_image

resize_dim_in = (448, 448)
resize_dim_disp = (540, 960)


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


def load_dreyeve_sample(frames_list, sample, mean_dreyeve_image, frames_per_seq=16, h=448, w=448):
    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = torch.zeros(size=(1, 3, 1, h, w), dtype=torch.float32)
    I_s = torch.zeros(size=(1, 3, frames_per_seq, h_s, w_s), dtype=torch.float32)
    I_c = torch.zeros(size=(1, 3, frames_per_seq, h_c, w_c), dtype=torch.float32)

    for fr in range(frames_per_seq):
        offset = sample - frames_per_seq + 1 + fr  # tricky

        x_in = cv2.resize(frames_list[offset], dsize=resize_dim_in[::-1], interpolation=cv2.INTER_LINEAR)
        x_in = np.transpose(x_in, (2, 0, 1)).astype(np.float32)
        x_in -= mean_dreyeve_image
        x_in = ToTensor()(x_in)

        I_s[0, :, fr, :, :] = resize_tensor(x_in, new_size=(h_s, w_s))

        x_disp = cv2.resize(frames_list[offset], dsize=resize_dim_disp[::-1], interpolation=cv2.INTER_LINEAR)

    I_ff[0, :, 0, :, :] = x_in

    return [I_ff, I_s, I_c], x_disp


if __name__ == '__main__':
    frames_per_seq, h, w = 16, 448, 448
    verbose = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_in", type=str)
    parser.add_argument('--gt_type', type=str)
    args = parser.parse_args()

    assert args.vid_in is not None, 'Please provide a correct video path'

    print("Reading video...")
    frames, pred_list = [], []
    vidcap = cv2.VideoCapture(args.vid_in)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        frames.append(image)
        success, image = vidcap.read()

    print('Finished reading video!')

    print('Now starting prediction...')

    demo_dir = 'demo_images'

    # load mean dreyeve image
    mean_dreyeve_image = read_image(join(demo_dir, 'dreyeve_mean_frame.png'),
                                    channels_first=True, resize_dim=(h, w))

    # get the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_branch = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='image').to(device)
    criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam(image_branch.parameters(), lr=1e-3)
    #if args.gt_type == 'sage':
    #    # load weights here
    #else:
    #    # load weights here

    image_branch.eval()
    with torch.no_grad():
        for sample in range(15, len(frames)):
            X, im = load_dreyeve_sample(sequence_dir=frames, sample=sample, mean_dreyeve_image=mean_dreyeve_image,
                                        frames_per_seq=frames_per_seq, h=h, w=w)

            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y_image = image_branch(X[:3])  # predict on image
            Y_image = Y_image.cpu().numpy()


