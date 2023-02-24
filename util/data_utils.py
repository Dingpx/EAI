import numpy as np
import torch
import logging
from copy import copy
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def prepare_params(params, frame_mask, dtype=np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def append2dict(source, data):
    for k in data.keys():
        if isinstance(data[k], list):
            source[k] += data[k].astype(np.float32)
        else:
            source[k].append(data[k].astype(np.float32))


def np2torch(item):
    out = {}
    for k, v in item.items():
        if v == []:
            continue
        if isinstance(v, list):
            try:
                out[k] = torch.from_numpy(np.concatenate(v))
            except:
                out[k] = torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v)
    return out


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array


def makepath(desired_path, isfile=False):
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def makelogger(log_dir, mode='w'):
    makepath(log_dir, isfile=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s' % log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def euler(rots, order='xyz', units='deg'):
    rots = np.asarray(rots)
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz, order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis == 'x':
                r = np.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), r)
            if axis == 'y':
                r = np.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), r)
            if axis == 'z':
                r = np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats


def create_video(path, fps=30, name='movie'):
    import os
    import subprocess

    src = os.path.join(path, '%*.png')
    movie_path = os.path.join(path, '%s.mp4' % name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path, '%s_%02d.mp4' % (name, i))
        i += 1

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)

    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue

import torch
import os


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma): 
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, file_name=['ckpt_best.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[0])
    torch.save(state, file_path)


contact_ids = {'Body': 1,
               'L_Thigh': 2,
               'R_Thigh': 3,
               'Spine': 4,
               'L_Calf': 5,
               'R_Calf': 6,
               'Spine1': 7,
               'L_Foot': 8,
               'R_Foot': 9,
               'Spine2': 10,
               'L_Toes': 11,
               'R_Toes': 12,
               'Neck': 13,
               'L_Shoulder': 14,
               'R_Shoulder': 15,
               'Head': 16,
               'L_UpperArm': 17,
               'R_UpperArm': 18,
               'L_ForeArm': 19,
               'R_ForeArm': 20,
               'L_Hand': 21,
               'R_Hand': 22,
               'Jaw': 23,
               'L_Eye': 24,
               'R_Eye': 25,
               'L_Index1': 26,
               'L_Index2': 27,
               'L_Index3': 28,
               'L_Middle1': 29,
               'L_Middle2': 30,
               'L_Middle3': 31,
               'L_Pinky1': 32,
               'L_Pinky2': 33,
               'L_Pinky3': 34,
               'L_Ring1': 35,
               'L_Ring2': 36,
               'L_Ring3': 37,
               'L_Thumb1': 38,
               'L_Thumb2': 39,
               'L_Thumb3': 40,
               'R_Index1': 41,
               'R_Index2': 42,
               'R_Index3': 43,
               'R_Middle1': 44,
               'R_Middle2': 45,
               'R_Middle3': 46,
               'R_Pinky1': 47,
               'R_Pinky2': 48,
               'R_Pinky3': 49,
               'R_Ring1': 50,
               'R_Ring2': 51,
               'R_Ring3': 52,
               'R_Thumb1': 53,
               'R_Thumb2': 54,
               'R_Thumb3': 55}


def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
    """Intialization of layers with normal distribution with mean and bias"""
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        layer.weight.data.normal_(mean_, sd_)
        if norm_bias:
            layer.bias.data.normal_(bias, 0.05)
        else:
            layer.bias.data.fill_(bias)


def weight_init(
        module,
        mean_=0,
        sd_=0.004,
        bias=0.0,
        norm_bias=False,
        init_fn_=normal_init_):
    """Initialization of layers with normal distribution"""
    moduleclass = module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    init_fn_(l, mean_, sd_, bias, norm_bias)
            else:
                init_fn_(layer, mean_, sd_, bias, norm_bias)
    except TypeError:
        init_fn_(module, mean_, sd_, bias, norm_bias)


def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
        nn.init.xavier_uniform_(layer.weight.data)
        if norm_bias:
            layer.bias.data.normal_(0, 0.05)
        else:
            layer.bias.data.zero_()


def create_dir_tree(base_dir):
    dir_tree = ['models', 'tf_logs', 'config', 'std_log']
    for dir_ in dir_tree:
        os.makedirs(os.path.join(base_dir, dir_), exist_ok=True)


def create_look_ahead_mask(seq_length, is_nonautoregressive=False):
    """Generates a binary mask to prevent to use future context in a sequence."""
    if is_nonautoregressive:
        return np.zeros((seq_length, seq_length), dtype=np.float32)
    x = np.ones((seq_length, seq_length), dtype=np.float32)
    mask = np.triu(x, 1).astype(np.float32)
    return mask  # (seq_len, seq_len)


RED = (0, 1, 1)
ORANGE = (20/360, 1, 1)
YELLOW = (60/360, 1, 1)
GREEN = (100/360, 1, 1)
CYAN = (175/360, 1, 1)
BLUE = (210/360, 1, 1)

RED_DARKER = (0, 1, 0.25)
ORANGE_DARKER = (20/360, 1, 0.25)
YELLOW_DARKER = (60/360, 1, 0.25)
GREEN_DARKER = (100/360, 1, 0.25)
CYAN_DARKER = (175/360, 1, 0.25)
BLUE_DARKER = (210/360, 1, 0.25)
class Grab_Skeleton_55:
    num_joints = 55
    start_joints = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 18, 17, 19, 15, 15, 15, 
                    21, 21, 21, 21, 21, 52, 53, 40, 41, 43, 44, 49, 50, 46, 47, 
                    20, 20, 20, 20, 20, 37, 38, 25, 26, 28, 29, 34, 35, 31, 32] 
    end_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 19, 21, 22, 23, 24,  
                  40, 43, 49, 46, 52, 53, 54, 41, 42, 44, 45, 50, 51, 47, 48,  
                  37, 25, 28, 34, 31, 38, 39, 26, 27, 29, 30, 35, 36, 32, 33,  
                  ]
    bones = list(zip(start_joints,end_joints))


def define_actions(action='all'):
    if action == 'all':
        return ['airplane-fly-1', 'airplane-lift-1', 'airplane-pass-1', 'alarmclock-lift-1', 'alarmclock-pass-1',
                'alarmclock-see-1', 'apple-eat-1', 'apple-pass-1', 'banana-eat-1', 'banana-lift-1', 'banana-pass-1',
                'banana-peel-1', 'banana-peel-2', 'binoculars-lift-1', 'binoculars-pass-1', 'binoculars-see-1',
                'bowl-drink-1',
                'bowl-drink-2', 'bowl-lift-1', 'bowl-pass-1', 'camera-browse-1', 'camera-pass-1',
                'camera-takepicture-1',
                'camera-takepicture-2', 'camera-takepicture-3', 'cubelarge-inspect-1', 'cubelarge-lift-1',
                'cubelarge-pass-1',
                'cubemedium-inspect-1', 'cubemedium-lift-1', 'cubemedium-pass-1', 'cubesmall-inspect-1',
                'cubesmall-lift-1',
                'cubesmall-pass-1', 'cup-drink-1', 'cup-drink-2', 'cup-lift-1', 'cup-pass-1', 'cup-pour-1',
                'cylinderlarge-inspect-1', 'cylinderlarge-lift-1', 'cylinderlarge-pass-1', 'cylindermedium-inspect-1',
                'cylindermedium-pass-1', 'cylindersmall-inspect-1', 'cylindersmall-pass-1', 'doorknob-lift-1',
                'doorknob-use-1', 'doorknob-use-2', 'duck-pass-1', 'elephant-inspect-1', 'elephant-pass-1',
                'eyeglasses-wear-1', 'flashlight-on-1', 'flashlight-on-2', 'flute-pass-1', 'flute-play-1',
                'fryingpan-cook-1',
                'fryingpan-cook-2', 'gamecontroller-lift-1', 'gamecontroller-pass-1', 'gamecontroller-play-1',
                'hammer-lift-1',
                'hammer-pass-1', 'hammer-use-1', 'hammer-use-2', 'hammer-use-3', 'hand-inspect-1', 'hand-lift-1',
                'hand-pass-1', 'hand-shake-1', 'headphones-lift-1', 'headphones-pass-1', 'headphones-use-1',
                'knife-chop-1',
                'knife-pass-1', 'knife-peel-1', 'lightbulb-pass-1', 'lightbulb-screw-1', 'mouse-lift-1', 'mouse-pass-1',
                'mouse-use-1', 'mug-drink-1', 'mug-drink-2', 'mug-lift-1', 'mug-pass-1', 'mug-toast-1', 'phone-call-1',
                'phone-lift-1', 'phone-pass-1', 'piggybank-pass-1', 'piggybank-use-1', 'pyramidlarge-pass-1',
                'pyramidmedium-inspect-1', 'pyramidmedium-lift-1', 'pyramidmedium-pass-1', 'pyramidsmall-inspect-1',
                'scissors-pass-1', 'scissors-use-1', 'spherelarge-inspect-1', 'spherelarge-lift-1',
                'spherelarge-pass-1',
                'spheremedium-inspect-1', 'spheremedium-lift-1', 'spheremedium-pass-1', 'spheresmall-inspect-1',
                'spheresmall-pass-1', 'stamp-lift-1', 'stamp-pass-1', 'stamp-stamp-1', 'stanfordbunny-inspect-1',
                'stanfordbunny-lift-1', 'stanfordbunny-pass-1', 'stapler-lift-1', 'stapler-pass-1', 'stapler-staple-1',
                'stapler-staple-2', 'teapot-pass-1', 'teapot-pour-1', 'teapot-pour-2', 'toothpaste-lift-1',
                'toothpaste-pass-1', 'toothpaste-squeeze-1', 'toothpaste-squeeze-2', 'toruslarge-inspect-1',
                'toruslarge-lift-1', 'toruslarge-pass-1', 'torusmedium-inspect-1', 'torusmedium-lift-1',
                'torusmedium-pass-1',
                'torussmall-inspect-1', 'torussmall-lift-1', 'torussmall-pass-1', 'train-lift-1', 'train-pass-1',
                'train-play-1', 'watch-pass-1', 'waterbottle-drink-1', 'waterbottle-pass-1', 'waterbottle-pour-1',
                'wineglass-drink-1', 'wineglass-drink-2', 'wineglass-lift-1', 'wineglass-pass-1', 'wineglass-toast-1']
    else:
        return action

if __name__ == "__main__":
    skeleton = Grab_Skeleton_55
    print(skeleton.bones)
