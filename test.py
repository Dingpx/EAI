from multiprocessing.util import is_exiting
import numpy
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from util.grab import Grab

from model_others.EAI import GCN_EAI

from util.opt import Options
import util.data_utils as data_utils
from util import utils_utils as utils
from util import loss_func
from tqdm import tqdm
import pdb
import shutil
import random

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_DDP(backend="nccl", verbose=False):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend)
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device

def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def main(opt):

    is_cuda = torch.cuda.is_available()
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n

    print(">>> creating model")
    model = GCN_EAI(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout, num_stage=opt.num_stage, lh_node_n=opt.num_lh*3, rh_node_n=opt.num_rh*3,b_node_n=opt.num_body*3) 
    model_name = '{}'.format(opt.model_type)

    if is_cuda:
        model.cuda()

    dct_trans_funcs = {
        'Norm': get_dct_norm,
        'No_Norm': get_dct,
    }
    idct_trans_funcs = {
        'Norm': get_idct_norm,
        'No_Norm': get_idct,
    }

    if opt.is_hand_norm:
        dct_trans = dct_trans_funcs['Norm']
        idct_trans = idct_trans_funcs['Norm']
    else:
        dct_trans = dct_trans_funcs['No_Norm']
        idct_trans = idct_trans_funcs['No_Norm']

    train_expid = opt.exp
    test_expid = 'TEST_' + opt.exp[6:]

    test_script_name = os.path.basename(__file__).split('.')[0]
    train_script_name = 'train_' + test_script_name[5:]
    train_script_name = "ckpt_eai_dct_n{:d}_out{:d}_dctn{:d}".format(input_n, output_n, all_n)
    test_script_name = "ckpt_eai_dct_n{:d}_out{:d}_dctn{:d}".format(input_n, output_n, all_n)

    train_ckpt_path = './checkpoint/{}/{}_best.pth.tar'.format(train_expid,train_script_name)
    test_csv_path = './checkpoint/{}'.format(test_expid)

    print(">>> loading ckpt len from '{}'".format(train_ckpt_path))
    ckpt = torch.load(train_ckpt_path) if is_cuda else torch.load(train_ckpt_path, map_location='cpu')

    lr = ckpt['lr']
    start_epoch = ckpt['epoch']
    train_loss = ckpt['train_loss']

    new_ckpt_state_dict = {}
    for i in ckpt['state_dict'].keys():
        new_ckpt_state_dict[i[7:]] = ckpt['state_dict'][i]

    model.load_state_dict(new_ckpt_state_dict)
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, train_loss))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_dataset = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=2, debug= opt.is_debug, using_saved_file=opt.is_using_saved_file,using_noTpose2=opt.is_using_noTpose2)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch, shuffle=False, num_workers=0,pin_memory=True)  
    test_loss, test_body_loss, test_lhand_loss, test_rhand_loss,test_lhand_rel_loss,test_rhand_rel_loss,_ = test_split(test_loader, model=model, device=device,dct_trans=dct_trans, idct_trans=idct_trans)

    eval_frame = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    print('>>> Frames  |>>> whole body |>>> main body |>>> left hand |>>> right hand|>>> rel left hand |>>> rel right hand')
    for i, f in enumerate(eval_frame):
        print('>>> {}       ｜>>> {:.3f}     |>>> {:.3f}     |>>> {:.3f}     |>>> {:.3f}     |>>> {:.3f}         |>>> {:.3f}   '\
        .format(f, test_loss[i],test_body_loss[i],test_lhand_loss[i],test_rhand_loss[i],test_lhand_rel_loss[i],test_rhand_rel_loss[i]))

        
def get_dct(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

def get_idct(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d

def get_dct_norm(out_joints):

    batch, frame, node, dim = out_joints.data.shape

    out_joints[:,:,25:40,:] = out_joints[:,:,25:40,:] - out_joints[:,:,20:21,:]
    out_joints[:,:,40:,:] = out_joints[:,:,40:,:] - out_joints[:,:,21:22,:]

    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

def get_idct_norm(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    # 50,32*55*3
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    # 32,162,50

    outputs_p3d[:,:,25:40,:] = outputs_p3d[:,:,25:40,:] + outputs_p3d[:,:,20:21,:]
    outputs_p3d[:,:,40:,:] = outputs_p3d[:,:,40:,:] + outputs_p3d[:,:,21:22,:]

    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d


def test_split(test_loader, model, device, dct_trans,idct_trans):
    N = 0

    eval_frame = [32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
    model.eval()
    t_posi = np.zeros(len(eval_frame))  

    t_body_posi = np.zeros(len(eval_frame)) 
    t_lhand_posi = np.zeros(len(eval_frame)) 
    t_rhand_posi = np.zeros(len(eval_frame))  

    t_lhand_rel_posi = np.zeros(len(eval_frame))  
    t_rhand_rel_posi = np.zeros(len(eval_frame))  

    with torch.no_grad():
        for i, (input_pose, target_pose) in enumerate(test_loader):
            model_input = dct_trans(input_pose)
            n = input_pose.shape[0]
            if torch.cuda.is_available():
                model_input = model_input.to(device).float()
                target_pose = target_pose.to(device).float()
            out_pose,_,_,_ = model(model_input)
            pred_3d, targ_3d = idct_trans(y_out=out_pose, out_joints=target_pose, device=device)

            rel_pred_3d = pred_3d.clone()
            rel_targ_3d = targ_3d.clone()

            rel_pred_3d[:,:,25:40] = rel_pred_3d[:,:,25:40] - rel_pred_3d[:,:,20:21]
            rel_pred_3d[:,:,40:] = rel_pred_3d[:,:,40:] - rel_pred_3d[:,:,21:22]

            rel_targ_3d[:,:,25:40] = rel_targ_3d[:,:,25:40] - rel_targ_3d[:,:,20:21]
            rel_targ_3d[:,:,40:] = rel_targ_3d[:,:,40:] - rel_targ_3d[:,:,21:22]

            for k in np.arange(0, len(eval_frame)):  
                j = eval_frame[k]

                test_out, test_joints = pred_3d[:, j, :, :], targ_3d[:, j, :, :]
                loss_wholebody, _ = loss_func.joint_body_loss_test(test_out, test_joints)
                t_posi[k] += loss_wholebody.cpu().data.numpy() * n * 100

                test_body_out, test_body_joints = pred_3d[:, j, :25, :], targ_3d[:, j, :25, :]
                t_body_posi[k] += loss_func.joint_loss(test_body_out, test_body_joints).cpu().data.numpy() * n * 100

                test_lhand_out, test_lhand_joints = pred_3d[:, j, 25:40, :], targ_3d[:, j, 25:40, :]
                t_lhand_posi[k] += loss_func.joint_loss(test_lhand_out, test_lhand_joints).cpu().data.numpy() * n * 100

                test_rhand_out, test_rhand_joints = pred_3d[:, j, 40:, :], targ_3d[:, j, 40:, :]
                t_rhand_posi[k] += loss_func.joint_loss(test_rhand_out, test_rhand_joints).cpu().data.numpy() * n * 100

                test_lhand_rel_out, test_lhand_rel_joints = rel_pred_3d[:, j, 25:40, :], rel_targ_3d[:, j, 25:40, :]
                t_lhand_rel_posi[k] += loss_func.joint_loss(test_lhand_rel_out, test_lhand_rel_joints).cpu().data.numpy() * n * 100

                test_rhand_rel_out, test_rhand_rel_joints = rel_pred_3d[:, j, 40:, :], rel_targ_3d[:, j, 40:, :]
                t_rhand_rel_posi[k] += loss_func.joint_loss(test_rhand_rel_out, test_rhand_rel_joints).cpu().data.numpy() * n * 100
        
            N += n
    return t_posi / N,t_body_posi / N,t_lhand_posi / N,t_rhand_posi / N,t_lhand_rel_posi / N,t_rhand_rel_posi / N, N


def get_dct_matrix(N):
    dct_m = np.eye(N)  
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)  
    return dct_m, idct_m

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    option = Options().parse()
    main(option)

