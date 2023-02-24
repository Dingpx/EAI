# coding=utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@IDE: PyCharm
@author: dpx
@contact: dingpx2015@gmail.com
@time: 2022,9月
Copyright (c), xiaohongshu

@Desc:

"""
from multiprocessing.util import is_exiting
import os
import pdb
import numpy
import torch
import shutil
import random
import torch.optim
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from util import loss_func
from util.opt import Options
from util.grab import Grab
from torch.autograd import Variable
from util import utils_utils as utils
from model_others.EAI import GCN_EAI
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



def main(opt, rank, local_rank, world_size, device):

    # 初始化参数
    setup_seed(opt.seed)
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr

    # 加载数据集
    print(">>> loading train_data")
    train_dataset = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=0, debug= opt.is_debug, using_saved_file=opt.is_using_saved_file, using_noTpose2=opt.is_using_noTpose2)
    print(">>> loading val_data")
    val_dataset   = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=1, debug= opt.is_debug, using_saved_file=opt.is_using_saved_file,using_noTpose2=opt.is_using_noTpose2)
    print(">>> making dataloader")

    # 多GPU分布式训练的数据处理
    batch_size = opt.train_batch // world_size  # [*] // world_size
    train_sampler = DistributedSampler(train_dataset, shuffle=True)  # [*]
    val_sampler = DistributedSampler(val_dataset, shuffle=False)  # [*]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)  # [*] sampler=...
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)  # [*] sampler=...
    print(">>> train data {}".format(train_dataset.__len__()))  
    print(">>> validation data {}".format(val_dataset.__len__()))  

    # 加载模型
    print(">>> creating model")
    model = GCN_EAI(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout, num_stage=opt.num_stage, lh_node_n=opt.num_lh*3, rh_node_n=opt.num_rh*3,b_node_n=opt.num_body*3) 
    model_name = '{}'.format(opt.model_type)
    if opt.is_exp:
        ckpt = opt.ckpt + opt.exp
    else:
        ckpt = opt.ckpt + model_name
    
    # 将模型迁移到GPU上
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        if_find_unused_parameters = False
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=if_find_unused_parameters)  # [*] DDP(...)
    print_only_rank0(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 加载优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr*world_size)

    # continue from checkpoint
    script_name = "eai_dct_n{:d}_out{:d}_dctn{:d}".format(input_n, output_n, all_n)
    print_only_rank0(">>> is_load {}".format(opt.is_load))
    if opt.is_load:
        model_path_len = '{}/ckpt_{}_best.pth.tar'.format(ckpt, script_name)
        print_only_rank0(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt_model = torch.load(model_path_len)
        else:
            ckpt_model = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt_model['epoch']
        err_best = ckpt_model['train_loss']
        lr_now = ckpt_model['lr']
        model.load_state_dict(ckpt_model['state_dict'])
        optimizer.load_state_dict(ckpt_model['optimizer'])
        print_only_rank0(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    else:
        print_only_rank0(">>> loading ckpt from scratch")
        # 新建/覆盖ckpt文件
        if dist.get_rank() == 0:
            if os.path.exists(ckpt):
                shutil.rmtree(ckpt)
            os.makedirs(ckpt,exist_ok=True)

    # start training
    print(">>> err_best", err_best)

    dct_trans_funcs = {
        'Norm': get_dct_norm,
        'No_Norm': get_dct,
    }
    idct_trans_funcs = {
        'Norm': get_idct_norm,
        'No_Norm': get_idct,
    }

    # flag设定：是否要对手部做norm
    print('>>> whether hand norm:{}'.format(opt.is_hand_norm))
    if opt.is_hand_norm:
        dct_trans = dct_trans_funcs['Norm']
        idct_trans = idct_trans_funcs['Norm']
    else:
        dct_trans = dct_trans_funcs['No_Norm']
        idct_trans = idct_trans_funcs['No_Norm']

    # 训练
    for epoch in range(start_epoch, opt.epochs):

        # sampler重采样dataloader
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # 学习率衰减设置
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print_only_rank0('=====================================')
        print_only_rank0('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))

        # csv初始化设置
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])

        # 训练
        Ir_now, t_l, = train(train_loader, model, optimizer, device=device, lr_now=lr_now, max_norm=opt.max_norm,dct_trans=dct_trans,idct_trans=idct_trans,is_boneloss=opt.is_boneloss,is_weighted_jointloss=opt.is_weighted_jointloss)
        # 训练结果
        print_only_rank0("train_loss:{}".format(t_l))
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        # 验证
        v_loss = validate(val_loader, model, device=device,dct_trans=dct_trans,idct_trans=idct_trans)
        # 短时结果
        print_only_rank0("v_loss:{}".format(v_loss))
        ret_log = np.append(ret_log, [v_loss])
        head = np.append(head, ['v_loss'])

        ########################################################################################################################
        # 以下是短时的ckpt保存的代码
        if not np.isnan(v_loss):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = v_loss < err_best  # err_best=10000
            err_best = min(v_loss, err_best)
        else:
            is_best = Falsecd
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        if epoch == start_epoch:
            df.to_csv(ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        file_name = ['ckpt_' + script_name + '_epoch_{}.pth.tar'.format(epoch+1), 'ckpt_']

        if dist.get_rank() == 0:
            file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
            utils.save_ckpt({'epoch': epoch + 1,
                            'lr': lr_now,
                            'train_loss': t_l,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                        ckpt_path=ckpt,
                        is_best=is_best,
                        file_name=file_name)   
    

def train(train_loader, model, optimizer, device, lr_now, max_norm, dct_trans, idct_trans, is_boneloss,is_weighted_jointloss):
    print_only_rank0("进入train")
    # 初始化
    iter_num = 0
    t_l = utils.AccumLoss()
    model.train()

    for (input_pose, target_pose) in tqdm(train_loader):
        # 加载数据
        model_input = dct_trans(input_pose)
        n = input_pose.shape[0]  # 16
        if torch.cuda.is_available():
            model_input = model_input.to(device).float()
            target_pose = target_pose.to(device).float()
        
        # 前向传播过程
        out_pose,  mmdloss_ab, mmdloss_ac, mmdloss_bc = model(model_input)
        pred_3d, targ_3d = idct_trans(y_out=out_pose, out_joints=target_pose, device=device)

        # loss计算
        if is_weighted_jointloss:
            loss_jt = loss_func.weighted_joint_loss(pred_3d, targ_3d, ratio=0.6)
        else:
            loss_jt = loss_func.joint_loss(pred_3d, targ_3d)
            loss_pjt = loss_func.relative_hand_loss(pred_3d, targ_3d)
    
        if is_boneloss:
            loss_bl = loss_func.bone_loss(pred_3d, targ_3d, device)
            loss = loss_jt + 0.1 * loss_bl + 0.1 * loss_pjt
        else:
            loss = loss_jt 
        loss = loss + 0.001 * (mmdloss_ab+mmdloss_ac+mmdloss_bc)

        # 反向传播过程
        optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
        loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值

        # 更新总体loss结果
        t_l.update(loss.cpu().data.numpy() * n, n)

    return lr_now, t_l.avg

def validate(val_loader, model, device, dct_trans, idct_trans):
    print_only_rank0("进入val")
    # 初始化
    t_l = utils.AccumLoss()
    model.eval()

    for i, (input_pose, target_pose) in enumerate(val_loader):
        
        # 加载数据
        model_input = dct_trans(input_pose)
        n = input_pose.shape[0]  # 64
        if torch.cuda.is_available():
            model_input = model_input.to(device).float()
            target_pose = target_pose.to(device).float()

        # 前向传播过程
        out_pose,  _, _, _ = model(model_input)

        # DCT 转 3D结果
        pred_3d, targ_3d = idct_trans(y_out=out_pose, out_joints=target_pose, device=device)
        
        # 短时的ckpt挑选
        pred_3d = pred_3d
        targ_3d = targ_3d
        loss= loss_func.joint_loss(pred_3d, targ_3d)
        t_l.update(loss.cpu().data.numpy() * n, n)



    return t_l.avg

# 一维DCT变换
def get_dct_matrix(N):
    dct_m = np.eye(N)  # 返回one-hot数组
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  # 2/35开更
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)  # 矩阵求逆
    return dct_m, idct_m

# 设定种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # 会降低训练速度
     torch.backends.cudnn.deterministic = True

# 多GPU分布式训练的初始化
def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device

# 多GPU分布式训练时候只打印第0个GPU的结果
def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)

# 相对plevis的坐标系下：3D转DCT
def get_dct(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

# 相对plevis的坐标系下：DCT转3D
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

# 身体的关节，相对plevis的坐标系下：3D转DCT； 针对手部，相对wrist关节的坐标系下：3D转DCT；
def get_dct_norm(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    out_joints[:,:,25:40,:] =  out_joints[:,:,25:40,:] - out_joints[:,:,20:21,:]
    out_joints[:,:,40:,:] = out_joints[:,:,40:,:] - out_joints[:,:,21:22,:]
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

# 身体的关节，相对plevis的坐标系下：DCT转3D； 针对手部，相对wrist关节的坐标系下：DCT转3D；
def get_idct_norm(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    outputs_p3d[:,:,25:40,:] = outputs_p3d[:,:,25:40,:] + outputs_p3d[:,:,20:21,:]
    outputs_p3d[:,:,40:,:] = outputs_p3d[:,:,40:,:] + outputs_p3d[:,:,21:22,:]
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d


if __name__ == "__main__":
    option = Options().parse()
    # 初始化ddp的代码
    rank, local_rank, world_size, device = setup_DDP(verbose=True)
    main(option, rank, local_rank, world_size, device)
