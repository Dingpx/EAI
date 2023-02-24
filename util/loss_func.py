import torch
import numpy as np
from .data_utils import Grab_Skeleton_55


def poses_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    loss = torch.mean(torch.norm(y_out - out_poses, 2, 1))
    return loss

def joint_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    return torch.mean(torch.norm(y_out - out_poses, 2, 1))

def relative_hand_loss(y_out, out_poses):

    y_out_rel_lhand = y_out[:,:,25:40,:] - y_out[:,:,20:21,:]
    y_out_rel_rhand = y_out[:,:,40:,:] - y_out[:,:,21:22,:]

    out_poses_rel_lhand = out_poses[:,:,25:40,:] - out_poses[:,:,20:21,:]
    out_poses_rel_rhand = out_poses[:,:,40:,:] - out_poses[:,:,21:22,:]

    loss_rel_lhand = joint_loss(y_out_rel_lhand,out_poses_rel_lhand)
    loss_rel_rhand = joint_loss(y_out_rel_rhand,out_poses_rel_rhand)

    return loss_rel_lhand + loss_rel_rhand

def joint_body_loss(y_out, out_poses):

    y_out_wrist = y_out[:,:,20:22,:]
    out_poses_wrist = out_poses[:,:,20:22,:]

    y_out_wrist = y_out_wrist.reshape(-1, 3)
    out_poses_wrist = out_poses_wrist.reshape(-1, 3)

    l_wrist = torch.mean(torch.norm(y_out_wrist - out_poses_wrist, 2, 1))

    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    return torch.mean(torch.norm(y_out - out_poses, 2, 1)),l_wrist

def joint_body_loss_test(y_out, out_poses):
    
    y_out_wrist = y_out[:,20:22,:]
    out_poses_wrist = out_poses[:,20:22,:]

    y_out_wrist = y_out_wrist.reshape(-1, 3)
    out_poses_wrist = out_poses_wrist.reshape(-1, 3)

    l_wrist = torch.mean(torch.norm(y_out_wrist - out_poses_wrist, 2, 1))

    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    return torch.mean(torch.norm(y_out - out_poses, 2, 1)),l_wrist

def bone_length_error(joints, input_bone_lengths, skeleton_cls):
    bone_lengths = calculate_bone_lengths(joints, skeleton_cls)
    return np.sum(np.abs(np.array(input_bone_lengths) - bone_lengths))

def calculate_bone_lengths(joints, skeleton_cls):
    return np.array([np.linalg.norm(joints[bone[0]] - joints[bone[1]] + 0.001) for bone in skeleton_cls.bones])

def bone_loss(raw,predict,device):

	raw_bone_length = cal_bone_loss(raw,device)
	pred_bone_length = cal_bone_loss(predict,device)

	diff = torch.abs(pred_bone_length - raw_bone_length) 
	loss = torch.mean(diff) 

	return loss

def cal_bone_loss(x,device):
    # KCS 
    batch_num = x.size()[0]
    frame_num = x.size()[1]
    joint_num = x.size()[2]

    Ct = get_matrix(device)

    x_ = x.transpose(2, 3) # b, t, 3, 55
    x_ = torch.matmul(x_, Ct)  # b, t, 3, 54
    bone_length = torch.norm(x_, 2, 2) # b, t, 54

    return bone_length

def get_matrix(device,type='all'):

    S_of_lhand = [20, 20, 20, 20, 20, 37, 38, 25, 26, 28, 29, 34, 35, 31, 32]  
    S_of_rhand = [21, 21, 21, 21, 21, 52, 53, 40, 41, 43, 44, 49, 50, 46, 47]  
    S_of_body = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7,   8,  9,  9,  9, 12,   13, 14, 16, 18, 17,  19, 15, 15, 15]  
    E_of_lhand = [37, 25, 28, 34, 31, 38, 39, 26, 27, 29, 30, 35, 36, 32, 33]  
    E_of_rhand = [40, 43, 49, 46, 52, 53, 54, 41, 42, 44, 45, 50, 51, 47, 48]  
    E_of_body = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15,  16, 17, 18, 20, 19,  21, 22, 23, 24]  

    if type=='all':
        E = np.hstack((E_of_body, E_of_lhand, E_of_rhand))
        S = np.hstack((S_of_body, S_of_lhand, S_of_rhand))
        matrix = torch.zeros([55,54])
    elif type=='lhand':
        E = E_of_lhand
        S = S_of_lhand
        matrix = torch.zeros([55,15])
    elif type=='rhand':
        E = E_of_rhand
        S = S_of_rhand
        matrix = torch.zeros([55,15])
    elif type=='body':
        E = E_of_body
        S = S_of_body
        matrix = torch.zeros([55,24])

    for i in range(S.shape[0]):
        matrix[S[i].tolist(),i] = 1
        matrix[E[i].tolist(),i] = -1
    
    matrix = matrix.to(device)

    return matrix
