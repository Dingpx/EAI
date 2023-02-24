from __future__ import absolute_import  
from __future__ import print_function  

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math, copy
from torch.nn import functional as F

from model_others.GCN import *
import util.data_utils as utils

class TempSoftmaxFusion_2(nn.Module):
    def __init__(self, channels, detach_inputs=False, detach_feature=False):
        super(TempSoftmaxFusion_2, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l+1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))

    def forward(self, x, y, work=True):
        b, n, f = x.shape
        x = x.reshape(-1, f)
        y = y.reshape(-1, f)
        f_in = torch.cat([x, y], dim=1)
        if self.detach_inputs:
            f_in = f_in.detach()
        f_temp = self.layers(f_in)
        f_weight = F.softmax(f_temp*self.temperature, dim=1)
        if self.detach_feature:
            x = x.detach()
            y = y.detach()
        f_out = f_weight[:,[0]]*x + f_weight[:,[1]]*y
        f_out = f_out.view(b,-1,f)
        return f_out

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def get_mmdloss(source, target,kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, pde_qk, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        query = query.transpose(0,1)
        key = key.transpose(0,1)
        value = value.transpose(0,1)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn = attn.transpose(0,1)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class SelfAttention_block(nn.Module):
    def __init__(self, input_feature=60, hidden_feature=256, p_dropout=0.5, num_stage=12, node_n=15*3):
        super(SelfAttention_block, self).__init__()
        self.gcn = GCN(input_feature, hidden_feature, p_dropout, num_stage, node_n)

    def forward(self, x):
        y = self.gcn(x)
        return y

class CrossAttention_block(nn.Module):
    def __init__(self,
                 input_dim=60,
                 head_num=3,
                 dim_ffn=256,
                 dropout=0.2,
                 init_fn=utils.normal_init_):
        super(CrossAttention_block, self).__init__()
        self._model_dim = input_dim
        self._dim_ffn = dim_ffn
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self.inner_att = MultiheadAttention(input_dim, head_num, attn_dropout=dropout)
        self._linear1 = nn.Linear(self._model_dim, self._dim_ffn)
        self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
        self._norm2 = nn.LayerNorm(self._model_dim, eps=1e-5)

        utils.weight_init(self._linear1, init_fn_=init_fn)
        utils.weight_init(self._linear2, init_fn_=init_fn)

    def forward(self, x, y, pdm_xy=None):
        query =x
        key = y
        value = y
        attn_output, _ = self.inner_att(
            query,
            key,
            value,
            pdm_xy
        )
        norm_attn_ = self._dropout_layer(attn_output) + query
        norm_attn = self._norm2(norm_attn_)
        output = self._linear1(norm_attn)
        output = self._relu(output)
        output = self._dropout_layer(output)
        output = self._linear2(output)
        output = self._dropout_layer(output) + norm_attn_
        return output

class DA_Norm(nn.Module):
  def __init__(self,num_features):
    super().__init__()

    shape = (1,1,num_features)
    shape2 = (1,1,num_features)

    self.gamma = nn.Parameter(torch.ones(shape))
    self.beta = nn.Parameter(torch.zeros(shape))
    self.gamma2 = nn.Parameter(torch.ones(shape))
    self.beta2 = nn.Parameter(torch.zeros(shape))

    moving_mean = torch.zeros(shape2)
    moving_var = torch.zeros(shape2)
    moving_mean2 = torch.zeros(shape2)
    moving_var2 = torch.zeros(shape2)

    self.register_buffer("moving_mean", moving_mean)  
    self.register_buffer("moving_var", moving_var)  
    self.register_buffer("moving_mean2", moving_mean2)  
    self.register_buffer("moving_var2", moving_var2)
    self.weight = nn.Parameter(torch.zeros(1))

  def forward(self,X, X2):
    if self.moving_mean.device != X.device:
      self.moving_mean = self.moving_mean.to(X.device)
      self.moving_var = self.moving_var.to(X.device)
      self.moving_mean2 = self.moving_mean2.to(X.device)
      self.moving_var2 = self.moving_var2.to(X.device)
    Y, Y2, self.moving_mean, self.moving_var,self.moving_mean2, self.moving_var2 = batch_norm(X,X2,self.gamma,self.beta,self.moving_mean,self.moving_var,self.gamma2,self.beta2,self.moving_mean2,self.moving_var2,self.weight,eps=1e-5,momentum=0.9)

    return Y,Y2

def batch_norm(X, X2,gamma,beta,moving_mean,moving_var,gamma2,beta2,moving_mean2,moving_var2,weight,eps,momentum):

  if not torch.is_grad_enabled():

    X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    X_hat2 = (X2 - moving_mean2) / torch.sqrt(moving_var2 + eps)
  else:
    weight = (F.sigmoid(weight)+1)/2

    mean = X.mean(dim=(0,1),keepdim=True)
    var = ((X - mean)**2).mean(dim=(0,1),keepdim=True)

    mean2 = X2.mean(dim=(0,1),keepdim=True)
    var2 = ((X2 - mean2)**2).mean(dim=(0,1),keepdim=True)

    mean_fa = weight * mean + (1-weight)* mean2
    mean_fb = weight * mean2 + (1-weight)* mean

    var_fa = weight * var + (1-weight)* var2
    var_fb = weight * var2 + (1-weight)* var

    X_hat = (X - mean_fa) / torch.sqrt(var_fa + eps)
    X_hat2 = (X2 - mean_fb) / torch.sqrt(var_fb + eps)

    moving_mean = momentum * moving_mean + (1.0 - momentum) * mean_fa
    moving_var = momentum * moving_var + (1.0 - momentum) * var_fa

    moving_mean2 = momentum * moving_mean2 + (1.0 - momentum) * mean_fb
    moving_var2 = momentum * moving_var2 + (1.0 - momentum) * var_fb

  Y = gamma * X_hat + beta
  Y2 = gamma2 * X_hat2 + beta2

  return Y, Y2, moving_mean.data, moving_var.data,moving_mean2.data, moving_var2.data

class Alignment_block(nn.Module):
    def __init__(self,
                 input_dim=256,
                 head_num=8,
                 dim_ffn=256,
                 dropout=0.2,
                 init_fn=utils.normal_init_,
                 src_len1=None,
                 src_len2=None):
        super(Alignment_block, self).__init__()
        self._model_dim = input_dim
        self._dim_ffn = dim_ffn

    def forward(self, x, x2, x3, mmd_flag=False):
        # Calculating MMD loss
        output_sa_x = x
        output_sa_x2 = x2
        output_sa_x3 = x3
        if mmd_flag:
            xa_f = torch.mean(output_sa_x,1)
            xb_f = torch.mean(output_sa_x2,1)
            xc_f = torch.mean(output_sa_x3,1)
            mmdlossab = get_mmdloss(xa_f,xb_f)
            mmdlossbc = get_mmdloss(xb_f,xc_f)
            mmdlossac = get_mmdloss(xc_f,xa_f)
        else:
            mmdlossab = 0
            mmdlossbc = 0 
            mmdlossac = 0

        return  output_sa_x, output_sa_x2, output_sa_x3, mmdlossab, mmdlossbc, mmdlossac

class GCN_EAI(nn.Module):
    def __init__(self, input_feature=60, hidden_feature=256, p_dropout=0.5, num_stage=12, lh_node_n=15*3, rh_node_n=15*3,b_node_n=25*3):
        super(GCN_EAI, self).__init__()

        # Individual Encoder
        num_stage_encoder = 12
        self.body_encoder = SelfAttention_block(input_feature, hidden_feature, p_dropout, num_stage_encoder, node_n=b_node_n)
        self.lhand_encoder = SelfAttention_block(input_feature, hidden_feature, p_dropout, num_stage_encoder, node_n=lh_node_n+3)
        self.rhand_encoder = SelfAttention_block(input_feature, hidden_feature, p_dropout, num_stage_encoder, node_n=rh_node_n+3)

        # Distribution Norm
        self._normab = DA_Norm(input_feature)
        self._normbc = DA_Norm(input_feature)
        self._normca = DA_Norm(input_feature)

        # Feature Alignment
        self.align_num_layers = 1
        head_num = 3
        self._align_layers = nn.ModuleList([])
        for i in range(self.align_num_layers):
            self._align_layers.append(Alignment_block(head_num=head_num,input_dim=input_feature,src_len1=b_node_n,src_len2=lh_node_n+3))

        # Semantic Interaction
        self.ca_num_layers = 5
        self._inter_body_lhand_layers = nn.ModuleList([])
        self._inter_body_rhand_layers = nn.ModuleList([])
        self._inter_lhand_body_layers = nn.ModuleList([])
        self._inter_lhand_rhand_layers = nn.ModuleList([])
        self._inter_rhand_lhand_layers = nn.ModuleList([])
        self._inter_rhand_body_layers = nn.ModuleList([])
        for i in range(self.ca_num_layers):
            self._inter_body_lhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_body_rhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_lhand_body_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_lhand_rhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_rhand_lhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_rhand_body_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))

        # Physical Interaction
        self.fusion_lwrist = TempSoftmaxFusion_2(channels=[input_feature*6,input_feature,2])
        self.fusion_rwrist = TempSoftmaxFusion_2(channels=[input_feature*6,input_feature,2])

        # Decoder
        self.body_decoder = nn.Linear(input_feature*3, input_feature)
        self.lhand_decoder = nn.Linear(input_feature*3, input_feature)
        self.rhand_decoder = nn.Linear(input_feature*3, input_feature)
        self.rwrist_decoder = nn.Linear(input_feature*3, input_feature)
        self.lwrist_decoder = nn.Linear(input_feature*3, input_feature)
        utils.weight_init(self.body_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.lhand_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.rhand_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.rwrist_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.lwrist_decoder, init_fn_= utils.normal_init_)

    def forward(self, x, action=None, pde_ml=None,pde_lm=None,pde_mr=None,pde_rm=None,pde_lr=None,pde_rl=None):

        # data process & wrist replicate 
        b, n, f = x.shape
        whole_body_x = x.view(b, -1, 3, f)
        lwrist = whole_body_x[:,20:21].detach()
        rwrist = whole_body_x[:,21:22].detach()
        b_x = whole_body_x[:,:25].view(b, -1, f)
        lh_x = torch.cat((lwrist,whole_body_x[:,25:40]),1)
        lh_x = lh_x.view(b, -1, f)
        rh_x = torch.cat((rwrist,whole_body_x[:,40:]),1)
        rh_x = rh_x.view(b, -1, f)

        # Encoding
        hbody = self.body_encoder(b_x)
        lhand = self.lhand_encoder(lh_x)
        rhand = self.rhand_encoder(rh_x)

        # Distribution Normalization
        hbody1,lhand1 = self._normab(hbody,lhand)
        lhand1,rhand1 = self._normbc(lhand1,rhand)
        rhand1,hbody1 = self._normca(rhand1,hbody1)

        # Feature Alignment
        hbody2, rhand2, lhand2, mmdloss_ab, mmdloss_ac, mmdloss_bc = self._align_layers[0](hbody1, rhand1,lhand1,mmd_flag = True)

        # Semantic Interaction
        rhand_2_hbody = hbody2
        lhand_2_hbody = hbody2
        lhand_2_rhand = rhand2
        hbody_2_rhand = rhand2
        rhand_2_lhand = lhand2
        hbody_2_lhand = lhand2

        for i in range(self.ca_num_layers):
            rhand_2_hbody = self._inter_body_rhand_layers[i](rhand_2_hbody, rhand2)
            lhand_2_hbody = self._inter_body_lhand_layers[i](lhand_2_hbody, lhand2)

            lhand_2_rhand = self._inter_rhand_lhand_layers[i](lhand_2_rhand, lhand2)
            hbody_2_rhand = self._inter_rhand_body_layers[i](hbody_2_rhand, hbody2)

            rhand_2_lhand = self._inter_lhand_rhand_layers[i](rhand_2_lhand, rhand2)
            hbody_2_lhand = self._inter_lhand_body_layers[i](hbody_2_lhand, hbody2)

        # Feature Concat
        fusion_body = torch.cat((hbody,rhand_2_hbody,lhand_2_hbody),dim=2)
        fusion_rhand = torch.cat((rhand,lhand_2_rhand,hbody_2_rhand),dim=2)
        fusion_lhand = torch.cat((lhand,rhand_2_lhand,hbody_2_lhand),dim=2)

        # Physical Interaction
        b, n, f1 = fusion_body.shape
        hbody_lwrist = fusion_body.view(b, -1, 3, f1)[:,20:21].view(b, -1, f1)
        hbody_rwrist = fusion_body.view(b, -1, 3, f1)[:,21:22].view(b, -1, f1)
        lhand_lwrist = fusion_lhand.view(b, -1, 3, f1)[:,:1].view(b, -1, f1)        
        rhand_rwrist = fusion_rhand.view(b, -1, 3, f1)[:,:1].view(b, -1, f1)
        fusion_lwrist = self.fusion_lwrist(hbody_lwrist,lhand_lwrist)
        fusion_rwrist = self.fusion_rwrist(hbody_rwrist,rhand_rwrist)

        hbody_no_wrist = torch.cat((fusion_body.view(b, -1, 3, f1)[:,:20],fusion_body.view(b, -1, 3, f1)[:,22:]),1).view(b, -1, f1)
        lhand_no_wrist = fusion_lhand.view(b, -1, 3, f1)[:,1:].view(b, -1, f1)
        rhand_no_wrist = fusion_rhand.view(b, -1, 3, f1)[:,1:].view(b, -1, f1)

        # Decoding
        hbody_no_wrist = self.body_decoder(hbody_no_wrist)
        lhand_no_wrist = self.lhand_decoder(lhand_no_wrist) 
        rhand_no_wrist = self.rhand_decoder(rhand_no_wrist)
        fusion_lwrist = self.lwrist_decoder(fusion_lwrist)
        fusion_rwrist = self.rwrist_decoder(fusion_rwrist)

        hbody_no_wrist = hbody_no_wrist.view(b, -1, 3, f)
        lhand_no_wrist = lhand_no_wrist.view(b, -1, 3, f)
        rhand_no_wrist = rhand_no_wrist.view(b, -1, 3, f)
        fusion_lwrist = fusion_lwrist.view(b, -1, 3, f)
        fusion_rwrist = fusion_rwrist.view(b, -1, 3, f)
        output = torch.cat([hbody_no_wrist[:,:20],fusion_lwrist,fusion_rwrist,hbody_no_wrist[:,20:],lhand_no_wrist,rhand_no_wrist],1).view(b, -1, f) + x

        return output, mmdloss_ab, mmdloss_ac, mmdloss_bc
