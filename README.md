# 时间原因，匆忙赶出来的简易版本，后续更新

# 训练: run_train.sh
# 测试: run.sh

# 数据集地址: ./Dataset_GRAB
# 测试集:grab_test.npy 
# 训练集:grab_dataloader_normalized_noTpose2_train.bin/npy
# 验证集:grab_dataloader_normalized_noTpose2_val.bin/npy
# 注: .bin二进制读取更快；训练和验证集均为去除完前后1s的数据；测试集为采样的数据，采样的index之前已经给您


# 训练好的ckpt 存储在checkpoint/TRAIN_modeltype_EAI_batchsize_64_lr_0.001下
# 注：运行run_train.sh时候会覆盖ckpt,使用注意备份

# 模型在model_others/EAI.py文件里

# 如有进一步的其他问题，可以联系我