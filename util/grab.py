import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class Grab(Dataset):
    def __init__(self, path_to_data, input_n, output_n, split=0, using_saved_file=True, using_noTpose2=False, norm=True, debug=False,opt=None,using_raw=False):
        tra_val_test = ['train', 'val', 'test']
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)

        data_size = {}
        data_size[0] = (176384, 60, 55, 3)
        data_size[1] = (52255, 60, 55, 3)

        if using_saved_file:
            if split==2:
                sampled_seq = np.load('{}/grab_{}.npy'.format(path_to_data,tra_val_test[split]))
            else:
                # print('>>> remove the first and last 1 second')
                # sampled_seq = np.load('{}/grab_dataloader_normalized_noTpose2_{}.npy'.format(path_to_data,tra_val_test[split]))
                tmp_bin_size = data_size[split]
                tmp_seq = np.memmap('{}/grab_dataloader_normalized_noTpose2_{}.bin'.format(path_to_data,tra_val_test[split]), dtype=np.float32, shape=tmp_bin_size)
                tem_res = np.frombuffer(tmp_seq, dtype=np.float32)
                sampled_seq= tem_res.reshape(tmp_bin_size)

            self.input_pose = torch.from_numpy(sampled_seq[:, i_idx])
            print("input", self.input_pose.shape)
            self.target_pose = torch.from_numpy(sampled_seq)
            print("target", self.target_pose.shape)

            import gc
            del sampled_seq
            gc.collect()
            return

    def gen_data(self):
        for input in self.input_pose:
            batch_samples = []
            while len(batch_samples) > 0:
                yield batch_samples.pop()
    def __len__(self):
        return np.shape(self.input_pose)[0]

    def __getitem__(self, item):
        return self.input_pose[item], self.target_pose[item]


