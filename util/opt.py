import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================

        self.parser.add_argument('--device', type=str, default='cuda:0',
                                 help='path to amass Synthetic dataset')
        self.parser.add_argument('--grab_data_dict', type=str, default='./Dataset_GRAB/',help='path to GRAB dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--model_type', type=str, default='LTD', help='path to save checkpoint')


        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')
        self.parser.add_argument('--num_body', type=int, default=25, help='# layers in linear model')
        self.parser.add_argument('--num_lh', type=int, default=15, help='# layers in linear model')
        self.parser.add_argument('--num_rh', type=int, default=15, help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=1.0e-3)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=30, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=30, help='future seq length')
        self.parser.add_argument('--all_n', type=int, default=60, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=64)
        self.parser.add_argument('--val_batch', type=int, default=128)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=0, help='subprocesses to use for data loading')
        self.parser.add_argument('--seed', type=int, default=1024, help='random seed')
        self.parser.add_argument("--local_rank", type=int, help="local rank")
        self.parser.add_argument('--W_pg', type=float, default=0.6, help='The weight of information propagation between part') 
        self.parser.add_argument('--W_p', type=float, default=0.6, help='The weight of part on the whole body')

        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--is_debug', dest='is_debug', action='store_true', help='wether to debug')
        self.parser.add_argument('--is_exp', dest='is_exp', action='store_true', help='wether to save different model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true',
                                 help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')
        self.parser.add_argument('--is_using_saved_file', dest='is_using_saved_file', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')
        self.parser.add_argument('--is_hand_norm', dest='is_hand_norm', action='store_true',help='')
        self.parser.add_argument('--is_hand_norm_split', dest='is_hand_norm_split', action='store_true',help='')
        self.parser.add_argument('--is_part', dest='is_part', action='store_true', help='')
        self.parser.add_argument('--part_type', type=str, default='lhand', help='')
        self.parser.add_argument('--is_boneloss', dest='is_boneloss', action='store_true', help='')
        self.parser.add_argument('--is_weighted_jointloss', dest='is_weighted_jointloss', action='store_true', help='')
        self.parser.add_argument('--is_using_noTpose2', dest='is_using_noTpose2', action='store_true', help='')
        self.parser.add_argument('--is_using_raw', dest='is_using_raw', action='store_true', help='')

        self.parser.add_argument('--J', type=int, default=1, help='The number of wavelet filters')                    
        self.parser.set_defaults(max_norm=True)


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        return self.opt
