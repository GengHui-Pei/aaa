# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import sys
import math
import numpy as np

class Config:
    
    ## dataset
    dataset = 'AMB' # InterHand2.6M, RHD, STB, AMB
    '''
    | TIME       | InterHand | RHD  | STB  |
    | ---------- | --------- | ---- | ---- |
    | **1080ti** | 4.65      | 0.24 | 0.17 |
    | **2080ti** | 3.64      |      | 0.14 |
    | **3090**   | 2.70      | 0.15 | 0.11 |
    '''
    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (64, 64, 64) # (depth, height, width)
    sigma = 5
    bbox_3d_size = 400 # depth axis
    bbox_3d_size_root = 400 # depth axis
    output_root_hm_shape = 64 # depth axis

    ## model
    resnet_type = 50 # 18, 34, 50, 101, 152

    ## training config
    lr_dec_epoch = [10, 20, 30, 40, 50] if dataset == 'InterHand2.6M' else [5, 10, 15, 20, 25]
    end_epoch = 20 if dataset == 'InterHand2.6M' else 50
    lr = 1e-4
    lr_backbone = 1e-5
    lr_dec_factor = 10
    train_batch_size = 12

    ## testing config
    test_batch_size = 32
    trans_test = 'rootnet' # gt, rootnet

    ## directory
    #os.path.abspath作用： 获取当前脚本的完整路径


    # cur_dir = osp.dirname(os.path.abspath(__file__))
    # root_dir = osp.join(cur_dir, '..')
    # data_dir = osp.join(root_dir, 'data')
    # background_img = f"{osp.join(cur_dir, '../../InterHand2.6M')}/ktools/background_img"
    # img_root_dir = f"{osp.join(cur_dir, '../../InterHand2.6M')}/syn_amb_dataset/synthetic_datasets"
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = '/root/Workspace/jyk/skeleton'
    data_dir = osp.join(root_dir, 'data')
    background_img = '/root/Workspace/jyk/background_img'
    img_root_dir = '/root/Workspace/jyk/synthetic_datasets/synthetic_datasets'

    # background_img = '/root/Workspace/InterHand2.6M/ktools/background_img'
    # img_root_dir = '/root/Workspace/InterHand2.6M/syn_amb_dataset/synthetic_datasets'
    #output_dir = osp.join(root_dir, 'output')
    output_dir ='/root/Workspace/jyk/skeleton/output'
    # model_dir = osp.join(output_dir, 'model_dump')
    # vis_dir = osp.join(output_dir, 'vis')
    # log_dir = osp.join(output_dir, 'log')
    # result_dir = osp.join(output_dir, 'result')
    model_dir = osp.join('/root/Workspace/jyk/skeleton/output', 'model_dump')
    vis_dir = osp.join('/root/Workspace/jyk/skeleton/output', 'vis')
    log_dir = osp.join('/root/Workspace/jyk/skeleton/output', 'log')
    result_dir = osp.join('/root/Workspace/jyk/skeleton/output', 'result')

    ## others
    num_thread = 24
    # num_gpus = 2
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

cfg = Config()
sys.path.insert(0, cfg.root_dir)
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset)) # /home/water/PycharmProjects/skeleton_net/common/../data/AMB
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

