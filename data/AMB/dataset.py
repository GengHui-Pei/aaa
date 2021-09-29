# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from skeleton.common.config import cfg
from skeleton.common.utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, \
    transform_input_to_output_space, generate_patch_image, trans_point2d
from skeleton.common.utils.transforms import world2cam, cam2pixel, pixel2cam
from skeleton.common.utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json,sys
import math
from pycocotools.coco import COCO
from skeleton.common.kvision import kprint,draw1_2d_skeleton,draw_2d_skeleton,skeleton_inter2mano,skeleton_mano2inter
from skeleton.common.prep_ambhand_data import get_bbox, cam2pix,random_bg_img
import sys
sys.path.append('/root/Workspace/jyk/skeleton')
class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        self.mode = mode
        self.root_path = f'{cfg.root_dir}'
        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(cfg.img_root_dir, 'skeleton.txt'), self.joint_num * 2)
        #记录的是skeleton名字，对应的关节数字，和父关节点数字
        #load_skeleton记录了每个关节名称，子节点数字，父节点，和自己的关节点。
        self.skeleton2sequence = np.load(f'{cfg.root_dir}/net/rs1_skeleton_to_sequence_256.npy',allow_pickle=True)
        #print(self.skeleton2sequence.shape)    256,3   256个中心坐标

        self.datalist = []
        db = COCO(f'{cfg.root_dir}/data/AMB/annotations/AmbHand_{mode}_data.json')
        with open(f'{cfg.root_dir}/data/AMB/annotations/AmbHand_{mode}_camera.json') as f:
            cameras = json.load(f)
        # with open('annotations/AmbHand_{mode}_MANO.json') as f:
        #     MANO_annot = json.load(f)
        with open(f'{cfg.root_dir}/data/AMB/annotations/AmbHand_{mode}_joint3d.json') as f:
            joint3d = json.load(f)
           # print(joint3d.keys())'subject_6', 'subject_7', 'subject_8', 'subject_9'...
        #print(db.anns.keys())从0到28795, 28796, 28797, 28798, 28799, 28800


        for aid in db.anns.keys():
            #什么是aid

            ann = db.anns[aid]
            #print(ann.keys()) dict_keys(['id', 'image_id', 'category_id', 'iscrowd', 'joint_valid', 'bbox', 'hand_type'])

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]


            img_path = os.path.join(cfg.img_root_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
            subject_name, gesture, frame_id, key = img['subject_id'], img['gesture'], img['frame_id'], img['camera_id']
            #print(subject_name, gesture, frame_id, key)6 0 0 rs1

            bbox = db.loadAnns(image_id)[0]['bbox']

            # bbox = process_bbox(bbox, (img_height, img_width))
            focal = np.array(cameras[f'subject_{subject_name}'][f'gesture_{gesture}'][str(frame_id)][key]['focal'],dtype=np.float32)
            princpt = np.array(cameras[f'subject_{subject_name}'][f'gesture_{gesture}'][str(frame_id)][key]['princpt'],dtype=np.float32)

            joint_cam = np.array(joint3d[f'subject_{subject_name}'][f'gesture_{gesture}'][str(frame_id)][key]['cam_joints'],dtype=np.float32)
            joint_cam = skeleton_mano2inter(joint_cam)
            #mano和interhand的关节编号不相同，这是转换
            joint_img = cam2pix(joint_cam, key)[:,:2]  # 相机内参变换 等价于渲染图像(RH.render)
            #print(joint_img.shape)    21,2
            joint_valid = np.array(db.loadAnns(image_id)[0]['joint_valid'])
            #print(joint_valid)  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
            # transform single hand data to double hand data structure
            hand_type = ann['hand_type']
            joint_img_dh = np.zeros((self.joint_num * 2, 2), dtype=np.float32)
            joint_cam_dh = np.zeros((self.joint_num * 2, 3), dtype=np.float32)
            joint_valid_dh = np.zeros((self.joint_num * 2), dtype=np.float32)
            #这里得查查数据结构
            joint_img_dh[self.joint_type[hand_type]] = joint_img
           # print(joint_img_dh)   42,2  然后填满前21个或者后21个

            joint_cam_dh[self.joint_type[hand_type]] = joint_cam
            joint_valid_dh[self.joint_type[hand_type]] = joint_valid
            joint_img = joint_img_dh;
            joint_cam = joint_cam_dh;
            joint_valid = joint_valid_dh;
            abs_depth = joint_cam[self.root_joint_idx[hand_type], 2]  # single hand abs depth

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'bbox': bbox, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type,
                    'abs_depth': abs_depth, 'subject': subject_name, 'gesture': gesture, 'frame': frame_id, 'camera_id': key,
                    'bg_idx': img['bg_idx']}
            self.datalist.append(data)


        self.datalist=random.sample(self.datalist, 100)


        # self.datalist = self.datalist[:320]

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type = data['img_path'], data['bbox'], data['joint'], data['hand_type']
        #使用copy会产生不同的地址，改变赋值量不会影响被赋值量
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        #None用于增加一维；concatenate要求两数组在axis维度上大小可以不一样，其余的维度一样的
        #print(joint_img.shape,joint_cam[:, 2, None].shape)   (42, 2) (42, 1)
        #print(joint_cam.shape)  42.3

        org_joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1) # (pix_x,pix_y,cam_z)

        # image load

        img = load_img(img_path)

        # cv2.imwrite('out1.jpg', img)
        # augmentation

        img, joint_coord_crop, joint_valid, hand_type, inv_trans = augmentation(img, bbox, org_joint_coord, joint_valid, hand_type, self.mode, self.joint_type, data['bg_idx'])

        #print(joint_coord.shape,inv_trans,joint_valid.shape,hand_type)
        #42,3   2*3的矩阵    42   [1,0]

        # cv2.imwrite('out2.jpg', img)
        #print(img.astype(np.float32)[0][0])经过totensor没有归一化到0-1

        img = self.transform(img.astype(np.float32))/255.
        rel_root_depth = np.zeros((1), dtype=np.float32)
        root_valid = np.zeros((1), dtype=np.float32)
        # transform to output heatmap space
        #下面两行程序的作用
        joint_coord_crop_space, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord_crop,joint_valid,rel_root_depth,root_valid,self.root_joint_idx,self.joint_type)

        #print(joint_coord.shape, joint_valid.shape, rel_root_depth.shape, root_valid.shape)(42, 3) (42,) (1,) (1,)
        joint_coord_kmean=np.zeros([42,3])

        joint_sequence = ((joint_coord_crop_space[:21, None, :] - self.skeleton2sequence[None, :, :]) ** 2).sum(-1).argmin(1)
        #print(joint_sequence.shape)  21
        joint_coord_kmean[:21]=self.skeleton2sequence[joint_sequence]
        '''
        print(joint_sequence)
        [179  24  57  32 254  92 215 237 172 150 230 174 202  47 155 121 194  99
         80 236 193 ]
        '''
        joint_kmeans = torch.from_numpy(self.skeleton2sequence[joint_sequence])*4
       # print(joint_kmeans.shape)torch.Size([21, 3])

        #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。

        inputs = {'img': img,
                  'img_path': img_path,
                  'bg_idx': data['bg_idx'],
                  'bbox': bbox,
                  'org_joint_coord': org_joint_coord }
        targets = {'joint_coord': joint_coord_crop_space,
                   'joint_coord_crop':joint_coord_crop,
                   'joint_sequence': joint_sequence,
                   'joint_kmeans': joint_kmeans,
                   'joint_coord_kmean':joint_coord_kmean,
                   'rel_root_depth': rel_root_depth,
                   'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid,
                     'root_valid': root_valid,
                     'hand_type_valid': 1,
                     'inv_trans': inv_trans,
                     'subject': data['subject'],
                     'gesture': data['gesture'],
                     'frame': data['frame'],
                     'camera_id': data['camera_id']}
        return inputs, targets, meta_info

    def evaluate(self, preds):

        print()
        print('Evaluation start...')

        gts = self.datalist
        #np.save('syn_gts.npy', gts)
        #print('Saved on:', 'syn_gts.npy')
        #exit()
        print(len(gts))

        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = \
            preds['joint_coord'], preds['rel_root_depth'], preds['hand_type'], preds['inv_trans']
        print(len(preds_joint_coord))
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        #print(len(preds_joint_coord))
       # exit()

        mpjpe = [[] for _ in range(self.joint_num)]  # treat right and left hand identical
        acc_hand_cls = 0
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], data['joint'], data['hand_type']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']

            # restore coordinates to original space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            for j in range(self.joint_num * 2):
                pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
            # restore depth to original camera space
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

            # add root joint depth
            pred_joint_coord_img[:, 2] += data['abs_depth']

            # back project to camera coordinate system
            img_path = data['img_path']
            cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            _img = cvimg[:, :, ::-1].transpose(2, 0, 1)
            # vis_keypoints(_img, pred_joint_coord_img, joint_valid, self.skeleton) # img:(3,540,960)
            # cv2.imshow('pred', draw1_2d_skeleton(_img, pred_joint_coord_img[:21]))
            # cv2.waitKey(0)
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            pred_joint_coord_cam[self.joint_type['right']] = pred_joint_coord_cam[self.joint_type['right']]\
                                                             - pred_joint_coord_cam[self.root_joint_idx['right'], None, :]

            gt_joint_coord[self.joint_type['right']] = gt_joint_coord[self.joint_type['right']] \
                                                       - gt_joint_coord[self.root_joint_idx['right'], None, :]


            # select right or left hand using groundtruth hand type
            pred_joint_coord_cam = pred_joint_coord_cam[self.joint_type[gt_hand_type]]
            gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
            joint_valid = joint_valid[self.joint_type[gt_hand_type]]
            #print(pred_joint_coord_cam.shape)  (21,3)





            # mpjpe save
            for j in range(self.joint_num):
                if joint_valid[j]:
                    mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2)))

            if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                acc_hand_cls += 1
            elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                acc_hand_cls += 1


            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                _img = cvimg[:, :, ::-1].transpose(2, 0, 1)
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                filename = 'out_' + str(n) + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton[:self.joint_num], filename=filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.png'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton[:self.joint_num], filename)



        print('Handedness accuracy: ' + str(acc_hand_cls / sample_num))

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
        print(eval_summary)
        print('MPJPE: %.2f' % (np.mean(mpjpe)))


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os.path as osp
from skeleton.common.utils.preprocessing import load_img, load_skeleton, process_bbox, transform_input_to_output_space, trans_point2d
from skeleton.common.utils.vis import vis_keypoints, vis_3d_keypoints
from  matplotlib import pyplot as plt
trainset_loader = Dataset(transforms.ToTensor(), "train")
# '''
# Train: 64800 (single camera: 16200)
#     subject: 6~37
#     gesture: 0~45
#     frame: 0~19
# ----------------------------
# Test: 16200 (single camera: 4050)
#     subject: 0~5
#     gesture: 0~45
#     frame: 0~9
# '''
# batch_generator = DataLoader(dataset=trainset_loader, batch_size=1#cfg.num_gpus * cfg.train_batch_size
#                              , shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
# for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
#     img = inputs['img'][0]  # [16, 3, 256, 256]
#     joint_coord = targets['joint_coord'][0]  # [16, 42, 3]
#     kmean_joint_coord = targets['joint_kmeans'][0]
#
#     rel_root_depth = targets['rel_root_depth']  # [16, 1]
#     hand_type = targets['hand_type']  # [16, 2]
#
#     joint_valid = meta_info['joint_valid'][0]  # [16, 42]
#     root_valid = meta_info['root_valid']  # [16, 1]
#     hand_type_valid = meta_info['hand_type_valid']  # [16]
#     inv_trans = meta_info['inv_trans']  # [16, 2, 3]
#     # vis_keypoints(img[0].numpy()*255., joint_coord[0].numpy()*256/64, joint_valid[0], trainset_loader.skeleton)
#     # vis_3d_keypoints(joint_coord[0].numpy(), joint_valid[0], trainset_loader.skeleton)
#     # plt.show()
#     # cv2.waitKey(0)
#     print(meta_info['subject'],meta_info['gesture'],meta_info['frame'],meta_info['camera_id'])
#     joint_coord = skeleton_inter2mano(joint_coord)
#     #cv2.imshow(f'Render_{(itr+1)%4}', draw1_2d_skeleton(img.numpy().transpose(1,2,0)[...,[2,1,0]], joint_coord[:21].numpy()*256/64))
#     im2 = draw1_2d_skeleton(img.numpy().transpose(1,2,0)[...,[2,1,0]], joint_coord[:21].numpy()*256/64).astype(np.uint8)[:, :, ::-1]  # transform image to rgb
#     # #dfdf=Image.open("/root/Workspace/jyk/background_img/0.jpg")
#     # #dfdf.show(dfdf)
#     # #plt.imshow([[[3,3,3],[3,3,3],[3,3,3]],[[4,4,4],[4,4,4],[4,4,4]],[[4,4,4],[4,4,4],[4,4,4]]])
#     plt.figure()
#     plt.imshow(im2)
#     #plt.show()
#     ## cv2.waitKey(0) if (itr+1)%4==0 else cv2.waitKey(2)
#
#     kmean_joint_coord = skeleton_inter2mano(kmean_joint_coord)
#     #cv2.imshow(f'kmean_Render_{(itr+1)%4}', draw1_2d_skeleton(img.numpy().transpose(1,2,0)[...,[2,1,0]], kmean_joint_coord[:21].numpy()))
#     im3 = draw1_2d_skeleton(img.numpy().transpose(1,2,0)[...,[2,1,0]], kmean_joint_coord[:21].numpy())[:, :,
#           ::-1]  # transform image to rgb
#     plt.figure()
#     plt.imshow(im3)
#     plt.show()
#     exit()
#     #cv2.waitKey(0) if (itr+1)%4==0 else cv2.waitKey(2)