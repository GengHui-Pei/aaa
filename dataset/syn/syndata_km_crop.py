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
from lib.common.config import cfg
from lib.common.utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, \
    transform_input_to_output_space, generate_patch_image, trans_point2d
from lib.common.utils.transforms import world2cam, cam2pixel, pixel2cam
from lib.common.utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json,sys
import math
from pycocotools.coco import COCO
from lib.common.kvision import kprint,draw1_2d_skeleton,draw_2d_skeleton,skeleton_inter2mano,skeleton_mano2inter
from lib.common.prep_ambhand_data import get_bbox, cam2pix,random_bg_img
import copy

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, annot_subset=None):
        self.mode = mode
        mode = 'test' if mode == 'test' else 'train'
        self.root_path = f'{cfg.root_dir}'
        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(cfg.img_root_dir, 'skeleton.txt'), self.joint_num * 2)
        self.skeleton2sequence = np.load(f'{cfg.root_dir}/dataset/syn/JointCrop2Squence256.npy',allow_pickle=True)
        self.datalist = []
        db = COCO(f'{cfg.root_dir}/dataset/syn/annotations/AmbHand_{mode}_data.json')
        with open(f'{cfg.root_dir}/dataset/syn/annotations/AmbHand_{mode}_camera.json') as f:
            cameras = json.load(f)
        # with open('annotations/AmbHand_{mode}_MANO.json') as f:
        #     MANO_annot = json.load(f)
        with open(f'{cfg.root_dir}/dataset/syn/annotations/AmbHand_{mode}_joint3d.json') as f:
            joint3d = json.load(f)

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            img_path = os.path.join(cfg.img_root_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
            subject_name, gesture, frame_id, key = img['subject_id'], img['gesture'], img['frame_id'], \
                                                   img['camera_id']
            bbox = db.loadAnns(image_id)[0]['bbox']
            # bbox = process_bbox(bbox, (img_height, img_width))
            focal = np.array(cameras[f'subject_{subject_name}'][f'gesture_{gesture}'][str(frame_id)]
                             [key]['focal'],dtype=np.float32)
            princpt = np.array(cameras[f'subject_{subject_name}'][f'gesture_{gesture}'][str(frame_id)][key]
                               ['princpt'],dtype=np.float32)

            joint_cam = np.array(joint3d[f'subject_{subject_name}'][f'gesture_{gesture}'][str(frame_id)][key]
                                 ['cam_joints'],dtype=np.float32)
            joint_cam = skeleton_mano2inter(joint_cam)

            joint_img = cam2pix(joint_cam, key)[:,:2]  # 相机内参变换 等价于渲染图像(RH.render)
            joint_valid = np.array(db.loadAnns(image_id)[0]['joint_valid'])


            # transform single hand data to double hand data structure
            hand_type = ann['hand_type']
            joint_img_dh = np.zeros((self.joint_num * 2, 2), dtype=np.float32)
            joint_cam_dh = np.zeros((self.joint_num * 2, 3), dtype=np.float32)
            joint_valid_dh = np.zeros((self.joint_num * 2), dtype=np.float32)
            joint_img_dh[self.joint_type[hand_type]] = joint_img
            joint_cam_dh[self.joint_type[hand_type]] = joint_cam
            joint_valid_dh[self.joint_type[hand_type]] = joint_valid
            joint_img = joint_img_dh
            joint_cam = joint_cam_dh
            joint_valid = joint_valid_dh
            abs_depth = joint_cam[self.root_joint_idx[hand_type], 2]  # single hand abs depth

            cam_param = {'focal': focal,
                         'princpt': princpt}
            joint = {'cam_coord': joint_cam,
                     'img_coord': joint_img,
                     'valid': joint_valid}
            data = {'img_path': img_path,
                    'bbox': bbox,
                    'cam_param': cam_param,
                    'joint': joint,
                    'hand_type': hand_type,
                    'abs_depth': abs_depth,
                    'subject': subject_name,
                    'gesture': gesture,
                    'frame': frame_id,
                    'camera_id': key,
                    'bg_idx': img['bg_idx']}
            self.datalist.append(data)
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
        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()
        hand_type = self.handtype_str2array(hand_type)
        org_joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1) # (pix_x,pix_y,cam_z)

        # image load
        img = load_img(img_path)
        # augmentation, now only use this func to crop image.
        img, joint_coord_crop, joint_valid, hand_type, inv_trans = augmentation(img, bbox, org_joint_coord,
                                                                           joint_valid, hand_type, self.mode,
                                                                           self.joint_type, data['bg_idx'])

        if self.mode == 'KMean':
            return joint_coord_crop[:21]

        img = self.transform(img.astype(np.float32))/255.
        rel_root_depth = np.zeros((1), dtype=np.float32)
        root_valid = np.zeros((1), dtype=np.float32)

        # transform to output heatmap space
        joint_coord_kmean = np.zeros([42, 3])
        joint_sequence = ((joint_coord_crop[:21, None, :] - self.skeleton2sequence[None, :, :]) ** 2).sum(-1).argmin(1)
        joint_coord_kmean[:21] = self.skeleton2sequence[joint_sequence]
        joint_coord_kmean_space, _, _, _ = transform_input_to_output_space(
            joint_coord_kmean.copy(),joint_valid.copy(),rel_root_depth.copy(),
            root_valid.copy(),self.root_joint_idx.copy(),self.joint_type.copy())

        joint_coord_crop_space, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(
            joint_coord_crop,joint_valid,rel_root_depth,root_valid,self.root_joint_idx,self.joint_type)

        inputs = {'img': img,
                  'img_path':img_path,
                  'bg_idx':data['bg_idx'],
                  'bbox':bbox,
                  'org_joint_coord':org_joint_coord }
        targets = {'joint_coord_crop': joint_coord_crop,
                   'joint_coord_crop_space': joint_coord_crop_space,
                   'joint_sequence': joint_sequence,
                   'joint_coord_kmean': joint_coord_kmean,
                   'joint_coord_kmean_space': joint_coord_kmean_space,
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

        gts = copy.deepcopy(self.datalist)
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = \
            preds['joint_coord'].copy(), preds['rel_root_depth'].copy(), \
            preds['hand_type'].copy(), preds['inv_trans'].copy()

        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)

        mpjpe = [[] for _ in range(self.joint_num)]  # treat right and left hand identical
        acc_hand_cls = 0
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type = data['bbox'], data['cam_param'], \
                                                   data['joint'], data['hand_type']
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
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton[:self.joint_num], filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.png'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton[:self.joint_num], filename)

        # print('Handedness accuracy: ' + str(acc_hand_cls / sample_num))

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
        print(eval_summary)
        print('MPJPE: %.2f' % (np.mean(mpjpe)))


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    import os.path as osp
    from lib.common.utils.preprocessing import load_img, load_skeleton, process_bbox, transform_input_to_output_space, trans_point2d
    from lib.common.utils.vis import vis_keypoints, vis_3d_keypoints
    from lib.net.main_net import heatmap2joint, joint2heatmap


    trainset_loader = Dataset(transforms.ToTensor(), "test")
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
    batch_generator = DataLoader(dataset=trainset_loader, batch_size=12#cfg.num_gpus * cfg.train_batch_size
                                 , shuffle=False, num_workers=12, pin_memory=True)


    for i in range(3):
        preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}
        for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
            img = inputs['img'][0]  # [16, 3, 256, 256]

            joint_coord = targets['joint_coord_crop']  # [16, 42, 3]
            joint_sp = targets['joint_coord_crop_space']
            joint_km = targets['joint_coord_kmean']
            joint_km_sp = targets['joint_coord_kmean_space']

            if itr % 100 == 0:
                print(itr)

            joint_km_sp = heatmap2joint(joint2heatmap(joint_km_sp.cuda()))

            preds['joint_coord'].append(joint_km_sp.cpu().numpy())
            preds['rel_root_depth'].append(targets['rel_root_depth'].cpu().numpy())
            preds['hand_type'].append(targets['hand_type'].cpu().numpy())
            preds['inv_trans'].append(meta_info['inv_trans'].cpu().numpy())

        preds = {k: np.concatenate(v) for k, v in preds.items()}
        trainset_loader.evaluate(preds)
