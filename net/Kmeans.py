from data.AMB.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from main.config import cfg
# import matplotlib
import cv2,torch
import numpy as np
# matplotlib.use('TkAgg')
from kvision import *
###########################
# 收集skeleton数据
###########################
trainset_loader = Dataset(transforms.ToTensor(), "train")
batch_generator = DataLoader(dataset=trainset_loader, batch_size=1#cfg.num_gpus * cfg.train_batch_size
                             ,shuffle=False, num_workers=24, pin_memory=True)
# px = torch.cat([targets['joint_coord'][:,:21].reshape(-1,3) for inputs, targets, meta_info in batch_generator],dim=0).float()
# np.save('test_rs1_skeleton',px)
# px = torch.from_numpy(np.load('test_rs1_skeleton.npy',allow_pickle=True))
# print(px.shape) # torch.Size([94500, 3])
############################################
# run kmeans to get our coordinate_to_dict
###########################################
def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    # 从px中随机选取512个点
    c = x[torch.randperm(N)[:ncluster]] # [512, 3]

    for i in range(niter):
        # assign all pixels to the closest codebook element
        # [94500, 512, 3]-->[94500, 512]-->[50000]
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c
# # dict_length
# ncluster = 256
# # start optim
# with torch.no_grad():
#     C = kmeans(px, ncluster, niter=8)  # [512, 3]
# np.save('rs1_skeleton_to_sequence_256', C)
# exit()
'''
C: [512, 3]
{"1": [x1, y1, z1]
 "2": [x2, y2, z2]
 "3": [x2, y2, z2]
 ...
 "512": [X512, y512, z512]
}
'''
###########################
# 可视化验证经过编码解码之后的坐标损失  (21, 3) -> (21, 512) -> (21, 3)
# encode the training examples with our codebook
# visualize how much we've lost in the discretization
###########################
for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
    img = inputs['img'][0]  # [16, 3, 256, 256]
    joint_coord = targets['joint_coord'][:, :21]  # [16, 42, 3]
    joint_sequence = targets['joint_sequence'][:, :21]
    kmean_joint_coord = torch.from_numpy(trainset_loader.skeleton2sequence[joint_sequence])
    kmean_error = (kmean_joint_coord - joint_coord)
    kmean_error = torch.sqrt(torch.sum(kmean_error ** 2) / 21)
    print(kmean_error)
    print(kmean_joint_coord.shape)
    print(joint_coord.shape)

    # joint_coord = skeleton_inter2mano(joint_coord[0])
    # kmean_joint_coord = skeleton_inter2mano(kmean_joint_coord[0])
    cv2.imshow('Render_kmean', draw1_2d_skeleton(img.numpy().transpose(1,2,0)[...,[2,1,0]], kmean_joint_coord[0].numpy()*256/64))
    cv2.imshow('Render', draw1_2d_skeleton(img.numpy().transpose(1,2,0)[...,[2,1,0]], joint_coord[0].numpy()*256/64))
    cv2.waitKey(0)
    # target_joint_heatmap = render_gaussian_heatmap(targets['joint_coord'][:,:21],sigma=0.5)
    # o3d.visualization.draw_geometries([show_3d_hm(target_joint_heatmap[0][i]) for i in range(21)])
    # show_o3d([show_3d_hm(target_joint_heatmap[0][i]) for i in range(21)],size=64)
