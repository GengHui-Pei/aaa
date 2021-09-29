from lib.dataset.syn.syndata_km_sp import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from main.config import cfg
# import matplotlib
import cv2,torch
import numpy as np
# matplotlib.use('TkAgg')
###########################
# 收集skeleton数据
###########################
def train_kmean(joints):
    print('KMean in process ...')

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


    ncluster = 256
    # start optim
    with torch.no_grad():
        C = kmeans(joints, ncluster, niter=8)  # [512, 3]
    np.save('./JointSP2Squence256.npy', C)
    print('KMean Done')

    return C


if __name__ == '__main__':
    trainset_loader = Dataset(transforms.ToTensor(), "KMean")
    batch_generator = DataLoader(dataset=trainset_loader, batch_size=128#cfg.num_gpus * cfg.train_batch_size
                                 ,shuffle=False, num_workers=12, pin_memory=True)
    print(trainset_loader.__len__())

    hand_joints = torch.cat([joint_crop for joint_crop in batch_generator],dim=0).float()

    np.save('test_rs1_skeleton',hand_joints)
    hand_joints = torch.from_numpy(np.load('test_rs1_skeleton.npy',allow_pickle=True))
    print(hand_joints.shape) # torch.Size([94500, 3])
    joints = hand_joints.clone().reshape(-1, 3)
    print('joints: ', joints.shape)


    # Train or load KMean.
    mode = 'Train'
    if mode == 'Train':
        C = train_kmean(joints)

    C = np.load('./JointSP2Squence256.npy')

    print('C: ', C.shape)


    # TODO: calculate kmean error
    joints_km_id = ((joints[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
    joints_km_id = joints_km_id.numpy().astype(int)
    joints_km = torch.from_numpy(C[joints_km_id])
    hand_joints_km = joints_km.reshape(-1, 21, 3)

    # hand_joints = hand_joints - hand_joints[:, [20], :]
    # hand_joints_km = hand_joints_km - hand_joints_km[:, [20], :]

    print(hand_joints_km.shape)

    Loss = torch.nn.L1Loss()
    kmean_error = Loss(hand_joints, hand_joints_km)
    print('kmean_error: ', kmean_error)

    kmean_error = torch.mean(torch.sqrt(((hand_joints.reshape([-1, 3]) -
                                          hand_joints_km.reshape([-1, 3])) ** 2).sum(-1)))
    print('kmean_error: ', kmean_error)
    exit()

