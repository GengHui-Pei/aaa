import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from net.main_net import ImagetoSkeleton
from net.misc import NestedTensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, img, steps=21, n_samples=16, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = 21
    model.eval()
    # kprint(img.shape) # [3, 256, 256]
    n_img = img.repeat(n_samples, 1, 1, 1) # [50, 3, 256, 256]
    x = torch.from_numpy(np.ones((n_samples,1))*256).to(n_img.device).to(torch.int64) # [50, 1](256)
    for k in range(steps):
        n_img_mask = NestedTensor(n_img, torch.zeros_like(n_img[:, 0, :, :], device=n_img.device) > 1)
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x_cond.transpose(0,1)
        logits, loss = model(n_img_mask, x_cond, mode='test')#(1,9,256)
        logits = logits.transpose(0,1)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options

        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    return x[:,1:]

from data.AMB.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common.kvision import *
from common.config import cfg


model = ImagetoSkeleton()
model = model.cuda()
model.load_state_dict(torch.load(
    '/home/water/PycharmProjects/skeleton_net/output/model_dump/'
    'SkeletonNet.trained.on.AMB/snapshot_49.pth.tar')['network'])

trainset_loader = Dataset(transforms.ToTensor(), "train")
batch_generator = DataLoader(dataset=trainset_loader, batch_size=1#cfg.num_gpus * cfg.train_batch_size
                             ,shuffle=False, num_workers=24, pin_memory=True)
print(trainset_loader.__len__())
n_samples = 50
regression_joint_annot = []
set_seed(0)
# for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
#         regression_joint_dict = {}
#         img = inputs['img'][0].cuda()  # [16, 3, 256, 256]
#         joint_sequence = targets['joint_sequence']
#         # start_pixel = joint_sequence[0][0].to(device)
#         # pixels = sample(model, img, steps=21, n_samples=n_samples,
#         #                 temperature=1.0, sample=True, top_k=50) # (n_sample, 21, 3)
#
#         joint_coord = targets['joint_coord'][:, :21]*4  # [16, 42, 3]
#         joint_sequence = targets['joint_sequence']
#         kmean_joint_coord = torch.from_numpy(trainset_loader.skeleton2sequence[joint_sequence])*4
#         kmean_error = (kmean_joint_coord - joint_coord)
#         kmean_error = torch.sqrt(torch.sum(kmean_error**2)/21)
#         kprint(itr)
#         print(f'kmean_error:{kmean_error}')
#         pred_joint_coords = torch.from_numpy(trainset_loader.skeleton2sequence[pixels.cpu()])*4
#         pre_error = (pred_joint_coords - joint_coord).reshape(n_samples,-1) # (b, 21, 3)
#         pre_error = torch.sqrt(torch.sum(pre_error**2, axis=1)/21)
#         print(f'pre_error:{pre_error.min().item()}, idx:{torch.argmin(pre_error).item()}')
#         regression_joint_dict['img_path'] = inputs['img_path'][0][77:]
#         regression_joint_dict['pre_joint'] = pred_joint_coords[torch.argmin(pre_error)]
#         regression_joint_dict['gt_joint'] = joint_coord
#         regression_joint_dict['bg_idx'] = inputs['bg_idx'][0]
#         regression_joint_dict['bbox'] = inputs['bbox']
#         regression_joint_dict['org_joint_coord'] = inputs['org_joint_coord'][0]
#         regression_joint_annot.append(regression_joint_dict)
# torch.save(regression_joint_annot,'test_regression_joint_annot')
# exit()


counts = torch.ones(256) # start counts as 1 not zero, this is called "smoothing"
rp = torch.randperm(len(trainset_loader))
n_samples = 9
top_kkk = 50
import matplotlib
matplotlib.use('TkAgg')
for itr, (inputs, targets, meta_info) in enumerate(batch_generator):
    # if meta_info['frame']>=10 and  meta_info['gesture'] in [3,7,9,10,15,17,18,21,22,25,
    #                                                         30,31,32,33,34,35,36,37,38,39,
    #                                                         40,41,42,43,44]:
    # if meta_info['frame'] == 18 and meta_info['gesture'] == 32 and meta_info['subject'] == 0:

        subject = meta_info['subject']
        gesture = meta_info['gesture']
        frame = meta_info['frame']
        img = inputs['img'][0].cuda()  # [16, 3, 256, 256]
        # start_pixel = joint_sequence[0][0].to(device)
        pixels = sample(model, img, steps=21, n_samples=n_samples,
                        temperature=1.0, sample=True, top_k=top_kkk)
        joint_coord = targets['joint_coord'][:,:21]  # [16, 42, 3]
        joint_sequence = targets['joint_sequence'][:,:21]
        kmean_joint_coord = torch.from_numpy(trainset_loader.skeleton2sequence[joint_sequence])
        joint_coord = skeleton_inter2mano(joint_coord[0])
        kmean_joint_coord = skeleton_inter2mano(kmean_joint_coord[0])


        kmean_error = calc_joint_error(kmean_joint_coord.numpy()*256/cfg.output_hm_shape,joint_coord.numpy()*256/cfg.output_hm_shape)
        cv2.imshow(f'Render_kmean{kmean_error:.2f}', draw1_2d_skeleton(img.cpu().numpy().transpose(1,2,0)[...,[2,1,0]], kmean_joint_coord[:21].numpy()*256/cfg.output_hm_shape))
        cv2.moveWindow(f'Render_kmean{kmean_error:.2f}', 50 + 380 * 3, 50 + 300 )
        cv2.imshow(f'{subject}:{gesture}:{frame}',draw1_2d_skeleton(img.cpu().numpy().transpose(1,2,0)[...,[2,1,0]], joint_coord[:21].numpy()*256/cfg.output_hm_shape))
        cv2.moveWindow(f'{subject}:{gesture}:{frame}', 50 + 380 * 3, 50 + 300 *2)
        pred_joint_coords = torch.from_numpy(trainset_loader.skeleton2sequence[pixels.cpu()])
        for i in range(n_samples):
            pred_joint_coord = skeleton_inter2mano(pred_joint_coords[i])
            pre_error = calc_joint_error(pred_joint_coord.numpy() * 256 / cfg.output_hm_shape,joint_coord.numpy() * 256 / cfg.output_hm_shape)
            kprint(pre_error)
            kprint(pred_joint_coord)
            kprint(joint_coord)
            cv2.imshow(f'pred_amb{i}_{pre_error:.2f}', draw1_2d_skeleton(img.cpu().numpy().transpose(1, 2, 0)[..., [2, 1, 0]],
                     pred_joint_coord.numpy() * 256 / cfg.output_hm_shape))
            cv2.moveWindow(f'pred_amb{i}_{pre_error:.2f}', 50+380*int(i%3), 50+300*int(i/3))
            # draw_3d_skeleton(pred_joint_coord, 331+i)
            # if itr % 4 == 1: cv2.moveWindow(f'pred_amb{itr % 4}', 430, 50)
            # if itr % 4 == 2: cv2.moveWindow(f'pred_amb{itr % 4}', 50, 350)
            # if itr % 4 == 3: cv2.moveWindow(f'pred_amb{itr % 4}', 430, 350)
        cv2.waitKey(2)
        plt.show()
        cv2.waitKey(0)
# 0, 32, 18
# 2, 39, 12

