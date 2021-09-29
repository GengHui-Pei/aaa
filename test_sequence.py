import random
import torch
from torch.nn import functional as F
from net.main_net import ImagetoSkeleton
import torch.backends.cudnn as cudnn
from net.misc import NestedTensor
from common.base import Trainer
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

    #print(n_img.shape)torch.Size([50, 3, 256, 256])
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
from net.misc import NestedTensor
from common.utils.preprocessing import *
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=str(0), type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset')
    parser.add_argument('--dataset', type=str, dest='dataset')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
args = parse_args()
cfg.set_args(args.gpu_ids, args.continue_train)
cudnn.benchmark = True
trainer = Trainer()
trainer._make_batch_generator()
if args.dataset=='interhand' :

    trainer._make_model(pre_train=cfg.model_dir + '/SkeletonNet.trained.on.AMB/snapshot_39.pth.tar')
else:
    trainer._make_model(pre_train=cfg.model_dir + '/SkeletonNet.trained.on.AMB/o_snapshot_35.pth.tar')

trainer._make_test_batch_generator()


preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}
n_samples = 1
for itr, (inputs, targets, meta_info) in enumerate(trainer.test_batch_generator):
    with torch.no_grad():
        img = inputs['img'][0].cuda()
        joint_sequence = targets['joint_sequence']
        joint_coord = targets['joint_coord'][:, :21]  # [16, 42, 3]

        joint_sequence = targets['joint_sequence']
        # print(joint_sequence.shape)   (1,21)
        pixels = sample(trainer.model, img, steps=21, n_samples=n_samples,
                        temperature=1.0, sample=False, top_k=None)  # (n_sample, 21, 3)
        print(pixels)

        pred_joint_coords = torch.from_numpy(trainer.testset_loader.skeleton2sequence[pixels.cpu()])
        joint_coord = np.zeros([pred_joint_coords.shape[0], 42, 3])
        joint_coord[:, :21] = pred_joint_coords.detach().cpu().numpy()

        preds['joint_coord'].append(joint_coord)  # targets['joint_coord_kmean_space'].cpu().numpy())
        preds['rel_root_depth'].append(targets['rel_root_depth'].cpu().numpy())
        preds['hand_type'].append(targets['hand_type'].cpu().numpy())
        preds['inv_trans'].append(meta_info['inv_trans'].cpu().numpy())

preds = {k: np.concatenate(v) for k, v in preds.items()}
trainer.testset_loader.evaluate(preds)




















