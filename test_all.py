import random
import torch
from torch.nn import functional as F
from net.main_net import ImagetoSkeleton
from net.misc import NestedTensor
from data.AMB.datas import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common.base import Refine_Trainer
import torch.backends.cudnn as cudnn
from net.misc import NestedTensor
from common.utils.preprocessing import *
import argparse
from skeleton.common.utils.vis import vis_keypoints, vis_3d_keypoints
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    #print(v.shape,logits.shape)torch.Size([10, 50]) torch.Size([10, 256])
    #这里的logits是一张图片的十个结果，256是256个点中每个点的概率

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
    #print(x.shape) torch.Size([10, 1])


    for k in range(steps):
        n_img_mask = NestedTensor(n_img, torch.zeros_like(n_img[:, 0, :, :], device=n_img.device) > 1)
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed

        x_cond = x_cond.transpose(0,1)
        logits, loss = model(n_img_mask, x_cond, mode='test')
        logits = logits.transpose(0,1)
        #print(logits.shape)torch.Size([10, 1, 256])

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

args = parse_args()
model = ImagetoSkeleton()
model = model.cuda()
if args.dataset=='interhand' :
    model.load_state_dict(torch.load(
        '/root/Workspace/jyk/skeleton/output/model_dump/SkeletonNet.trained.on.AMB/'
        'snapshot_39.pth.tar')['network'])
else:
    model.load_state_dict(torch.load(
        '/root/Workspace/jyk/skeleton/output/model_dump/SkeletonNet.trained.on.AMB/'
        'o_snapshot_35.pth.tar')['network'])


trainset_loader = Dataset(transforms.ToTensor(), "test")
batch_generator = DataLoader(dataset=trainset_loader, batch_size=1#cfg.train_batch_size
                             ,shuffle=False, num_workers=12, pin_memory=True
                             )
print(trainset_loader.__len__())
n_samples = 1
regression_joint_annot = []
regression_joint_dict = {}
set_seed(0)

cfg.set_args(args.gpu_ids, args.continue_train)
cudnn.benchmark = True

trainer = Refine_Trainer()
trainer._make_batch_generator()
trainer._make_test_batch_generator()
if args.dataset=='interhand' :
    trainer._make_model(pre_train=cfg.model_dir + '/SkeletonNet.trained.on.AMB/inter_refine_snapshot_49.pth.tar')
else:
    trainer._make_model(pre_train=cfg.model_dir + '/SkeletonNet.trained.on.AMB/refine_snapshot_6.pth.tar')

preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}

mpjpe = [[] for _ in range(21)]  # treat right and left hand identical
for itr, (inputs, targets, meta_info) in enumerate(trainer.test_batch_generator):
        regression_joint_dict = {}

        addr='/root/Workspace/jyk/skeleton/output/result/'+str(itr)
        #os.makedirs('/root/Workspace/jyk/skeleton/output/result/'+str(itr), exist_ok=True)
        #cv2.imwrite(addr+'/ori_img.jpg',(np.array(inputs['img'][0]).transpose((1,2,0))[:,:,[2,1,0]]*255).astype('uint8'))

        # if itr>1000: break
        img = inputs['img'].cuda()

        joint_coord = (targets['joint_coord'][:,:21]).cuda()  # [16, 42, 3]

        #print(joint_sequence.shape)   (1,21)
        pixels = sample(model, img, steps=21, n_samples=n_samples,
                        temperature=1.0, sample=False, top_k=None)  # (n_sample, 21, 3)
        #print(pixels.shape)torch.Size([50, 21])
        #print(pixels.shape)torch.Size([10, 21])
        joint_valid=trainset_loader.datalist[itr]["joint"]['valid']

        joint_valid = joint_valid[np.arange(0,21)]
        pred_joint_coords = torch.from_numpy(trainset_loader.skeleton2sequence[pixels.cpu()])
        pred_joint_coords = pred_joint_coords.cuda()
        #print(pred_joint_coords.shape)torch.Size([10, 21, 3])


        # for nn in range(n_samples) :
        #
        #
        #     refine_heatmap, loss, refine_joint = trainer.model(img,pred_joint_coords[nn][None,:,:], joint_coord, 'train')
        #     joint_coord1 = np.zeros([ 21, 3])
        #     joint_coord1 = refine_joint[0].detach().cpu().numpy()
        #     vis = True
        #     if vis:
        #         filename = addr+'/out_' + str(nn) + '_3d.png'
        #         vis_3d_keypoints(joint_coord1, joint_valid, trainset_loader.skeleton[:trainset_loader.joint_num], filename)




        refine_heatmap, loss, refine_joint = trainer.model(img, pred_joint_coords, joint_coord, 'train')


        joint_coord = np.zeros([refine_joint.shape[0], 42, 3])
        joint_coord[:, :21] = pred_joint_coords.detach().cpu().numpy()
        # print(joint_coord.shape)(12, 42, 3)
        # exit()
        preds['joint_coord'].append(joint_coord)  # targets['joint_coord_kmean_space'].cpu().numpy())
        preds['rel_root_depth'].append(targets['rel_root_depth'].cpu().numpy())
        preds['hand_type'].append(targets['hand_type'].cpu().numpy())
        preds['inv_trans'].append(meta_info['inv_trans'].cpu().numpy())

preds = {k: np.concatenate(v) for k, v in preds.items()}
trainer.testset_loader.evaluate(preds)






