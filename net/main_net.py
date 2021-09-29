from .misc import NestedTensor, nested_tensor_from_tensor_list
import torch,torchvision,math
from common.config import cfg
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from common.kvision import *
from .backbone import build_backbone
from typing import Optional
from net.resnet import ResNetBackbone
from net.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from torchvision.models import resnet50
from torch.utils.data.sampler import WeightedRandomSampler
# vocab_size = 32*32*32 #
# block_size = 21 # joints_num

# mconf = GPTConfig(vocab_size, block_size, n_layer=6, n_head=8, n_embd=512)
# gpt_net = GPT(mconf)

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ImagetoSkeleton(nn.Module):
    def __init__(self,):
        super(ImagetoSkeleton, self).__init__()
        ####################
        # backbone
        ####################
        self.img_vocab_size = 512
        self.backbone = build_backbone(self.img_vocab_size)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.img_vocab_size, kernel_size=1)

        ####################
        # transformer
        ####################
        # Encoder
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.img_vocab_size, nhead=8)
        self.transformer = nn.Transformer(d_model=self.img_vocab_size, nhead=8)
        # Decoder
        self.joint_vocab_size = 256+1
        self.hidden = 512
        self.joint_block_size = 21
        self.tok_emb = nn.Embedding(self.joint_vocab_size, self.hidden)
        #print(self.tok_emb.shape)   0-256个词，每个词编码512个向量
        #exit()
        self.pos_decoder = PositionalEncoding(self.hidden, dropout=0.1)
        # self.joint_pos_emb = nn.Parameter(torch.zeros(21, 1, self.joint_vocab_size))
        # self.transformer_decoder = nn.TransformerDecoderLayer(d_model=self.joint_vocab_size, nhead=8)
        self.decoder = nn.Linear(self.hidden, self.joint_vocab_size-1)
        # mask
        self.register_buffer("mask", torch.tril(torch.ones(self.joint_block_size, self.joint_block_size))
                                     .view(1, 1, self.joint_block_size, self.joint_block_size))
        self.skeleton2sequence = torch.tensor(np.load(f'{cfg.root_dir}/net/rs1_skeleton_to_sequence_256.npy', allow_pickle=True)).cuda()

    def square_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def top_k_logits(self,logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out

    def forward(self, samples: NestedTensor,
                targets: Optional[Tensor] = None,
                mode:str='train')->Tensor:
        '''
        Args:
            samples: samples.tensors=img (b,3,256,256)
            targets: (21, b) onehot [0, 512]
            lossfunction: Cross Entropy

        Returns:
            joint_sequence: (b, 21, 512)
        '''
        features, pos = self.backbone(samples)

        img_features, mask = features[-1].decompose()  # [b, 2048, 16, 16])
        #print(len(features))    1
        print(img_features.shape)
        print(img_features[0][0])
       # print(pos[-1].shape)  torch.Size([48, 512, 16, 16])
        #print(img_features.shape)  torch.Size([48, 2048, 16, 16])
        #print(img_features[0][0])
        img_features = self.input_proj(img_features,)
        #print(img_features.shape)torch.Size([48, 512, 16, 16])
        img_features = img_features.flatten(2).permute(2, 0, 1)
        #print(img_features.shape) 256,48,512
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
        src = img_features+pos_embed
        #print(src.shape)torch.Size([256, 48, 512])
        #print(targets.shape)  21,48
        #print(targets[:-1,:].shape) 20,48
        #print(targets[:-1].shape)    20,48
        # train
        if mode == 'train':
           # print(torch.ones_like(targets[0,:]).shape)

            '''
            torch.ones_like(targets[0,:][None,:]).shape   1,48
            torch.ones_like(targets[:-1, :]).shape  20,48
            torch.ones_like(targets[0,:]   48
              
            '''
            joint_squence = torch.cat((torch.ones_like(targets[0,:][None,:])*256, targets[:-1, :]), 0)  # (21, b)
            #这个是要送入解码器的query。在训练过程中，第一个送进去256，后面送入的20个是tagets里面的值。
        else:
            joint_squence = targets
        #print(joint_squence.shape) torch.Size([21, 48])

        token_embeddings = self.tok_emb(joint_squence)  # [21, b, 512]
        #print(token_embeddings.shape)   torch.Size([21, 48, 512])
        #也就是说embedding的init的第一个值是forward中输入的值的范围
        token_embeddings = self.pos_decoder(token_embeddings)  # [21, b, 512]
        #print(token_embeddings.shape)   torch.Size([21, 48, 512])

        tgt_mask = self.square_mask(len(token_embeddings)).to(token_embeddings.device) # [21, 21]
        transformer_feature = self.transformer(src, token_embeddings,  tgt_mask=tgt_mask)
        #src(256,48,512)  tgt/token_embeddings  21,48,512
        #print(transformer_feature.shape)torch.Size([21, 48, 512])


        pre_joint_sequence = self.decoder(transformer_feature)
        #print(pre_joint_sequence.shape)torch.Size([21, 48, 256])

        loss = None
        if mode == 'train':
            loss = {}
            #print(pre_joint_sequence.size(-1))   256
            #print(pre_joint_sequence.reshape(-1, pre_joint_sequence.size(-1)).shape)[1008, 256]
            #print(targets.reshape(-1).shape)    1008
            #print(targets.reshape(0).shape)   不能这么用
            #exit()
            loss['sequence'] = F.cross_entropy(pre_joint_sequence.reshape(-1, pre_joint_sequence.size(-1)), targets.reshape(-1))
        return pre_joint_sequence, loss

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)

    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat


class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num  # single hand

        self.joint_deconv_1 = make_deconv_layers([2048, 256, 256, 256])
        self.joint_conv_1 = make_conv_layers([256, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1,
                                             padding=0, bnrelu_final=False)
        self.joint_deconv_2 = make_deconv_layers([2048, 256, 256, 256])
        self.joint_conv_2 = make_conv_layers([256, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1,
                                             padding=0, bnrelu_final=False)

        self.root_fc = make_linear_layers([2048, 512, cfg.output_root_hm_shape], relu_final=False)
        self.hand_fc = make_linear_layers([2048, 512, 2], relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, img_feat):  # [b, 2048, 8, 8] [b, 21, 32, 32, 32]
        joint_img_feat_1 = self.joint_deconv_1(img_feat)  # [b, 256, 64, 64]
        # heatmap3d [b, 21, 64, 64, 64]
        joint_heatmap3d_1 = self.joint_conv_1(joint_img_feat_1).view(-1, self.joint_num, cfg.output_hm_shape[0],
                                                                     cfg.output_hm_shape[1], cfg.output_hm_shape[2])

        # joint_img_feat_2 = self.joint_deconv_2(img_feat)
        # joint_heatmap3d_2 = self.joint_conv_2(joint_img_feat_2).view(-1, self.joint_num, cfg.output_hm_shape[0],
        #                                                              cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        # kprint(joint_heatmap3d_2.shape)
        # joint_heatmap3d = torch.cat((joint_heatmap3d_1, joint_heatmap3d_2), 1)
        # kprint(joint_heatmap3d.shape)

        # img_feat_gap = F.avg_pool2d(img_feat, (img_feat.shape[2], img_feat.shape[3])).view(-1, 2048)
        # root_heatmap1d = self.root_fc(img_feat_gap)
        # root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)
        # hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))
        # return joint_heatmap3d, root_depth, hand_type
        return joint_heatmap3d_1
# print(PoseNet(21))
# exit()
def joint2heatmap(joint_coord):
    x = torch.arange(cfg.output_hm_shape[2])
    y = torch.arange(cfg.output_hm_shape[1])
    z = torch.arange(cfg.output_hm_shape[0])
    zz, yy, xx = torch.meshgrid(z, y, x)
    xx = xx[None, None, :, :, :].cuda().float();
    yy = yy[None, None, :, :, :].cuda().float();
    zz = zz[None, None, :, :, :].cuda().float();

    x = joint_coord[:, :, 0, None, None, None];
    y = joint_coord[:, :, 1, None, None, None];
    z = joint_coord[:, :, 2, None, None, None];
    heatmap = torch.exp(
        -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
    heatmap = heatmap * 255
    heatmap = heatmap.float()
    return heatmap


def heatmap2joint(heatmap):
    val_z, idx_z = torch.max(heatmap, 2)
    val_zy, idx_zy = torch.max(val_z, 2)
    val_zyx, joint_x = torch.max(val_zy, 2)
    joint_x = joint_x[:, :, None]
    joint_y = torch.gather(idx_zy, 2, joint_x)
    joint_z = torch.gather(idx_z, 2, joint_y[:, :, :, None].repeat(1, 1, 1, cfg.output_hm_shape[1]))[:, :, 0, :]
    joint_z = torch.gather(joint_z, 2, joint_x)
    joint_coord_out = torch.cat((joint_x, joint_y, joint_z), 2).float()
    return joint_coord_out

class Refinenet(nn.Module):
    def __init__(self,  PoseNet):
        super(Refinenet, self).__init__()
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc
        self.posenet = PoseNet
        self.joint_conv_1 = make_conv_layers([21 * cfg.output_hm_shape[0], 256], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.joint_conv_2 = make_conv_layers([512, 256], kernel=1, stride=1, padding=0, bnrelu_final=False)


    def forward(self,img: Tensor,
                pre_joint: Optional[Tensor] = None,  # (prep)
                gt_joint: Optional[Tensor] = None,  # (prep)
                mode:str='train')->Tensor:
        pre_heatmap = joint2heatmap(pre_joint).reshape(-1, 21*cfg.output_hm_shape[2], cfg.output_hm_shape[1], cfg.output_hm_shape[0])  # [b, 21, 64, 64, 64]
        pre_heatmap = self.joint_conv_1(pre_heatmap)  # [b, 256, 64, 64]
        x = self.backbone.conv1(img)  # [2, 64, 128, 128]
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # [2, 64, 64, 64]
        x = self.backbone.layer1(x)  # [2, 256, 64, 64]
        x = torch.cat((pre_heatmap, x), 1)  # [2, 512, 64, 64]
        x = self.joint_conv_2(x)  # [b, 256, 64, 64]

        x = self.backbone.layer2(x)  # [2, 512, 32, 32]
        x = self.backbone.layer3(x)  # [2, 1024, 16, 16]
        img_feat = self.backbone.layer4(x)  # [b, 2048, 8, 8]
        # # convert from 2048 to 256 feature planes for the transformer
        # h = self.conv(x)

        refine_residual = self.posenet(img_feat)
        # kprint(refine_residual.shape)
        # kprint(pre_joint.shape)
        # kprint(gt_joint.shape)
        # kprint(self.joint2heatmap(pre_joint).shape)
        refine_heatmap = refine_residual * joint2heatmap(pre_joint)
        # print('Kmeans: ', F.smooth_l1_loss(self.joint2heatmap(pre_joint), self.joint2heatmap(gt_joint)))
        # print('Refine: ', F.smooth_l1_loss(refine_heatmap, self.joint2heatmap(gt_joint)).item())
        loss = None
        if mode == 'train':
            loss = {}
            refine_joint=heatmap2joint(refine_heatmap)
            loss['heatmap1'] = F.smooth_l1_loss(refine_heatmap, joint2heatmap(gt_joint))

        if mode == 'test':
            val_z, idx_z = torch.max(refine_heatmap, 2)
            val_zy, idx_zy = torch.max(val_z,2)
            val_zyx, joint_x = torch.max(val_zy,2)
            joint_x = joint_x[:,:,None]
            joint_y = torch.gather(idx_zy, 2, joint_x)
            joint_z = torch.gather(idx_z, 2, joint_y[:,:,:,None].repeat(1,1,1,cfg.output_hm_shape[1]))[:,:,0,:]
            joint_z = torch.gather(joint_z, 2, joint_x)
            joint_coord_out = torch.cat((joint_x, joint_y, joint_z), 2).float()
            return joint_coord_out, refine_heatmap

        return refine_heatmap, loss,refine_joint

# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from data.AMB.dataset import Dataset
# from data.AMB.reg_dataset import Reg_Dataset
#
# trainset_loader = Reg_Dataset(root=cfg.root_dir+'/data/AMB/kmean_train_joint_annot')
# batch_generator = DataLoader(dataset=trainset_loader, batch_size=2, shuffle=True, num_workers=24#, pin_memory=True
#                              )
# posenet = PoseNet(21)
# def init_weights(m):
#     if type(m) == nn.ConvTranspose2d:
#         nn.init.normal_(m.weight,std=0.001)
#     elif type(m) == nn.Conv2d:
#         nn.init.normal_(m.weight,std=0.001)
#         nn.init.constant_(m.bias, 0)
#     elif type(m) == nn.BatchNorm2d:
#         nn.init.constant_(m.weight,1)
#         nn.init.constant_(m.bias,0)
#     elif type(m) == nn.Linear:
#         nn.init.normal_(m.weight,std=0.01)
#         nn.init.constant_(m.bias,0)
# posenet.apply(init_weights)
# model_path = '/home/water/PycharmProjects/skeleton_net/common/../output/model_dump/SkeletonNet.trained.on.AMB/refine_snapshot_49.pth.tar'
# refinenet = Refinenet(posenet).cuda()
# ckpt = torch.load(model_path)
# refinenet.load_state_dict(ckpt['network'])
# kmean_error = 0
# refine_error = 0
# for itr, (inputs, pre_joint, gt_joint) in enumerate(batch_generator):
#     pre_joint = pre_joint.cuda()
#     gt_joint = gt_joint.cuda()
#     refine_joint, refine_heatmap = refinenet(inputs.cuda(), pre_joint, gt_joint, mode='test')
#
#     pre_joint = pre_joint.cuda()*4
#     gt_joint = gt_joint.cuda()*4
#     refine_joint = refine_joint*4
#     kmean_error = torch.sqrt(torch.sum((pre_joint - gt_joint) ** 2) / 21)
#     refine_error = torch.sqrt(torch.sum((refine_joint - gt_joint) ** 2) / 21)
#     # print(f'Base_error: {kmean_error/(itr+1)} | refine: {refine_error/(itr+1)}')
#     print(f'Base_error: {kmean_error} | refine: {refine_error}')
#
#     print(pre_joint.shape)
#     pre_joint = skeleton_inter2mano(pre_joint[0])
#     gt_joint = skeleton_inter2mano(gt_joint[0])
#     refine_joint = skeleton_inter2mano(refine_joint[0])
#     if kmean_error-refine_error > 6:
#         cv2.imshow(f'kmean_amb',
#                    draw1_2d_skeleton(inputs[0].numpy().transpose(1, 2, 0)[..., [2, 1, 0]],
#                                      pre_joint.cpu().detach().numpy()))
#         cv2.imshow(f'gt_amb',
#                    draw1_2d_skeleton(inputs[0].numpy().transpose(1, 2, 0)[..., [2, 1, 0]],
#                                      gt_joint.cpu().detach().numpy()))
#         cv2.imshow(f'refine_amb',
#                    draw1_2d_skeleton(inputs[0].numpy().transpose(1, 2, 0)[..., [2, 1, 0]],
#                                      refine_joint.cpu().detach().numpy()))
#         cv2.waitKey(0)
#     # print(loss['heatmap1'].shape)
#     # print(loss['heatmap1'].max())
#     # print(refine_joint.shape)
#     # exit()
# # exit()




