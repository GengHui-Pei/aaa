import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from net.transformer import Transformer
from net.backbone import build_backbone
from net.misc import NestedTensor, nested_tensor_from_tensor_list

class DETRdemo(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)  # ([1, 64, 400, 533])
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)# [1, 64, 200, 267]
        x = self.backbone.layer1(x) # [1, 256, 200, 267]
        x = self.backbone.layer2(x) # [1, 512, 100, 134]
        x = self.backbone.layer3(x) # [1, 1024, 50, 67]
        x = self.backbone.layer4(x) # [1, 2048, 25, 34]
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x) # [1, 256, 25, 34]
        # construct positional encodings
        H, W = h.shape[-2:] # 25 34
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),   # [25, 34, 256/2]
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),   # [25, 34, 256/2]
        ], dim=-1).flatten(0, 1).unsqueeze(1)                  # [850, 1, 256]
        # propagate through the transformer
        # pos: [850, b, 256] + h: [850, b, 256]
        # query_pos: [b, 100, 256]
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1) # h [b, 100, 256]
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), # [1, 100, n_class+1])
                'pred_boxes': self.linear_bbox(h).sigmoid()} # [1, 100,4]


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

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.joint_encoder = MLP(3, 256, hidden_dim, 3)  # input_dim, hidden_dim, output_dim, num_layers
        self.joint_decoder = MLP(hidden_dim, 256, 3, 3)  # input_dim, hidden_dim, output_dim, num_layers

    def forward(self,samples,pre_joint,targets,mode):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        joint_feature = self.joint_encoder(pre_joint) #in [21, b, 3] #out [21, b, 512])
        hs = self.transformer(self.input_proj(src), mask, joint_feature, pos[-1])[0]
        refine_residual = self.joint_decoder(hs) # [21, b, 3]
        refine_joint = refine_residual + pre_joint
        loss = None
        if mode == 'train':
            loss = {}
            loss['joint_residual'] = F.smooth_l1_loss(refine_joint, targets)
        return refine_joint, loss



# backbone = build_backbone(256)
# transformer =  Transformer(d_model=256,dropout=0.1,nhead=8,dim_feedforward=2048,
#         num_encoder_layers=6,num_decoder_layers=6,
#         normalize_before=True,return_intermediate_dec=True)
#
# detr = DETR(backbone=backbone, transformer=transformer, num_classes=2, num_queries=100, aux_loss=False)
#
# from data.AMB.dataset import Dataset
# image,_,_ = Dataset(torchvision.transforms.ToTensor(), "test")[0]
# img = image['img'][None,:,:,:]
# img_mask = NestedTensor(img, torch.zeros_like(image['img'][0][None,:,:])>1)
# output = detr(img_mask)
# print(output.shape)
# exit()
