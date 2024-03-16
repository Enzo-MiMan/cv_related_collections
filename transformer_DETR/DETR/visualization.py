# #------------------------------------------------------------#
# 可视化Detr方法：
# spatial attention weight : (cq + oq)*pk
# combined attention weight: (cq + oq)*(memory + pk)
# 其中:
#     pk:原始特征图的位置编码;
#     oq:训练好的object queries
#     cq:decoder最后一层self-attn中的输出query
#     memory:encoder的输出
# #------------------------------------------------------------#
# 在此基础上只要稍微修改便可可视化ConditionalDetr的Fig1特征图
# #------------------------------------------------------------#
# 代码参考自:https://github.com/facebookresearch/detr/tree/colab
# #------------------------------------------------------------#

import numpy as np
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torch.nn.functional import dropout, linear, softmax
import torch.nn.functional as F

torch.set_grad_enabled(False)
import matplotlib


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = transforms.Compose([transforms.Resize(800),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 加载线上的模型
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# 获取训练好的参数
for name, parameters in model.named_parameters():
    # 获取训练好的object queries，即pq:[100,256]
    if name == 'query_embed.weight':
        pq = parameters  # q_position
    # 获取 decoder 最后一层的交叉注意力模块中q和k的线性权重和偏置:[256*3,256]，[768]
    if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_weight':
        in_proj_weight = parameters
    if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_bias':
        in_proj_bias = parameters

# 线上下载图像
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with 0.9+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# use lists to store the outputs via up-values
conv_features = []
enc_attn_weights = []
dec_attn_weights = []
cq = []  # 存储detr中的 cq(q_content)
pk = []  # 存储detr中的 encoder pos(k_position)
memory = []  # 存储encoder的输出特征图memory

# 注册hook
hooks = [
    # 获取resnet最后一层特征图
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    # 获取encoder的图像特征图memory
    model.transformer.encoder.register_forward_hook(
        lambda self, input, output: memory.append(output)
    ),
    # 获取encoder的最后一层layer的self-attn weights
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    # 获取decoder的最后一层layer中交叉注意力的 weights
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
    # 获取decoder最后一层self-attn的输出cq
    model.transformer.decoder.layers[-1].norm1.register_forward_hook(
        lambda self, input, output: cq.append(output)  #
    ),
    # 获取图像特征图的位置编码pk
    model.backbone[-1].register_forward_hook(
        lambda self, input, output: pk.append(output)
    ),
]

# propagate through the model
outputs = model(img)

# 用完的hook后删除
for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]  # [1,2048,25,34]
enc_attn_weights = enc_attn_weights[0]  # [1,850,850]   : [N,L,S]
dec_attn_weights = dec_attn_weights[0]  # [1,100,850]   : [N,L,S] --> [batch, tgt_len, src_len]
memory = memory[0]  # [850,1,256]
cq = cq[0]  # decoder 的 self_attn:最后一层输出[100,1,256]
pk = pk[0]  # [1,256,25,34]

# 绘制postion embedding
pk = pk.flatten(-2).permute(2, 0, 1)  # [1,256,850] --> [850,1,256]
pq = pq.unsqueeze(1).repeat(1, 1, 1)  # [100,1,256]
q = pq + cq
# ------------------------------------------------------#
#   1) k = pk，则可视化： (cq + oq)*pk
#   2_ k = pk + memory，则可视化 (cq + oq)*(memory + pk)
#   读者可自行尝试
# ------------------------------------------------------#
k = pk
# k = memory
# k = pk + memory   #这就是正常，完整的k了
# ------------------------------------------------------#

# 将q和k完成线性层的映射，代码参考自nn.MultiHeadAttn()
_b = in_proj_bias
_start = 0
_end = 256
_w = in_proj_weight[_start:_end, :]
if _b is not None:
    _b = _b[_start:_end]
q = F.linear(q, _w, _b)

_b = in_proj_bias
_start = 256
_end = 256 * 2
_w = in_proj_weight[_start:_end, :]
if _b is not None:
    _b = _b[_start:_end]
k = F.linear(k, _w, _b)

scaling = float(256) ** -0.5
q = q * scaling
q = q.contiguous().view(100, 8, 32).transpose(0, 1)
k = k.contiguous().view(-1, 8, 32).transpose(0, 1)
attn_output_weights = torch.bmm(q, k.transpose(1, 2))

attn_output_weights = attn_output_weights.view(1, 8, 100, 850)
attn_output_weights = attn_output_weights.view(1 * 8, 100, 850)
attn_output_weights = softmax(attn_output_weights, dim=-1)
attn_output_weights = attn_output_weights.view(1, 8, 100, 850)

# 后续可视化各个头
attn_every_heads = attn_output_weights  # [1,8,100,850]
attn_output_weights = attn_output_weights.sum(dim=1) / 8  # [1,100,850]

# -----------#
#   可视化
# -----------#
# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]

fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=10, figsize=(22, 28))  # [11,2]
colors = COLORS * 100

# 可视化
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    # 可视化decoder的注意力权重
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}', fontsize=30)
    # 可视化框和类别
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[probas[idx].argmax()], fontsize=30)
    # 分别可视化8个头部的位置特征图
    for head in range(2, 2 + 8):
        ax = ax_i[head]
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        heatmap = attn_every_heads[0, head - 2, idx].view(h, w)
        # 将一些小的权重值置为0，以免后面合成的时候干扰原图
        heatmap.masked_fill_((heatmap < torch.mean(heatmap)), 0.)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * 100 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_DEEPGREEN)  # [h,w,3]

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap = heatmap[:, :, (1, 0, 2)]

        # overlapping = img
        overlapping = cv2.addWeighted(img, 1.0, heatmap, 0.8, 0)
        overlapping = cv2.cvtColor(overlapping, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(overlapping)

        ax.imshow(img)

        ax.axis('off')
        ax.set_title(f'head:{head - 2}', fontsize=30)

fig.tight_layout()  # 自动调整子图来使其填充整个画布
plt.show()
plt.savefig('spatial_attn.jpg')



