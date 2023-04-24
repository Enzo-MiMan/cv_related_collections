import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch


def bilinear_interpolation(image, out_height, out_width, corner_align=False):
    # 获取输入图像的宽高
    height, width = image.shape[:2]

    # 创建输出图像
    output_image = np.zeros((out_height, out_width), dtype=np.float32)

    # 计算x、y轴缩放因子
    scale_x_corner = float(width - 1) / (out_width - 1)   # (3 - 1) / (5 - 1) = 0.5
    scale_y_corner = float(height - 1) / (out_height - 1)   # (3 - 1) / (5 - 1) = 0.5

    scale_x = float(width) / out_width   # 3 / 5 = 0.6
    scale_y = float(height) / out_height   # 3 / 5 = 0.6

    # 遍历输出图像的每个像素，分别计算其在输入图像中最近的四个像素的坐标，然后按照加权值计算当前像素的像素值
    for out_y in range(out_height):
        for out_x in range(out_width):
            if corner_align == True:
                # 计算当前像素在输入图像中的位置
                x = out_x * scale_x_corner
                y = out_y * scale_y_corner
            else:
                x = (out_x + 0.5) * scale_x - 0.5
                y = (out_y + 0.5) * scale_y - 0.5
                x = np.clip(x, 0, width-1)
                y = np.clip(y, 0, height-1)

            # 计算当前像素在输入图像中最近的四个像素的坐标
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1

            # 对原图像边缘进行特殊处理
            if x0 == width - 1:
                x0 = width - 2
                x1 = width - 1
            if y0 == height - 1:
                y0 = height - 2
                y1 = height - 1

            xd = x - x0
            yd = y - y0
            p00 = image[y0, x0]
            p01 = image[y0, x1]
            p10 = image[y1, x0]
            p11 = image[y1, x1]
            x0y = p01 * xd + (1 - xd) * p00
            x1y = p11 * xd + (1 - xd) * p10
            output_image[out_y, out_x] = x1y * yd + (1 - yd) * x0y

    return output_image


# 读取原始图像
image_array = np.arange(0, 9).reshape((3, 3))

# ========= 角对齐 =========
output_array_corner = bilinear_interpolation(image_array, 5, 5, corner_align=True)
print(output_array_corner)
print('*'*20)

# =========  边对齐 =========
output_array = bilinear_interpolation(image_array, 5, 5, corner_align=False)
print(output_array)
print('*'*20)

# ========= 用 torch.nn.functional 实现双线性插值 ================

image = torch.as_tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
result_corner = F.interpolate(image, size=(5, 5), mode='bilinear', align_corners=True)
print(result_corner)

image = torch.as_tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
result = F.interpolate(image, size=(5, 5), mode='bilinear', align_corners=False)
print(result)

