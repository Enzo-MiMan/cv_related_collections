import numpy as np
from PIL import Image


def nearest_neighbor_interpolation(image, scale_factor):
    """
    最邻近插值算法
    :param input_array: 输入图像数组
    :param output_shape: 输出图像的 shape
    :return: 输出图像数组
    """
    # 输入图像、输出图像的宽高
    height, width = image.shape[:2]
    out_height, out_width = int(height * scale_factor), int(width * scale_factor)

    # 创建输出图像
    output_image = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # 遍历输出图像的每个像素，分别计算其在输入图像中最近的像素坐标，并将其像素值赋值给当前像素
    for out_y in range(out_height):
        for out_x in range(out_width):
            # 计算当前像素在输入图像中的坐标
            input_x = round(out_x / scale_factor)
            input_y = round(out_y / scale_factor)
            # 判断计算出来的输入像素坐标是否越界，如果越界则赋值为边界像素
            input_x = min(input_x, width - 1)
            input_y = min(input_y, height - 1)
            # 将输入像素的像素值赋值给输出像素
            output_image[out_y, out_x] = image[input_y, input_x]
    return output_image


# 读取原始图像
input_image = Image.open('original_image.jpg')
image_array = np.array(input_image)

# 输出缩放后的图像
output_array = nearest_neighbor_interpolation(image_array, 1.5)
output_image = Image.fromarray(output_array)

input_image.show()
output_image.show()







