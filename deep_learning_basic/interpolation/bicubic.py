import torch
import torch.nn.functional as F
import numpy as np


def bicubic_weight_function(x):
  # the coefficients of the bicubic polynomial.
  a = -0.75
  x = torch.abs(x)
  # Calculate the bicubic weight.
  weight = ((a + 2.0) * torch.pow(x, 3) - (a + 3.0) * torch.pow(x, 2) + 1.0) * (x <= 1.0) + \
           (a * torch.pow(x, 3) - 5.0 * a * torch.pow(x, 2) + 8.0 * a * x - 4.0 * a) * ((x > 1.0) & (x <= 2.0))
  return weight.repeat(4, 1)


if __name__ == '__main__':

    image = torch.tensor([[[[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]]]], dtype=torch.float32)

    out_height, out_width = 5, 5

    # Get the input dimensions.
    _, _, height, width = image.shape

    scale_y = height / out_height
    scale_x = width / out_width

    image_pad = F.pad(image, (2, 2, 2, 2), mode='replicate')

    # 创建输出图像
    output_image = np.zeros((out_height, out_width), dtype=np.float32)

    for out_y in range(out_height):
        for out_x in range(out_width):

            x = (out_x + 0.5) * scale_x + 1.5
            y = (out_y + 0.5) * scale_y + 1.5

            # calculate weights
            delta_x = x % 1
            delta_y = y % 1
            distance_x = torch.tensor([delta_x+1, delta_x, 1-delta_x, 2-delta_x])
            distance_y = torch.tensor([delta_y+1, delta_y, 1-delta_y, 2-delta_y])

            weight_x = bicubic_weight_function(distance_x)
            weight_y = bicubic_weight_function(distance_y).T

            # calculate index
            index_x = round(x+0.5)
            index_y = round(y+0.5)
            source = image_pad[:, :, index_y-2:index_y+2, index_x-2:index_x+2].squeeze()

            output_image[out_y, out_x] = torch.multiply(torch.multiply(source, weight_x), weight_y).sum()
    print(output_image)


    # =======================================================================

    result = F.interpolate(image, (out_height, out_width), mode='bicubic', align_corners=False)
    print(result)



