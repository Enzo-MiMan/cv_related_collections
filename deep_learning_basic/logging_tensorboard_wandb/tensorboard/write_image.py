from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np
import os


file_name = ['image1', 'image2', 'image3']
file_list = [os.path.join('./images', i+'.jpg') for i in file_name]

writer = SummaryWriter(comment='_images')
for i in range(0, 3):
    writer.add_image('images',
                     np.array(Image.open(file_list[i])),
                     global_step=i,
                     dataformats='HWC')
writer.close()
