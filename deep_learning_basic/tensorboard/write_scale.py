from tensorboardX import SummaryWriter
import torch

writer = SummaryWriter()

for i in range(10):
    writer.add_scalar('quadratic', torch.tensor(i**2), global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)
writer.close()

# 查看方式
# step 1 ： cd 到生成的 runs 同级目录下
# step 2 ： 在终端输入
# step 3 ： 再浏览器中输入地址： http://localhost:6006/
