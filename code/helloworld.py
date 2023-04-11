import torch.nn as nn
import torch

a = torch.tensor([[1,2,3,4],
                 [5,6,7,8]])
print(a)
print(torch.max(a, 1)[0])
print(a.data.max(1, keepdim=True)[0])  # 0输出最大值
print(a.data.max(1, keepdim=True)[1])  # 1输出最大值的位置