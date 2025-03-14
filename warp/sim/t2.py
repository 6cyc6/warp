import numpy as np
import torch
import warp as wp
import copy
import time

wp.init()

a = torch.rand((3000, 3), dtype=torch.float32, device='cuda:0')
b = torch.rand((3000, 3), dtype=torch.float32, device='cuda:0')
c = torch.rand((3000, 3), dtype=torch.float32, device='cuda:0')

# np_a = np.random.random((3000, 3))
# np_b = np.random.random((3000, 3))
# np_c = np.random.random((3000, 3))
#
# wa = wp.from_numpy(np_a, device='cuda')
# wb = wp.from_numpy(np_b, device='cuda')
# wc = wp.from_numpy(np_c, device='cuda')
wa = wp.from_torch(a)
wb = wp.from_torch(b)
wc = wp.from_torch(c)

k = 0
for i in range(500):
    k += i

s = time.time()

# ta = wp.to_torch(wa)
# tb = wp.to_torch(wb)
# tc = wp.to_torch(wc)
#
# ta[:, 2] = ta[:, 2] + 0.02
# tb[:, 2] = tb[:, 2] + 0.02
# tb[:, 2] = tb[:, 2] + 0.02
#
# wa = wp.from_torch(ta)
# wb = wp.from_torch(tb)
# wc = wp.from_torch(tc)
np_a = wa.numpy()
np_b = wb.numpy()
np_c = wc.numpy()
#
# a = torch.from_numpy(np_a).to("cuda:0")
# b = torch.from_numpy(np_b).to("cuda:0")
# c = torch.from_numpy(np_c).to("cuda:0")

e = time.time()
print(e - s)
