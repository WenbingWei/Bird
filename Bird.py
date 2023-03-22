import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open("/home3/weiwb/code/Multi-Channel-Model-master/bird.jfif")
s = np.array(img.convert('L'))

# plt.imshow(s, cmap='gray')
# plt.imsave("Gray_bird.jpg", s)
# pylab.show()
seed_torch()
img1 = s.astype(np.float32)
img = torch.from_numpy(np.asarray(img1)).float()
img = img.unsqueeze(dim=0)
img = img.unsqueeze(dim=0)  # [1,1,600,960]
# img = img.permute(0, 3, 1, 2)  # [B,C,H,W]
encoder = UnetEncoder()
out = encoder(img)

out = out.squeeze()
# out = out.permute(1, 2, 0)
out = out.detach().numpy()
out = (out).astype(np.uint8)

plt.imshow(out)  # 显示卷积后的图片
plt.imsave("UnetEncoder.jpg", out)
pylab.show()
