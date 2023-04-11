#!/usr/bin/python
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rle_img import rle_encode
import cv2
from torchvision import transforms as T

IMAGE_SIZE = 256

model_root = '../user_data/model_data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load(os.path.join(model_root, 'model.pth'))
print(model)
model = model.eval()
model.to(DEVICE)

test_path = '../tcdata/test_a'
test_mask = pd.read_csv('../tcdata/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
# test_mask['name'] = test_mask['name'].apply(lambda x: '../tcdata/test_a/' + x)

'''
img = cv2.imread(dataset.paths[0])  # 512*512*3
plt.figure()
plt.imshow(img)
plt.show()
'''

trfm = T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize([0.625, 0.448, 0.688],
                    [0.131, 0.177, 0.101]),
    ])
n = 0

subm = []
for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
    image = cv2.imread(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]
        score = model(image)['out'][0][0]
        score_sigmoid = score.sigmoid().cpu().numpy()
        score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
        score_sigmoid = cv2.resize(score_sigmoid, (512, 512))
        # break
    subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
subm = pd.DataFrame(subm)
subm.to_csv('../prediction_result/result.csv', index=None, header=None, sep='\t')
'''
for name in test_mask['name']:

    image = Image.open(test_path + '/' + name)  # 512*512
    image = np.array(image)
    image = as_tensor(image)  # tensor 3*256*256
    image = image.unsqueeze(dim=0)  # tensor 1*3*256*256
    image = image.to(DEVICE)

    y = model(image)['out']
    y = y.cpu()
    y = y.squeeze(dim=0).squeeze(dim=0)  # 1*1*256*256 -> 256*256
    y = y * 255
    y = np.where(y >= 108.5, 255, 0)
    y = y.astype('uint8')
    im = Image.fromarray(y)
    
    plt.figure()
    plt.imshow(im)
    plt.show()
    
    y = np.where(y > 122.5, 1, 0)
    # y = y[:, :, 0]
    rle = rle_encode(y)
    test_mask['mask'][n] = rle
    n += 1

test_mask.to_csv('../prediction_result/result.csv', index=None, header=None, sep='\t')
'''
