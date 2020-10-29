from glob import glob
import os
import numpy as np
path = os.path.abspath(os.path.dirname(__file__))
images = glob(os.path.join(path, 'train/*.jpg'))

shuffle = np.random.permutation(len(images))

try:
    os.mkdir(os.path.join(path,'valid'))

    for t in ['train', 'valid']:
        for folder in ['dog', 'cat']:
            os.mkdir(os.path.join(path,t,folder))


except:
    pass
  
for i in shuffle[:2000]:
    folder = images[i].split('/')[-1].split('.')[0]
    image = images[i].split('/')[-1]
    print(images[i])
    print(os.path.join(path, folder, image))
    os.rename(images[i], os.path.join(path,'valid', folder, image))

for i in shuffle[2000:]:
    folder = images[i].split('/')[-1].split('.')[0]
    image = images[i].split('/')[-1]
    os.rename(images[i], os.path.join(path,'train', folder, image))
