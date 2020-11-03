import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
  
import math

from PIL import Image
import numbers

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import os
if not os.path.exists('../output'):
    os.mkdir('../output')

train_df = pd.read_csv('./mnist_train.csv')
test_df = pd.read_csv('./mnist_test.csv')

n_train = len(train_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))

print('Number of training samples: {0}'.format(n_train))
print('Number of training pixels: {0}'.format(n_pixels))
print('Number of classes: {0}'.format(n_class))
random_sel = np.random.randint(n_train, size=8)

grid = make_grid(torch.Tensor((train_df.iloc[random_sel, 1:].as_matrix()/255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)
plt.rcParams['figure.figsize'] = (16, 2)
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
plt.show()
print(*list(train_df.iloc[random_sel, 0].values), sep = ', ')

plt.rcParams['figure.figsize'] = (8, 5)
plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts())
plt.xticks(np.arange(n_class))
plt.xlabel('Class', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.grid('on', axis='y')
plt.show()

transform = transforms.COmpose([
  transforms.ToPILImage(),
  transforms.ToTensor(),
  transforms.Normalize(mean(0.5), std=(0.5))
])
class MNIST_data(Dataset):
  def __init__(self,file_path,transform=transform):
    df = pd.read_csv(file_path)

    if len(df.columns) == n_pixels:
      self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
      self.y = None
    else:
      self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
      self.y = torch.from_numpy(df.iloc[:,0].values)
      self.transform = transform

  def __len__(self):
    return len(self.X)
  def __getitem__(self,idx):
    if self.y is not None:
      return self.transform(self.X[idx]),self.y[idx]
    else:
      return self.transform(self.X[idx])

class RandomRotation(object):
  def __init__(self,degrees,resample=False, expand=False,center=None):
    if isinstance(degrees, numbers.Number):
      if degrees <0:
        raise ValueError("If degrees is a single number, it must be positive")
      self.degrees = (-degrees,degrees)
    else:
      if len(degrees) !=2:
        raise ValueError("If degrees is a sequence, it must be of len2")
      self.degrees = degrees
    
    self.resample = resample
    self.expand = expand
    self.center = center

    @staticmethod
    def get_params(degrees):
      angle = np.random.uniform(degrees[0],degrees[1])
      return angle

    def __call__(self,img):
      def rotate(img,angle,resample=False,expand=False,center=None):
        return img.rotate(angle,resample,expand,center)

      angle = self.get_params(self.degrees)
      return rotate(img,angle,self.resample,self.expand,self.center)

class RandomShift(object):
  def __init__(self, shift):
    self.shift = shift

  @staticmethod
  def get_params(shift):
    hshift, vshift = np.random.uniform(-shift,shift,size=2)
    return hshift,vshift

  def __call__(self,img):
    hshift,vshift = self.get_params(self.shift)
    return img.transform(img.size,Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC,fill=1)
    
batch_size = 64
train_dataset = MNIST_data('./mnist_train.csv',transform = transforms.Compose(
  [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
  transforms.ToTensor(),transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data('./mnist_test.csv')

train_loader = torch.utils.data.DataLoader(dataset=train_datset, batch-size = batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

rotate = RandomRotation(20)
shift = RandomShift(3)
composed = transforms.Compose([RandomRotation(20),RandomShift(3)])

fig = plt.figure()
sample = transforms.ToPILImage()(train_df.iloc[65,1:].reshape((28,28)).astype(np.uint8)[:,:,None])
for i, tsfrm in enumerate([rotate,shift,composed]):
  transformed_sample = tsfrm(sample)

  ax = plt.subplot(1,3,i+1)
  plt.tight_layout()
  ax.set_title(type(tsfrm).__name__)
  ax.imshow(np.reshqpe(np.array(list(transformed_sample.getdata())),(-1,28)),cmap='gray')

plt.show()

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32,32,kenel_size=3,stride=1,padding =1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2,stride=2),
      nn.Conv2d(32,64,kenel_size=3,padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64,64,kernel_size=3,padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2,stride=2),
    )

    self.Classifier = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(64*7*7,512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(512,512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),
      nn.Linear(512,10),
    )

    for m in self.features.children():
      if isinstance(m,nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1]*m.out_channels
        m.weight.data.normal_(0,math.sqrt(2. /n))
      elif isinstance(m,nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    
    for m in self.Classifier.children():
      if isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight)
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self,x):
    x = self.features(x)
    x =  x.view(x.size(0), -1)
    x = self.Classifier(x)

    return x

model = Net()
optimizer = optim.Adam(model.parameters(),lr=0.003)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma =0.1)

if torch.cuda.is_available():
  model = model.cuda()
  criterion = criteroin.cuda()

def train(epoch):
    model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data))
    
    torch.save(model.state_dict(), "../output_MNIST/model.pt")


def evaluate(data_loader):
  model.eval()
  loss = 0
  correct = 0

  for data, target in  data_loader:
    data,target = Variable(data,volatile=True),Variable(target)
    if torch.cuda.is_available():
      data = data.cuda()
      target = target.cuda()
    output = model(data)
    loss +=F.cross_entropy(output, target, size_average= False).data
    pred = output.data.max(1,keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
  
  loss /= len(data_loader.dataset)

  print('\nAverage loss : {:4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    loss, correct, len(data_loader.dataset),
    100. * correct / len(data_loader.dataset)))
n_epochs = 1

for epoch in range(n_epochs):
  train(epoch)
  evaluate(train_loader)

def prediction(data_loader):
  model.eval()
  test_pred = torch.LongTensor()

  for i,data in enumerate(data_loader):
    data = Variable(data, volatile=True)
    if torch.cuda.is_available():
      data = data.cuda()

    output = model(data)

    pred = output.cpu().data.max(1, keepdim=True)[1]
    test_pred = torch.cat((test_pred, pred), dim=0)

  return test_pred
  
test_pred = prediciton(test_loader)
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], test_pred.numpy()], 
                      columns=['ImageId', 'Label'])


print(out_df.head())

out_df.to_csv('./output/submission.csv',index=False)






















