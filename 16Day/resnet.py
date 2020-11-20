import torch, time
import torch.nn as nn
import torch.optim as optim
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


# def __init__(self, dir, size=(244, 244)):
#     self.files = glob(dir)
#     self.size = size 

# def __getitem__(self, idx):
#     img = np.array(Image.open(self.files[idx]).resize(self.size))
#     label = self.files[idx].split('/')[-2]
#     return img, label

# def __len__(self):
#     return len(self.files)
is_cuda = False

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
# 이미지 데이터
train = ImageFolder('./16Day/train/', transform=transform)
valid = ImageFolder('./16Day/valid/', transform=transform)

# 데이터 로더
trainloader = DataLoader(train, batch_size=64, num_workers=2, shuffle=True)
validloader = DataLoader(valid, batch_size=64, num_workers=2, shuffle=True)

dataloader = {'train': trainloader, 'valid': validloader}
datalen = {'train': len(trainloader.dataset), 'valid': len(validloader.dataset)}

#모델 생성(pretrained)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

#GPU 사용
if is_cuda:
    model_ft = model_ft.cuda()

#최적화, 손실함수 정의
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, schedular, epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        print('Epoch {:2d}/{}'.format(epoch, epochs))
        print('-' * 20)

        for phase in ['train', 'valid']:
            if phase == 'train':
                schedular.step()
                model.train(True)
            
            else:
                model.train(False)
            
            running_loss = 0.0 # 손실
            running_corrects = 0 # 맞은 갯수

            for data in dataloader[phase]:
                inputs, labels = data         

                if is_cuda:
                    inputs = inputs.cuda()
                    outputs = outputs.cuda()
                else:
                    pass
                #파라미터 기울기 최적화
                optimizer.zero_grad()
                #포워딩
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                #학습 단계(train만 수행)
                if(phase=='train'):
                    loss.backward()
                    optimizer.step()

                #통계
                running_loss += loss.item()      
                runnin
                    epoch_loss = running_loss/datalen[phase]
                epoch_acc = running_corrects/datalen[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc =epoch_acc
                best_model_wts = model.state_dict()

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #최적 가중치 로드
    model.load_state_dict(best_model_wts)
    return model


train_model(model_ft,criterion,optimizer,exp_lr_schedular)
