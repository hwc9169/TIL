from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import numpy as np
transformation = transforms.Compose(
    [transforms.Resize((224,224)),transforms.ToTensor()]
)
is_cuda = False
vgg = models.vgg16(pretrained=True)
features = vgg.features

train = ImageFolder('../16Day/train', transform=transformation)
valid = ImageFolder('../16Day/valid', transform=transformation)
train_loader = DataLoader(train, batch_size=32, shuffle=False, num_workers=3)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False, num_workers=3)

def preconvfeat(dataset, model):
    conv_features = []
    labels_list = []
    for data in dataset:
        inputs, labels = data
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        conv_features.extend(output.data.numpy())
        labels_list.extend(labels.data.numpy())
    conv_features = np.concatenate([[f] for f in conv_features])
    
    return (conv_features, labels_list)

conv_feat_train, labels_train = preconvfeat(train_loader, features)
conv_feat_val, labels_val = preconvfeat(valid_loader, features)

print(conv_feat_train.shape)

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train = MyDataset(conv_feat_train, labels_train)
valid = MyDataset(conv_feat_valid, labels_valid)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid, batch_size=64, shuffle=True)