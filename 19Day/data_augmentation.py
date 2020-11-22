from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
transformation1 = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomRotation(32),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
transformation2 = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])                                     

train = ImageFolder('./16Day/train', transform=transformation2)
valid = ImageFolder('./16Day/valid', transform=transformation2)
train_aug = ImageFolder('./16Day/train', transform=transformation1)
valid_aug = ImageFolder('./16Day/valid', transform=transformation1)


train_aug_loader = DataLoader(train_aug, shuffle=False, batch_size=32)
test_aug_loader = DataLoader(valid_aug, shuffle=False, batch_size=32)
train_loader = DataLoader(train, shuffle=False, batch_size=32)
test_loader = DataLoader(valid, shuffle=False, batch_size=32)
import numpy as np
def plot_img(img1, img2):
    image1 = img1.numpy()[0]
    image2 = img2.numpy()[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image1)
    ax1.set_title("No Augmentation")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(image2)
    ax1.set_title("Augmentation")
    plt.show()

sample_aug_data = next(iter(train_aug_loader))
sample_data = next(iter(train_loader))

for i in range(30):
    plot_img(sample_data[0][i], sample_aug_data[0][i])

print(train)
