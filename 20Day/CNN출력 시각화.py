from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
transformation = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor()
])

vgg = models.vgg16(pretrained=True)
img = ImageFolder('../16Day/valid', transform=transformation)
img_loader = DataLoader(img, shuffle=True, batch_size=1)
img_loader = next(iter(img_loader))

def grid_show(act):
    fig = plt.figure(figsize=(20,50))
    fig.subplots_adjust(left=0, right=0.9, bottom=0, top=0.95, hspace=0.1, wspace=0.2)
    for i in range(1, 31):
        ax = fig.add_subplot(6, 5, i)
        ax.imshow(act[0][i].detach().numpy(), cmap='gray')
    plt.show()

class LayerActivations():
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove
        

conv_out = LayerActivations(vgg.features, 5)
o = vgg(img_loader[0])
act1 = conv_out.features
conv_out.remove()

conv_out = LayerActivations(vgg.features, 30)
o = vgg(img_loader[0])
act2 = conv_out.features
conv_out.remove()


plt.imshow(img_loader[0][0].numpy().transpose(1,2,0))
plt.show()
grid_show(act1)
grid_show(act2)