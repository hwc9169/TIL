from torchvision import models
import matplotlib.pyplot as plt

vgg = models.vgg16(pretrained=True)

cnn_weight = vgg.state_dict()['features.0.weight']

fig = plt.figure(figsize=(10,40))
fig.subplots_adjust(left=0, right=0.9, bottom=0, top=0.9, hspace=0.05, wspace=0.2)
for i in range(1, 31):
    ax = fig.add_subplot(5, 6 ,i)
    ax.imshow(cnn_weight[0])
plt.show()
