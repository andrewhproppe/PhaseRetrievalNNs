import torchvision
import torchvision.datasets as datasets
import os

from PRNN.utils import get_system_and_backend
get_system_and_backend()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

images = mnist_trainset.data

for i, img in enumerate(images):
    name = 'mnist_%i' % i
    torchvision.io.write_png(img.unsqueeze(0), os.path.join('masks_mnist', name))
