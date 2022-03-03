from model.vgg import vgg19_bn, vgg19
from model.resnet import resnet18, resnet101
from model.mobilenetv2 import mobilenetv2


def create_model(name):
    if name == 'vgg19_bn':
        return vgg19_bn
    elif name == 'vgg19':
        return vgg19
    elif name == 'resnet18':
        return resnet18
    elif name == 'resnet101':
        return resnet101
    elif name == 'mobilenetv2':
        return mobilenetv2


