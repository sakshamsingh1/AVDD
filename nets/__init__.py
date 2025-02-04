import torch
import torch.nn as nn

from .audio_net import ConvNet
from .vision_net import ConvNet_v
from .cls_net import Classifier_Ensemble
from .criterion import BCELoss, CELoss

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet6', weights='', channel=1, im_size=(96,64)):
        if arch == "convNet":
            net_sound = ConvNet(channel, im_size)
        else:
            raise Exception('Architecture undefined!')
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))
        if torch.cuda.device_count() > 1:
            net_sound = nn.DataParallel(net_sound)
        return net_sound

    def build_frame(self, arch='resnet18', weights='', channel=3, im_size=(224,224)):
        if arch == "convNet":
            net = ConvNet_v(channel, im_size)
        else:
            raise Exception('Architecture undefined!')
        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        return net

    def build_classifier(self, arch, cls_num, weights='', input_modality='av'):
        if arch == 'ensemble':
            net = Classifier_Ensemble(cls_num, input_modality)
        else:
            raise Exception('Architecture undefined!')
        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_grounding')
            net.load_state_dict(torch.load(weights))
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'ce':
            net = CELoss()
        else:
            raise Exception('Architecture undefined!')
        return net
