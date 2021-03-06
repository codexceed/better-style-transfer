from collections import namedtuple
import torch
from torchvision import models

"""
    More detail about the VGG architecture (if you want to understand magic/hardcoded numbers) can be found here:
    
    https://github.com/pytorch/vision/blob/3c254fb7af5f8af252c24e89949c54a3461ff0be/torchvision/models/vgg.py
"""


class Vgg16(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True, progress=show_progress).features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_feature_maps_index = 1  # relu2_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # all layers used for style representation

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


class Vgg16Experimental(torch.nn.Module):
    """Everything exposed so you can play with different combinations for style and content representation"""
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True, progress=show_progress).features
        self.layer_names = ['relu1_1', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu4_1', 'relu4_3', 'relu5_1']
        self.content_feature_maps_index = 4
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # all layers used for style representation

        self.conv1_1 = vgg_pretrained_features[0]
        self.relu1_1 = vgg_pretrained_features[1]
        self.conv1_2 = vgg_pretrained_features[2]
        self.relu1_2 = vgg_pretrained_features[3]
        self.max_pooling1 = vgg_pretrained_features[4]
        self.conv2_1 = vgg_pretrained_features[5]
        self.relu2_1 = vgg_pretrained_features[6]
        self.conv2_2 = vgg_pretrained_features[7]
        self.relu2_2 = vgg_pretrained_features[8]
        self.max_pooling2 = vgg_pretrained_features[9]
        self.conv3_1 = vgg_pretrained_features[10]
        self.relu3_1 = vgg_pretrained_features[11]
        self.conv3_2 = vgg_pretrained_features[12]
        self.relu3_2 = vgg_pretrained_features[13]
        self.conv3_3 = vgg_pretrained_features[14]
        self.relu3_3 = vgg_pretrained_features[15]
        self.max_pooling3 = vgg_pretrained_features[16]
        self.conv4_1 = vgg_pretrained_features[17]
        self.relu4_1 = vgg_pretrained_features[18]
        self.conv4_2 = vgg_pretrained_features[19]
        self.relu4_2 = vgg_pretrained_features[20]
        self.conv4_3 = vgg_pretrained_features[21]
        self.relu4_3 = vgg_pretrained_features[22]
        self.max_pooling4 = vgg_pretrained_features[23]
        self.conv5_1 = vgg_pretrained_features[24]
        self.relu5_1 = vgg_pretrained_features[25]
        self.conv5_2 = vgg_pretrained_features[26]
        self.relu5_2 = vgg_pretrained_features[27]
        self.conv5_3 = vgg_pretrained_features[28]
        self.relu5_3 = vgg_pretrained_features[29]
        self.max_pooling5 = vgg_pretrained_features[30]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1_1(x)
        conv1_1 = x
        x = self.relu1_1(x)
        relu1_1 = x
        x = self.conv1_2(x)
        conv1_2 = x
        x = self.relu1_2(x)
        relu1_2 = x
        x = self.max_pooling1(x)
        x = self.conv2_1(x)
        conv2_1 = x
        x = self.relu2_1(x)
        relu2_1 = x
        x = self.conv2_2(x)
        conv2_2 = x
        x = self.relu2_2(x)
        relu2_2 = x
        x = self.max_pooling2(x)
        x = self.conv3_1(x)
        conv3_1 = x
        x = self.relu3_1(x)
        relu3_1 = x
        x = self.conv3_2(x)
        conv3_2 = x
        x = self.relu3_2(x)
        relu3_2 = x
        x = self.conv3_3(x)
        conv3_3 = x
        x = self.relu3_3(x)
        relu3_3 = x
        x = self.max_pooling3(x)
        x = self.conv4_1(x)
        conv4_1 = x
        x = self.relu4_1(x)
        relu4_1 = x
        x = self.conv4_2(x)
        conv4_2 = x
        x = self.relu4_2(x)
        relu4_2 = x
        x = self.conv4_3(x)
        conv4_3 = x
        x = self.relu4_3(x)
        relu4_3 = x
        x = self.max_pooling4(x)
        x = self.conv5_1(x)
        conv5_1 = x
        x = self.relu5_1(x)
        relu5_1 = x
        x = self.conv5_2(x)
        conv5_2 = x
        x = self.relu5_2(x)
        relu5_2 = x
        x = self.conv5_3(x)
        conv5_3 = x
        x = self.relu5_3(x)
        relu5_3 = x
        x = self.max_pooling5(x)
        # expose only the layers that you want to experiment with here
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_1, relu2_1, relu2_2, relu3_1, relu3_2, relu4_1, relu4_3, relu5_1)

        return out


class Vgg19(torch.nn.Module):
    """
    Used in the original NST paper, only those layers are exposed which were used in the original paper

    'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1' were used for style representation
    'conv4_2' was used for content representation (although they did some experiments with conv2_2 and conv5_2)
    """
    def __init__(self, content_layer=-1, style_layers=-1, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features
        if use_relu:  # use relu or as in original paper conv layers
            self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
            self.offset = 1
        else:
            self.layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
            self.offset = 0

        # Set content feature map and style feature maps
        if content_layer == -1:
            content_layer = 4
        if style_layers == -1:
            style_layers = len(self.layer_names)

        self.content_feature_maps_index = content_layer
        self.style_feature_maps_indices = list(range(style_layers))
        if style_layers > content_layer:
            self.style_feature_maps_indices.remove(content_layer) 


        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1+self.offset):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1+self.offset, 6+self.offset):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6+self.offset, 11+self.offset):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11+self.offset, 20+self.offset):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20+self.offset, 22):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29++self.offset):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out


class Resnet50(torch.nn.Module):
    """
    Implement resnet50 for style transfer
    """
    def __init__(self, content_layer=-1, style_layers=-1, requires_grad=False, show_progress=False):
        super().__init__()
        self.resnet_pretrained = models.resnet50(pretrained=True, progress=show_progress)
        self.layer_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']

        # Set content feature map and style feature maps
        if content_layer == -1:
            content_layer = 3
        if style_layers == -1:
            style_layers = len(self.layer_names)

        self.content_feature_maps_index = content_layer
        self.style_feature_maps_indices = list(range(style_layers))
        if style_layers > content_layer:
            self.style_feature_maps_indices.remove(content_layer)  

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x = self.resnet_pretrained.conv1(x)
        x = self.resnet_pretrained.bn1(x)
        x = self.resnet_pretrained.relu(x)
        x = self.resnet_pretrained.maxpool(x)
        
        layer0 = x
        x = self.resnet_pretrained.layer1(x)
        layer1 = x
        x = self.resnet_pretrained.layer2(x)
        layer2 = x
        x = self.resnet_pretrained.layer3(x)
        layer3 = x
        x = self.resnet_pretrained.layer4(x)
        layer4 = x
        
        x = self.resnet_pretrained.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet_pretrained.fc(x)
        
        resnet_outputs = namedtuple("ResnetOutputs", self.layer_names)
        out = resnet_outputs(layer0, layer1, layer2, layer3, layer4)
        return out


class InceptionV3(torch.nn.Module):
    """
    Implement inceptionV3 for style transfer
    """
    def __init__(self, content_layer=-1, style_layers=-1, requires_grad=False, show_progress=False):
        super().__init__()
        self.inception_pretrained = models.inception_v3(pretrained=True, progress=show_progress)
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7']

        # Set content feature map and style feature maps
        if content_layer == -1:
            content_layer = 5
        if style_layers == -1:
            style_layers = len(self.layer_names)

        self.content_feature_maps_index = content_layer
        self.style_feature_maps_indices = list(range(style_layers))
        if style_layers > content_layer:
            self.style_feature_maps_indices.remove(content_layer) 

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        
        x = self.inception_pretrained.Conv2d_1a_3x3(x)
        layer1 = x
        x = self.inception_pretrained.Conv2d_2a_3x3(x)
        x = self.inception_pretrained.Conv2d_2b_3x3(x)
        layer2 = x
        x = self.inception_pretrained.maxpool1(x)
        x = self.inception_pretrained.Conv2d_3b_1x1(x)
        layer3 = x
        x = self.inception_pretrained.Conv2d_4a_3x3(x)
        layer4 = x
        x = self.inception_pretrained.maxpool2(x)
        x = self.inception_pretrained.Mixed_5b(x)
        x = self.inception_pretrained.Mixed_5c(x)
        x = self.inception_pretrained.Mixed_5d(x)
        layer5 = x
        x = self.inception_pretrained.Mixed_6a(x)
        x = self.inception_pretrained.Mixed_6b(x)
        x = self.inception_pretrained.Mixed_6c(x)
        x = self.inception_pretrained.Mixed_6d(x)
        x = self.inception_pretrained.Mixed_6e(x)
        layer6 = x
        x = self.inception_pretrained.Mixed_7a(x)
        x = self.inception_pretrained.Mixed_7b(x)
        x = self.inception_pretrained.Mixed_7c(x)
        layer7 = x
        x = self.inception_pretrained.avgpool(x)
        x = self.inception_pretrained.dropout(x)
        x = torch.flatten(x, 1)
        x = self.inception_pretrained.fc(x)
        
        resnet_outputs = namedtuple("ResnetOutputs", self.layer_names)
        out = resnet_outputs(layer1, layer2, layer3, layer4, layer5, layer6, layer7)
        return out