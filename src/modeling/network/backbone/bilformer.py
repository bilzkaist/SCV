import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

from src.utils.registry import REGISTRY

@REGISTRY.register('bilformer')
class BilFormer(nn.Module):
    def __init__(self, num_classes=5, classify=True):
        super(BilFormer, self).__init__()

        # Load a pre-trained Swin Transformer model
        self.swin = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=num_classes, pretrained=True)

        # Modify the last layer to match the number of classes
        if not classify:
            self.swin.head = nn.Identity()

    def forward(self, inputs):
        return self.swin(inputs)

    def get_layer_groups(self):
        linear_layers = [param for name, param in self.named_parameters() if 'head' in name]
        feature_layers = [param for name, param in self.named_parameters() if 'head' not in name]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': feature_layers 
        }
        return param_groups

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from src.utils.registry import REGISTRY

# # Define ResNet block
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# @REGISTRY.register('bilformer')
# # Define ResNet model
# class BilFormer(nn.Module):
#     def __init__(self, block, layers, num_classes=5, classify=True):
#         super(BilFormer, self).__init__()
#         self.in_channels = 64

#         # Initial conv layer
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # ResNet layers
#         self.layer1 = self.make_layer(block, 64, layers[0])
#         self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

#         # Classification layer
#         if classify:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#             self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def make_layer(self, block, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#             self.in_channels = out_channels * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.maxpool(out)

#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)

#         if hasattr(self, 'avgpool'):
#             out = self.avgpool(out)
#             out = torch.flatten(out, 1)
#             out = self.fc(out)

#         return out

#     def get_layer_groups(self):
#         linear_layers = [param for name, param in self.named_parameters() if 'fc' in name]
#         feature_layers = [param for name, param in self.named_parameters() if 'fc' not in name]
#         param_groups = {
#             'classifier': linear_layers,
#             'feature_extractor': feature_layers 
#         }
#         return param_groups
