from torch import nn
from torchvision import models

from src.utils.registry import REGISTRY

@REGISTRY.register('bilnet')
class BilNet(nn.Module):
    def __init__(self, num_classes=5, classify=True):
        super(BilNet, self).__init__()
        self.bilnet = models.squeezenet1_1(pretrained=True)
        
        if classify:
            self.bilnet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.bilnet.num_classes = num_classes
        else:
            self.bilnet = nn.Sequential(*(list(self.bilnet.children())[:-1] + [nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()]))
    
    def forward(self, inputs):
        return self.bilnet(inputs)
    
    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in filter(lambda param_tuple: isinstance(param_tuple[1], nn.Conv2d), self.bilnet.named_parameters())]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: not isinstance(param_tuple[1], nn.Conv2d), self.bilnet.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers 
        }
        return param_groups




# from torch import nn
# from torchvision import models

# from src.utils.registry import REGISTRY

# @REGISTRY.register('bilnet')
# class BilNet(nn.Module):
#     def __init__(self, num_classes=5, classify=True):
#         super(BilNet, self).__init__()
#         self.bilnet = models.resnet18(pretrained=True)
        
#         if classify:
#             self.bilnet.fc = nn.Linear(512, num_classes)
#         else:
#             self.bilnet = nn.Sequential(*(list(self.bilnet.children())[:-1] + [nn.Flatten()]))
    
#     def forward(self, inputs):
#         return self.bilnet(inputs)
    
#     def get_layer_groups(self):
#         linear_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' in param_tuple[0], self.bilnet.named_parameters())]
#         other_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' not in param_tuple[0], self.bilnet.named_parameters())]
#         param_groups = {
#             'classifier': linear_layers,
#             'feature_extractor': other_layers 
#         }
#         return param_groups


# import torch.nn.functional as F

# from torch import nn
# import torch
# from torchvision import models

# from src.utils.registry import REGISTRY

# class MultiModel(nn.Module):
#     def __init__(self, num_classes):
#         super(MultiModel, self).__init__()
#         self.resnet = models.resnet50(pretrained=True)
#         self.densenet = models.densenet201(pretrained=True)
#         self.alexnet = models.alexnet(pretrained=True)
#         self.squeezenet = models.squeezenet1_1(pretrained=True)
        
#         self.fc1 = nn.Linear(2048 + 1920 + 256 + 512, 512)
#         self.fc2 = nn.Linear(512, num_classes)
    
#     def forward(self, inputs):
#         x1 = self.resnet(inputs)
#         x2 = self.densenet(inputs)
#         x3 = self.alexnet(inputs)
#         x4 = self.squeezenet(inputs)
        
#         # Concatenate the output of all models
#         x = torch.cat((x1, x2, x3, x4), dim=1)
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         return x
    
#     def get_layer_groups(self):
#         param_groups = {
#             'classifier': [self.fc1, self.fc2],
#             'feature_extractor': list(self.resnet.children())[:-2] + list(self.densenet.children())[:-1] + list(self.alexnet.features.children()) + list(self.squeezenet.children())[:-1]
#         }
#         return param_groups


# @REGISTRY.register('bilnet')
# class BilNet(nn.Module):
#     def __init__(self, num_classes=5, classify=True):
#         super(BilNet, self).__init__()
        
#         self.bilnet = MultiModel(num_classes)
        
#         if not classify:
#             self.bilnet = nn.Sequential(*(list(self.bilnet.children())[:-2]))
    
#     def forward(self, inputs):
#         return self.bilnet(inputs)
    
#     def get_layer_groups(self):
#         return self.bilnet.get_layer_groups()

# import torch.nn.functional as F

# from torch import nn
# import torch
# from torchvision import models

# from src.utils.registry import REGISTRY


# class MultiModel(nn.Module):
#     def __init__(self, model_name, num_classes):
#         super(MultiModel, self).__init__()
#         self.model_name = model_name
#         if model_name == 'resnet50':
#             self.model = models.resnet50(pretrained=True)
#             num_features = self.model.fc.in_features
#         elif model_name == 'densenet':
#             self.model = models.densenet121(pretrained=True)
#             num_features = self.model.classifier.in_features
#         elif model_name == 'alexnet':
#             self.model = models.alexnet(pretrained=True)
#             num_features = self.model.classifier[6].in_features
#         elif model_name == 'squeezenet':
#             self.model = models.squeezenet1_1(pretrained=True)
#             num_features = self.model.classifier[1].in_channels
#         else:
#             raise ValueError(f"Invalid model name: {model_name}")
        
#         self.fc1 = nn.Linear(num_features, 512)
#         self.fc2 = nn.Linear(512, num_classes)
    
#     def forward(self, inputs):
#         if self.model_name == 'squeezenet':
#             x = self.model.features(inputs)
#             x = F.relu(x)
#             x = F.avg_pool2d(x, kernel_size=13).view(inputs.size(0), -1)
#         else:
#             x = self.model(inputs)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
#     def get_layer_groups(self):
#         if self.model_name == 'resnet50':
#             return {
#                 'classifier': [self.fc1, self.fc2],
#                 'feature_extractor': list(self.model.children())[:-2]
#             }
#         elif self.model_name == 'densenet':
#             return {
#                 'classifier': [self.fc1, self.fc2],
#                 'feature_extractor': list(self.model.features.children())
#             }
#         elif self.model_name == 'alexnet':
#             return {
#                 'classifier': [self.fc1, self.fc2],
#                 'feature_extractor': list(self.model.features.children())
#             }
#         elif self.model_name == 'squeezenet':
#             return {
#                 'classifier': [self.fc1, self.fc2],
#                 'feature_extractor': list(self.model.features.children())
#             }
#         else:
#             raise ValueError(f"Invalid model name: {self.model_name}")


# @REGISTRY.register('bilnet')
# class BilNet(nn.Module):
#     def __init__(self, model_name='resnet50', num_classes=5, classify=True):
#         super(BilNet, self).__init__()
#         self.bilnet = MultiModel(model_name, num_classes)
        
#         if not classify:
#             self.bilnet = nn.Sequential(*(list(self.bilnet.children())[:-2]))
    
#     def forward(self, inputs):
#         return self.bilnet(inputs)
    
#     def get_layer_groups(self):
#         return self.bilnet.get_layer_groups()
