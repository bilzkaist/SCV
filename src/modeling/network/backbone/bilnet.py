from torch import nn
from torchvision import models

from src.utils.registry import REGISTRY

@REGISTRY.register('bilnet')
class BilNet(nn.Module):
    def __init__(self, num_classes=5, classify=True):
        super(BilNet, self).__init__()
        self.bilnet = models.resnet18(pretrained=True)
        
        if classify:
            self.bilnet.fc = nn.Linear(512, num_classes)
        else:
            self.bilnet = nn.Sequential(*(list(self.bilnet.children())[:-1] + [nn.Flatten()]))
    
    def forward(self, inputs):
        return self.bilnet(inputs)
    
    def get_layer_groups(self):
        linear_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' in param_tuple[0], self.bilnet.named_parameters())]
        other_layers = [elem[1] for elem in filter(lambda param_tuple: 'fc' not in param_tuple[0], self.bilnet.named_parameters())]
        param_groups = {
            'classifier': linear_layers,
            'feature_extractor': other_layers 
        }
        return param_groups
