import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

class OtolithResNet(nn.Module):
    def __init__(self, model_name, out_features=1, train_indices=None, val_indices=None):
        super(OtolithResNet, self).__init__()

        self.model_name = model_name

        self.train_indices = train_indices
        self.val_indices = val_indices

        # Loading desired ResNet architecture from torchvision.
        if model_name == 'res18':
            self.resnet = resnet18(weights='IMAGENET1K_V1')
        elif model_name == 'res34':
            self.resnet = resnet34(weights='IMAGENET1K_V1')
        elif model_name == 'res50':
            self.resnet = resnet50(weights='IMAGENET1K_V2')
        elif model_name == 'res101':
            self.resnet = resnet101(weights='IMAGENET1K_V2')
        elif model_name == 'res152':
            self.resnet = resnet152(weights='IMAGENET1K_V2')
        
        # Reshaping final linear layer to have desired output size, i.e., 1 for regression, number of classes for classification.
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=out_features)

    def forward(self, x):
        return self.resnet(x).squeeze()
    
    def unfreeze(self, layers, learning_rates):
        # Unfreeze (enable gradient computation) given layers with given learning rates, making them available to be optimized.
        to_unfreeze = []
        for layer_string, learning_rate in zip(layers, learning_rates):
            layer = getattr(self.resnet, layer_string)
            layer.requires_grad_(True)

            to_unfreeze.append(
                {
                    'params': layer.parameters(),
                    'lr': learning_rate
                }
            )
        
        return to_unfreeze
    
    def freeze(self, layers):
        # Freeze (disable gradient computation) given layers, making them unable to be optimized.
        for layer_string in layers:
            layer = getattr(self.resnet, layer_string)
            layer.requires_grad_(False)