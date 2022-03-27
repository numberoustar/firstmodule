import torch.nn as nn
import timm
class Load_a_net(nn.Module):
    def __init__(self, model_name='seresnext50_32x4d', out_features=176,
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x