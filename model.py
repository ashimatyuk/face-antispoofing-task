import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0(nn.Module):

    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.efficientnet_b0 = efficientnet_b0(weights='DEFAULT')
        self.efficientnet_b0.classifier[1] = nn.Linear(in_features=1280, out_features=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, inputs):
        output = self.efficientnet_b0(inputs)
        output = self.dropout(output)

        return output
