import torch
from torch import nn
from torch.nn import functional as F
from models.efficientnet_pytorch.model import EfficientNet

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.model1 = EfficientNet.from_name('efficientnet-b0', in_channels=1, num_classes=1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(1280 * 3, 1)
        
        
    def forward(self, x):
        
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x3 = x[:, 2, :, :].unsqueeze(1)


        out1 = self.model1.forward2(x1)
        out2 = self.model1.forward2(x2)
        out3 = self.model1.forward2(x3)

        out = torch.cat([out1, out2, out3], dim = 1)
        out, feat = self.fusion(out)
        
        
        return out
        
        
    def fusion(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """

        feat = inputs.flatten(start_dim=1)
        x = self._dropout(feat)
        x = self._fc(x)
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x, feat
    
