import torch.nn as nn
from IPython import embed
import numpy as np

def _get_cnn_weights(model):
    params = list(model.parameters())
    return np.squeeze(params[-1].data.numpy()) # returning last layer weights.

class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.weight = _get_cnn_weights(self.submodule)
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "classfier":
                x = x.view(x.size(0),-1)
            if name == "base":
                for block_name, cnn_block in module._modules.items():
                    x = cnn_block(x)
                    if block_name in self.extracted_layers:
                        outputs.append(x)
        return outputs, self.weight