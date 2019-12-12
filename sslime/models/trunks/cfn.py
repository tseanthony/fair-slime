import torch
import torch.nn as nn
import torchvision.models as models


class CFN(nn.Module):
    def __init__(self):
        super(CFN, self).__init__()
        alexnet = models.alexnet()
        alexnet.features[0].stride = (2, 2)
        self.features = alexnet.features
        self.fc6 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        rs = []
        for i in range(x.size(1)):
            tx = self.features(x[:, i])
            tx = torch.flatten(tx, 1)
            tx = self.fc6(tx)
            rs.append(tx)
        return [torch.cat(rs, 1)]
