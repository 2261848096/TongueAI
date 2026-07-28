import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, feature_dim, num_classes):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                feature_dim,
                num_classes
            )
        )


    def forward(self, x):

        return self.fc(x)