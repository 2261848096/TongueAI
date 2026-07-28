import torch.nn as nn

from .efficientnet import EfficientNetBackbone
from .classification_head import ClassificationHead



class TongueMultiTaskModel(nn.Module):

    def __init__(self,cfg):

        super().__init__()


        # backbone

        self.backbone = EfficientNetBackbone(
            cfg["model"]["pretrained"]
        )


        feature_dim = self.backbone.feature_dim


        num_classes = cfg["model"]["num_classes"]


        # 五任务head

        self.heads = nn.ModuleDict({

            "tongue_color":
                ClassificationHead(
                    feature_dim,
                    num_classes["tongue_color"]
                ),


            "coat_color":
                ClassificationHead(
                    feature_dim,
                    num_classes["coat_color"]
                ),


            "coat_thickness":
                ClassificationHead(
                    feature_dim,
                    num_classes["coat_thickness"]
                ),


            "crack":
                ClassificationHead(
                    feature_dim,
                    num_classes["crack"]
                ),


            "teeth_mark":
                ClassificationHead(
                    feature_dim,
                    num_classes["teeth_mark"]
                )

        })



    def forward(self,x):

        feature = self.backbone(x)

        outputs={}

        for name,head in self.heads.items():

            outputs[name]=head(feature)


        return outputs