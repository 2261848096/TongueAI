import torch
import torch.nn as nn
import timm

from safetensors.torch import load_file



class EfficientNetBackbone(nn.Module):


    def __init__(

            self,

            weight_path=None

    ):

        super().__init__()



        self.model = timm.create_model(

            "tf_efficientnet_b3_ns",

            pretrained=False,

            num_classes=0

        )



        if weight_path:


            print(
                f"Loading pretrained weight: {weight_path}"
            )


            weights = load_file(
                weight_path
            )


            msg = self.model.load_state_dict(

                weights,
  
                strict=False

            )


            print(
                "Weight loaded"
            )

            print(msg)



        self.feature_dim = self.model.num_features




    def forward(self,x):

        return self.model(x)