import torch.nn as nn



class MultiTaskLoss(nn.Module):


    """
    多任务分类Loss

    total_loss =
        Σ weight_i * loss_i

    """



    def __init__(

            self,

            weights

    ):

        super().__init__()



        self.loss_fn = nn.CrossEntropyLoss()



        self.weights = weights




    def forward(

            self,

            outputs,

            targets

    ):


        total_loss = 0


        loss_dict = {}



        for task in outputs:


            task_loss = self.loss_fn(

                outputs[task],

                targets[task]

            )


            loss_dict[task] = task_loss.item()



            total_loss += (

                self.weights[task]

                *

                task_loss

            )



        loss_dict["total"] = total_loss.item()



        return total_loss, loss_dict