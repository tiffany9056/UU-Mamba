import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from nnunetv2.nets.UMambaEnc import get_umamba_enc_from_plans
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from nnunetv2.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunetv2.training.loss.compound_losses import AutoWeighted_DC_and_CE_and_Focal_loss
from nnunetv2.training.loss.sam import SAM

class nnUNetTrainerUMambaEnc(nnUNetTrainer):
    """
    UMmaba Encoder + Residual Decoder + Skip Connections
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.loss  = AutoWeighted_DC_and_CE_and_Focal_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                                           {},
                                                           {'alpha':0.5, 'gamma':2, 'smooth':1e-5},
                                                           ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
#         self.loss = RobustCrossEntropyLoss()
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_umamba_enc_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)
        
#         print("UMambaEnc: {}".format(model))

        return model
 
# ########## Uncertainty-Aware loss ##########
#     def configure_optimizers(self):
#         param_groups = [
#             {'params': self.network.parameters(), 'weight_decay': self.weight_decay, 'lr': self.initial_lr},
#             {'params': self.loss.awl.parameters(), 'weight_decay': 0, 'lr': self.initial_lr}
#         ]
        
#         optimizer = torch.optim.SGD(param_groups, momentum=0.99, nesterov=True)
#         lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

#         return optimizer, lr_scheduler

# ########## Uncertainty-Aware loss + SAM optimizers ##########
    def configure_optimizers(self):
        param_groups = [
            {'params': self.network.parameters(), 'weight_decay': self.weight_decay, 'lr': self.initial_lr},
            {'params': self.loss.awl.parameters(), 'weight_decay': 0, 'lr': self.initial_lr}
        ]
        base_optimizer = torch.optim.SGD
        optimizer = SAM(param_groups, base_optimizer, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer.base_optimizer, self.initial_lr, self.num_epochs)
    
        return optimizer, lr_scheduler