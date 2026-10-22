import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

class CustomTrainerLR01Epoch50CE(nnUNetTrainer):
    def __init__(
        self, plans, configuration, fold,
        dataset_json, device=torch.device("cuda")
    ):
        super().__init__(
            plans, configuration, fold,
            dataset_json, device
        )
        self.initial_lr = 0.005
        self.num_epochs = 50
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 25
    def _build_loss(self):
        loss = super()._build_loss()

        # Preserve deep supervision
        if isinstance(loss, DeepSupervisionWrapper):
            base_loss = loss.loss
        else:
            base_loss = loss

        # Use Soft Dice only
        base_loss.weight_dice = 0.0
        base_loss.weight_ce = 1.0

        return loss
