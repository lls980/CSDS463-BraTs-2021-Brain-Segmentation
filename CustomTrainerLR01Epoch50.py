from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs

class CustomTrainerLR01Epoch50(nnUNetTrainer_250epochs):
    def __init__(self, plans, configuration, fold,
        dataset_json, device):
        super().__init__(
            plans, configuration, fold,
            dataset_json, device
        )
        self.initial_lr = 0.005
        self.num_epochs = 50
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 25
