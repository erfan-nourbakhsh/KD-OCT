"""Training configuration for UCSD OCT Classification."""


class TrainingConfig:
    """Enhanced configuration with new hyperparameters."""

    def __init__(self):
        # Data paths
        self.train_dir = '../../../Dataset/UCSD/CellData/OCT/train'
        self.test_dir = '../../../Dataset/UCSD/CellData/OCT/test'
        self.val_split = 0.2  
        
        self.image_size = 384
        self.batch_size = 64
        self.accumulation_steps = 1
        self.num_workers = 16

        # Model
        self.num_classes = 4  # CNV, DME, DRUSEN, NORMAL
        self.backbone = 'convnextv2_large.fcmae_ft_in22k_in1k'
        self.dropout = 0.3
        self.drop_path_rate = 0.3

        # Training
        self.epochs = 150
        self.warmup_epochs = 10
        self.learning_rate_head = 1e-4
        self.learning_rate_backbone = 2e-5
        self.min_lr = 1e-7
        self.weight_decay = 0.05
        self.freeze_backbone_epochs = 5
        self.early_stopping_patience = 25
        
        # SWA
        self.use_swa = True
        self.swa_start_epoch = 100
        self.swa_lr = 1e-5

        # Mixup/CutMix & Label smoothing
        self.use_mixup = True
        self.mixup_alpha = 0.4
        self.cutmix_alpha = 1.0
        self.mixup_prob = 0.8
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'
        self.label_smoothing = 0.1

        # Loss
        self.use_focal_loss = True
        self.focal_gamma = 2.0
        
        # TTA
        self.use_tta = True
        
        # Advanced augmentation
        self.use_randaugment = True

        # Output
        self.output_dir = 'results'
        self.save_best = True
        self.log_interval = 10

        self.random_state = 43

