"""Configuration for MedSigLIP training."""


class MedSigLIPConfig:
    """Configuration for MedSigLIP training."""
    
    def __init__(self):
        # Data
        self.data_root = '../../../Dataset/NEH/NEH_UT_2021RetinalOCTDataset'
        self.csv_path = '../../../Dataset/NEH/data_information.csv'
        self.image_size = 448  
        self.batch_size = 8  
        self.num_workers = 8
        
        # Model
        self.num_classes = 3
        self.dropout = 0.3
        self.freeze_encoder_initially = True  
        self.use_gradient_checkpointing = True
        
        # Training - Progressive strategy
        self.epochs_frozen = 20  
        self.epochs_unfrozen = 100  
        self.total_epochs = self.epochs_frozen + self.epochs_unfrozen
        
        # Optimization
        self.learning_rate_head = 1e-3  
        self.learning_rate_encoder = 5e-5  
        self.weight_decay = 1e-4
        self.patience = 25
        self.min_lr = 1e-7
        self.warmup_epochs = 5
        self.grad_clip = 1.0
        
        self.use_amp = True
        self.use_mixup = True
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.cutmix_prob = 0.5
        self.use_ema = True
        self.ema_decay = 0.9997
        
        # Loss
        self.label_smoothing = 0.1
        self.focal_gamma = 2.0
        
        # Test-time augmentation
        self.tta_transforms = 8
        
        self.output_dir = 'results'
        self.save_best = True
        
        self.n_folds = 5
        self.random_state = 43

