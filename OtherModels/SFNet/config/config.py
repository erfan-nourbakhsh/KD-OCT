"""Configuration for SF-Net model and training."""


class SFNetConfig:
    """SF-Net model architecture configuration."""
    
    def __init__(self):
        # Backbone
        self.backbone = 'convnext_base'  # or 'convnext_small', 'convnext_large'
        self.pretrained = True
        
        # Multi-scale fusion
        self.use_multiscale_fusion = True
        self.fusion_target_channels = 384
        
        # Classification head
        self.num_classes = 3
        self.head_hidden_dims = [768, 512, 256]
        self.dropout_rate = 0.3
        self.use_attention_head = True
        
        # Stochastic depth
        self.drop_path_rate = 0.2


class SFNetTrainingConfig:
    """Training hyperparameters for SF-Net."""
    
    def __init__(self):
        # Data paths
        self.data_root = '../../../Dataset/NEH/NEH_UT_2021RetinalOCTDataset'
        self.csv_path = '../../../Dataset/NEH/data_information.csv'
        self.output_dir = 'results'
        
        # Model config
        self.model_config = SFNetConfig()
        self.image_size = 224  # SF-Net can use 224 (faster) or 384 (better quality)
        
        # Training params (Based on paper)
        self.epochs = 300  # Paper uses 300 epochs
        self.batch_size = 64  # Paper uses batch size 64
        self.num_workers = 8
        self.learning_rate = 0.0005  # Paper's initial LR
        self.weight_decay = 0.05  # Paper's weight decay
        self.warmup_epochs = 10
        self.min_lr = 1e-6
        self.patience = 50  # More patience due to longer training
        self.grad_clip = 1.0
        
        # Advanced training
        self.use_amp = True  # Mixed precision
        self.use_class_weights = True  # Critical for imbalanced data
        
        # Cross-validation
        self.n_folds = 5
        self.random_state = 43
        
        # Data augmentation
        self.use_augmentation = True

