"""Configuration for student model training with knowledge distillation."""


class StudentConfig:
    """Configuration for student model training"""
    
    def __init__(self):
        # Data paths
        self.data_root = '../../../Dataset/NEH/NEH_UT_2021RetinalOCTDataset'
        self.csv_path = '../../../Dataset/NEH/data_information.csv'
        self.teacher_model_dir = '../../../Teacher-Training/NEH/results'
        
        # Model architectures
        self.student_backbone = 'efficientnet_b2'  
        self.teacher_backbone = 'convnextv2_large.fcmae_ft_in22k_in1k'
        
        # Model settings
        self.num_classes = 3
        self.image_size = 384
        self.dropout = 0.3
        
        # Training settings
        self.batch_size = 8  
        self.accumulation_steps = 2
        self.num_workers = 4
        self.epochs = 100
        self.warmup_epochs = 5
        
        # Optimizer
        self.learning_rate = 1e-3  
        self.min_lr = 1e-6
        self.weight_decay = 0.01
        
        # Distillation hyperparameters
        self.temperature = 4.0  
        self.alpha = 0.7  # Weight for soft target loss
        self.beta = 0.3   # Weight for hard label loss
        
        # Augmentation
        self.use_randaugment = True
        
        # Output
        self.output_dir = 'results'
        self.save_best = True
        self.log_interval = 10
        self.early_stopping_patience = 20
        
        # Cross-validation
        self.n_folds = 5
        self.random_state = 43

