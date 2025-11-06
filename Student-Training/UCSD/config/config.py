"""Configuration for student model training with knowledge distillation."""


class StudentConfig:
    """Configuration for student model training"""
    
    def __init__(self):
        # Data paths
        self.train_dir = '../../../Dataset/UCSD/CellData/OCT/train'
        self.test_dir = '../../../Dataset/UCSD/CellData/OCT/test'
        self.teacher_model_dir = '../../../Teacher-Training/UCSD/results'  
        
        self.student_backbone = 'efficientnet_b2'  
        self.teacher_backbone = 'convnextv2_large.fcmae_ft_in22k_in1k'
        
        # Model settings
        self.num_classes = 4  # CNV, DME, DRUSEN, NORMAL
        self.image_size = 384
        self.dropout = 0.3
        
        # Training settings
        self.batch_size = 16  
        self.accumulation_steps = 2
        self.num_workers = 8
        self.epochs = 100
        self.warmup_epochs = 5
        
        # Optimizer
        self.learning_rate = 1e-3  # Higher LR for student
        self.min_lr = 1e-6
        self.weight_decay = 0.01
        
        # Distillation hyperparameters
        self.temperature = 4.0  
        self.alpha = 0.7  
        self.beta = 0.3  
        
        self.use_randaugment = True
        
        # Output
        self.output_dir = 'results'
        self.save_best = True
        self.log_interval = 10
        self.early_stopping_patience = 20
        
        # Cross-validation
        self.n_folds = 5
        self.val_split = 0.2  # From training set in each fold
        self.random_state = 43

