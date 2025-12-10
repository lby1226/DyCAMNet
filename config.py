class Config:
    # 数据集配置
    data_config = {
        'train_dir': '',
        'test_dir': '',
        'img_size': 224,
        'train_val_split': 0.8,
        'num_workers': 4
    }
    
    # 训练配置
    train_config = {
        'seed': 42,
        'batch_size': 32,
        'num_epochs': 100,
        'base_lr': 0.0001,
        'weight_decay': 0.01,
        'min_lr': 1e-6,
        'T_max': 100,
        

        'random_rotate': 10,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2
        },
        

        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }
    

    model_config = {

        'block_layers': [3, 4, 6, 3],
        'num_classes': 2,
        'zero_init_residual': False,
        'groups': 1,
        'width_per_group': 64,
        'pretrained': True,

        'dynamic_conv': {
            'kernel_size': 3,
            'reduction_ratio': 4,
            'num_groups': 4,
            'bias': True
        },

        'dynamic_attn': {
            'num_heads': 4,
            'qk_scale': None,
            'attn_drop': 0,
            'sr_ratio': 1
        },
        

        'multi_scale_conv': {
            'kernel_sizes': [7, 5, 3, 1],
            'reduction_ratio': 4,
            'num_groups': 4,
            'use_bias': True,
            'gn_groups': 8,
            'use_residual': True
        },
        

        'fusion': {
            'reduction_ratio': 4
        }
    }
    

    save_config = {
        'save_dir': '',
        'model_name': 'best_model.pth',
        'save_freq': 1
    }
    

    test_config = {
        'batch_size': 1,
        'num_vis_samples': 10,
        'pred_save_dir': 'pred_samples',
        'confusion_matrix_name': 'confusion_matrix.png'
    }
    

    log_config = {
        'log_dir': 'logs',
        'log_freq': 10
    }

    @staticmethod
    def get_dynamic_conv_config():
        return Config.model_config['dynamic_conv']
    
    @staticmethod
    def get_dynamic_attn_config():
        return Config.model_config['dynamic_attn']
    
    @staticmethod
    def get_multi_scale_conv_config():
        return Config.model_config['multi_scale_conv']
    
    @staticmethod
    def get_fusion_config():
        return Config.model_config['fusion']
    
    @staticmethod
    def get_train_transform_config():
        return {
            'img_size': Config.data_config['img_size'],
            'random_rotate': Config.train_config['random_rotate'],
            'color_jitter': Config.train_config['color_jitter'],
            'normalize_mean': Config.train_config['normalize_mean'],
            'normalize_std': Config.train_config['normalize_std']
        }
    
    @staticmethod
    def get_test_transform_config():
        return {
            'img_size': Config.data_config['img_size'],
            'normalize_mean': Config.train_config['normalize_mean'],
            'normalize_std': Config.train_config['normalize_std']
        }