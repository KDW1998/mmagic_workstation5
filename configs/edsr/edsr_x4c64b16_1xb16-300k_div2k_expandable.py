_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/sisr_x4_test_config.py'
]

experiment_name = 'edsr_x4c64b16_1xb16-300k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

load_from = None  # Pretrained x2 weights can be loaded here if available

scale = 4

# 모델 정의
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='EDSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,
        res_scale=1,
        rgb_mean=[0.4488, 0.4371, 0.4040],
        rgb_std=[1.0, 1.0, 1.0]
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR'], crop_border=scale),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    )
)

# 학습 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=196),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]

# 검증 pipeline
val_pipeline = [
    dict(type='LoadImageFromFile', key='img', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='LoadImageFromFile', key='gt', color_type='color', channel_order='rgb', imdecode_backend='cv2'),
    dict(type='PackInputs')
]

# 학습 데이터 경로 리스트 (확장 가능)
train_data_roots = [
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/BR/br_합성전_100개단위/br105_to_finish',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/BR/br_합성전_100개단위/br_19111_01_to_br_19111_104/leftImg8bit',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/BR/br_합성전_100개단위/DJI_P34_P35/leftImg8bit',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/BR/br_합성전_100개단위/DJI_P36_P37/leftImg8bit',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/탄천2고가교',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/배수관',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/1차년도_raw_image',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/2022_현장촬영이미지',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/2022onsiteimgsplit',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/2023현장촬영',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/P4윈드삭고흥-D면가드레일',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Joint/한국도로공사_split',
    '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/leakage'

    # '/path/to/dataset3', 등 추가 가능
]

# 학습용 dataloader - ConcatDataset 구성
train_dataloader = dict(
    num_workers=8,
    batch_size=16,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='BasicImageDataset',
                ann_file=f'{root}/train/meta_info_train.txt',
                metainfo=dict(dataset_type='div2k', task_name='sisr'),
                data_root=root,
                data_prefix=dict(img='train/LR', gt='train/HR'),
                filename_tmpl=dict(img='{}', gt='{}'),
                pipeline=train_pipeline
            ) for root in train_data_roots
        ]
    )
)

# 검증용 dataloader
val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='mydata_val', task_name='sisr'),
        data_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/SRData_total',
        data_prefix=dict(img='val/LR', gt='val/HR'),
        pipeline=val_pipeline
    )
)

# 평가 metric
val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ]
)

# 학습 설정
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=3000)
val_cfg = dict(type='MultiValLoop')

# Optimizer 설정
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
)

# Learning rate 스케줄러
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    milestones=[200000],
    gamma=0.5
)

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
