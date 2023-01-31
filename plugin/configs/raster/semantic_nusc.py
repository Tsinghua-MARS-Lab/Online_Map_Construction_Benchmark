_base_ = [
    '../_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_size = (128, 352)

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_classes = max(cat2id.values()) + 1

# rasterize configs
roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
canvas_size = (400, 200) # bev feature size
coords_dim = 2 # polylines coordinates dimension, 2 or 3
thickness = 3 # thickness of rasterized polylines

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_external=False,
    output_format='raster')

# model config
model = dict(
    type='HDMapNet_Semantic',
    backbone_cfg=dict(
        type='HDMapNetBackbone',
        img_res=img_size, 
        out_channels=64, 
        canvas_size=canvas_size,
        n_views=7,
    ),
    head_cfg=dict(
        type='BevEncode',
        in_channels=64, 
        out_channels=num_classes,
        ),
    loss_cfg=dict(
        type='SimpleLoss',
        pos_weight=2.13,
        loss_weight=1.0
        ),
)

# data processing pipelines
train_pipeline = [
    dict(
        type='RasterizeMap',
        roi_size=roi_size,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
    ),
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # (H, W)
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=True),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'semantic_mask'], meta_keys=(
        'token', 'cam_intrinsics', 'cam_extrinsics'))
]

# configs for evaluation code
# DO NOT CHANGE
eval_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuScenes',
    ann_file='./datasets/nuScenes/nuscenes_map_infos_val.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='RasterizeMap',
            roi_size=roi_size,
            coords_dim=coords_dim,
            canvas_size=canvas_size,
            thickness=thickness,
        ),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['semantic_mask'], meta_keys=['token'])
    ],
    interval=1,
)

# dataset configs
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='NuscDataset',
        data_root='./datasets/nuScenes',
        ann_file='./datasets/nuScenes/nuscenes_map_infos_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        interval=1,
    ),
    val=dict(
        type='NuscDataset',
        data_root='./datasets/nuScenes',
        ann_file='./datasets/nuScenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        eval_config=eval_config,
        test_mode=True,
        interval=1,
    ),
    test=dict(
        type='NuscDataset',
        data_root='./datasets/nuScenes',
        ann_file='./datasets/nuScenes/nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        eval_config=eval_config,
        test_mode=True,
        interval=1,
    ),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    paramwise_cfg=dict(
    custom_keys={
        # 'backbone': dict(lr_mult=0.1),
    }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    step=[9, 11])
checkpoint_config = dict(interval=3)
total_epochs = 12

# kwargs for dataset evaluation
eval_kwargs = dict()
evaluation = dict(
    interval=3, 
    **eval_kwargs)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

find_unused_parameters = True
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
