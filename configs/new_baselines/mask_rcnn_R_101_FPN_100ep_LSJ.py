from .mask_rcnn_R_50_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

dataloader.train.total_batch_size = 4
model.backbone.bottom_up.stages.depth = 101
