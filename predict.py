# Comet must come first
# from comet_ml import Experiment

# Import all other modules
import os
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
import torchio as tio
from hydra import compose, initialize
from omegaconf import DictConfig
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from torch import nn, optim

from bagginghsf.data.loader import load_from_config
from bagginghsf.models.losses import FocalTversky_loss
from bagginghsf.models.models import SegmentationModel

initialize(config_path="conf")
cfg = compose(config_name="config")
# print(OmegaConf.to_yaml(cfg))

seg_loss = FocalTversky_loss({"apply_nonlin": None})
optimizer = optim.AdamW
scheduler = partial(optim.lr_scheduler.CosineAnnealingLR,
                    T_max=cfg.lightning.max_epochs)
learning_rate = 1e-4
# classes_names = None

# Load and setup model
if cfg.models.n == 1:
    model_name = list(cfg.models.models.keys())[0]
    hparams = cfg.models.models[model_name].hparams
    is_capsnet = cfg.models.models[model_name].is_capsnet
else:
    raise NotImplementedError

model = SegmentationModel.load_from_checkpoint(
    checkpoint_path="pretrained/resdunet_partial.ckpt",
    hparams=hparams,
    seg_loss=seg_loss,
    optimizer=optimizer,
    scheduler=scheduler,
    learning_rate=learning_rate,
    is_capsnet=is_capsnet)

sub = tio.Subject(mri=tio.ScalarImage(
    "/home/cp264607/Datasets/hippocampus_memodev_3T/sub04/sub04_t2_bet_hippocampus_left_AffineFast_crop.nii.gz"
))

preprocessing_pipeline = tio.Compose([
    tio.ToCanonical(),
    tio.ZNormalization(),
    tio.EnsureShapeMultiple(8),
])

sub = preprocessing_pipeline(sub)
sub.plot()
segmentation = model(sub["mri"]["data"].unsqueeze(0))
_, seg = segmentation.max(dim=1)

from bagginghsf.utils import visualize
plot_volume = visualize.plot_volume_interactive

plot_volume(sub.mri.numpy().squeeze())

output = tio.LabelMap(seg[0])
sub.add_image(output, "segmentation")
sub.segmentation.plot()
