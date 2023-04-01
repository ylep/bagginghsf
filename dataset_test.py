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

mri_datamodule = load_from_config(cfg.datasets)(
    preprocessing_pipeline=tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(),
        tio.EnsureShapeMultiple(8),
    ]),
    augmentation_pipeline=tio.Compose([
        tio.RandomAffine(scales=.1, degrees=5, translation=3, p=.1),
        tio.RandomAnisotropy(p=.1),
        tio.transforms.RandomElasticDeformation(num_control_points=4,
                                                max_displacement=3,
                                                locked_borders=0,
                                                p=.05),
        tio.RandomFlip(axes=('LR',), flip_probability=.2),
        # tio.RandomMotion(degrees=5, translation=5, num_transforms=2, p=.01),
        # tio.RandomSpike(p=.01),
        # tio.RandomBiasField(coefficients=.2, p=.01),
        tio.RandomBlur(std=(0, 0.1), p=.01),
        tio.RandomNoise(mean=0, std=0.1, p=.1),
        tio.RandomGamma(log_gamma=0.1, p=.1),
    ]),
    postprocessing_pipeline=tio.Compose([tio.OneHot()]))
mri_datamodule.setup()

for _ in range(5):
    batch = next(iter(mri_datamodule.train_dataloader()))

    subject = tio.utils.get_subjects_from_batch(batch)[0]
    subject.plot()

# batch["label"]["data"].shape
