import torchio as tio
from torchio.transforms.preprocessing.intensity.z_normalization import ZNormalization

sub = tio.Subject(mri=tio.ScalarImage(
    "/home/cp264607/Datasets/hippocampus_hiplay_7T/sub-22C/sub_22_T2_bet_hippocampus_left_AffineFast_crop.nii.gz"
))
sub.plot()

# tr = tio.RandomMotion(degrees=5, translation=5, num_transforms=3)
tr = tio.Compose([
    tio.ZNormalization(),
    tio.RandomFlip(axes=('LR',)),
    tio.RandomMotion(degrees=5, translation=5, num_transforms=3),
    tio.RandomBlur(std=(0, 0.5)),
    tio.RandomNoise(mean=0, std=0.5),
    tio.RandomGamma(log_gamma=0.4),
    tio.RandomAffine(scales=.2, degrees=15, translation=3, isotropic=False),
    # # tio.RandomAnisotropy(p=.1, scalars_only=False),
    tio.transforms.RandomElasticDeformation(num_control_points=5,
                                            max_displacement=4,
                                            locked_borders=0),
    # tio.RandomFlip(axes=('LR',)),
    # tio.RandomSpike(p=.01),
    # tio.RandomBiasField(coefficients=.2, p=.01),
])

for _ in range(5):
    tr(sub).plot()
