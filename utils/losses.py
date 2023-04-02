from monai import losses
from torch import nn


def get_segmentation_loss(loss_name, **kwargs):
    if loss_name in ['Dice', 'Focal', 'DiceCE', 'DiceFocal', 'Tversky',
                     'GeneralizedDiceFocal', 'GeneralizedWassersteinDice', 'GeneralizedDice']:
        loss = losses.__dict__.get(f"{loss_name}Loss")
        return loss(**kwargs)
    else:
        return nn.CrossEntropyLoss(**kwargs)
