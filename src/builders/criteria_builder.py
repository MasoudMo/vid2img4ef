import torch.nn as nn
from src.core.criteria import GANLoss


def build(config, logger):

    if config['mode'] == 'generator':
        criteria = {'GAN': GANLoss(config['gan_mode']),
                    'L1': nn.L1Loss()}
    else:
        criteria = {'L1': nn.L1Loss()}

    logger.info_important('criteria is built for {} mode.'.format(config['mode'].upper()))

    return criteria
