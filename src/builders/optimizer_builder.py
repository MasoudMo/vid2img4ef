import logging
import torch.optim
from torch import optim
from copy import deepcopy


def build(config,
          model,
          logger):

    optimizer = dict()

    if config['mode'] == 'generator':
        optimizer['gen'] = torch.optim.Adam(list(model['encoder'].parameters()) + list(model['decoder'].parameters()),
                                            lr=config['lr'],
                                            betas=(config['beta1'], 0.999))
        optimizer['disc'] = torch.optim.Adam(model['disc'].parameters(),
                                             lr=config['lr'],
                                             betas=(config['beta1'], 0.999))
    else:
        optimizer['encoder'] = torch.optim.Adam(model['encoder'].parameters(),
                                                lr=config['lr'],
                                                betas=(config['beta1'], 0.999))

    logger.info_important('Optimizer is built for {} mode.'.format(config['mode'].upper()))

    return optimizer
