from src.core.models import Conv3DEncoder, ConvTransposeDecoder, PixelDiscriminator, init_net, PoolLinearRegression, \
    Conv3DPoolLinearRegression
from copy import deepcopy


REGRESSORS = {'linear': PoolLinearRegression,
              'conv3d': Conv3DPoolLinearRegression}


def build(config,
          logger):

    """
    Builds the models dict

    :param config: dict, model config dict
    :param logger: logging.Logger, custom logger
    :return: Pytorch Module
    """

    config = deepcopy(config)

    model = dict()

    # Create the model
    if config['mode'] == 'generator':
        model['encoder'] = init_net(Conv3DEncoder(**config['generator']['encoder']),
                                    config['init_type'],
                                    config['init_gain'],
                                    config['gpu_ids'])

        model['decoder'] = init_net(ConvTransposeDecoder(**config['generator']['decoder']),
                                    config['init_type'],
                                    config['init_gain'],
                                    config['gpu_ids'])

        model['disc'] = init_net(PixelDiscriminator(**config['disc']),
                                 config['init_type'],
                                 config['init_gain'],
                                 config['gpu_ids'])
    else:

        model['encoder'] = init_net(Conv3DEncoder(**config['generator']['encoder']),
                                    config['init_type'],
                                    config['init_gain'],
                                    config['gpu_ids'])
        freeze_encoder = config['regressor'].pop('freeze_encoder')
        if freeze_encoder:
            for param in model['encoder'].parameters():
                param.requires_grad = False

        regressor_name = config['regressor'].pop('type')
        model['regressor'] = init_net(REGRESSORS[regressor_name](**config['regressor']),
                                      config['init_type'],
                                      config['init_gain'],
                                      config['gpu_ids'])

    logger.info_important('Model is built for {} mode.'.format(config['mode'].upper()))

    return model
