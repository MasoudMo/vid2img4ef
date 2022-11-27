from src.core.models import Conv3DEncoder, ConvTransposeDecoder, PixelDiscriminator, init_net, PoolLinearRegression, \
    Conv3DPoolLinearRegression, move_model_to_device
from copy import deepcopy
import torch
import os


REGRESSORS = {'linear': PoolLinearRegression,
              'conv': Conv3DPoolLinearRegression}


def build(config,
          logger,
          checkpoint_path):

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
                                    config['init_gain'])

        model['decoder'] = init_net(ConvTransposeDecoder(**config['generator']['decoder']),
                                    config['init_type'],
                                    config['init_gain'])

        model['disc'] = init_net(PixelDiscriminator(**config['disc']),
                                 config['init_type'],
                                 config['init_gain'])

        if checkpoint_path is not None:
            logger.info_important("Loading pretrained encoder/decoder/disc weights for the generator mode.")
            model['encoder'].load_state_dict(torch.load(os.path.join(checkpoint_path, 'encoder.pth')))
            model['decoder'].load_state_dict(torch.load(os.path.join(checkpoint_path, 'decoder.pth')))
            model['disc'].load_state_dict(torch.load(os.path.join(checkpoint_path, 'disc.pth')))

        model['encoder'] = move_model_to_device(model['encoder'], config['gpu_ids'])
        model['decoder'] = move_model_to_device(model['decoder'], config['gpu_ids'])
        model['disc'] = move_model_to_device(model['disc'], config['gpu_ids'])

    else:

        model['encoder'] = init_net(Conv3DEncoder(**config['generator']['encoder']),
                                    config['init_type'],
                                    config['init_gain'])
        freeze_encoder = config['regressor'].pop('freeze_encoder')
        if freeze_encoder:
            for param in model['encoder'].parameters():
                param.requires_grad = False

        regressor_name = config['regressor'].pop('type')
        model['regressor'] = init_net(REGRESSORS[regressor_name](**config['regressor']),
                                      config['init_type'],
                                      config['init_gain'])

        if checkpoint_path is not None:
            logger.info_important("Loading pretrained encoder/regressor weights for the generator mode.")

            model['encoder'].load_state_dict(torch.load(os.path.join(checkpoint_path, 'encoder.pth')))

            if os.path.exists(os.path.join(checkpoint_path, 'regressor.pth')):
                model['regressor'].load_state_dict(torch.load(os.path.join(checkpoint_path, 'regressor.pth')))
            else:
                logger.info_important("Regressor weights not present. Only loading encoder weights...")

        model['encoder'] = move_model_to_device(model['encoder'], config['gpu_ids'])
        model['regressor'] = move_model_to_device(model['regressor'], config['gpu_ids'])

    logger.info_important('Model is built for {} mode.'.format(config['mode'].upper()))

    return model
