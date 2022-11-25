import logging
from colorlog import ColoredFormatter
import torch
import os


def create_logger(name: str) -> logging.Logger:
    """
    Creates a custom logger

    :param name: str, name for the logger
    :return: A custom logging.logger object
    """

    # Define a new level for the logger
    def _info_important(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOIMPORTANT':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.propagate = False
    logger.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOIMPORTANT')
    logging.Logger.info_important = _info_important

    return logger


def reset_meters(meters):

    for meter in meters.keys():
        meters[meter].reset()


def update_meters(meters, values):
    for meter in meters.keys():
        meters[meter].update(values[meter])


def to_train(model):
    for key in model.keys():
        model[key].train()


def to_eval(model):
    for key in model.keys():
        model[key].eval()


def update_learning_rate(optimizer, scheduler, config, metric=None):
    """Update learning rates for all the networks; called at the end of every epoch"""
    keys = list(optimizer.keys())
    old_lr = optimizer[keys[0]].param_groups[0]['lr']
    for key in scheduler.keys():
        if config['lr_policy'] == 'plateau':
            scheduler[key].step(metric)
        else:
            scheduler[key].step()

    lr = optimizer[keys[0]].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))


def move_to_device(data_list):
    if torch.cuda.is_available():
        for data in data_list:
            data = data.cuda()


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_networks(models, path, gpu_ids, mode):
    for (name, model) in models.items():
        if len(gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(model.module.cpu().state_dict(), os.path.join(path, mode + '_' + name + '.pth'))
            model.cuda(gpu_ids[0])
        else:
            torch.save(model.cpu().state_dict(), os.path.join(path, mode + '_' + name + '.pth'))
