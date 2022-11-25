from src.core.meters import AverageEpochMeter


def build(logger,
          mode):

    loss_meters = dict()

    if mode == 'generator':
        loss_meters['gen'] = AverageEpochMeter('Generator Loss', logger)
        loss_meters['disc'] = AverageEpochMeter('Discriminator Loss', logger)
    else:
        loss_meters['ef'] = AverageEpochMeter('EF L1 Loss', logger)

    logger.info_important('Loss meters are built.')

    return loss_meters
