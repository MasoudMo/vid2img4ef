from src.core.datasets import EchoNetEfDataset


def build(config,
          logger):

    dataset = EchoNetEfDataset(**config)

    logger.info_important('Dataset is built.')

    return dataset
