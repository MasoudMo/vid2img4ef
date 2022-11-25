from torch.utils.data import SubsetRandomSampler, DataLoader


def build(config,
          dataset,
          logger,
          phase='training') -> dict:

    dataloaders = {}

    # Create samplers for each split
    if phase == 'training':
        train_sampler = SubsetRandomSampler(dataset.train_idx)

        dataloaders['training'] = DataLoader(dataset,
                                             batch_size=config['batch_size'],
                                             drop_last=True,
                                             sampler=train_sampler,
                                             num_workers=config['num_workers'],
                                             pin_memory=True)

        dataloaders['val'] = DataLoader(dataset[dataset.val_idx],
                                        batch_size=1,
                                        drop_last=False,
                                        num_workers=config['num_workers'],
                                        pin_memory=True)

        logger.info_important('Dataloaders are built with {} training '
                              'and {} validation samples.'.format(len(train_sampler),
                                                                  len(dataset.val_idx)))

    elif phase == 'test':
        dataloaders['test'] = DataLoader(dataset[dataset.test_idx],
                                         batch_size=1,
                                         drop_last=False,
                                         num_workers=config['num_workers'])

        logger.info_important('Test Phase Dataloader is built.')
        logger.info_important('Using {} test samples.'.format(len(dataset.text_idx)))

    return dataloaders
