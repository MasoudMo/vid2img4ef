import torch
import yaml
import os
import wandb
from src.builders import dataloader_builder, dataset_builder, criteria_builder, model_builder, optimizer_builder, \
    scheduler_builder, meter_builder
from src.utils import reset_meters, update_meters, to_train, to_eval, update_learning_rate, move_to_device, \
    set_requires_grad, save_networks
import matplotlib.pyplot as plt
from tqdm import tqdm


class Engine(object):

    def __init__(self,
                 config_path,
                 logger,
                 save_dir):

        self.logger = logger
        self.save_dir = save_dir

        # Load and process the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config

        # Set up Wandb if required
        self.use_wandb = config['train']['use_wandb']
        if config['train']['use_wandb']:
            wandb.init(project=config['train']['wandb_project_name'],
                       name=config['train']['wandb_run_name'],
                       config=config,
                       mode=config['train']['wandb_mode'])

            wandb.define_metric("batch_train/step")
            wandb.define_metric("batch_valid/step")
            wandb.define_metric("epoch")
            # set all other metrics to use the corresponding step
            wandb.define_metric("batch_train/*", step_metric="batch_train/step")
            wandb.define_metric("batch_valid/*", step_metric="batch_valid/step")
            wandb.define_metric("epoch/*", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")

    def train(self):

        self._process_config_file(self.config, phase='training')
        self._build(phase='training')
        self._train()

        if self.train_config['use_wandb']:
            wandb.finish()

    def evaluate(self):

        self._process_config_file(self.config, phase='test')
        self._build(phase='test')
        self._evaluate_once(epoch=0, phase='test')

        if self.train_config['use_wandb']:
            wandb.finish()

    def _process_config_file(self, config, phase='training'):

        # Extract configurations for each component
        self.data_config = config['data']
        self.train_config = config['train']
        self.model_config = config['model']

        # Add training mode to configs that need it
        self.train_config['criteria'].update({'mode': self.train_config['mode']})
        self.train_config['optimizer'].update({'mode': self.train_config['mode'],
                                               'n_epochs': self.train_config['n_initial_epochs'],
                                               'n_epochs_decay': self.train_config['n_decay_epochs']})
        self.model_config.update({'mode': self.train_config['mode']})

    def _build(self, phase):

        # Build the datasets
        dataset = dataset_builder.build(config=self.data_config,
                                        logger=self.logger)

        # Build the dataloaders
        self.dataloader = dataloader_builder.build(config=self.train_config,
                                                   dataset=dataset,
                                                   logger=self.logger,
                                                   phase=phase)

        # Build the model
        self.model = model_builder.build(config=self.model_config,
                                         logger=self.logger)

        # Create the criteria
        self.criteria = criteria_builder.build(config=self.train_config['criteria'],
                                               logger=self.logger)

        # Create the optimizers
        self.optimizer = optimizer_builder.build(config=self.train_config['optimizer'],
                                                 model=self.model,
                                                 logger=self.logger)

        # Create the schedulers
        self.scheduler = scheduler_builder.build(config=self.train_config['optimizer'],
                                                 optimizer=self.optimizer,
                                                 logger=self.logger)

        # Create the loss meter
        self.loss_meters = meter_builder.build(logger=self.logger, mode=self.train_config['mode'])

    def _train(self):

        for epoch in range(self.train_config['n_initial_epochs'] + self.train_config['n_decay_epochs']  + 1):

            # Reset meters
            reset_meters(self.loss_meters)

            # Train for one epoch
            self._train_one_epoch(epoch)

            # Save model after each epoch
            save_networks(self.model, self.save_dir, self.model_config['gpu_ids'])

            # Reset meters
            reset_meters(self.loss_meters)

            # Validation epoch
            self._evaluate_once(epoch, 'val')

            # (to be updated to save best checkpoints)

    def _train_one_epoch(self, epoch):
        # move models to train mode
        to_train(self.model)

        # Update model's learning rate
        update_learning_rate(self.optimizer, self.scheduler, self.train_config['optimizer'])

        epoch_steps = len(self.dataloader['training'])

        data_iter = iter(self.dataloader['training'])
        iterator = tqdm(range(epoch_steps), dynamic_ncols=True)
        for i in iterator:
            (cine_vid, ed_frame, es_frame, label) = next(data_iter)

            if len(self.model_config['gpu_ids']) > 0 and torch.cuda.is_available():
                cine_vid = cine_vid.cuda()
                ed_frame = ed_frame.cuda()
                es_frame = es_frame.cuda()
                label = label.cuda()

            loss_G, loss_D, fake_img = self._forward_optimize(cine_vid, ed_frame, es_frame, label)

            with torch.no_grad():
                update_meters(self.loss_meters, {'gen': loss_G, 'disc': loss_D})

                self.set_tqdm_description(iterator, 'training', epoch, {'gen': loss_G,'disc': loss_D})

                step = (epoch * epoch_steps + i) * self.train_config['batch_size']

                if self.train_config['use_wandb']:

                    if i % self.train_config['wandb_iters_per_log'] == 0:
                        self.log_wandb({'gen': loss_G, 'disc': loss_D}, {'step': step}, mode='batch_train')
                        self.log_wandb_img(torch.cat((ed_frame, es_frame, fake_img),
                                                     dim=1).detach().cpu().numpy(),
                                           {'step': step},
                                           mode='batch_train')

        # Epoch stats
        if self.train_config['mode'] == 'generator':
            total_loss = self.loss_meters['gen'].avg + self.loss_meters['disc'].avg
            losses = {'gen': self.loss_meters['gen'].avg, 'disc': self.loss_meters['disc'].avg}

            self.logger.info_important("Training Epoch {} - Total loss: {}, "
                                       "Generator loss: {}, Discriminator loss: {}".format(epoch,
                                                                                           total_loss,
                                                                                           losses['gen'],
                                                                                           losses['disc']))

            if self.train_config['use_wandb']:
                self.log_wandb(losses, {"epoch": epoch}, mode='epoch/train')

        else:
            pass

    def _forward_optimize(self, cine_vid, ed_frame, es_frame, label=None, phase='training'):

        if self.train_config['mode'] == 'generator':
            img_embedding = self.model['encoder'](cine_vid)
            fake_img = self.model['decoder'](img_embedding)

            if phase == 'training':

                ################# Discriminator step ##################
                set_requires_grad(self.model['disc'], True)
                self.optimizer['disc'].zero_grad()

                pred_fake = self.model['disc'](torch.cat((ed_frame, es_frame, fake_img), 1).detach())
                loss_D_fake = self.criteria['GAN'](pred_fake, False)

                real = self.model['disc'](torch.cat((ed_frame, es_frame, ed_frame, es_frame), 1))
                loss_D_real = self.criteria['GAN'](real, True)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimizer['disc'].step()

                ################## Generator step ###################
                set_requires_grad(self.model['disc'], False)
                self.optimizer['gen'].zero_grad()

                pred_fake = self.model['disc'](torch.cat((ed_frame, es_frame, fake_img), 1).detach())
                loss_G_GAN = self.criteria['GAN'](pred_fake, True)

                loss_G_L1 = self.criteria['L1'](fake_img, torch.cat((ed_frame, es_frame), 1))

                loss_G = loss_G_GAN * self.train_config['criteria']['GAN_lambda'] + \
                         loss_G_L1 * self.train_config['criteria']['l1_lambda']

                loss_G.backward()
                self.optimizer['gen'].step()

        else:
            loss_G = 0
            loss_D = 0
            pass

        return loss_G.detach().item(), loss_D.detach().item(), fake_img.detach()

    def log_wandb(self, losses, step_metric, mode='batch_train'):

        if not self.train_config['use_wandb']:
            return

        step_name, step_value = step_metric.popitem()

        if "batch" in mode:
            log_dict = {f'{mode}/{step_name}': step_value}
        elif "epoch" in mode:
            log_dict = {f'{step_name}': step_value,   # both train and valid x axis are called epoch
                        'lr': self.optimizer['gen'].param_groups[0]['lr']}  # record the Learning Rate
        else:
            raise "invalid mode for wandb logging"

        for loss_name, loss in losses.items():
            loss = loss.item() if type(loss) == torch.Tensor else loss
            log_dict.update({f'{mode}/{loss_name}': loss})

        wandb.log(log_dict)

    def log_wandb_img(self, img, step_metric, mode='batch_train'):
        step_name, step_value = step_metric.popitem()

        fig = plt.figure()

        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        ax1.title.set_text('Real ED')
        ax2.title.set_text('Real ES')
        ax3.title.set_text('Fake ED')
        ax4.title.set_text('Fake ES')

        ax1.imshow(img[0][0]*255, cmap='gray', vmin=0, vmax=255)
        ax2.imshow(img[0][1]*255, cmap='gray', vmin=0, vmax=255)
        ax3.imshow(img[0][2]*255, cmap='gray', vmin=0, vmax=255)
        ax4.imshow(img[0][3]*255, cmap='gray', vmin=0, vmax=255)

        wandb.log({f'{mode}/vis': fig,
                   f'{mode}/{step_name}': step_value})
        plt.close()

    def _evaluate_once(self, epoch=0, phase='test'):
        pass

    def set_tqdm_description(self, iterator, mode, epoch, losses):

        if self.train_config['mode'] == 'generator':
            iterator.set_description("[Epoch {}] | {} | Generator Loss: {:.4f} | "
                                     "Discriminator Loss: {:.4f} | ".format(epoch,
                                                                            mode,
                                                                            losses['gen'],
                                                                            losses['disc']),
                                     refresh=True)
