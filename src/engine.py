import torch
import yaml
import wandb
from src.builders import dataloader_builder, dataset_builder, criteria_builder, model_builder, optimizer_builder, \
    scheduler_builder, meter_builder
from src.utils import reset_meters, update_meters, to_train, to_eval, update_learning_rate, \
    set_requires_grad, save_networks
from src.core.evaluators import R2Evaluator
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class Engine(object):

    def __init__(self,
                 config_path,
                 logger,
                 save_dir):

        self.logger = logger
        self.save_dir = save_dir

        self.best_error = 1000000

        # Load and process the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config

        # Set up Wandb if required
        self.use_wandb = config['train']['use_wandb']
        if config['train']['use_wandb']:
            wandb.init(project=config['train']['wand_project_name'],
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

        self.model_config['generator']['decoder'].update({'input_channels':
                                                              self.model_config['generator']['encoder']['num_conv_filters'][-1]})
        self.model_config['regressor'].update({'input_channels':
                                                              self.model_config['generator']['encoder']['num_conv_filters'][-1]})

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
                                         logger=self.logger,
                                         checkpoint_path=self.model_config['checkpoint_path'])

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

        # Create the R2 evaluator
        if self.train_config['mode'] == 'ef':
            self.r2_evaluator = R2Evaluator()

        # Create the loss meter
        self.loss_meters = meter_builder.build(logger=self.logger, mode=self.train_config['mode'])

    def _train(self):

        for epoch in range(self.train_config['n_initial_epochs'] + self.train_config['n_decay_epochs']  + 1):

            # Reset meters/evaluators
            reset_meters(self.loss_meters)
            if self.train_config['mode'] == 'ef':
                self.r2_evaluator.reset()

            # Train for one epoch
            self._train_one_epoch(epoch)

            # Save model after each epoch
            save_networks(self.model, self.save_dir, self.model_config['gpu_ids'], mode='last')

            # Reset meters/evaluators
            reset_meters(self.loss_meters)
            if self.train_config['mode'] == 'ef':
                self.r2_evaluator.reset()

            # Validation epoch
            error = self._evaluate_once(epoch, 'val')

            # (to be updated to save best checkpoints)
            if error < self.best_error:
                save_networks(self.model, self.save_dir, self.model_config['gpu_ids'], mode='best')
                self.best_error = error


    def _train_one_epoch(self, epoch):
        # move models to train mode
        to_train(self.model)

        # Update model's learning rate
        if epoch != 0:
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

            loss_G, loss_D, loss_ef, fake_img = self._forward_optimize(cine_vid, ed_frame, es_frame, label)

            with torch.no_grad():
                update_meters(self.loss_meters, {'gen': loss_G, 'disc': loss_D, 'ef': loss_ef})

                self.set_tqdm_description(iterator, 'training', epoch, {'gen': loss_G, 'disc': loss_D, 'ef': loss_ef})

                step = (epoch * epoch_steps + i) * self.train_config['batch_size']

                if self.train_config['use_wandb']:

                    if i % self.train_config['wandb_iters_per_log'] == 0:
                        self.log_wandb({'gen': loss_G, 'disc': loss_D, 'ef': loss_ef}, {'step': step},
                                       mode='batch_train')

                        if self.train_config['mode'] == 'generator':
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
            total_loss = self.loss_meters['ef'].avg
            losses = {'ef': self.loss_meters['ef'].avg,
                      'r2': self.r2_evaluator.compute()}

            self.logger.info_important("Training Epoch {} - Total loss: {}, "
                                       "EF loss: {}, R2 Score: {}".format(epoch,
                                                                          total_loss,
                                                                          losses['ef'],
                                                                          losses['r2']))

            if self.train_config['use_wandb']:
                self.log_wandb(losses, {"epoch": epoch}, mode='epoch/train')

    def _forward_optimize(self, cine_vid, ed_frame, es_frame, label=None, phase='training'):

        img_embedding = self.model['encoder'](cine_vid)

        if phase == 'training':
            for key in self.optimizer.keys():
                self.optimizer[key].zero_grad()

        if self.train_config['mode'] == 'generator':
            fake_img = self.model['decoder'](img_embedding.squeeze(2))

            loss_ef = torch.zeros(1)

            ################# Discriminator step ##################
            set_requires_grad(self.model['disc'], True)
            self.optimizer['disc'].zero_grad()

            pred_fake = self.model['disc'](torch.cat((ed_frame, es_frame, fake_img), 1).detach())
            loss_D_fake = self.criteria['GAN'](pred_fake, False)

            real = self.model['disc'](torch.cat((ed_frame, es_frame, ed_frame, es_frame), 1))
            loss_D_real = self.criteria['GAN'](real, True)

            loss_D = (loss_D_real + loss_D_fake) * 0.5

            if phase == 'training':
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

            if phase == 'training':
                loss_G.backward()
                self.optimizer['gen'].step()

        else:
            loss_G = torch.zeros(1)
            loss_D = torch.zeros(1)
            fake_img = torch.zeros(1)

            pred_ef = self.model['regressor'](img_embedding.squeeze(2))

            loss_ef = self.criteria['L1'](pred_ef.squeeze(), label.squeeze())

            if phase == 'training':
                loss_ef.backward()
                self.optimizer['ef'].step()

            with torch.no_grad():
                self.r2_evaluator.update(pred_ef.squeeze().detach().cpu().numpy(),
                                         label.squeeze().detach().cpu().numpy())

        return loss_G.detach().item(), loss_D.detach().item(), loss_ef.detach().item(), fake_img.detach()

    def log_wandb(self, losses, step_metric, mode='batch_train'):

        if not self.train_config['use_wandb']:
            return

        step_name, step_value = step_metric.popitem()

        if "batch" in mode:
            log_dict = {f'{mode}/{step_name}': step_value}
        elif "epoch" in mode:
            log_dict = {f'{step_name}': step_value,   # both train and valid x axis are called epoch
                        'lr': self.optimizer[list(self.optimizer.keys())[0]].param_groups[0]['lr']}  # record the Learning Rate
        else:
            raise("invalid mode for wandb logging")

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

        with torch.no_grad():
            # move models to train mode
            to_eval(self.model)

            epoch_steps = len(self.dataloader[phase])

            data_iter = iter(self.dataloader[phase])
            iterator = tqdm(range(epoch_steps), dynamic_ncols=True)
            for i in iterator:
                (cine_vid, ed_frame, es_frame, label) = next(data_iter)

                if len(self.model_config['gpu_ids']) > 0 and torch.cuda.is_available():
                    cine_vid = cine_vid.cuda()
                    ed_frame = ed_frame.cuda()
                    es_frame = es_frame.cuda()
                    label = label.cuda()

                loss_G, loss_D, loss_ef, fake_img = self._forward_optimize(cine_vid, ed_frame, es_frame, label, phase)

                update_meters(self.loss_meters, {'gen': loss_G, 'disc': loss_D, 'ef': loss_ef})

                self.set_tqdm_description(iterator, 'validation/test', epoch, {'gen': loss_G,
                                                                               'disc': loss_D,
                                                                               'ef': loss_ef})

                step = (epoch * epoch_steps + i) * self.train_config['batch_size']

                if self.train_config['use_wandb']:

                    if i % self.train_config['wandb_iters_per_log'] == 0:
                        self.log_wandb({'gen': loss_G, 'disc': loss_D, 'ef': loss_ef}, {'step': step},
                                       mode='batch_val')

                        if self.train_config['mode'] == 'generator':
                            self.log_wandb_img(torch.cat((ed_frame, es_frame, fake_img),
                                                         dim=1).detach().cpu().numpy(),
                                               {'step': step},
                                               mode='batch_val')

            # Epoch stats
            if self.train_config['mode'] == 'generator':
                total_loss = self.loss_meters['gen'].avg + self.loss_meters['disc'].avg
                losses = {'gen': self.loss_meters['gen'].avg, 'disc': self.loss_meters['disc'].avg}

                self.logger.info_important("Validation/Test Epoch {} - Total loss: {}, "
                                           "Generator loss: {}, Discriminator loss: {}".format(epoch,
                                                                                               total_loss,
                                                                                               losses['gen'],
                                                                                               losses['disc']))

                if self.train_config['use_wandb']:
                    self.log_wandb(losses, {"epoch": epoch}, mode='epoch/val')

                return total_loss

            else:
                total_loss = self.loss_meters['ef'].avg
                losses = {'ef': self.loss_meters['ef'].avg,
                          'r2': self.r2_evaluator.compute()}

                self.logger.info_important("Validation/Test Epoch {} - Total loss: {}, "
                                           "EF loss: {}, R2 Score: {}".format(epoch,
                                                                              total_loss,
                                                                              losses['ef'],
                                                                              losses['r2']))

                if self.train_config['use_wandb']:
                    self.log_wandb(losses, {"epoch": epoch}, mode='epoch/val')

                return total_loss

    def set_tqdm_description(self, iterator, mode, epoch, losses):

        if self.train_config['mode'] == 'generator':
            iterator.set_description("[Epoch {}] | {} | Generator Loss: {:.4f} | "
                                     "Discriminator Loss: {:.4f} | ".format(epoch,
                                                                            mode,
                                                                            losses['gen'],
                                                                            losses['disc']),
                                     refresh=True)
        else:
            iterator.set_description("[Epoch {}] | {} | EF Loss: {:.4f} | ".format(epoch,
                                                                            mode,
                                                                            losses['ef']),
                                     refresh=True)
