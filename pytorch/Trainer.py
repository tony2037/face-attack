import torch
from torch.utils.tensorboard  import SummaryWriter
from torchvision import transforms

from tqdm import tqdm
import datetime
import os, sys

class Trainer:
    
    def __init__(self, Attacker, Victim, optimizer_Attacker, optimizer_Victim,\
                                criterion, dataloader, device, print_every,\
                                save_every, save_img_every, img_dictionary, model_dictionary,\
                                scheduler, monitor, *kargs):

        self.Attacker = Attacker
        self.Victim = Victim
        self.optimizer_Attacker = optimizer_Attacker
        self.optimizer_Victim = optimizer_Victim
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.print_every = print_every
        self.save_every = save_every
        self.save_img_every = save_img_every
        self.img_dictionary = img_dictionary
        self.model_dictionary = model_dictionary
        self.scheduler = scheduler
        self.monitor = monitor
        self.real_label = 1
        self.fake_label = 0
        self.epoch = 0
        self.G_losses = []
        self.D_losses = []
        self.transformer = transforms.ToPILImage()

    def run_epoch(self):

        for data in tqdm(dataloader, 0):

            img_a, img_b = data
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            sel.Victim.zero_grad()
            # Format batch
            b_size = img_a.size(0)
            label = torch.full((b_size,), self.real_label, device=self.device)
            # Forward pass real batch through D
            output = self.Victim(img_a).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            fake = self.Attacker(img_a, img_b)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.Victim(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizer_Victim.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.Attacker.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.Victim(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizer_Attacker.step()

            # Output training stats
            if self.epoch % self.print_every == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if self.epoch % self.save_img_every == 0:
                with torch.no_grad():
                    fake = self.Attacker.(img_a, img_b).detach().cpu()
                self._save_image(fake, os.path.join(self.img_dictionary, 'epoch:%s.fake' % str(self.epoch)))
                self._save_image(img_a, os.path.join(self.img_dictionary, 'epoch:%s.A' % str(self.epoch)))
                self._save_image(img_b, os.path.join(self.img_dictionary, 'epoch:%s.B' % str(self.epoch)))
                self._save_model()

            iters += 1

        self.epoch += 1

    def train(self, epochs):
        
        for epoch in epochs:
            self.Attacker.train()
            self.Victim.train()
            run_epoch()

    def _save_model(self):
        
        torch.save(self.Attacker, os.path.join(self.model_dictionary, 'Attacker.model'))
        torch.save(self.Victim, os.path.join(self.model_dictionary, 'Victim.model'))

    def _elapsed_time(self):

        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds

    def _save_image(self, tensor, save_name='example'):
        """
        args:
            @tensor (tensor)
        """
        image = self.transformer(tensor)
        image.save('{}.jpg'.format(save_name))
