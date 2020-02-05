from torch.utils.data import DataLoader
import torch

from DatasetAB import DatasetAB
from generator import Generator
from vgg import vgg16
from Trainer import Trainer

if __name__ == '__main__':
    dataset = DatasetAB('../faces94/malestaff/tony/', '../faces94/malestaff/voudcx/')
    dataloader = DataLoader(dataset)
    Attacker = Generator(180, 200, 3, 64, 256)
    Victim = vgg16(num_classes=2)
    criterion = torch.nn.BCELoss()

    trainer = Trainer(Attacker, Victim, optimizer_Attacker, optimizer_Victim,\
                                criterion, dataloader, device, print_every,\
                                save_every, save_img_every, img_dictionary, model_dictionary,\
                                scheduler, monitor, *kargs)
