from torch.utils.data import DataLoader

from DatasetAB import DatasetAB
from generator import Generator
from vgg import vgg16

dataset = DatasetAB('../faces94/malestaff/tony/', '../faces94/malestaff/voudcx/')
dataloader = DataLoader(dataset)
Attacker = Generator(180, 200, 3, 64, 256)
Victim = vgg16(num_classes=2)

print(Attacker)
print('=' * 35)
print(Victim)

for data in dataloader:
    img_a, img_b = data
    masked_img = Attacker(img_a, img_b)
