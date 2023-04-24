import torchvision.transforms as transforms
from PIL import Image
import numpy as np



img = Image.open("dinosaur.jpg")

trans = transforms.Compose([transforms.RandomResizedCrop((640, 640)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.ColorJitter(0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

output = trans(img)

