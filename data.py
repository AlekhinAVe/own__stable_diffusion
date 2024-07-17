import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

class DATA():

    def __init__(self, path):
        self.path = path
        self.data = []
        self.data_train = []

    def create_data(self):
        i = 0
        for obj in os.listdir(self.path):
            if i<1000:
                try:
                    sample = Image.open(os.path.join(self.path, obj))
                    self.data.append(sample)
                    i+=1
                except:
                    continue
            else:
                break


    def load_transformed_dataset(self, IMG_SIZE=512):
        data_transforms = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]
        data_transform = transforms.Compose(data_transforms)
        self.data = [data_transform(img) for img in self.data]
        #self.data = self.data[:100]

    def create_dataloader(self, BATCH_SIZE=64):
        self.load_transformed_dataset()
        self.data = DataLoader(self.data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
