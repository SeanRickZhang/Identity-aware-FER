import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torch.utils import data

class datasets_7label(Dataset):
    def __init__(self, image_dir, attr_path, transform, mode, just_real_test):

        self.mode = mode
        self.image_dir = image_dir
        self.transform = transform
        self.just_real_test = just_real_test

        self.train_dataset = []
        self.test_dataset = []
        self.X = []
        self.Y = []
        self.org_image = ['00', '01', '02', '03', '04', '05', '06', '07']

        self.X, self.Y = self.preprocess()
        if self.mode == 'train':
            for i in range(len(self.X)):
                self.train_dataset.append([self.X[i], self.Y[i]])
            self.num_images = len(self.train_dataset)
        else:
            for i in range(len(self.X)):
                self.test_dataset.append([self.X[i], self.Y[i]])
            self.num_images = len(self.test_dataset)


    def preprocess(self):
        dirnames = []
        label = []
        filename = []
        for dirname in os.listdir(self.image_dir):
            dirnames.append(os.path.join(self.image_dir, dirname))
        for dirname in dirnames:
            for subdirname in os.listdir(dirname):
                if subdirname in self.org_image:
                    temp = int(subdirname[1])
                    label.append(temp)
                    filename.append(dirname)
                else:
                    continue
        # print(filename, label, identity)
        return filename,label


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image_0 = Image.open(os.path.join(filename, str(0), 'primary_image.jpg'))
        image_1 = Image.open(os.path.join(filename, str(1), 'primary_image.jpg'))
        image_2 = Image.open(os.path.join(filename, str(2), 'primary_image.jpg'))
        image_3 = Image.open(os.path.join(filename, str(3), 'primary_image.jpg'))
        image_4 = Image.open(os.path.join(filename, str(4), 'primary_image.jpg'))
        image_5 = Image.open(os.path.join(filename, str(5), 'primary_image.jpg'))
        image_6 = Image.open(os.path.join(filename, str(6), 'primary_image.jpg'))
        for temp in self.org_image:
            direct = os.path.join(filename, temp, 'primary_image.jpg')
            if os.path.exists(direct):
                image_primary = Image.open(direct)
        # print(filename,label)
        # print(identity)
        return  self.transform(image_0), self.transform(image_1), self.transform(image_2), self.transform(image_3),\
                self.transform(image_4), self.transform(image_5), self.transform(image_6), self.transform(image_primary), \
                torch.LongTensor([label])

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class datasets_6label(Dataset):
    def __init__(self, image_dir, attr_path, transform, mode, just_real_test):

        self.mode = mode
        self.image_dir = image_dir
        self.transform = transform
        self.just_real_test = just_real_test

        self.train_dataset = []
        self.test_dataset = []
        self.X = []
        self.Y = []
        self.org_image = ['00', '01', '02', '03', '04', '05', '06', '07']

        self.X, self.Y = self.preprocess()
        if self.mode == 'train':
            for i in range(len(self.X)):
                self.train_dataset.append([self.X[i], self.Y[i]])
            self.num_images = len(self.train_dataset)
        else:
            for i in range(len(self.X)):
                self.test_dataset.append([self.X[i], self.Y[i]])
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        dirnames = []
        label = []
        filename = []
        for dirname in os.listdir(self.image_dir):
            dirnames.append(os.path.join(self.image_dir, dirname))
        for dirname in dirnames:
            for subdirname in os.listdir(dirname):
                if subdirname in self.org_image:
                    temp = int(subdirname[1])
                    label.append(temp)
                    filename.append(dirname)
                else:
                    continue

        return filename, label

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]

        image_0 = Image.open(os.path.join(filename, str(0), str(label)+'.jpg'))
        image_1 = Image.open(os.path.join(filename, str(1), str(label)+'.jpg'))
        image_2 = Image.open(os.path.join(filename, str(2), str(label)+'.jpg'))
        image_3 = Image.open(os.path.join(filename, str(3), str(label)+'.jpg'))
        image_4 = Image.open(os.path.join(filename, str(4), str(label)+'.jpg'))
        image_5 = Image.open(os.path.join(filename, str(5), str(label)+'.jpg'))
        for temp in self.org_image:
            direct = os.path.join(filename, temp, str(label)+'.jpg')
            if os.path.exists(direct):
                image_primary = Image.open(direct)

        return self.transform(image_0), self.transform(image_1), self.transform(image_2), self.transform(image_3), \
               self.transform(image_4), self.transform(image_5), self.transform(image_primary), torch.LongTensor([label])

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def get_loader(image_dir, batch_size=16, dataset = 'OuluCASIA', mode='train', num_workers=0, resize=True, just_real_test=False):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Grayscale(1))
    if resize==True:
        transform.append(T.Resize(32))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=[0.5], std=[0.5]))
    print('image processing')

    if dataset == 'CKplus':
        transform = T.Compose(transform)
        dataset = datasets_7label(image_dir, None, transform, mode, just_real_test=just_real_test)
    elif dataset == 'MMI_CKplus' or 'OuluCasIA' in dataset:
        transform = T.Compose(transform)
        dataset = datasets_6label(image_dir, None, transform, mode, just_real_test=just_real_test)

    return data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)