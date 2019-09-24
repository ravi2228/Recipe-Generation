import os
import json
import ijson
import nltk
import torch
import torchvision
import torchvision.transforms as transforms
import gensim.models.keyedvectors as word2vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from skimage import io, transform

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
EPOCHS = 20

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class Image2VecDataset(Dataset):
    """Image2Vec dataset."""

    def __init__(self,
        layer1_json='./1-batch.json',
        layer2_json='./layer2.json',
        train_dir='./Recipe-1M/recipe1M_images_train/train',
        vec_model='./GoogleNews-vectors-negative300.bin',
        saved_df=None,
        transform=None):
        """
        Args:
            layer1_json (string): Path to the json file with annotations.
            layer2_json (string): Path to the json file with image paths.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_dir = train_dir
        self.transform = transform

        if saved_df is not None:
            self.landmarks_frame = pd.read_pickle(saved_df)
        else:
            self.landmarks_frame = pd.DataFrame(columns=['id', 'vector', 'image'])
            with open(layer2_json, 'r') as image_path_file:
                image_paths = json.load(image_path_file)

            curr_row = 0
            model = word2vec.KeyedVectors.load_word2vec_format(vec_model, binary=True)
            with open(layer1_json, 'r') as recipe_file:
                for index, item in enumerate(ijson.items(recipe_file, "item")):
                    if item['partition'] != 'train':
                        continue
                    tokenized_title = nltk.word_tokenize(item['title'])
                    vectors = []
                    for token in tokenized_title:
                        try:
                            vectors.append(model[token])
                        except:
                            pass
                    if len(vectors) > 0:
                        vectors = np.array(vectors, dtype=np.float)
                    else:
                        vectors = np.ones((300,), dtype=np.float)

                    for path in image_paths:
                        if path['id'] == item['id']:
                            print(item['id'])
                            for path in path['images']:
                                self.landmarks_frame.loc[curr_row] = [item['id'], np.average(vectors, axis=0), path['id']]
                                curr_row += 1
                            break
            self.landmarks_frame.to_pickle('./train_df.pkl')

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        file_name = self.landmarks_frame.iloc[idx, 2]
        img_name = os.path.join(self.train_dir,
                                file_name[0:1],
                                file_name[1:2],
                                file_name[2:3],
                                file_name[3:4],
                                file_name)
        image = io.imread(img_name)
        target_vector = self.landmarks_frame.iloc[idx, 1]
        sample = {'image': image, 'vector': target_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, vector = sample['image'], sample['vector']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if not isinstance(vector, np.ndarray):
            vector = np.ones((300,), dtype=np.float32)
        return {'image': torch.from_numpy(image).float(),
                'vector': torch.from_numpy(vector).float()}

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, vector = sample['image'], sample['vector']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'vector': vector}

class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, vector = sample['image'], sample['vector']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'vector': vector}

class Net(nn.Module):
    def __init__(self, vec_dim=300):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*28*28, vec_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    """def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(2560, 640)
        self.fc2 = nn.Linear(640, 300)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        #x = x.view(-1, 2560)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x.shape)
        #return F.softmax(x, dim=1)
        return x.view(BATCH_SIZE, -1)"""

def test():
    correct_guesses = 0

    for inputs, labels in testloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        predictions = outputs.max(1, keepdim=True)[1]
        correct_guesses += predictions.eq(labels.view_as(predictions)).int().sum()

    total_inputs = len(testloader.dataset)
    print("{}/{} correct, accuracy: {}".format(int(correct_guesses), total_inputs, float(correct_guesses) / total_inputs))

if __name__ == "__main__":
    trainset = Image2VecDataset(saved_df="train_df.pkl",
                                transform=transforms.Compose([
                                   Rescale(256),
                                   RandomCrop(224),
                                   ToTensor()
                                ]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2)

    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for epoch in range(EPOCHS):
        print("EPOCH " + str(epoch))
        for data in trainloader:
            inputs = data['image']
            labels = data['vector']

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = F.mse_loss(outputs, labels)
            print(loss.item())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "./saved_model.pt")
    plt.plot(losses)
    plt.show()
        #test()
