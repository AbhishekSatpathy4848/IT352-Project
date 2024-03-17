import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_json('data/train.json')

unique_labels = []
for labels in df['labels']:
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

print(len(unique_labels))
print(unique_labels)

binary_dfs = {}
for label in unique_labels:
    df_new = df.copy()[['text', 'labels']]
    df_new['labels'] = df_new['labels'].apply(lambda x: torch.Tensor([1.0, 0.0]) if label in x else torch.Tensor([0.0, 1.0]))
    binary_dfs[label] = df_new

class_weights = {}
tot = 7000
for label in unique_labels:
    df_new = binary_dfs[label]
    pos = 0
    for labels in df_new['labels']:
        if labels[0] == 1.0:
            pos += 1
    class_weights[label] = [tot / pos, tot / (tot - pos)]

from torch.utils import data as data
from PIL import Image

class Subtask1Dataset(data.Dataset):
    def _init_(self, labels: list, text_tensor: torch.Tensor, transform = None):
        self.labels = labels
        self.label_len = 2
        self.text_tensor = text_tensor

    def _len_(self):
        return self.text_tensor.shape[0]

    def _getitem_(self, index):
        return self.text_tensor[index], self.labels[index]
    
import torch.nn as nn
from transformers import AutoModelForImageClassification

class SingleClassClassifier(nn.Module):
  def _init_(self):
    super()._init_()
    self.fc1 = nn.Linear(768, 512)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(512, 128)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(128, 2)

  def forward(self, text):
    text = self.fc1(text)
    text = self.relu1(text)
    text = self.fc2(text)
    text = self.relu2(text)
    text = self.fc3(text)
    return text
  
def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (text, labels) in enumerate(train_loader):
        text = text.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    train_losses = []
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print("Epoch: {} | Train Loss: {}".format(epoch, train_loss))
    return train_losses

text_tensor = torch.cat(torch.load('text_tensors/train.pt'))

#form a loop to train on all unique labels and save the models with the corresponding label names
for label in unique_labels:
    model = SingleClassClassifier().to(device)
    labels = binary_dfs[label]['labels'].tolist()
    dataset = Subtask1Dataset(labels, text_tensor)
    train_loader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights[label]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, None, criterion, optimizer, device, 75)
    #remove '/' characters from label names
    label_new = label.replace('/', '')
    path = 'models/' + label_new + '.pt'
    torch.save(model.state_dict(), path)