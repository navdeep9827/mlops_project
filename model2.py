import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import joblib

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# Define your ResNet9 model here
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64, pool=True) # 64 x 64 x 64
        self.conv2 = conv_block(64, 128, pool=True) # 128 x 32 x 32
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) # 128 x 32 x 32
        self.conv3 = conv_block(128, 256, pool=True) # 256 x 16 x 16
        self.conv4 = conv_block(256, 512, pool=True) # 512 x 8 x 8
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) # 512 x 8 x 8
        self.conv5 = conv_block(512, 512, pool=True) # 512 x 4 x 4
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.classifier(out)
        return out

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# Define additional functions here
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Setting up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Setting up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def predict_image(img, model, device):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_data.classes[preds[0].item()]

# Main code block
if __name__ == '__main__':
    data_dir = 'E:/MLOps Plaksha/mlops_project/data/'
    classes = os.listdir(data_dir)
    print(classes)

    A_file = os.listdir(data_dir + "A")
    print("NO. of Training examples for A:", len(A_file))
    print(A_file[:5])

    di = {}
    for i in classes:
        di[i] = len(os.listdir(data_dir + i))
    print(di)

    target_num = len(classes)

    raw_images = ImageFolder(data_dir, tt.ToTensor())

    image, label = raw_images[0]
    print("Dimension:", image.shape)
    plt.imshow(image.permute(1, 2, 0))

    raw_dl = DataLoader(raw_images, 400, shuffle=True, num_workers=2, pin_memory=True)

    show_batch(raw_dl)

    average = torch.Tensor([0, 0, 0])
    standard_dev = torch.Tensor([0, 0, 0])

    for image, labels in raw_images:
        average += image.mean([1, 2])
        standard_dev += image.std([1, 2])
    stats_avgs = (average / len(raw_images)).tolist()
    stats_stds = (standard_dev / len(raw_images)).tolist()
    stats_avgs, stats_stds

    stats = (stats_avgs, stats_stds)
    train_tfms = tt.Compose([tt.RandomHorizontalFlip(),
                             tt.ToTensor(),
                             tt.Normalize(*stats, inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    train_data = ImageFolder(data_dir, transform=train_tfms)
    valid_data = ImageFolder(data_dir, transform=valid_tfms)
    test_data = ImageFolder(data_dir, transform=valid_tfms)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    valid_size = 0.15
    test_size = 0.10
    val_split = int(np.floor(valid_size * num_train))
    test_split = int(np.floor(test_size * num_train))
    valid_idx, test_idx, train_idx = indices[:val_split], indices[val_split:val_split+test_split], indices[val_split+test_split:]

    batch_size = 250
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=2, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
        sampler=test_sampler, num_workers=2, pin_memory=True)

    del raw_images, average, standard_dev, raw_dl

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = to_device(ResNet9(3, target_num), device)

    history = [evaluate(model, valid_dl)]
    epochs = 10
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    # Commented out IPython magic to ensure Python compatibility.
    # %%time
    # history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
    #                          grad_clip=grad_clip,
    #                          weight_decay=weight_decay,
    #                          opt_func=opt_func)

    def plot_accuracies(history):
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')

    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')

    def plot_lrs(history):
        lrs = np.concatenate([x.get('lrs', []) for x in history])
        plt.plot(lrs)
        plt.xlabel('Batch no.')
        plt.ylabel('Learning rate')
        plt.title('Learning Rate vs. Batch no.')

    correct = []
    for i in test_idx:
        img, lab = test_data[i]
        xb = to_device(img.unsqueeze(0), device)
        yb = model(xb)
        _, preds = torch.max(yb, dim=1)
        correct.append(preds[0].item() == lab)
    print(f"Accuracy [Test Data]: {sum(correct) / len(test_idx) * 100} %")

    n_rows, n_cols, i = 3, 5, 1
    fig = plt.figure(figsize=(16, 10))
    for index in test_idx[:15]:
        img, label = test_data[index]
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img.permute(1, 2, 0).clamp(0,1))
        ax.set_title(f"Label: {test_data.classes[label]} , Predicted: {predict_image(img, model, device)}")
        i += 1

    img, label = valid_data[19872]
    plt.imshow(img.permute(1, 2, 0))
    print('Label:', valid_data.classes[label], ', Predicted:', predict_image(img, model, device))

    torch.save(model.state_dict(), 'ISN-2-custom-resnet.pth')
    filename = 'ISN-1-custom-resnet.sav'
    joblib.dump(model, filename)
