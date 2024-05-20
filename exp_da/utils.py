import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=np.nan) -> Deprecated
import torch
import os
import torchvision.utils as vutils
from data_loader import get_data
from data_loader import get_mnist_loader, get_usps_loader
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

def print_out(losses, curr_iter, total_iter, tbx_writer=None):
    msg = 'Step [%d/%d], ' % (curr_iter, total_iter)
    for k, v in losses.items():
        msg += '%s: %.4f ' % (k, v)
        if tbx_writer is not None:
            tbx_writer.add_scalar(k, v, curr_iter)
    print(msg)

def print_opts(config):
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

def print_accuracy(config, model, disp=True):
    classifier = KNeighborsClassifier(n_neighbors=1)
    X, y = get_data(config)
    X, y = to_var(X), to_var(y)
    if model is not None:
        X = model.g(X)
    X, y = to_data(X), to_data(y)
    classifier.fit(X.reshape(X.shape[0], -1), y.reshape(-1))

    Xhat, yhat = get_data(config, train=False)
    Xhat, yhat = to_var(Xhat), to_var(yhat)
    Xhat, yhat = to_data(Xhat), to_data(yhat)
    pred = classifier.predict(Xhat.reshape(Xhat.shape[0], -1))
    accuracy = (pred == yhat.reshape(-1)).astype(float).mean()
    if disp:
      print('Classification accuracy: %0.4f' % (accuracy))
    else:
      return accuracy

def visualize_iter(images, dir, step, config, data_range=(-1, 1)):
    for k, image in images.items():
        vutils.save_image(image.cpu().data, os.path.join(dir, '%s_%06d.png' % (k, step)), normalize=True, value_range=data_range, nrow=int(np.sqrt(config.batch_size)))

def visualize_single(image, path, config, data_range=(-1, 1)):
    vutils.save_image(image.cpu().data, path, normalize=True, value_range=data_range, nrow=int(np.sqrt(config.batch_size)))

def print_networks(networks):
    for name, net in networks.items():
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %d' % (name, num_params))
        print('-----------------------------------------------')

def save_networks(networks, model_dir):
    for k, v in networks.items():
        torch.save(v.state_dict(), os.path.join(model_dir, ('%s.pkl' % k)))

def unsqueeze(tensor, ndim=2):
    for it in range(ndim-1):
        tensor = tensor.unsqueeze(1)
    return tensor

def get_ones(size):
    ones = torch.ones(size)
    if torch.cuda.is_available():
        ones = ones.cuda()
    return ones

def to_var(x, requires_grad=False):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

# Train SimpleCNN 
def trainSimpleCNN(model, config):
  """Train a CNN to be a classifier. model is the simple CNN."""
  if config.direction == 'usps-mnist':
          train_loader = get_mnist_loader(config, batch_size=2007, train=False)
  elif config.direction == 'mnist-usps':
          train_loader = get_usps_loader(config, batch_size=10000, train=False)
  num_epochs = config.cnn_epochs
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

  # Save the trained model
  torch.save(model.state_dict(), 'simple_cnn.pth')
