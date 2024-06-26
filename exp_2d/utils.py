import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=np.nan) -> Deprecated
import torch
import os
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment as linear_assignment
# Get Python Optimal Transport (make sure you run pip install pot on the command line before running this code)
import ot
# from sklearn.utils.linear_assignment_ import linear_assignment -> Deprecated
DISPLAY_NUM = 150

def solve_assignment(z, r, cost, batch_size):
    rep_r = r.repeat(batch_size, 1)
    rep_z = z.unsqueeze(1).repeat(1, batch_size, 1).view(batch_size**2, 2)
    c = cost(rep_r, rep_z).view(batch_size, batch_size)
    indices = linear_assignment(to_data(c))
    # Convert indices to numpy array
    indices = np.array(indices)
    approx_tz = r[indices[1, :]]
    return approx_tz

def visualize_iter(images, dir, step, config, data_range=(-2, 2)):
    """ visualization for 2D experiment in separate images """
    """ here we assume the following coding and colors:
    X  :  real data  (green color)
    Y  :  fake data  (red color)
    ZY :  noise data (magenta color)
    """
    x, y = to_data(images['X']), to_data(images['Y'])
    fx, fy = to_data(images['TX']), to_data(images['TY'])
    fig, ax = plt.subplots()
    scatter_ax(ax, x=x, y=y, fx=fx, c_x='g', c_y='r', c_l='k', data_range=data_range)
    plt.savefig(os.path.join(dir, 'tx_%06d.png' % (step)), bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots()
    scatter_ax(ax, x=y, y=x, fx=fy, c_x='r', c_y='g', c_l='k', data_range=data_range)
    plt.savefig(os.path.join(dir, 'ty_%06d.png' % (step)), bbox_inches='tight')
    plt.clf()

    if config.gen:
        z = to_data(images['ZY'])
        fig, ax = plt.subplots()
        scatter_ax(ax, x=z, y=x, fx=y, c_x='m', c_y='g', c_l='0.5', data_range=data_range)
        plt.savefig(os.path.join(dir, 'gz_%06d.png' % (step)), bbox_inches='tight')
        plt.clf()
    return

def visualize_single(x, y, fx, path, data_range=(-2, 2)):
    fig, ax = plt.subplots()
    scatter_ax(ax, x=x, y=y, fx=fx, c_x='g', c_y='r', c_l='k', data_range=data_range)
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def scatter_ax(ax, x, y, fx, c_x, c_y, c_l, data_range):
    data_min = data_range[0]
    data_max = data_range[1]
    ax.scatter(x[:, 0], x[:, 1], s=1, c=c_x)
    ax.scatter(y[:, 0], y[:, 1], s=1, c=c_y)
    if fx is not None:
        for i in range(DISPLAY_NUM):
          # Debugging purposes
          #print(x)
          #print(fx)
          ax.arrow(x[i, 0], x[i, 1], fx[i, 0]-x[i, 0], fx[i, 1]-x[i, 1],
                    head_width=0.03, head_length=0.05, fc=c_l, ec=c_l)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)
    
def print_opts(config):
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

def print_out(losses, curr_iter, total_iter, tbx_writer=None):
    msg = 'Step [%d/%d], ' % (curr_iter, total_iter)
    for k, v in losses.items():
        msg += '%s: %.4f ' % (k, v)
        if tbx_writer is not None:
            tbx_writer.add_scalar(k, v, curr_iter)
    print(msg)

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
  
def shuffle(z, r):
  """Takes in inputs z and r and shuffles the order in z such that z[i] is
  closest to r[i], as per squared euclidean distance. """
  # Get optimal transport matrix, rows correspond to different r[i] while columns to z[i]
  #z = z.detach().numpy() # First convert into np arrays
  #r = r.detach().numpy()
  z = to_data(z)
  r = to_data(r)
  M = ot.dist(r,z) # Get OT matrix
  row_ind, col_ind = linear_assignment(M) # Solve for corresponding indices
  z = z[col_ind] # Shuffle z matrix
  z = torch.tensor(z, dtype = torch.float32)
  r = torch.tensor(r, dtype = torch.float32) # Convert back to tensor
  if torch.cuda.is_available():
    z = z.cuda()
    r = r.cuda()
  return z
