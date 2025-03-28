import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import re
def show_batch(results, channels=4):
    x = results.cpu().clone().permute((0, 2, 3, 1)).detach().numpy()
    plt.figure(2)
    plt.clf()
    num = results.shape[0]
    if num > 8:
        num = 8
    for i in range(num):
        img = x[i, :, :, 0:channels]
        plt.figure(2)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)


def get_batch(pool, x_prime, batch_size):
    idxs = np.random.randint(0, pool.shape[0], batch_size)
    batch = pool[idxs, :, :, :]
    batch[0:2, :, :, :] = x_prime
    return batch, idxs


def update_pool(pool, results, idxs):
    pool[idxs] = results.clone().detach()
    return pool


def plot_data_with_moving_mean_and_range(data, window_size=100, title="Pre Convolution Mask Training Performance",
                                         num=1, subplots=3, legend= None, color = "r", plot_num =1, ylim= None):
    moving_mean = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    moving_std = np.std([data[i:i + window_size] for i in range(len(data) - window_size + 1)], axis=1)

    x_moving = np.arange(window_size - 1, len(data))

    plt.figure(plot_num,figsize=(10, 6))
    plt.subplot(1, subplots, num)

    plt.plot(x_moving, moving_mean, color=color, linewidth=2, label = legend )

    plt.fill_between(x_moving, moving_mean - moving_std, moving_mean + moving_std, color=color, alpha=0.3,
                     )

    plt.title(title)
    if ylim is not None:
        plt.ylim([3,9])
    plt.ylabel('Log10(loss)')
    plt.xlabel('Training Step')
    plt.legend()
    plt.grid(True)



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_circular_mask(image_size, center_x, center_y, radius):
    height, width = image_size
    y_grid, x_grid = torch.meshgrid(torch.arange(height, device="cuda:0"), torch.arange(width, device="cuda:0"),
                                    indexing='ij')
    distance_squared = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    mask = distance_squared <= radius ** 2
    return mask

def project_sort(x, proj):
  return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

def ot_loss(source, target, proj_n=32):
  ch, n = source.shape[-2:]
  projs = torch.nn.functional.normalize(torch.randn(ch, proj_n, device="cuda:0"), dim=0)
  source_proj = project_sort(source, projs)
  target_proj = project_sort(target, projs)
  target_interp = torch.nn.functional.interpolate(target_proj, n, mode='nearest')
  return (source_proj-target_interp).square().sum()

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def get_image(path,height=50, width=50, padding =0):
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = cv2.resize(base, (int(height), int(width)), interpolation=cv2.INTER_LINEAR)
    base_2 = base / 255
    base_2[..., :3] *= base_2[..., 3:]
    base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).cuda()
    base_torch = torch.nn.functional.pad(base_torch, [padding,padding,padding,padding ])
    base_tt = base_torch.cpu().permute((1, 2, 0)).clone().detach().numpy()
    return base_torch,base_tt


def get_reference_image_and_seed(path, height = 50, width =50, channels =16):
    base = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base = cv2.resize(base, (int(height), int(width)), interpolation=cv2.INTER_LINEAR)
    base_2 = base / 255
    base_2[..., :3] *= base_2[..., 3:]
    base_torch = torch.tensor(base_2, dtype=torch.float32, requires_grad=True).permute((2, 0, 1)).cuda()
    x_prime = torch.zeros((1,channels, height, width), dtype=torch.float32).cuda()
    x_prime[:,:, int(height / 2), int(width / 2)] = 1.0
    return base_torch, x_prime

def to_vue_image(tensor):
    return tensor.cpu().permute((1, 2, 0)).clone().detach().numpy()

def make_gene_pool(gene_location,pool_size = 1333, height = 50, width= 50, channels = 12, device = "cuda:0", gene_size = 3):
    seed = torch.zeros((channels, height, width), device=device)
    for gene_loc in gene_location:
        seed[channels-1
             -gene_loc, height // 2, width // 2] = 1
    seed[3:-gene_size, height // 2, width // 2] = 1
    pool = seed.tile(pool_size,1,1,1)
    return pool

def get_gene_pool(pools, partitions, seeds):
    idxs = []
    pool_tot = []
    for part, pool, seed in zip(partitions, pools, seeds):
        idx = np.random.choice(pool.shape[0], part, replace=False)
        idxs.append(idx)
        p = pool[idx]
        p[0:1] = seed.clone()
        pool_tot.append(p)
    return idxs, torch.cat(pool_tot,dim=0)

def udate_gene_pool(pools,results, idxs, partitions):
    pool_new =[]
    cum_idx = 0
    for pool, idx, part in zip(pools, idxs, partitions):
        pool[idx] = results[cum_idx:part+cum_idx]
        cum_idx+=part
        pool_new.append(pool)
    return pool_new