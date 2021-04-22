import os
import gzip
import math
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample

class AUGMENT_MNIST():
    """
    ==========
    simulate an augmented dataset based on MNIST by
    putting each digit on a canvas with some offset from the center of the canvas
    ==========
    """
    def __init__(self, frame_size, chunk_size):
        super(AUGMENT_MNIST, self).__init__()
        self.frame_size = frame_size
        self.mnist_size = 28 ## by default
        self.chunk_size = chunk_size ## datasets are dividied into pieces with this number and saved separately

    def load_mnist(self, MNIST_DIR):
        MNIST_PATH = os.path.join(MNIST_DIR, 'train-images-idx3-ubyte.gz')
        if not os.path.exists(MNIST_PATH):
            print('===Downloading MNIST train dataset from \'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\'===')
            if not os.path.exists(MNIST_DIR):
                os.makedirs(MNIST_DIR)
            r = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
            open(MNIST_PATH,'wb').write(r.content)
            print('===Saved to \'%s\'===' % MNIST_PATH)

        with gzip.open(MNIST_PATH, 'rb') as f:
            mnist = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist = mnist.reshape(-1, 28, 28)
        return mnist

    def sim_positions(self, save_flag=False):
        positions = Uniform(-1, 1).sample((self.num_digits, 2))
    
        if save_flag:
            np.save('pos', positions.data.numpy())
        return positions

    def sim_one_multimnist(self, mnist, mnist_index):
        '''
        Put each digit in one canvas to create one image.
        '''
        s_factor = self.frame_size / self.mnist_size
        t_factor = (self.frame_size - self.mnist_size) / self.mnist_size
        multimnist = []
        positions = self.sim_positions()
        templates = torch.from_numpy(mnist[mnist_index] / 255.0).float()
        S = torch.Tensor([[s_factor, 0], [0, s_factor]]).repeat(self.num_digits, 1, 1)
        Thetas = torch.cat((S, positions.unsqueeze(-1) * t_factor), -1) # K * 2 * 3
        grid = affine_grid(Thetas, torch.Size((self.num_digits, 1, self.frame_size, self.frame_size)))
        canvas = grid_sample(templates.unsqueeze(1), grid, mode='nearest')
        canvas = canvas.squeeze(1).sum(0).clamp(min=0.0, max=1.0)
        return canvas

    def sim_save_data(self, num_canvases, MNIST_DIR, PATH):
        """
        ==========
        it saves data:
        if num_canvases <= N, then one round of indexing is enough
        if num_canvases > N, then more than one round is needed
        ==========
        """
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        mnist = self.load_mnist(MNIST_DIR=MNIST_DIR)
        N = mnist.shape[0]
        assert num_canvases > 0, 'number of canvases must be a positive number'
        assert isinstance(num_canvases, int)
        consecutive = torch.arange(N).repeat(1, self.num_digits).squeeze(0)
        mnist_indices = consecutive[torch.randperm(N*self.num_digits)].view(N, self.num_digits) ## generate random indices
        num_canvases_left = num_canvases
        print('Start to simulate augmented images...')
        counter = 1
        while(num_canvases_left > 0):
            time_start = time.time()
            num_current_iteration = min(self.chunk_size, num_canvases_left)
            multimnists = []
            if mnist_indices.shape[0] < num_current_iteration: ## more indices are needed for this round
                new_indices = consecutive[torch.randperm(N*self.num_digits)].view(N, self.num_digits)
                mnist_indices = torch.cat((mnist_indices, new_indices), 0)
            for i in range(num_current_iteration):
                canvas = self.sim_one_multimnist(mnist=mnist, mnist_index=mnist_indices[i])
                multimnists.append(canvas.unsqueeze(0))
            mnist_indices = mnist_indices[num_current_iteration:]
            multimnists = torch.cat(multimnists, 0)
            assert multimnists.shape == (num_current_iteration, self.frame_size, self.frame_size), "ERROR! unexpected chunk shape."
            incremental_PATH = PATH + 'ob-%d' % counter
            np.save(incremental_PATH, multimnists)
            counter += 1
            num_canvases_left = max(num_canvases_left - num_current_iteration, 0)
            time_end = time.time()
            print('(%ds) Simulated %d multi-mnist images, saved to \'%s\', %d images left.' % ((time_end - time_start), num_current_iteration, incremental_PATH, num_canvases_left))

    def viz_data(self, MNIST_DIR, num_canvases=50, fs=15):
        mnist = self.load_mnist(MNIST_DIR=MNIST_DIR)
        N = mnist.shape[0]
        mnist_indices = torch.arange(N).repeat(1, self.num_digits).squeeze(0)
        mnist_indices = mnist_indices[torch.randperm(N*self.num_digits)].view(N, self.num_digits)
        num_cols = 10
        num_rows = int(num_canvases / num_cols)
        if (num_cols * num_rows) < num_canvases:
            num_rows += 1
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs, fs * num_rows / num_cols))
        for i in range(num_canvases):
            canvas = self.sim_one_multimnist(mnist, mnist_indices[i])
            ax = fig.add_subplot(gs[int(i/num_cols), int(i%num_cols)])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(canvas, cmap='gray', vmin=0.0, vmax=1.0)
