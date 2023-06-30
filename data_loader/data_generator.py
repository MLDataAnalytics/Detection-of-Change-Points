import numpy as np
import scipy.io as sio


class DataGenerator():
    def __init__(self, config):
        self.config = config
        
        # load data here
        self.input = sio.loadmat(self.config.train_data)

        self.x = self.input["t_x"]
        self.y = self.input["t_y"]
        self.seq_len = self.input["t_len"].flatten()
        self.y_mask = self.input["t_mask"]

        self.N = self.x.shape[0]


    def next_batch(self, batch_size):
        idx = np.random.choice(self.N, batch_size)
        
        yield self.x[idx], self.y[idx], self.seq_len[idx], self.y_mask[idx]
