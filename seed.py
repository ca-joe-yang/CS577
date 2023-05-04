import numpy as np
import random
import torch

def set_random_seed(seed):
    # set randoom seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # pl.utilities.seed.seed_everything(seed)

