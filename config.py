import torch
import torch_geometric

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Learning configuration
LEARNING_RATE = 0.001
DECAY_RATE = 0
BATCH_SIZE = 16
VAL_BATCH_SIZE = 2
EPOCHS = 10
# GNN BaseModel
EMBEDDING_SIZE = 8
NUM_LAYERS = 2
DROPOUT = 0.0
ACT = torch.nn.PReLU()
BN = torch_geometric.nn.norm.BatchNorm(EMBEDDING_SIZE)
JK_MODE = 'cat'
# GNN ReadoutModel
MLP_EMBEDDING = 8
MLP_LAYERS = 2
# Regularization parameters
WEIGHT_DECAY = 0
SEED = 24
