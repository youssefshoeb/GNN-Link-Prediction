import torch
import torch_geometric
import numpy as np
import mlflow

from dataset import GNNC21Dataset
from model import HetroGIN
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Learning configuration
LEARNING_RATE = 0.01
DECAY_RATE = 0  # 0.5
BATCH_SIZE = 16
VAL_BATCH_SIZE = 4
EPOCHS = 10
# GNN BaseModel
EMBEDDING_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.0
ACT = torch.nn.PReLU()
BN = torch_geometric.nn.norm.BatchNorm(EMBEDDING_SIZE)
JK_MODE = 'cat'
# GNN ReadoutModel
MLP_EMBEDDING = 8
MLP_LAYERS = 2
# Regularization parameters
WEIGHT_DECAY = 0  # 0.01#1e-5


def flatten(list):
    return [item for sublist in list for item in sublist]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def init_params(model, train_loader):
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(DEVICE)
    model(batch.x_dict, batch.edge_index_dict)


def mape(y_predict, y_true):
    '''
    Mean Absolute Percentage Error
    '''
    return np.average(np.abs((y_predict - y_true) / np.abs(y_true))) * 100


def calculate_metrics(y_pred, y_true, epoch, type):
    acc = mape(y_pred, y_true)
    print(f"MAPE: {acc}")
    mlflow.log_metric(key=f"MAPE-{type}", value=float(acc), step=epoch)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for _, batch in enumerate(train_loader):
        # Use GPU if available
        batch.to(DEVICE)

        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        pred = model(batch.x_dict, batch.edge_index_dict)
        # print('Shape', pred['link'].shape)
        # pred = model_readout(pred)
        print(pred.shape)

        # Calculating the loss and gradients
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float)
        print(label.shape)
        loss = loss_fn(torch.squeeze(pred), label)
        loss.backward()
        optimizer.step()

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

        break  # TODO

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step


def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(DEVICE)
        pred = model(batch.x_dict, batch.edge_index_dict)
        # pred = model_readout(pred['path'])
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float)
        loss = loss_fn(torch.squeeze(pred), label)

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())
        break  # TODO

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print("Predictions", all_preds[:10])
    print("Labels", all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return running_loss / step


if __name__ == "__main__":

    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    '''
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, -1), name="x"),
                       TensorSpec(np.dtype(np.int32), (2, -1), name="edge_index")])

    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])
    SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)
    '''
    # Specify tracking server
    mlflow.set_experiment(experiment_name="GNNs")

    # Loading the dataset
    print("Loading dataset...")
    train_dataset = GNNC21Dataset(root='data/', filename='gnnet_data_set_training')
    val_dataset = GNNC21Dataset(root='data/', filename='gnnet_data_set_validation', val=True)
    test_dataset = GNNC21Dataset(root='data/', filename='gnnet_data_set_evaluation_delays', test=True)
    data = train_dataset[0]

    # Prepare training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True)

    # Loading the model
    print("Loading model...")
    # Define a homogeneous GNN model
    # print('data', data['link']['x'].shape[1])
    input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
    model = HetroGIN(input_channels=input_channels, embedding_size=EMBEDDING_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
                     act=ACT, norm=BN, jk=JK_MODE, post_hidden_layer_size=MLP_EMBEDDING, post_num_layers=MLP_LAYERS)
    # Convert a homogeneous GNN model into its heterogeneous equivalent
    # model = to_hetero(model, data.metadata(), aggr='sum')
    model = model.to(DEVICE)

    # Initialize parameters.
    # init_params(model, train_loader) # not needed since we are not using lazy initialization
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")

    with mlflow.start_run(run_name="GIN Run: 1"):
        mlflow.log_param("num_params", num_params)
        mlflow.log_param("embedding_size", EMBEDDING_SIZE)
        mlflow.log_param("num_mp_layers", NUM_LAYERS)
        mlflow.log_param("mlp_embedding_size", MLP_EMBEDDING)
        mlflow.log_param("mlp_layers", MLP_LAYERS)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("act", str(ACT))
        mlflow.log_param("bn", str(BN))
        mlflow.log_param("jk", JK_MODE)

        # Training paramerers
        loss_fn = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(list(model.parameters()) + list(model_readout.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=LEARNING_RATE)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("decay_rate", DECAY_RATE)
        mlflow.log_param("weight_decay", WEIGHT_DECAY)
        mlflow.log_param("epochs", EPOCHS)

        # Start training
        for epoch in range(EPOCHS):
            # Training
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch} | Train Loss {loss}")
            mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

            # Evaluation
            model.eval()
            loss = test(epoch, model, val_loader, loss_fn)
            print(f"Epoch {epoch} | Test Loss {loss}")
            mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

            break  # TODO

            # scheduler.step()

        # Save the currently best model
        # mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
