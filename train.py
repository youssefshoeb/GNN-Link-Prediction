import torch
import torch_geometric
import numpy as np
import mlflow

from dataset import GNNC21Dataset
from model import HetroGIN
from torch_geometric.loader import DataLoader
from config import *


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
    print(f"MAPE-{type}: {acc}")
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

        # Calculating the loss and gradients
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float)
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


def test(epoch, model, test_loader, loss_fn, mode):
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
    calculate_metrics(all_preds, all_labels, epoch, mode)
    return running_loss / step


if __name__ == "__main__":

    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

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
    input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
    model = HetroGIN(input_channels=input_channels, embedding_size=EMBEDDING_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
                     act=ACT, norm=BN, jk=JK_MODE, post_hidden_layer_size=MLP_EMBEDDING, post_num_layers=MLP_LAYERS)
    # Convert a homogeneous GNN model into its heterogeneous equivalent
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

        best_loss = np.inf

        # Start training
        for epoch in range(EPOCHS):
            # Training
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch+1} | Train Loss {loss}")
            mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

            # Evaluation on validation set
            model.eval()
            loss = test(epoch, model, val_loader, loss_fn, "validation")
            print(f"Epoch {epoch+1} | Validation Loss {loss}")
            mlflow.log_metric(key="Validation loss", value=float(loss), step=epoch)

            # Update best loss
            if loss < best_loss:
                best_loss = loss
                #  Save the current best model
                print("Saving new best model ...")
                mlflow.pytorch.log_model(model, "best_model")

            break  # TODO

            # scheduler.step()

        # Save the final model
        mlflow.pytorch.log_model(model, "last_model")

        # Test best model
        model_uri = "runs:/{}/best_model".format(mlflow.active_run().info.run_id)
        best_model = mlflow.pytorch.load_model(model_uri)
        loss = test(0, best_model, test_loader, loss_fn, "test")
        print(f"Test Loss {loss}")
