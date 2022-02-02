import torch
import torch_geometric
import numpy as np
import mlflow
import tqdm

from dataset import HETROGNNC21Dataset
from model import HetroGIN
from torch_geometric.loader import DataLoader
from config import *


K_FOLD = 10


def flatten(list):
    return [item for sublist in list for item in sublist]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mape(y_predict, y_true):
    '''
    Mean Absolute Percentage Error
    '''
    # double check why one of the labels is zero ?
    # return np.average(np.abs((y_predict - y_true) / (np.abs(y_true) + 0.0000001))) * 100
    return torch.mean(torch.abs((y_predict - y_true) / (torch.abs(y_true) + 0.0000001))) * 100


def calculate_metrics(y_pred, y_true, epoch, type):
    acc = mape(y_pred, y_true)
    print(f"MAPE-{type}: {acc}")
    mlflow.log_metric(key=f"MAPE-{type}", value=float(acc), step=epoch)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    mapes = []
    running_loss = 0.0
    step = 0

    for _, batch in tqdm.tqdm(enumerate(train_loader)):
        # Use GPU if available
        batch.to(DEVICE)

        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        pred = model(batch.x_dict, batch.edge_index_dict)

        # Calculating the loss and gradients
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float).to(DEVICE)
        loss_value = loss_fn(torch.squeeze(pred), label)

        """
        Root mean squared error (should we square the error first )?
        However squaring the error will give more weight to larger errors than smaller ones,
        skewing the error estimate towards the odd outliers, Do we want this ?
        Currently I'm keeping the mean absolute error however I am taking the square root of
        the value does this make sense ? or should I remove the square root ?
        """
        loss = torch.sqrt(loss_value)
        loss.backward()
        optimizer.step()

        # Update tracking
        running_loss += loss_value.item()
        step += 1
        # all_preds.append(pred.cpu().detach().numpy())
        # all_labels.append(label.cpu().detach().numpy())
        mapes.append(loss_value.cpu().detach().numpy())
        # break # TODO

    # all_preds = np.concatenate(all_preds).ravel()
    # all_labels = np.concatenate(all_labels).ravel()
    # calculate_metrics(all_preds, all_labels, epoch, "train")
    print(f"MAPE-train: {np.average(mapes)}")
    mlflow.log_metric(key=f"MAPE-train", value=float(np.average(mapes)), step=epoch)
    return running_loss / step


def test(epoch, model, test_loader, loss_fn, mode):
    all_preds = []
    all_labels = []
    mapes = []
    running_loss = 0.0
    step = 0
    for batch in tqdm.tqdm(test_loader):
        batch.to(DEVICE)
        pred = model(batch.x_dict, batch.edge_index_dict)
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float).to(DEVICE)
        loss = loss_fn(torch.squeeze(pred), label)

        # Update tracking
        running_loss += loss.item()
        step += 1
        mapes.append(loss.cpu().detach().numpy())

    print(f"MAPE-{mode}: {np.average(mapes)}")
    mlflow.log_metric(key=f"MAPE-{mode}", value=float(np.average(mapes)), step=epoch)
    return running_loss / step


if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    # Loading the dataset
    print("Loading dataset...")
    train_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-train')
    data = train_dataset[0]

    # k-fold cross validation
    train_score = []
    val_score = []

    total_size = len(train_dataset)
    fraction = 1 / K_FOLD
    seg = int(total_size * fraction)

    for i in range(K_FOLD):
        print(f"### Fold {i+1}: ###")

        # Loading the model
        print("Loading model...")

        # Define a homogeneous GNN model
        input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1], 'node': data['node']['x'].shape[1]}
        model = HetroGIN(input_channels=input_channels, embedding_size=EMBEDDING_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
                         act=ACT, norm=BN, jk=JK_MODE, post_hidden_layer_size=MLP_EMBEDDING, post_num_layers=MLP_LAYERS)
        model = model.to(DEVICE)

        num_params = count_parameters(model)
        print(f"Number of parameters: {num_params}")

        # Training paramerers
        loss_fn = mape
        optimizer = torch.optim.Adam(list(model.parameters()), lr=LEARNING_RATE)

        # Dataset split
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))

        train_set = torch.utils.data.dataset.Subset(train_dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(train_dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=True)

        best_loss = np.inf
        best_train_loss = 0

        for epoch in range(EPOCHS):
            # Training
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch+1} | Train Loss {loss}")

            train_loss = loss

            # Evaluation
            model.eval()
            loss = test(epoch, model, val_loader, loss_fn, "validation")
            print(f"Epoch {epoch+1} | Validation Loss {loss}")

            # Update best loss
            if loss < best_loss:
                best_loss = loss
                best_train_loss = train_loss
                print(f"Best model found at Epoch {epoch +1} ...")

        train_score.append(best_train_loss)
        val_score.append(best_loss)

    print("Train Loss:", train_score)
    print("Val Loss:", val_score)
    print("Mean Train Loss", np.mean(train_score))
    print("Mean Val Loss", np.mean(val_score))
