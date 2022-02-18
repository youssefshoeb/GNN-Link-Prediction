import os
import argparse
import json
import torch
import torch_geometric
import numpy as np
import mlflow
import tqdm
import random
import wandb

from dataset import HETROGNNC21Dataset, GNNC21Dataset
from model import HetroGIN, HetroFastIDLineGIN
from torch_geometric.loader import DataLoader


def flatten(list):
    return [item for sublist in list for item in sublist]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mape(y_predict, y_true):
    '''
    Mean Absolute Percentage Error
    '''
    # double check why one of the labels is zero ?
    loss = torch.mean(torch.abs((y_predict - y_true) / (torch.abs(y_true) + 0.000000001))) * 100

    return loss


def save_best_model(mode):
    print("Saving new best model ...")
    model = model.to("cpu")

    if mode == 'wandb':
        if not os.path.exists(f'runs/{run.name}'):
            os.mkdir(f'runs/{run.name}')
        torch.save(model.state_dict(), f'runs/{run.name}/best_model.pth')
        model = model.to(DEVICE)
    else:
        mlflow.pytorch.log_model(model, "best_model")
        model = model.to(DEVICE)


def load_datasets(name):
    if name == "hetero":
        train_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-train')
        val_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-validation', val=True)
        test_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-test-with-labels', test=True)
    else:
        train_dataset = GNNC21Dataset(root='data/GNN-CH21/', filename='gnnet-ch21-dataset-train')
        val_dataset = GNNC21Dataset(root='data/GNN-CH21/', filename='gnnet-ch21-dataset-validation', val=True)
        test_dataset = GNNC21Dataset(root='data/GNN-CH21/', filename='gnnet-ch21-dataset-test-with-labels', test=True)

    return train_dataset, val_dataset, test_dataset


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, wandb_run=False, k=1):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    # Enumerate over the data
    for batch in tqdm.tqdm(train_loader):
        # Use GPU if available
        batch.to(DEVICE)

        # Reset gradients
        optimizer.zero_grad()

        # get model output
        pred = model(batch.x_dict, batch.edge_index_dict)

        # Calculating the loss and gradients
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float).to(DEVICE)
        loss_value = loss_fn(torch.squeeze(pred), label)
        loss = torch.sqrt(loss_value)
        loss.backward()
        optimizer.step()

        # Update tracking
        running_loss += loss_value.item()
        step += 1
        all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    mape_value = mape(torch.from_numpy(all_preds), torch.from_numpy(all_labels))
    print(f"MAPE-train: {mape_value}")

    if wandb_run:
        wandb.log({f"MAPE-train -{k}": float(mape_value), "Epoch": epoch + 1})
    else:
        mlflow.log_metric(key=f"MAPE-train-{k}", value=float(mape_value), step=epoch + 1)

    return running_loss / step


def test(epoch, model, test_loader, loss_fn, mode, wandb_run=False, k=1):
    all_preds = []
    all_labels = []

    running_loss = 0.0
    step = 0

    # Enumerate over the data
    for batch in tqdm.tqdm(test_loader):
        # Use GPU if available
        batch.to(DEVICE)

        # Get model output
        pred = model(batch.x_dict, batch.edge_index_dict)

        # Get label
        label = torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float).to(DEVICE)

        # Calculate loss
        loss = loss_fn(torch.squeeze(pred), label)

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    mape_value = mape(torch.from_numpy(all_preds), torch.from_numpy(all_labels))
    print(f"MAPE-{mode}: {mape_value}")

    if wandb_run:
        wandb.log({f"MAPE-{mode}-{k}": float(mape_value), "Epoch": epoch + 1})
    else:
        mlflow.log_metric(key=f"MAPE-{mode}-{k}", value=float(mape_value), step=epoch + 1)

    return running_loss / step


if __name__ == "__main__":

    # instantiate parser
    parser = argparse.ArgumentParser(description='Required files for training.')

    # Required hyperparameters path argument
    parser.add_argument('config', type=str,
                        help='Config json file containing hyperparameters')

    # Switch
    parser.add_argument('--wandb', action='store_true',
                        help='Whether Wandb is used or not (default is False)')

    # Switch
    parser.add_argument('--cv', action='store_true',
                        help='Whether cross validation is performed (default: False)')

    # Optional argument
    parser.add_argument('--project', type=str,
                        help='Wandb/MlFlow project name')

    # Cross validate (haykoon bardo 3andaha etenin)
    parser.add_argument('--fold', type=int, default=10,
                        help='Number of folds in cross validation (default: 10)')

    args = parser.parse_args()

    if args.project is None:
        raise IOError("Project name required (--project)")

    # load hyperparameters
    with open(args.config) as data_file:
        json_config = json.load(data_file)

    # Ensure deterministic behavior by setting random seed
    torch.backends.cudnn.deterministic = True
    random.seed(json_config['SEED'])
    np.random.seed(json_config['SEED'])
    torch.manual_seed(json_config['SEED'])
    torch.cuda.manual_seed_all(json_config['SEED'])

    torch.backends.cudnn.benchmark = False

    DEVICE = eval(json_config['DEVICE'])

    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    if args.cv:
        K_FOLD = args.fold
        if args.wandb:  # K-fold on wandb
            with wandb.init(project=args.project, entity="youssefshoeb", config=json_config) as run:
                # Access all hyperparameters through wandb.config, so logging matches execution
                config = wandb.config

                # Loading the dataset
                print("Loading dataset...")
                train_dataset, _, _ = load_datasets(json_config["DATASET_REP"])
                data = train_dataset[0]

                # Prepare training
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

                # k-fold cross validation scores and segment size
                val_score = []

                total_size = len(train_dataset)
                fraction = 1 / K_FOLD
                seg = int(total_size * fraction)

                # Loading the model
                print("Loading model...")
                if config["DATASET_REP"] == "hetero":
                    input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1], 'node': data['node']['x'].shape[1]}
                    model = HetroGIN(input_channels=input_channels, embedding_size=config.EMBEDDING_SIZE, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
                                     act=eval(config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(config.EMBEDDING_SIZE), jk=config.JK_MODE,
                                     post_hidden_layer_size=config.MLP_EMBEDDING, post_num_layers=config.MLP_LAYERS, post_head_act=eval(config.MLP_HEAD_ACT), concat_path=config.CONCAT_PATH)
                else:
                    input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
                    model = HetroFastIDLineGIN(input_channels=input_channels, embedding_size=config.EMBEDDING_SIZE, num_layers=config.NUM_LAYERS,
                                               dropout=config.DROPOUT, act=eval(config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(config.EMBEDDING_SIZE),
                                               jk=config.JK_MODE, post_hidden_layer_size=config.MLP_EMBEDDING,
                                               post_num_layers=config.MLP_LAYERS, post_head_act=eval(config.MLP_HEAD_ACT), concat_path=config.CONCAT_PATH)

                model = model.to(DEVICE)

                num_params = count_parameters(model)
                # print(f"Number of parameters: {num_params}")
                wandb.log({"num_params": num_params})

                # Training paramerers
                loss_fn = eval(config['LOSS'])  # torch.nn.MSELoss()

                if config.OPTIMIZER == "adam":
                    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.LEARNING_RATE)
                else:
                    optimizer = torch.optim.SGD(list(model.parameters()), lr=config.LEARNING_RATE)

                # Watch gradients and weights of model
                wandb.watch(model, loss_fn, log="all", log_freq=10)

                for i in range(K_FOLD):
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

                    # Prepare training
                    train_loader = DataLoader(train_set, batch_size=json_config['BATCH_SIZE'], shuffle=True)
                    val_loader = DataLoader(val_set, batch_size=json_config['BATCH_SIZE'], shuffle=True)

                    # Start training
                    best_loss = np.inf
                    for epoch in range(config.EPOCHS):
                        # Training
                        model.train()
                        loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, wandb_run=True, k=i + 1)
                        print(f"Epoch {epoch+1} | Train Loss {loss}")
                        wandb.log({f"Train loss {i + 1}": float(loss), "Epoch": epoch + 1})

                        # Evaluation on validation set
                        model.eval()
                        loss = test(epoch, model, val_loader, loss_fn, "validation", wandb_run=True, k=i + 1)
                        print(f"Epoch {epoch+1} | Validation Loss {loss}")
                        wandb.log({f"Validation loss {i + 1}": float(loss), "Epoch": epoch + 1})

                        # Update best loss
                        if loss < best_loss:
                            best_loss = loss

                    # Record the best validation error
                    wandb.log({f"Best MAPE-validation": best_loss, "Fold": i + 1})
                    val_score.append(best_loss)
                    torch.cuda.empty_cache()

                # Record the best validation error
                wandb.log({f"Average Best MAPE-validation": float(np.mean(val_score)), "Epoch": 0})
                print("Mean Val Loss", np.mean(val_score))

        else:  # K-fold cv on MLflow
            # Loading the dataset
            print("Loading dataset...")
            train_dataset, _, _ = load_datasets(json_config["DATASET_REP"])
            data = train_dataset[0]

            # k-fold cross validation scores and segment size
            val_score = []

            total_size = len(train_dataset)
            fraction = 1 / K_FOLD
            seg = int(total_size * fraction)

            with mlflow.start_run(run_name=args.project):
                mlflow.log_param("num_folds", K_FOLD)
                for k, v in json_config.items():
                    mlflow.log_param(k, v)

                for i in range(K_FOLD):
                    print(f"### Fold {i+1}: ###")

                    # Loading the model
                    print("Loading model...")
                    if json_config["DATASET_REP"] == "hetero":
                        input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1], 'node': data['node']['x'].shape[1]}
                        model = HetroGIN(input_channels=input_channels, embedding_size=json_config['EMBEDDING_SIZE'], num_layers=json_config['NUM_LAYERS'], dropout=json_config['DROPOUT'],
                                         act=eval(json_config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(json_config['EMBEDDING_SIZE']), jk=json_config['JK_MODE'],
                                         post_hidden_layer_size=json_config['MLP_EMBEDDING'], post_num_layers=json_config['MLP_LAYERS'], post_head_act=eval(json_config['MLP_HEAD_ACT'], concat_path=json_config['CONCAT_PATH']))
                    else:
                        input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
                        model = HetroFastIDLineGIN(input_channels=input_channels, embedding_size=json_config['EMBEDDING_SIZE'], num_layers=json_config['NUM_LAYERS'],
                                                   dropout=json_config['DROPOUT'], act=eval(json_config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(json_config['EMBEDDING_SIZE']),
                                                   jk=json_config['JK_MODE'], post_hidden_layer_size=json_config['MLP_EMBEDDING'],
                                                   post_num_layers=json_config['MLP_LAYERS'], post_head_act=eval(json_config['MLP_HEAD_ACT']), concat_path=json_config['CONCAT_PATH'])

                    model = model.to(DEVICE)
                    if i == 0:
                        num_params = count_parameters(model)
                        # print(f"Number of parameters: {num_params}")
                        mlflow.log_param("num_params", num_params)

                    # Training paramerers
                    loss_fn = eval(json_config['LOSS'])

                    if json_config['OPTIMIZER'] == "adam":
                        optimizer = torch.optim.Adam(list(model.parameters()), lr=json_config['LEARNING_RATE'])
                    else:
                        optimizer = torch.optim.SGD(list(model.parameters()), lr=json_config['LEARNING_RATE'])

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

                    # Prepare training
                    train_loader = DataLoader(train_set, batch_size=json_config['BATCH_SIZE'], shuffle=True)
                    val_loader = DataLoader(val_set, batch_size=json_config['BATCH_SIZE'], shuffle=True)

                    best_loss = np.inf
                    for epoch in range(json_config['EPOCHS']):
                        # Training
                        model.train()
                        loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, k=i + 1)
                        print(f"Epoch {epoch+1} | Train Loss {loss}")
                        mlflow.log_metric(key=f"Train loss {i + 1}", value=float(loss), step=epoch + 1)

                        # Evaluation on validation set
                        model.eval()
                        loss = test(epoch, model, val_loader, loss_fn, "validation", k=i + 1)
                        print(f"Epoch {epoch+1} | Validation Loss {loss}")
                        mlflow.log_metric(key=f"Validation loss {i + 1}", value=float(loss), step=epoch + 1)

                        # Update best loss
                        if loss < best_loss:
                            best_loss = loss

                    # Record the best validation error
                    mlflow.log_metric(f"Best MAPE-validation", value=float(best_loss), step=i + 1)
                    val_score.append(best_loss)
                    torch.cuda.empty_cache()

                # Record the best validation error
                mlflow.log_param(f"Average Best MAPE-validation", float(np.mean(val_score)))
                print("Mean Val Loss", np.mean(val_score))

    else:
        if args.wandb:  # Wandb training
            with wandb.init(project=args.project, entity="youssefshoeb", config=json_config) as run:
                # Access all hyperparameters through wandb.config, so logging matches execution
                config = wandb.config

                # Prepare training
                print("Loading dataset...")
                train_dataset, val_dataset, _ = load_datasets(json_config["DATASET_REP"])
                data = train_dataset[0]
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=True)

                # Loading the model
                print("Loading model...")
                if config["DATASET_REP"] == "hetero":
                    input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1], 'node': data['node']['x'].shape[1]}
                    model = HetroGIN(input_channels=input_channels, embedding_size=config.EMBEDDING_SIZE, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT,
                                     act=eval(config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(config.EMBEDDING_SIZE), jk=config.JK_MODE,
                                     post_hidden_layer_size=config.MLP_EMBEDDING, post_num_layers=config.MLP_LAYERS, post_head_act=eval(config.MLP_HEAD_ACT), concat_path=config.CONCAT_PATH)
                else:
                    input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
                    model = HetroFastIDLineGIN(input_channels=input_channels, embedding_size=config.EMBEDDING_SIZE, num_layers=config.NUM_LAYERS,
                                               dropout=config.DROPOUT, act=eval(config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(config.EMBEDDING_SIZE),
                                               jk=config.JK_MODE, post_hidden_layer_size=config.MLP_EMBEDDING,
                                               post_num_layers=config.MLP_LAYERS, post_head_act=eval(config.MLP_HEAD_ACT))

                model = model.to(DEVICE)

                num_params = count_parameters(model)
                print(f"Number of parameters: {num_params}")
                wandb.log({"num_params": num_params})

                # Training paramerers
                loss_fn = eval(config['LOSS'])  # torch.nn.MSELoss()

                if config.OPTIMIZER == "adam":
                    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.LEARNING_RATE)
                else:
                    optimizer = torch.optim.SGD(list(model.parameters()), lr=config.LEARNING_RATE)

                # Watch gradients and weights of model
                wandb.watch(model, loss_fn, log="all", log_freq=10)

                # Start training
                best_loss = np.inf
                for epoch in range(config.EPOCHS):
                    # Training
                    model.train()
                    loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, wandb_run=True)
                    print(f"Epoch {epoch+1} | Train Loss {loss}")
                    wandb.log({"Train loss": float(loss), "Epoch": epoch + 1})

                    # Evaluation on validation set
                    model.eval()
                    loss = test(epoch, model, val_loader, loss_fn, "validation", wandb_run=True)
                    print(f"Epoch {epoch+1} | Validation Loss {loss}")
                    wandb.log({"Validation loss": float(loss), "Epoch": epoch + 1})

                    # Update best loss
                    if loss < best_loss:
                        best_loss = loss
                        save_best_model('wandb')

                # Record the best validation error
                wandb.log({f"Best MAPE-validation": best_loss, "Epoch": 0})

        else:  # Mlflow training
            # Prepare training
            print("Loading dataset...")
            train_dataset, val_dataset, _ = load_datasets(json_config["DATASET_REP"])
            data = train_dataset[0]
            train_loader = DataLoader(train_dataset, batch_size=json_config['BATCH_SIZE'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=json_config['VAL_BATCH_SIZE'], shuffle=True)

            # Loading the model
            print("Loading model...")
            if json_config["DATASET_REP"] == "hetero":
                input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1], 'node': data['node']['x'].shape[1]}
                model = HetroGIN(input_channels=input_channels, embedding_size=json_config['EMBEDDING_SIZE'], num_layers=json_config['NUM_LAYERS'], dropout=json_config['DROPOUT'],
                                 act=eval(json_config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(json_config['EMBEDDING_SIZE']), jk=json_config['JK_MODE'],
                                 post_hidden_layer_size=json_config['MLP_EMBEDDING'], post_num_layers=json_config['MLP_LAYERS'], post_head_act=eval(json_config['MLP_HEAD_ACT'], concat_path=json_config['CONCAT_PATH']))
            else:
                input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
                model = HetroFastIDLineGIN(input_channels=input_channels, embedding_size=json_config['EMBEDDING_SIZE'], num_layers=json_config['NUM_LAYERS'],
                                           dropout=json_config['DROPOUT'], act=eval(json_config['ACT']), norm=torch_geometric.nn.norm.BatchNorm(json_config['EMBEDDING_SIZE']),
                                           jk=json_config['JK_MODE'], post_hidden_layer_size=json_config['MLP_EMBEDDING'],
                                           post_num_layers=json_config['MLP_LAYERS'], post_head_act=eval(json_config['MLP_HEAD_ACT']), concat_path=json_config['CONCAT_PATH'])

            model = model.to(DEVICE)

            num_params = count_parameters(model)
            print(f"Number of parameters: {num_params}")

            with mlflow.start_run(run_name=args.project):
                mlflow.log_param("num_params", num_params)
                for k, v in json_config.items():
                    mlflow.log_param(k, v)

                # Training paramerers
                loss_fn = eval(json_config['LOSS'])

                if json_config['OPTIMIZER'] == "adam":
                    optimizer = torch.optim.Adam(list(model.parameters()), lr=json_config['LEARNING_RATE'])
                else:
                    optimizer = torch.optim.SGD(list(model.parameters()), lr=json_config['LEARNING_RATE'])

                # Start training
                best_loss = np.inf
                for epoch in range(json_config['EPOCHS']):
                    # Training
                    model.train()
                    loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                    print(f"Epoch {epoch+1} | Train Loss {loss}")
                    mlflow.log_metric(key=f"Train loss", value=float(loss), step=epoch + 1)

                    # Evaluation on validation set
                    model.eval()
                    loss = test(epoch, model, val_loader, loss_fn, "validation")
                    print(f"Epoch {epoch+1} | Validation Loss {loss}")
                    mlflow.log_metric(key=f"Validation loss", value=float(loss), step=epoch + 1)

                    # Update best loss
                    if loss < best_loss:
                        best_loss = loss
                        save_best_model('mlflow')

                # Record the best validation error
                mlflow.log_param(f"Best MAPE-validation", best_loss)

    torch.cuda.empty_cache()
