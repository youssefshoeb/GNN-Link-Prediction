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
import re
import pandas as pd
import os.path as osp
import datanetAPI

from dataset import NEWGNNCH21Dataset, HETROGNNC21Dataset, GNNC21Dataset
from model import HetroGIN, HetroFastIDLineGIN
from torch_geometric.loader import DataLoader

HETERO_CONVERTED_DIRS = {'train': './dataset/converted_train',
                         'val': './dataset/converted_validation',
                         'test': './dataset/converted_test'
                         }

LINE_CONVERTED_DIRS = {'train': './dataset/LINEconverted_train',
                       'val': './dataset/LINEconverted_validation',
                       'test': './dataset/LINEconverted_test'
                       }


RAW_DIRS = {'train': f'./dataset/gnnet-ch21-dataset-{"train"}',
            'val': f'./dataset/gnnet-ch21-dataset-{"validation"}',
            'test': f'./dataset/gnnet-ch21-dataset-{"test"}'}

DEBUG = False


def flatten(list):
    return [item for sublist in list for item in sublist]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(json_config):
    # Ensure deterministic behavior by setting random seed
    torch.backends.cudnn.deterministic = True
    random.seed(json_config['SEED'])
    np.random.seed(json_config['SEED'])
    torch.manual_seed(json_config['SEED'])
    torch.cuda.manual_seed_all(json_config['SEED'])

    torch.backends.cudnn.benchmark = False

    return


def get_optmizer(model, config):
    if config['OPTIMIZER'] == "adam":
        return torch.optim.Adam(list(model.parameters()), lr=config['LEARNING_RATE'])
    else:
        return torch.optim.SGD(list(model.parameters()), lr=config['LEARNING_RATE'])


def mape(y_predict, y_true):
    '''
    Mean Absolute Percentage Error
    '''
    # double check why one of the labels is zero ?
    loss = torch.mean(torch.abs((y_predict - y_true) / (torch.abs(y_true) + 0.000000001))) * 100

    return loss


def save_best_model(model, mode, run=None):
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


def converted_filenames_metadata(filenames, path_to_original_dataset):
    '''
    Divide  validation and test sets into 3
    '''

    def m(f):
        g = re.match("(validation|train|test)\_(\d+)\_(\d+).*", f).groups()
        g = [g[0], int(g[1]), int(g[2])]
        return g

    matches = [m(f) for f in filenames]
    reader = datanetAPI.Datanet21API(path_to_original_dataset)
    files_num = np.array([m[1] for m in matches], dtype=np.int32)
    samples_num = np.array([m[2] for m in matches], dtype=np.int32)

    all_paths = np.array(reader.get_available_files())

    df = pd.DataFrame(index=filenames, columns=['full_path', 'num_nodes', 'validation_setting'])
    df['full_path'] = all_paths[files_num, 0]
    df['sample_num'] = samples_num
    df['file_num'] = files_num

    df['num_nodes'] = np.array([osp.split(f)[-1] for f in df['full_path'].values], dtype=np.int32)

    if matches[0][0] in ['validation', 'test']:
        df['validation_setting'] = np.array([osp.split(f)[-2][-1] for f in df['full_path'].values], dtype=np.int32)
    else:
        df['validation_setting'] = -1

    """
        Put it in correct order
    """
    df = df.sort_values(by=['validation_setting', 'num_nodes', 'file_num', 'sample_num'])
    return df


def load_datasets(name):
    if name == "hetero":

        # Divide Dataset
        ds_val = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['val'], graph_type='hetero', convert_files=False)

        df_val = converted_filenames_metadata(ds_val.filenames, RAW_DIRS['val'])
        df_val['filenames'] = df_val.index.values
        # df_val = df_val.groupby('full_path').head(10) # This makes validation set smaller

        # Initialize Dataset
        train_dataset = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['train'], graph_type='hetero', convert_files=False)
        val_dataset = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['val'], graph_type='hetero', convert_files=False)
        test_dataset = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['val'], graph_type='hetero', convert_files=False)

        which_files = list(df_val[df_val['validation_setting'] == 1]['filenames'].values)
        val_1_dataset = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['val'], graph_type='hetero', convert_files=False, filenames=which_files)

        which_files = list(df_val[df_val['validation_setting'] == 2]['filenames'].values)
        val_2_dataset = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['val'], graph_type='hetero', convert_files=False, filenames=which_files)

        which_files = list(df_val[df_val['validation_setting'] == 3]['filenames'].values)
        val_3_dataset = NEWGNNCH21Dataset(root_dir=HETERO_CONVERTED_DIRS['val'], graph_type='hetero', convert_files=False, filenames=which_files)

        # train_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-train')
        # val_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-validation', val=True)
        # test_dataset = HETROGNNC21Dataset(root='data/GNN-CH21-H/', filename='gnnet-ch21-dataset-test-with-labels', test=True)
    else:

        # Divide Dataset
        ds_val = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['val'], graph_type='line', convert_files=False)

        df_val = converted_filenames_metadata(ds_val.filenames, RAW_DIRS['val'])
        df_val['filenames'] = df_val.index.values
        # df_val = df_val.groupby('full_path').head(10) #This makes validation set smaller

        # Initialize Dataset
        train_dataset = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['train'], graph_type='line', convert_files=False)
        val_dataset = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['val'], graph_type='line', convert_files=False)
        test_dataset = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['val'], graph_type='line', convert_files=False)

        which_files = list(df_val[df_val['validation_setting'] == 1]['filenames'].values)
        val_1_dataset = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['val'], graph_type='line', convert_files=False, filenames=which_files)

        which_files = list(df_val[df_val['validation_setting'] == 2]['filenames'].values)
        val_2_dataset = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['val'], graph_type='line', convert_files=False, filenames=which_files)

        which_files = list(df_val[df_val['validation_setting'] == 3]['filenames'].values)
        val_3_dataset = NEWGNNCH21Dataset(root_dir=LINE_CONVERTED_DIRS['val'], graph_type='line', convert_files=False, filenames=which_files)

        # train_dataset = GNNC21Dataset(root='data/GNN-CH21/', filename='gnnet-ch21-dataset-train')
        # val_dataset = GNNC21Dataset(root='data/GNN-CH21/', filename='gnnet-ch21-dataset-validation', val=True)
        # test_dataset = GNNC21Dataset(root='data/GNN-CH21/', filename='gnnet-ch21-dataset-test-with-labels', test=True)

    return train_dataset, val_1_dataset, val_2_dataset, val_3_dataset, val_dataset, test_dataset


def load_model(data, config):
    # Loading the model
    print("Loading model...")
    if config["DATASET_REP"] == "hetero":
        input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1], 'node': data['node']['x'].shape[1]}
        model = HetroGIN(input_channels=input_channels, embedding_size=config['EMBEDDING_SIZE'], num_layers=config['NUM_LAYERS'], dropout=config['DROPOUT'],
                         act=eval(config['ACT']), norm=eval(config['BN']), jk=config['JK_MODE'],
                         post_hidden_layer_size=config['MLP_EMBEDDING'], post_num_layers=config['MLP_LAYERS'], post_head_act=eval(config['MLP_HEAD_ACT']), concat_path=config['CONCAT_PATH'])
    else:
        input_channels = {'link': data['link']['x'].shape[1], 'path': data['path']['x'].shape[1]}
        model = HetroFastIDLineGIN(input_channels=input_channels, embedding_size=config['EMBEDDING_SIZE'], num_layers=config['NUM_LAYERS'],
                                   dropout=config['DROPOUT'], act=eval(config['ACT']), norm=eval(config['BN']),
                                   jk=config['JK_MODE'], post_hidden_layer_size=config['MLP_EMBEDDING'],
                                   post_num_layers=config['MLP_LAYERS'], post_head_act=eval(config['MLP_HEAD_ACT']), concat_path=config['CONCAT_PATH'])
    return model


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, wandb_run=False, k=None):
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
        # pred = (pred * 10**3).round() / (10**3)

        # Calculating the loss and gradients
        label = batch['path'].y   # torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float).to(DEVICE)
        # label = (label * 10**3).round() / (10**3)

        loss_value = loss_fn(torch.squeeze(pred), label)
        loss = loss_value  # torch.sqrt(loss_value)
        loss.backward()
        optimizer.step()
        # break

        # Update tracking
        running_loss += loss_value.item()
        step += 1
        all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    if DEBUG:
        print('Predictions', all_preds[0:10])
        print('Labels', all_labels[0:10])

    mape_value = mape(torch.from_numpy(all_preds), torch.from_numpy(all_labels))

    average_loss = running_loss / step
    if wandb_run:
        if k is not None:
            print(f"Epoch {epoch+1} | Train Loss {average_loss}")
            wandb.log({f"MAPE-Train -{k}": float(mape_value), "Epoch": epoch + 1})
            wandb.log({f"Train loss {k}": float(average_loss), "Epoch": epoch + 1})
        else:
            print(f"Epoch {epoch+1} | Train Loss {average_loss}")
            wandb.log({f"MAPE-Train": float(mape_value), "Epoch": epoch + 1})
            wandb.log({"Train loss": float(average_loss), "Epoch": epoch + 1})
    else:
        if k is not None:
            print(f"Epoch {epoch+1} | Train Loss {average_loss}")
            mlflow.log_metric(key=f"MAPE-Train-{k}", value=float(mape_value), step=epoch + 1)
            mlflow.log_metric(key=f"Train loss {k}", value=float(average_loss), step=epoch + 1)
        else:
            print(f"Epoch {epoch+1} | Train Loss {average_loss}")
            mlflow.log_metric(key=f"MAPE-Train", value=float(mape_value), step=epoch + 1)
            mlflow.log_metric(key=f"Train loss", value=float(average_loss), step=epoch + 1)

    print(f"MAPE-Train: {mape_value}")

    return


def test(epoch, model, test_loader, loss_fn, mode, wandb_run=False, k=None):
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
        # pred = (pred * 10**3).round() / (10**3)

        # Get label
        label = batch['path'].y  # torch.tensor(np.array(flatten(batch['path'].y)), dtype=torch.float).to(DEVICE)
        # label = (label * 10**3).round() / (10**3)

        # Calculate loss
        loss = loss_fn(torch.squeeze(pred), label)

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    if DEBUG:
        print('Predictions', all_preds[0:10])
        print('Labels', all_labels[0:10])

    mape_value = mape(torch.from_numpy(all_preds), torch.from_numpy(all_labels))

    average_loss = running_loss / step

    if wandb_run:
        if k is not None:
            print(f"Epoch {epoch+1} | Validation Loss {average_loss}")
            wandb.log({f"MAPE-{mode}-{k}": float(mape_value), "Epoch": epoch + 1})
            wandb.log({f"Validation loss {k}": float(average_loss), "Epoch": epoch + 1})
        else:
            print(f"Epoch {epoch+1} | {mode} Loss {average_loss}")
            wandb.log({f"MAPE-{mode}": float(mape_value), "Epoch": epoch + 1})
            wandb.log({f"{mode} loss": float(average_loss), "Epoch": epoch + 1})
    else:
        if k is not None:
            print(f"Epoch {epoch+1} | Validation Loss {average_loss}")
            mlflow.log_metric(key=f"MAPE-{mode}-{k}", value=float(mape_value), step=epoch + 1)
            mlflow.log_metric(key=f"Validation loss {k}", value=float(average_loss), step=epoch + 1)
        else:
            print(f"Epoch {epoch+1} | {mode} Loss {average_loss}")
            mlflow.log_metric(key=f"MAPE-{mode}", value=float(mape_value), step=epoch + 1)
            mlflow.log_metric(key=f"{mode} loss", value=float(average_loss), step=epoch + 1)

    print(f"MAPE-{mode}: {mape_value}")
    return mape_value


def cross_validate(args):
    K_FOLD = args.fold
    if args.wandb:  # K-fold on wandb
        with wandb.init(project=args.project, entity="youssefshoeb", config=json_config) as run:
            # Access all hyperparameters through wandb.config, so logging matches execution
            config = wandb.config

            # Loading the dataset
            print("Loading dataset...")
            train_dataset, _, _, _, _, _ = load_datasets(json_config["DATASET_REP"])
            data = train_dataset[0]

            # Prepare training
            # Intialize Dataloaders
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)

            # k-fold cross validation scores and segment size
            val_score = []

            total_size = len(train_dataset)
            fraction = 1 / K_FOLD
            seg = int(total_size * fraction)

            for i in range(K_FOLD):
                model = load_model(data, config)

                model = model.to(DEVICE)

                num_params = count_parameters(model)
                # print(f"Number of parameters: {num_params}")
                wandb.log({"num_params": num_params})

                # Training paramerers
                loss_fn = eval(config['LOSS'])  # torch.nn.MSELoss()

                optimizer = get_optmizer(model, config)

                # Watch gradients and weights of model
                wandb.watch(model, loss_fn, log="all", log_freq=10)

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
                    train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, wandb_run=True, k=i + 1)

                    # Evaluation on validation set
                    model.eval()
                    loss = test(epoch, model, val_loader, loss_fn, "validation", wandb_run=True, k=i + 1)

                    # Update best loss
                    if loss < best_loss:
                        best_loss = loss

                # Record the best validation error
                wandb.log({f"Best MAPE-validation": best_loss, "Fold": i + 1})
                val_score.append(best_loss)
                # free cache
                torch.cuda.empty_cache()
                del model

            # Record the best validation error
            wandb.log({f"Average Best MAPE-validation": float(np.mean(val_score)), "Epoch": 0})
            print("Mean Val Loss", np.mean(val_score))

    else:  # K-fold cv on MLflow
        # Loading the dataset
        print("Loading dataset...")
        train_dataset, _, _, _, _, _ = load_datasets(json_config["DATASET_REP"])
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

                model = load_model(data, json_config)

                model = model.to(DEVICE)
                if i == 0:
                    num_params = count_parameters(model)
                    # print(f"Number of parameters: {num_params}")
                    mlflow.log_param("num_params", num_params)

                # Training paramerers
                loss_fn = eval(json_config['LOSS'])

                optimizer = get_optmizer(model, json_config)

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
                    train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, k=i + 1)

                    # Evaluation on validation set
                    model.eval()
                    loss = test(epoch, model, val_loader, loss_fn, "validation", k=i + 1)

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


def train(args):
    if args.wandb:  # Wandb training
        with wandb.init(project=args.project, entity="youssefshoeb", config=json_config) as run:
            # Access all hyperparameters through wandb.config, so logging matches execution
            config = wandb.config

            # Prepare training
            print("Loading dataset...")
            train_dataset, val_1_dataset, val_2_dataset, val_3_dataset, val_dataset, _ = load_datasets(json_config["DATASET_REP"])
            data = train_dataset[0]
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            if DEBUG:
                val_1_loader = DataLoader(val_1_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=True)
                val_2_loader = DataLoader(val_2_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=True)
                val_3_loader = DataLoader(val_3_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=True)

            # Loading the model
            model = load_model(data, config)

            model = model.to(DEVICE)

            num_params = count_parameters(model)
            print(f"Number of parameters: {num_params}")
            wandb.log({"num_params": num_params})

            # Training paramerers
            loss_fn = eval(config['LOSS'])  # torch.nn.MSELoss()

            optimizer = get_optmizer(model, config)

            # Watch gradients and weights of model
            wandb.watch(model, loss_fn, log="all", log_freq=10)

            # Start training
            best_loss = np.inf
            for epoch in range(config.EPOCHS):
                # Training
                model.train()
                train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, wandb_run=True)

                model.eval()
                if DEBUG:
                    # Evaluation on validation set 1
                    test(epoch, model, val_1_loader, loss_fn, "Validation_1", wandb_run=True)

                    # Evaluation on validation set 2
                    test(epoch, model, val_2_loader, loss_fn, "Validation_2", wandb_run=True)

                    # Evaluation on validation set 3
                    test(epoch, model, val_3_loader, loss_fn, "Validation_3", wandb_run=True)

                # Evaluation on validation set
                loss = test(epoch, model, val_loader, loss_fn, "Validation", wandb_run=True)

                # Update best loss
                if loss < best_loss:
                    best_loss = loss
                    save_best_model(model, 'wandb', run)

            # Record the best validation error
            wandb.log({f"Best MAPE-Validation": best_loss, "Epoch": 0})

    else:  # Mlflow training
        # Prepare training
        print("Loading dataset...")
        train_dataset, val_1_dataset, val_2_dataset, val_3_dataset, val_dataset, _ = load_datasets(json_config["DATASET_REP"])
        data = train_dataset[0]
        train_loader = DataLoader(train_dataset, batch_size=json_config['BATCH_SIZE'], shuffle=True)
        if DEBUG:
            val_1_loader = DataLoader(val_1_dataset, batch_size=json_config['VAL_BATCH_SIZE'], shuffle=True)
            val_2_loader = DataLoader(val_2_dataset, batch_size=json_config['VAL_BATCH_SIZE'], shuffle=True)
            val_3_loader = DataLoader(val_3_dataset, batch_size=json_config['VAL_BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=json_config['VAL_BATCH_SIZE'], shuffle=True)

        # Loading the model
        model = load_model(data, json_config)

        model = model.to(DEVICE)

        num_params = count_parameters(model)
        print(f"Number of parameters: {num_params}")

        with mlflow.start_run(run_name=args.project):
            mlflow.log_param("num_params", num_params)
            for k, v in json_config.items():
                mlflow.log_param(k, v)

            # Training paramerers
            loss_fn = eval(json_config['LOSS'])

            optimizer = get_optmizer(model, json_config)

            # Start training
            best_loss = np.inf
            for epoch in range(json_config['EPOCHS']):
                # Training
                model.train()
                train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)

                model.eval()
                if DEBUG:
                    # Evaluation on validation set 1
                    test(epoch, model, val_1_loader, loss_fn, "Validation_1")

                    # Evaluation on validation set 2
                    test(epoch, model, val_2_loader, loss_fn, "Validation_2")

                    # Evaluation on validation set 3
                    test(epoch, model, val_3_loader, loss_fn, "Validation_3")

                # Evaluation on validation set
                loss = test(epoch, model, val_loader, loss_fn, "Validation")

                # Update best loss
                if loss < best_loss:
                    best_loss = loss
                    save_best_model(model, 'mlflow')

            # Record the best validation error
            mlflow.log_param(f"Best MAPE-validation", best_loss)


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

    # Cross validate
    parser.add_argument('--fold', type=int, default=10,
                        help='Number of folds in cross validation (default: 10)')

    args = parser.parse_args()

    if args.project is None:
        raise IOError("Project name required (--project)")

    # load hyperparameters
    with open(args.config) as data_file:
        json_config = json.load(data_file)

    set_random_seed(json_config)

    DEVICE = eval(json_config['DEVICE'])

    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    if args.cv:
        cross_validate(args)
    else:
        train(args)

    torch.cuda.empty_cache()
