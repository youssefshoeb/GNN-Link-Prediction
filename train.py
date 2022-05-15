import torch
import wandb
import os
import torch_geometric
from tqdm import tqdm
from dataset import initDataset
import numpy as np
from models import HetroGAT,HetroGIN



def mape(preds, actuals):
    return 100.0 * torch.mean(torch.abs((preds - actuals) / actuals))


def train_one_epoch(epoch, loss_func, opt, dataloader, model, k=None):
    running_loss = 0.0
    step = 0

    running_loss_mape = 0.0
    step_mape = 0

    # Enumerate over the data
    total = len(dataloader)
    for sample in tqdm(dataloader, total=total):
        # Train model
        with torch.set_grad_enabled(True):
            sample.cuda()

            # Reset Gradients
            opt.zero_grad()

            # Get Model Output
            out = model(sample.x_dict, sample.edge_index_dict, sample["path"].batch)
            # print(out.shape)

            # Calculate loss and gradients
            label = sample['path'].y.reshape(-1, 1)
            # print(label.shape)
            loss_value = loss_func(out, label)

            loss = torch.sqrt(loss_value)
            loss.backward()
            opt.step()

            # Update Tracking
            running_loss += loss_value
            step += 1

            running_loss_mape += mape(out, label).item() * sample["path"].x.shape[0]  # TODO Double check this
            step_mape += sample["path"].x.shape[0]

    average_loss = running_loss / step
    mape_loss = running_loss_mape / step_mape

    print(f"Epoch {epoch+1} | Train Loss {average_loss}")
    print(f"MAPE-Train: {mape_loss}")
    if k is not None:
        wandb.log({f"MAPE-Train - {k}": float(mape_loss), "Epoch": epoch + 1})
        wandb.log({f"Train loss - {k}": float(average_loss), "Epoch": epoch + 1})
    else:
        wandb.log({f"MAPE-Train": float(mape_loss), "Epoch": epoch + 1})
        wandb.log({"Train loss": float(average_loss), "Epoch": epoch + 1})

    torch.cuda.empty_cache()

    return


def test(epoch, loss_func, dataloader, model, mode, k=None):
    all_preds = []
    all_labels = []

    running_loss = 0.0
    step = 0

    running_loss_mape = 0.0
    step_mape = 0

    with torch.no_grad():
        # Enumerate over the data
        for sample in tqdm(dataloader):
            sample.cuda()
            with torch.set_grad_enabled(False):
                out = model(sample.x_dict, sample.edge_index_dict, sample["path"].batch)

                # Get label
                label = sample['path'].y.reshape(-1, 1)
                loss_value = loss_func(out, label)

                # Update Tracking
                running_loss += loss_value.item()
                step += 1

                running_loss_mape += mape(out, label).item() * sample["path"].x.shape[0]
                step_mape += sample["path"].x.shape[0]
                all_preds.append(out.cpu().detach().numpy())
                all_labels.append(label.cpu().detach().numpy())

        average_loss = running_loss / step
        mape_loss = running_loss_mape / step_mape

        print(f"Epoch {epoch+1} | {mode} Loss {average_loss}")
        print(f"MAPE-{mode}: {mape_loss}")

        if k is not None:
            wandb.log({f"MAPE-{mode}-{k}": float(mape_loss), "Epoch": epoch + 1})
            wandb.log({f"Validation loss-{k}": float(average_loss), "Epoch": epoch + 1})
        else:
            wandb.log({f"MAPE-{mode}": float(mape_loss), "Epoch": epoch + 1})
            wandb.log({f"{mode} loss": float(average_loss), "Epoch": epoch + 1})

        return average_loss


def load_model(config, datasets):
    input_channels = {'link': datasets["train"][0]['link']['x'].shape[1],
                      'path': datasets["train"][0]['path']['x'].shape[1],
                      'node': datasets["train"][0]['node']['x'].shape[1]}
    if config['MODEL'] == "GAT":
        model = HetroGAT(input_channels=input_channels, node_embedding_size=config['NODE_EMBEDDING_SIZE'],
                         message_passing_layers=config['MP_LAYERS'], dropout=config['DROPOUT'], heads=config['HEADS'],
                         concat_path=config['CONCAT_PATH'],bl_features=config["BL_FEATURES"], divided_features=config["DIVIDED_FEATURES"],
                         global_feats=config['GLOBAL_FEATS'], mlp_layers=config['MLP_LAYERS'],
                         act=config['MLP_ACT'], mlp_bn=config['MLP_BN'], mlp_head_act=config['MLP_HEAD_ACT'])

    elif config['MODEL'] == "GIN":
        model = HetroGIN(input_channels=input_channels, node_embedding_size=config['NODE_EMBEDDING_SIZE'],
                         message_passing_layers=config['MP_LAYERS'], dropout=config['DROPOUT'], concat_path=config['CONCAT_PATH'],
                         bl_features=config["BL_FEATURES"],divided_features=config["DIVIDED_FEATURES"],
                         global_feats=config['GLOBAL_FEATS'], mlp_layers=config['MLP_LAYERS'],
                         act=config['MLP_ACT'], mlp_bn=config['MLP_BN'], mlp_head_act=config['MLP_HEAD_ACT'])

    else:
        raise IOError("Model not implemented")

    return model


def load_optmizer(config, model):
    if config['OPTIMIZER'] == 'adam':
        return torch.optim.Adam(lr=config["LEARNING_RATE"], params=model.parameters(), weight_decay=config['WEIGHT_DECAY'])

    if config['OPTIMIZER'] == 'adamW':
        return torch.optim.AdamW(lr=config["LEARNING_RATE"], params=model.parameters(), weight_decay=config['WEIGHT_DECAY'])

    if config['OPTIMIZER'] == 'sgd':
        return torch.optim.SGD(lr=config["LEARNING_RATE"], params=model.parameters(), weight_decay=config['WEIGHT_DECAY'])


def save_best_model(model, run):
    os.makedirs("./runs", exist_ok=True)

    print("Saving new best model ...")
    model = model.to("cpu")

    if not os.path.exists(f'runs/{run.name}'):
        os.mkdir(f'runs/{run.name}')
    torch.save(model.state_dict(), f'runs/{run.name}/best_model.pth')
    model = model.cuda()


def train(config):
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    with wandb.init(project=config['PROJECT_NAME'], entity="youssefshoeb", config=config) as run:
        # Access all hyperparameters through wandb.config, so logging matches execution
        config = wandb.config

        print("Loading Dataset...")
        datasets, dataloaders = initDataset(config)

        print("Loading model...")
        model = load_model(config, datasets)
        model.cuda()

        # Hyperparameters
        num_epochs = config["EPOCHS"]
        loss_func = eval(config["LOSS"])

        opt = load_optmizer(config, model)

        # Start training
        best_loss = np.inf
        for epoch in range(num_epochs):
            model.train()
            # Train
            train_one_epoch(epoch, loss_func, opt, dataloaders['train'], model)

            model.eval()
            # Evaluation on validation set 1
            test(epoch, loss_func, dataloaders['val_1'], model, "Validation_1")

            # Evaluation on validation set 2
            test(epoch, loss_func, dataloaders['val_2'], model, "Validation_2")

            # Evaluation on validation set 3
            test(epoch, loss_func, dataloaders['val_3'], model, "Validation_3")

            # Evaluation on validation set
            loss = test(epoch, loss_func, dataloaders['val'], model, "Validation")

            # Update best loss
            if loss < best_loss:
                best_loss = loss
                save_best_model(model, run)

        # evaluate best model
        evaluate(config, run.name)



def test_baseline(config):
    _, dataloaders = initDataset(config)

    # Train
    all_preds = []
    all_labels = []

    for sample in tqdm(dataloaders['val'], total=len(dataloaders['val'])):
        with torch.set_grad_enabled(False):
            b_out = sample['path'].x[:, 3]

            all_preds.append(torch.squeeze(b_out).cpu().detach().numpy())
            all_labels.append(sample['path'].y.cpu().detach().numpy())

    all_preds = torch.from_numpy(np.concatenate(all_preds).ravel())
    all_labels = torch.from_numpy(np.concatenate(all_labels).ravel())

    loss = mape(all_preds, all_labels)
    print('Test', loss)

    torch.cuda.empty_cache()

    """
    Training tensor(10.5695)
    Val tensor(10.0665)
    Val_1 tensor(11.2141)
    Val_2 tensor(10.7487)
    Val_3 tensor(9.3497)
    Test tensor(9.3962)
    """


def cross_validate(config):
    K_FOLD = 10
    with wandb.init(project=config['PROJECT_NAME'], entity="youssefshoeb", config=config):
        # Access all hyperparameters through wandb.config, so logging matches execution
        config = wandb.config

        print("Loading Dataset...")
        datasets, _ = initDataset(config)

        # Hyperparameters
        num_epochs = config["EPOCHS"]
        loss_func = eval(config["LOSS"])

        # k-fold cross validation scores and segment size
        val_score = []

        total_size = len(datasets['train'])
        fraction = 1 / K_FOLD
        seg = int(total_size * fraction)

        for i in range(K_FOLD):
            print(f"...Fold... {i + 1}")
            print("Loading model...")
            model = load_model(config, datasets)
            model.cuda()

            opt = load_optmizer(config, model)

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

            train_set = torch.utils.data.dataset.Subset(datasets['train'], train_indices)
            val_set = torch.utils.data.dataset.Subset(datasets['train'], val_indices)

            # Prepare training
            train_loader = torch_geometric.loader.DataLoader(train_set, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=True)
            val_loader = torch_geometric.loader.DataLoader(val_set, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=True)

            # Start training
            best_loss = np.inf
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_one_epoch(epoch, loss_func, opt, train_loader, model, k=i + 1)

                # Evaluation on validation set
                model.eval()
                loss = test(epoch, loss_func, val_loader, model, "Validation_1", k=i + 1)

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

def evaluate(config, filename):
    datasets, dataloaders = initDataset(config)

    model = load_model(config, datasets)

    model.load_state_dict(torch.load(f'./runs/{filename}/best_model.pth'))

    model.eval()
    running_loss = 0.0
    step = 0

    for _, sample in tqdm(enumerate(dataloaders['test']), total=len(dataloaders['test'])):
        with torch.set_grad_enabled(False):
            out = model(sample.x_dict, sample.edge_index_dict, sample["path"].batch)

            # Get label
            label = sample['path'].y.reshape(-1, 1)
            loss_value = mape(out, label)

            # Update Tracking
            running_loss += loss_value.item()
            step += 1

    average_loss = running_loss / step

    print('Test', average_loss)
    wandb.log({f"Test loss": float(average_loss), "Epoch": 0})