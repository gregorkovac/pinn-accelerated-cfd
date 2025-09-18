import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import random
import wandb
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

from data import PoissonPINNDataset, FullPINNDataset
from model import CNN
from loss import PoissonLoss, NavierStokesLoss
from visualization import plot_full, plot_poisson
from misc import str2bool

def main():
    # Initialize command line argument parser
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument("--model",
                        help="Which model to train",
                        choices=["poisson", "full"])
    parser.add_argument("--data_path",
                        help="Path to the data folder")
    parser.add_argument("--data_samples",
                        help="Number of data cases used for training",
                        default=1)
    parser.add_argument("--samples_per_case",
                        help="Number of samples per training case. -1 means all cases",
                        default=-1)
    parser.add_argument("--model_path",
                        help="Where to save the trained model")
    parser.add_argument("--epochs",
                        help="Number of training epochs",
                        default=100000)
    parser.add_argument("--lr",
                        help="Learning rate",
                        default=1e-3)
    parser.add_argument("--batch_size",
                        help="Batch size",
                        default=256)
    parser.add_argument("--num_layers",
                        help="Number of CNN layers",
                        default=3)
    parser.add_argument("--hidden_dim",
                        help="Hidden dimension of CNN",
                        default=64)
    parser.add_argument("--use_scheduler",
                        help="Whether to use learning rate scheduler or not",
                        type=str2bool,
                        default=False)
    parser.add_argument("--scheduler_gamma",
                        help="Gamma parameter of scheduler",
                        default=0.8)
    parser.add_argument("--scheduler_step",
                        help="Scheduler step",
                        default=100)
    parser.add_argument("--optimizer",
                        help="Which optimizer to use",
                        choices=["adam", "lbfgs"],
                        default="adam")
    parser.add_argument("--weight1",
                        help="Weights of loss 1",
                        default=1)
    parser.add_argument("--weight2",
                        help="Weights of loss 2",
                        default=1)
    parser.add_argument("--weight3",
                        help="Weights of loss 3",
                        default=1)
    parser.add_argument("--weight4",
                        help="Weights of loss 4",
                        default=1)
    parser.add_argument("--wandb",
                        help="Whether to use Weights and Biases for logging or not",
                        type=str2bool,
                        default=False)
    parser.add_argument("--wandb-project",
                        help="The name of the Weight and Biases project")
    
    # Parse command line arguments
    args = parser.parse_args()

    # Save parsed arguments to variables
    model_type = args.model
    data_path = args.data_path
    data_samples = int(args.data_samples)
    samples_per_case = int(args.samples_per_case)
    model_path = args.model_path
    epochs = int(args.epochs)
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_layers = int(args.num_layers)
    hidden_dim = int(args.hidden_dim)
    use_scheduler = bool(args.use_scheduler)
    scheduler_gamma = float(args.scheduler_gamma)
    scheduler_step = int(args.scheduler_step)
    optimizer_ = args.optimizer
    loss_weights = (float(args.weight1), float(args.weight2), float(args.weight3), float(args.weight4))
    use_wandb = args.wandb
    wandb_project = args.wandb_project

    # Set the device
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    # Set seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Set up Weights and Biases
    if use_wandb:
        wandb.init(
            project = wandb_project,
            config = {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "use_scheduler": use_scheduler,
                "scheduler_gamma": scheduler_gamma,
                "scheduler_step": scheduler_step,
                "optimizer": optimizer_,
                "weight1": loss_weights[0],
                "weight2": loss_weights[1],
                "weight3": loss_weights[2],
                "weight4": loss_weights[3]
            }
        )

    # Fetch and print parameters
    params = vars(args)
    print("PARAMETERS")
    print(params)

    if data_samples == 1:
        train_split = 1
    else:
        train_split = int(data_samples * 0.7)

    # Load data
    if model_type == "poisson":
        train_dataset = PoissonPINNDataset(data_path, train_split, samples_per_case, start_sample=0)

        if data_samples == 1:
            smp = 1
            start_sample = 0
        else:
            smp = data_samples - train_split
            start_sample = train_split

        val_dataset = PoissonPINNDataset(data_path, smp, samples_per_case, start_sample=start_sample)
    elif model_type == "full":
        train_dataset = FullPINNDataset(data_path, train_split, samples_per_case, start_sample=0)

        if data_samples == 1:
            smp = 0
            start_sample = 0
        else:
            smp = data_samples - train_split
            start_sample = train_split
            
        val_dataset = FullPINNDataset(data_path, smp, samples_per_case, start_sample=start_sample)

    # Get input data dimensions
    input_channels, input_height, input_width, output_channels = train_dataset.get_dims()

    # Initialize data loader
    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    # Initialize model
    model = CNN(input_channels=input_channels,
                output_channels=output_channels,
                num_layers=num_layers,
                hidden_channels=hidden_dim,
                input_width=input_width,
                input_height=input_height,
                enforceBC=train_dataset.enforceBC,
                use_skip=True)
    
    # Move model to device if possible
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Initialize training criterion
    if model_type == "poisson":
        criterion = PoissonLoss()
    else:
        criterion = NavierStokesLoss()

    # Define the optimizer
    if optimizer_ == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), 
                                      lr=lr, 
                                      max_iter=20, 
                                      history_size=100, 
                                      line_search_fn='strong_wolfe')
    elif optimizer_ == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=lr)

    if use_scheduler:
        scheduler = StepLR(optimizer,
                           step_size=scheduler_step,
                           gamma=scheduler_gamma)

    # Create the folder for saving the model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    model_path = os.path.join(model_path, model_type, timestamp)
    os.makedirs(model_path, exist_ok=True)

    # Define the variables for the best model
    best_loss = np.inf
    best_model_dict = None

    plt.ion()
    plt.tight_layout()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 7))
    ax = ax.flatten()
    im = None
    cbar = None

    print(f"DATA LENGTH: {len(train_dataset)}")

    # Training loop
    for epoch in range(epochs):
        running_loss = [0, 0, 0, 0, 0]
        val_running_loss = [0, 0, 0, 0, 0]

        # if loss_weights[2] < 1000:
        #     loss_weights = (loss_weights[0], loss_weights[1], loss_weights[2] * 1.1, loss_weights[3])
        #     if loss_weights[2] >= 1000:
        #         print("EXCEEDED")

        # Iterate through data loader
        for data in train_dataloader:
            # Extract data
            x, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask = data

            x = x.to(device).clone().requires_grad_(True)

            batch_loss = [0, 0, 0, 0, 0]

            def closure():
                optimizer.zero_grad()

                outputs = model(x)
                
                l = criterion(x, outputs, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask, loss_weights)
                l[0].backward()

                for i in range(len(l)):
                    batch_loss[i] = l[i].item()

                return l[0]
            
            optimizer.step(closure)

            for i in range(len(running_loss)):
                running_loss[i] += batch_loss[i] * x.size(0)


        loss = np.array(running_loss) / len(train_dataloader.dataset)

        if np.isnan(loss[0]):
            print("NaN LOSS - STOPPING EARLY")
            break

        # with torch.no_grad():
        for data in val_dataloader:
            x, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask = data
            x = x.to(device).clone().requires_grad_(True)

            outputs = model(x)

            l = criterion(x, outputs, delta_x, delta_y, t, delta_t, rho, mu, coords, boundary_mask, loss_weights)

            for i in range(len(val_running_loss)):
                val_running_loss[i] += l[i].item() * x.size(0)

        val_loss = np.array(val_running_loss) / len(val_dataloader.dataset)

        if val_loss[0] <= best_loss:
            best_loss = val_loss[0]
            best_model_dict = model.state_dict()

            torch.save(best_model_dict, os.path.join(model_path, "model.pt"))

            with open(os.path.join(model_path, "params.json"), "w") as f:
                json.dump(params, f)

        if use_scheduler:
            scheduler.step()

        # print(f"EPOCHS [{epoch + 1}/{epochs}]   TOTAL LOSS = {loss[0]:.10f}   LOSS PARTS = [{loss[1]:.10f}, {loss[2]:.10f}, {loss[3]:.10f}, {loss[4]:.10f}]")
        print(f"EPOCHS [{epoch + 1}/{epochs}]   TRAIN LOSS = {loss[0]:.10f}   VAL LOSS = {val_loss[0]:.10f}")

        if use_wandb:
            wandb.log({"loss": loss[0],
                       "loss1": loss[1],
                       "loss2": loss[2],
                       "loss3": loss[3],
                       "loss4": loss[4]})
            
        with torch.no_grad():
            x_, delta_x_, delta_y_, _, _, _, _, _, boundary_mask = train_dataset[min(0, len(train_dataset) - 1)]
            out = model(x_.unsqueeze(0)).detach().cpu().squeeze().numpy()

            if model_type == "full":
                cbar = plot_full(ax[:3], out, delta_x_.item(), cbar=cbar)
            else:
                phi = train_dataset.get_phi(min(10, len(train_dataset) - 1))
                cbar = plot_poisson(ax[:3], out, phi, boundary_mask, cbar=cbar)

            x_, delta_x_, delta_y_, _, _, _, _, _, boundary_mask = train_dataset[min(31, len(train_dataset) - 1)]
            out = model(x_.unsqueeze(0)).detach().cpu().squeeze().numpy()

            if model_type == "full":
                cbar = plot_full(ax[3:], out, delta_x_.item(), cbar=cbar)
            else:
                phi = train_dataset.get_phi(min(60, len(train_dataset) - 1))
                cbar = plot_poisson(ax[3:], out, phi, boundary_mask, cbar=cbar)

            plt.draw()
            plt.pause(0.001)
            
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()