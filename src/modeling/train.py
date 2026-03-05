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
import sys

from data import PoissonPINNDataset, FullPINNDataset, CaseBatchSampler
from model import CNN
from model_encdec import EncDecCNN
from loss import PoissonLoss, NavierStokesLoss
from visualization import plot_full, plot_poisson
from misc import str2bool

sys.path.insert(0, "..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from derivatives import *

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
    parser.add_argument("--num_workers",
                        help="The number of dataloader workers",
                        default=0)
    parser.add_argument("--activation",
                        help="Activation function of the model",
                        choices=["relu", "tanh", "silu"],
                        default="relu")
    parser.add_argument("--architecture",
                        help="The architecture of the model. Either a standard CNN or an encoder-decoder",
                        choices=["cnn", "encdec"],
                        default="cnn")
    parser.add_argument("--pooling",
                        help="Pooling type for the encoder-decoder model",
                        choices=["avg", "max", "stride"],
                        default="none")         
    parser.add_argument("--upsampling",
                        help="Upsampling type of the encoder-decoder model",
                        choices=["upsample", "convtranspose"],
                        default="none")
    parser.add_argument("--norm",
                        help="Normalization type in the model",
                        choices=["none", "batch", "group"],
                        default="none")
    
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
    num_workers = int(args.num_workers)
    activation = args.activation
    architecture = args.architecture
    pooling = args.pooling
    upsampling = args.upsampling
    norm = args.norm

    # Set the device
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    # Set seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.set_default_device("cuda")

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

    if data_samples == 1:
        train_split = 1
    else:
        train_split = int(data_samples * 0.7)

    # Load data
    if model_type == "poisson":
        train_dataset = PoissonPINNDataset(data_path, train_split, "cpu", samples_per_case, start_sample=0)

        if data_samples == 1:
            smp = 1
            start_sample = 0
        else:
            smp = data_samples - train_split
            start_sample = train_split

        val_dataset = PoissonPINNDataset(data_path, smp, "cpu", samples_per_case, start_sample=start_sample)
    elif model_type == "full":
        train_dataset = FullPINNDataset(data_path, train_split, "cpu", samples_per_case, start_sample=0)

        if data_samples == 1:
            smp = 0
            start_sample = 0
        else:
            smp = data_samples - train_split
            start_sample = train_split
            
        val_dataset = FullPINNDataset(data_path, smp, "cpu", samples_per_case, start_sample=start_sample)

    x_mean = train_dataset.x.mean(dim=(0, 2, 3), keepdim=True)
    x_std = train_dataset.x.std(dim=(0, 2, 3), keepdim=True)
    x_std[x_std < 1e-8] = 1.0


    # TODO: Normalization logic should probably be moved to the dataset
    if model_type == "poisson":
        # dont_normalize = [3, 4, 5, 8, 9]
        # dont_normalize = [0, 1, 2, 3, 4]
        dont_normalize = []
    else:
        dont_normalize = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
    x_mean[:, dont_normalize] = 0
    x_std[:, dont_normalize] = 1

    # Disable normalization
    # x_mean[...] = 0
    # x_std[...] = 1

    # train_dataset.x = (train_dataset.x - x_mean) / x_std
    # val_dataset.x = (val_dataset.x - x_mean) / x_std

    train_dataset.x.sub_(x_mean)
    train_dataset.x.div_(x_std)

    val_dataset.x.sub_(x_mean)
    val_dataset.x.div_(x_std)

    # Fetch and print parameters
    params = vars(args)
    params["x_mean"] = [float(x_) for x_ in x_mean.detach().numpy().flatten()]
    params["x_std"] = [float(x_) for x_ in x_std.detach().numpy().flatten()]
    print("PARAMETERS")
    print(params)

    # Get input data dimensions
    input_channels, input_height, input_width, output_channels = train_dataset.get_dims()

    pin_memory = num_workers > 0

    if samples_per_case > batch_size:
        sampler_batch_size = batch_size
    else:
        sampler_batch_size = samples_per_case


    # Initialize data loader
    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    
    # batch_sampler = CaseBatchSampler(len(train_dataset), sampler_batch_size)
    # train_dataloader = DataLoader(train_dataset,
    #                             batch_sampler=batch_sampler,
    #                             num_workers=num_workers,
    #                             pin_memory=pin_memory)
    

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

    if architecture == "cnn":
        # Initialize model
        model = CNN(input_channels=input_channels,
                    output_channels=output_channels,
                    num_layers=num_layers,
                    hidden_channels=hidden_dim,
                    input_width=input_width,
                    input_height=input_height,
                    enforceBC=train_dataset.enforceBC,
                    use_skip=True,
                    activation=activation,
                    architecture=architecture,
                    pooling=pooling,
                    upsampling=upsampling,
                    norm=norm)
    else:
        model = EncDecCNN(input_channels=input_channels,
                    output_channels=output_channels,
                    num_layers=num_layers,
                    hidden_channels=hidden_dim,
                    input_width=input_width,
                    input_height=input_height,
                    enforceBC=train_dataset.enforceBC,
                    use_skip=False,
                    activation=activation,
                    architecture=architecture,
                    pooling=pooling,
                    upsampling=upsampling,
                    norm=norm)
    
    # state_dict = torch.load(os.path.join("../../models/poisson/2025-11-18_13_28", "model.pt"), map_location=device)
    # model.load_state_dict(state_dict, strict=True)

    # Move model to device if possible
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Initialize training criterion
    if model_type == "poisson":
        criterion = PoissonLoss(enforceBC=train_dataset.enforceBC)
    else:
        criterion = NavierStokesLoss(enforceBC=train_dataset.enforceBC)

    criterion = criterion.to(device)

    # Define the optimizer
    if optimizer_ == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), 
                                      lr=lr, 
                                      max_iter=20, 
                                      history_size=10, 
                                      line_search_fn='strong_wolfe')
    elif optimizer_ == "adam":
        optimizer = torch.optim.Adam(model.parameters(), 
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
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(18, 7))
    ax = ax.flatten()
    im = None
    cbar = None

    print(f"DATA LENGTH: {len(train_dataset)}")

    prev_loss = 0

    # Training loop
    for epoch in range(epochs):
        running_loss = [0, 0, 0, 0, 0]
        val_running_loss = [0, 0, 0, 0, 0]

        # if loss_weights[2] < 1000:
        #     loss_weights = (loss_weights[0], loss_weights[1], loss_weights[2] * 1.1, loss_weights[3])
        #     if loss_weights[2] >= 1000:
        #         print("EXCEEDED")

        def closure():
            optimizer.zero_grad()
            total_loss_train = torch.tensor(0.0, device=device)

            for data in train_dataloader:
                x, _, _, _, _, _, _, _ = data
                x = x.to(device)
                
                outputs = model(x)
                x_norm = x * x_std.to(device) + x_mean.to(device)

                l = criterion(x_norm, outputs, loss_weights)

                loss_scaled = l[0] * (x.size(0) / len(train_dataset))
                loss_scaled.backward() 
                
                total_loss_train += loss_scaled.detach()

                # for i in range(len(running_loss)):
                #     running_loss[i] += l[i].item() * x.size(0)

            return total_loss_train
        

        optimizer.step(closure)

        model.eval() 
        with torch.no_grad():
            for data in train_dataloader:
                x = data[0].to(device)
                outputs = model(x)
                x_norm = x * x_std.to(device) + x_mean.to(device)
                l = criterion(x_norm, outputs, loss_weights)
                
                for i in range(len(l)):
                    running_loss[i] += l[i].item() * x.size(0)

        # # Iterate through data loader
        # for data in train_dataloader:
        #     # Extract data
        #     x, delta_x, delta_y, _, _, _, _, _ = data

        #     x = x.to(device) #.detach()#.clone()#.requires_grad_(True)
        #     delta_x = delta_x.to(device) #.detach()
        #     delta_y = delta_y.to(device) #.detach()

        #     batch_loss = [0, 0, 0, 0, 0]

        #     # outputs = model(x)

        #     # x_norm = x * x_std.to(device) + x_mean.to(device)

        #     # l = criterion(x, outputs, delta_x, delta_y, loss_weights,)

        #     # optimizer.zero_grad()
        #     # l[0].backward()
        #     # optimizer.step()

        #     # for i in range(len(l)):
        #     #     batch_loss[i] = l[i].item()


        #     def closure():
        #         optimizer.zero_grad()
        #         # total_loss = torch.tensor(0.0, device=device)

        #         with torch.enable_grad():
        #             outputs = model(x)

        #             x_norm = x * x_std.to(device) + x_mean.to(device)
                    
        #             l = criterion(x_norm, outputs, loss_weights)
        #             l[0].backward()

        #         for i in range(len(l)):
        #             batch_loss[i] = l[i].item()

        #         return l[0]
            
        #     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #     optimizer.step(closure)

        #     for i in range(len(running_loss)):
        #         running_loss[i] += batch_loss[i] * x.size(0)

        loss = np.array(running_loss) / len(train_dataloader.dataset)

        if np.isnan(loss[0]):
            print("NaN LOSS - STOPPING EARLY")
            break

        with torch.no_grad():
            for data in val_dataloader:
                x, delta_x, delta_y, _, _, _, _, _ = data
                x = x.to(device) #.detach()#.clone() #.requires_grad_(True)
                delta_x = delta_x.to(device) #.detach()
                delta_y = delta_y.to(device) #.detach()

                outputs = model(x)

                x_norm = x * x_std.to(device) + x_mean.to(device)

                l = criterion(x_norm, outputs, loss_weights)

                for i in range(len(val_running_loss)):
                    val_running_loss[i] += l[i].item() * x.size(0)

            val_loss = np.array(val_running_loss) / len(val_dataloader.dataset)

            val_loss_unweighted = val_loss[1] + val_loss[2] + val_loss[3] + val_loss[4]

            if val_loss_unweighted <= best_loss:
                best_loss = val_loss_unweighted
                best_model_dict = model.state_dict()

                torch.save(best_model_dict, os.path.join(model_path, "model.pt"))

                with open(os.path.join(model_path, "params.json"), "w") as f:
                    json.dump(params, f)

        if use_scheduler:
            scheduler.step()

        print(f"EPOCHS [{epoch + 1}/{epochs}]   TOTAL TRAIN LOSS = {loss[0]:.10f}   TRAIN LOSS PARTS = [{loss[1]:.5f}, {loss[2]:.5f}, {loss[3]:.5f}, {loss[4]:.5f}]   |   TOTAL VAL LOSS = {val_loss[0]:.10f}   VAL LOSS PARTS = [{val_loss[1]:.5f}, {val_loss[2]:.5f}, {val_loss[3]:.5f}, {val_loss[4]:.5f}]")
        print()
        # print(f"                                TOTAL VAL LOSS = {val_loss[0]:.10f}   VAL LOSS PARTS = [{val_loss[1]:.10f}, {val_loss[2]:.10f}, {val_loss[3]:.10f}, {val_loss[4]:.10f}]")
        # print(f"EPOCHS [{epoch + 1}/{epochs}]   TRAIN LOSS = {loss[0]:.10f}   VAL LOSS = {val_loss[0]:.10f}")

        # if np.abs(loss[0] - prev_loss) < 1e-6 and epoch != 0:
        #     print("LOSS STOPPED MINIMIZING - STOPPING TRAINING")
        #     break

        # if best_loss < val_loss_unweighted and np.abs(best_loss - val_loss_unweighted) > 0.01:
        #     print("OVERFITTING DETECTED - STOPPING TRAINING")
        #     break

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.abs().mean().item())

        prev_loss = loss[0]

        if use_wandb:
            wandb.log({"train_loss": loss[0],
                       "train_loss1": loss[1],
                       "train_loss2": loss[2],
                       "train_loss3": loss[3],
                       "train_loss4": loss[4],
                       "val_loss": val_loss_unweighted,
                       "val_loss1": val_loss[1],
                       "val_loss2": val_loss[2],
                       "val_loss3": val_loss[3],
                       "val_loss4": val_loss[4],
                       "log_val_loss": np.log(val_loss_unweighted)})
            
        with torch.no_grad():
            x_, delta_x_, delta_y_, delta_t_, rho_, _, _, boundary_mask = train_dataset[min(0, len(train_dataset) - 1)]
            x_ = x_.cpu().clone().detach().to(device)
            out = model(x_.unsqueeze(0)) #.detach().cpu().squeeze().numpy()

            # out = train_dataset.enforceBC(x_.unsqueeze(0), out)

            # print(out[0, :, 10, 0])
            # print(out[0, :, 0:3, 0:3])

            if model_type == "full":
                out = out.detach().cpu().squeeze().numpy()
                cbar = plot_full(ax[:4], out, delta_x_.item(), cbar=cbar, boundary_image=boundary_mask)
            else:
                out = out.detach().cpu().squeeze().numpy()
                phi = train_dataset.get_phi(min(10, len(train_dataset) - 1))
                x_ = (x_ * x_std.to(device) + x_mean.to(device)).squeeze()
                cbar = plot_poisson(ax[:4], out, phi, boundary_mask, cbar=cbar, in_=x_.cpu(), delta_t=delta_t_, rho=rho_, dx=delta_x_, dy=delta_y_, enforceBC=train_dataset.enforceBC)

            x_, delta_x_, delta_y_, delta_t_, rho_, _, _, boundary_mask = train_dataset[min(31, len(train_dataset) - 1)]
            x_ = x_.cpu().clone().detach().to(device)
            out = model(x_.unsqueeze(0)) #.detach().cpu().squeeze().numpy()

            # out = train_dataset.enforceBC(x_.unsqueeze(0), out)

            if model_type == "full":
                out = out.detach().cpu().squeeze().numpy()
                cbar = plot_full(ax[4:], out, delta_x_.item(), cbar=cbar, boundary_image=boundary_mask)
            else:
                out = out.detach().cpu().squeeze().numpy()
                phi = train_dataset.get_phi(min(60, len(train_dataset) - 1))
                x_ = (x_ * x_std.to(device) + x_mean.to(device)).squeeze()
                cbar = plot_poisson(ax[4:], out, phi, boundary_mask, cbar=cbar, in_=x_.cpu(), delta_t=delta_t_, rho=rho_, dx=delta_x_, dy=delta_y_, enforceBC=train_dataset.enforceBC)

            plt.draw()
            plt.pause(0.001)
            
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()