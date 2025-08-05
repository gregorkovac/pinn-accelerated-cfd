import argparse
import torch
import torch.nn as nn
import random
import wandb
import numpy as np

from misc import str2bool

def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--weight_bc",
                        help="Boundary condition loss weight",
                        default=10000)
    parser.add_argument("--weight_mse",
                        help="MSE loss weight",
                        default=10000)
    parser.add_argument("--weight_symmetry",
                        help="Symmetry loss weight",
                        default=10000)
    

    args = parser.parse_args()
    
    epochs = int(args.epochs)
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_layers = int(args.num_layers)
    hidden_dim = int(args.hidden_dim)
    use_scheduler = bool(args.use_scheduler)
    scheduler_gamma = float(args.scheduler_gamma)
    scheduler_step = int(args.scheduler_step)
    optimizer = args.optimizer
    loss_weights = (float(args.weight_bc), float(args.weight_mse), float(args.weight_symmetry))

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    input_width = 20
    input_height = 20

    # dataset = Data(N=47, width=input_width, height=input_height)
    dataset = Data(N=1, width=input_width, height=input_height)

    print(f"Data length: {len(dataset)}")

    # batch_size = len(dataset)

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    use_wandb = False

    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="poisson-solver",

            # track hyperparameters and run metadata
            config={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "use_scheduler": use_scheduler,
                "scheduler_gamma": scheduler_gamma,
                "scheduler_step": scheduler_step,
                "optimizer": optimizer,
                "weight_bc": loss_weights[0],
                "weight_mse": loss_weights[1],
                "weight_symmetry": loss_weights[2]
            }
        )

    params = vars(args)

    print(params)

    split = int(len(dataset) * 1)

    train_indices = np.arange(0, split)
    val_indices = np.arange(split, len(dataset))

    # indices = np.random.permutation(len(dataset))
    # train_indices = indices[:split]
    # val_indices = indices[split:]

    x_train = dataset.x[train_indices]
    x_mean = x_train.mean()
    x_std = x_train.std() + 1e-8
    # dataset.x = (dataset.x - x_mean) / x_std

    params["x_mean"] = x_mean.item()
    params["x_std"] = x_std.item()

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    dataloader_val = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    model = NN(input_channels=4,
               num_layers=num_layers,
               hidden_channels=hidden_dim,
               input_width=input_width,
               input_height=input_height)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = PoissonLoss(width=input_width, height=input_height, x_mean=x_mean, x_std=x_std)

    if optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
    elif optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        print("ERROR: optimizer name not valid")
        exit(0)

    best_val_loss = np.inf
    best_model_dict = None

    same_val = []

    adam = False
    adam_steps = 0

    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    plt.ion() 
    plt.tight_layout()
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(12, 7))
    ax = ax.flatten()
    plt_init = True
    ims = []
    # im_phi = None
    # im_phi_true = None
    # im_phi_trunc = None
    # im_phi_true_trunc = None

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        running_pde = 0
        running_bc = 0
        running_mse = 0
        running_sym = 0

        for x, y, delta_x, delta_t, rho, coords, boundary in dataloader_train:
            x, y = x.to(device), y.to(device)
            x = x.clone().detach().requires_grad_(True)
            coords = coords.clone().detach().requires_grad_(True)

            # with torch.no_grad():
            #     x_ = x.detach().cpu().squeeze().numpy()
            #     plt.imshow(x_[0, 0, ...])
            #     plt.show()

            coords = coords.to(device).clone().detach().requires_grad_(True)
            boundary = boundary.to(device)

            # plt.imshow(boundary.detach().numpy()[0][0])
            # plt.show()

            x_coords = torch.cat([x, coords], dim=1)

            batch_loss = [0]
            batch_pde = [0]
            batch_bc = [0]
            batch_mse = [0]
            batch_sym = [0]

            def closure():

                optimizer.zero_grad()
                outputs = model(x_coords)
                loss, loss_pde, loss_bc, loss_mse, loss_sym = criterion(x, outputs, y, delta_x, delta_t, rho, coords, loss_weights)
                loss.backward()

                batch_loss[0] = loss.item()
                batch_pde[0] = loss_pde.item()
                batch_bc[0] = loss_bc.item()
                batch_mse[0] = loss_mse.item()
                batch_sym[0] = loss_sym.item()

                return loss #.detach()
            
            optimizer.step(closure)

            running_loss += batch_loss[0] * x.size(0)
            running_pde += batch_pde[0] * x.size(0)
            running_bc += batch_bc[0] * x.size(0)
            running_mse += batch_mse[0] * x.size(0)
            running_sym += batch_sym[0] * x.size(0)

        train_loss = running_loss / len(dataloader_train.dataset)
        train_pde = running_pde / len(dataloader_train.dataset)
        train_bc = running_bc / len(dataloader_train.dataset)
        train_mse = running_mse / len(dataloader_train.dataset)
        train_sym = running_sym / len(dataloader_train.dataset)
        
        model.eval()
        running_loss = 0
        running_pde = 0
        running_bc = 0
        running_mse = 0
        running_sym = 0

        for x, y, delta_x, delta_t, rho, coords, boundary in dataloader_val:
            x, y = x.to(device), y.to(device)
            x = x.clone().detach().requires_grad_(True)
            coords = coords.to(device).clone().detach().requires_grad_(True)
            boundary = boundary.to(device)
            x_coords = torch.cat([x, coords], dim=1)

            outputs = model(x_coords)

            loss, loss_pde, loss_bc, loss_mse, loss_sym = criterion(x, outputs, y, delta_x, delta_t, rho, coords, loss_weights)
            
            running_loss += loss.item() * x.size(0)
            running_pde += loss_pde.item() * x.size(0)
            running_bc += loss_bc.item() * x.size(0)
            running_mse += loss_mse.item() * x.size(0)
            running_sym += loss_sym.item() * x.size(0)

        val_loss = running_loss / len(dataloader_val.dataset)
        val_pde = running_pde / len(dataloader_val.dataset)
        val_bc = running_bc / len(dataloader_val.dataset)
        val_mse = running_mse / len(dataloader_val.dataset)
        val_sym = running_sym / len(dataloader_val.dataset)
    
        if np.isnan(train_loss) or np.isnan(val_loss):
            print("NaN Loss - Stopping early")
            break

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_model_dict = model.state_dict()

            torch.save(best_model_dict, "nn.pt")

            with open("params.json", "w") as f:
                json.dump(params, f)

        if use_scheduler:
            scheduler.step()


        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.5f} [{train_pde:.5f}] [{train_bc:.5f}] [{train_mse:.5f}] [{train_sym:.5f}], Val Loss: {val_loss:.5f} [{val_pde:.5f}] [{val_bc:.5f}] [{val_mse:.5f}] [{val_sym:.5f}]')

        if use_wandb:
            wandb.log({"train_loss": train_loss, 
                       "val_loss": val_loss, 
                       "best_val_loss": best_val_loss})
            
        
        with torch.no_grad():
            phi = []
            for i in range(10):
                # first_batch = next(iter(dataloader_train))
                x, y, delta_x, delta_t, rho, coords, boundary = dataset[i]

                x, y = x.to(device), y.to(device)
                x = x.clone().detach().requires_grad_(True).unsqueeze(0)

                if i == 9:
                    phi.append(y.detach().squeeze().numpy())
                    continue

                coords = coords.to(device).clone().detach().requires_grad_(True).unsqueeze(0)

                x_coords = torch.cat([x, coords], dim=1)

                phi_ = model(x_coords).detach().cpu().squeeze().numpy()
                phi.append(phi_)

            vmax = max([np.max(phi[i]) for i in range(len(phi))])
            vmin = min([np.min(phi[i]) for i in range(len(phi))])

            if plt_init:
                for i in range(10):
                    ims.append(ax[i].imshow(phi[i], vmin=vmin, vmax=vmax))
                    ax[i].axis("off")

                # im_phi = ax[0].imshow(phi, vmin=vmin, vmax=vmax)
                # im_phi_true = ax[1].imshow(phi_true, vmin=vmin, vmax=vmax)
                # im_phi_trunc = ax[2].imshow(phi[1:-1, 1:-1], vmin=vmin, vmax=vmax)
                # im_phi_true_trunc = ax[3].imshow(phi_true[1:-1, 1:-1], vmin=vmin, vmax=vmax)

                # ax[0].set_title("Phi pred")
                # ax[1].set_title("Phi true")
                # ax[2].set_title("Phi pred (truncated)")
                # ax[3].set_title("Phi true (truncated)")

                fig.colorbar(ims[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

                plt_init = False
            else:
                for i in range(10):
                    ims[i].set_data(phi[i])
                    ims[i].set_clim(vmin, vmax)
            
            plt.draw()
            plt.pause(0.001)

    torch.save(best_model_dict, "nn.pt")
    
    with open("params.json", "w") as f:
        json.dump(params, f)

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()