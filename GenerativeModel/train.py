import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import TransformerModel, generate_square_subsequent_mask
from loaders.loader import CombinedDataset, collate_fn
from loaders.youtube_loader import FORMAT_RANGES, FORMAT_SIZE

# Hyperparameters
TOKEN_DIM = FORMAT_SIZE
MODEL_PARAMS = (TOKEN_DIM, 256, 16, 4, 1024)  # d_input, d_model, nhead, num_layers, dim_feedforward
INIT_LR = 1e-3
FINAL_LR = 2e-5
NUM_EPOCHS = 300
BATCH_SIZE = 90
RANDOM_MASK_PROB = 0.5
NUM_MODELS = 5 # Number of models in the ensemble

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def loss_fn(output, target, weight=5):
    output[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]] = weight * output[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]]
    output[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]] = weight * output[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
    target[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]] = weight * target[:, FORMAT_RANGES["p1_pad"][0]:FORMAT_RANGES["p1_pad"][1]]
    target[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]] = weight * target[:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
    return F.mse_loss(output, target)

def save_checkpoint(model, optimizer, lr_scheduler, epoch, filename):
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer, lr_scheduler):
    checkpoint = torch.load(filename)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    return checkpoint['epoch']

def train(rank, world_size):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)

    # Load data
    loader = CombinedDataset("data/recons", "data/recons_lab", "data/recons_test", "data/recons_lab_test", val_split=0.05)

    # Split the dataset into NUM_MODELS distinct subsets
    total_size = len(loader.train_dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    subsets_indices = np.array_split(indices, NUM_MODELS)
    train_datasets = [torch.utils.data.Subset(loader.train_dataset, idxs) for idxs in subsets_indices]

    # Create data loaders for each model
    train_loaders = []
    for i in range(NUM_MODELS):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_datasets[i],
            num_replicas=world_size,
            rank=rank
        )
        train_loader = torch.utils.data.DataLoader(
            train_datasets[i],
            batch_size=BATCH_SIZE // world_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collate_fn
        )
        train_loaders.append(train_loader)

    val_loader = torch.utils.data.DataLoader(
        loader.val_dataset,
        batch_size=BATCH_SIZE // world_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Device configuration
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Initialize N models with different random seeds
    models = []
    optimizers = []
    lr_schedulers = []
    for i in range(NUM_MODELS):
        # Set different seeds for different initializations
        torch.manual_seed(i + rank * NUM_MODELS)
        model = TransformerModel(*MODEL_PARAMS).to(device)
        model = DDP(model, device_ids=[rank])
        models.append(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
        optimizers.append(optimizer)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=FINAL_LR / INIT_LR, total_iters=NUM_EPOCHS)
        lr_schedulers.append(lr_scheduler)

    # Initialize loss trackers for each model
    train_losses = [[] for _ in range(NUM_MODELS)]  # List of lists
    val_losses = [[] for _ in range(NUM_MODELS)]    # List of lists
    best_val_loss = [float('inf')] * NUM_MODELS

    for epoch in range(NUM_EPOCHS):
        for i in range(NUM_MODELS):
            model = models[i]
            optimizer = optimizers[i]
            lr_scheduler = lr_schedulers[i]
            train_loader = train_loaders[i]
            model.train()
            epoch_train_loss = 0.0
            train_samples = 0
            train_loader.sampler.set_epoch(epoch)

            for batch, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Rank {rank} Model {i}", unit="batch", disable=rank!=0):
                batch, token_mask = batch[:, :, :TOKEN_DIM], batch[:, :, TOKEN_DIM:]
                if np.random.rand() < RANDOM_MASK_PROB:
                    x = batch * loader.random_mask
                else:
                    x = batch

                batch = batch.to(device)
                lengths = lengths.to(device)
                x = x.to(device)

                fps = batch[:, 0, -1] * 100.0
                token_mask = token_mask.to(device)
                future_mask = generate_square_subsequent_mask(x.shape[1]).to(device)

                lengths_mask = torch.arange(x.size(1)).unsqueeze(1).to(device) < lengths.unsqueeze(0)
                lengths_mask = lengths_mask[1:].permute(1, 0)

                optimizer.zero_grad()
                y = model(x, mask=future_mask, token_mask=token_mask, fps=fps)

                y, target, tm = y[:, :-1], batch[:, 1:], token_mask[:, 1:]
                y = y * tm
                target = target * tm
                y = y[lengths_mask]
                target = target[lengths_mask]

                loss = loss_fn(y, target)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                train_samples += 1

            # Average the loss over the batches
            epoch_train_loss /= train_samples
            train_losses[i].append(epoch_train_loss)

            # Validation loop
            model.eval()
            epoch_val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch, lengths in val_loader:
                    batch, token_mask = batch[:, :, :TOKEN_DIM], batch[:, :, TOKEN_DIM:]
                    if np.random.rand() < RANDOM_MASK_PROB:
                        x = batch * loader.random_mask
                    else:
                        x = batch

                    batch = batch.to(device)
                    lengths = lengths.to(device)
                    x = x.to(device)

                    fps = x[:, 0, -1] * 100.0
                    token_mask = token_mask.to(device)
                    future_mask = generate_square_subsequent_mask(x.shape[1]).to(device)
                    lengths_mask = torch.arange(x.size(1)).unsqueeze(1).to(device) < lengths.unsqueeze(0)
                    lengths_mask = lengths_mask[1:].permute(1, 0)

                    y = model(x, mask=future_mask, token_mask=token_mask, fps=fps)

                    y, target, tm = y[:, :-1], batch[:, 1:], token_mask[:, 1:]
                    y = y * tm
                    target = target * tm
                    y = y[lengths_mask]
                    target = target[lengths_mask]

                    loss = loss_fn(y, target)
                    epoch_val_loss += loss.item()
                    val_samples += 1

            # Average the validation loss over the batches
            epoch_val_loss /= val_samples
            val_losses[i].append(epoch_val_loss)

            # Update learning rate
            lr_scheduler.step()

            if rank == 0:
                print(f"Epoch {epoch+1} - Model {i}: Train Loss: {train_losses[i][-1]:.6f}, Val Loss: {val_losses[i][-1]:.6f}")

                # Save the best models
                if val_losses[i][-1] < best_val_loss[i]:
                    best_val_loss[i] = val_losses[i][-1]
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, f'models/ensemble/best_model_{i}.pth')
                    print(f"New best model {i} saved with validation loss: {best_val_loss[i]:.6f}")

                # Save a checkpoint for resuming training
                save_checkpoint(model, optimizer, lr_scheduler, epoch+1, f'models/ensemble/checkpoint_model_{i}.pth')

        if rank == 0:
            current_lr = optimizers[0].param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

            # Plotting the losses for each model
            plt.figure(figsize=(12, 6))
            for i in range(NUM_MODELS):
                plt.plot(train_losses[i], label=f"Model {i} Train")
                plt.plot(val_losses[i], label=f"Model {i} Val")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Losses")
            plt.legend()
            plt.grid(True)
            plt.savefig("track.png")
            plt.close()

    if rank == 0:
        print("Training completed.")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
