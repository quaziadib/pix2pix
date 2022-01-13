import imp
import torch
from util import save_checkpoint, load_checkpoint, save_some_example
import torch as nn 
import torch.optim as optim 
import config
from dataset import RainDataset 
from generator_model import Generator 
from discriminator_model import Discriminator
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torchvision.utils import save_image 

def train_fn(disc, gen, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    pass

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, beta=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, beta=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = RainDataset(root_dir="dataset\train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    test_dataset = RainDataset(root_dir="dataset\test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for epoch in config.NUM_EPOCHS:
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        
        save_some_example(gen, train_loader, epoch, folder="evaluation")






if __name__ == "__main__":
    main()