from synthetic_data_gen import SparsePointAssocDataset
from synthetic_data_gen import Tier1A, Tier1B
from synthetic_data_gen import Tier2A, Tier2B
from synthetic_data_gen import Tier3A, Tier3B
from synthetic_data_gen import Tier4A, Tier4B

from neural_nets import ShallowerSparsePointAssocNetwork
from neural_nets import DeeperSparsePointAssocNetwork

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim

'''
Authored by Gary Lvov
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

NUM_GEOS = 5
num_epochs = 1000

geos = []
for _ in range(NUM_GEOS):
    geos.append(SparsePointAssocDataset.generate_random_geometry())

networks = [ShallowerSparsePointAssocNetwork, DeeperSparsePointAssocNetwork]
network_names = ['Shallower', 'Deeper']
tiers = [Tier1A, Tier1B, Tier2A, Tier2B, Tier3A, Tier3B, Tier4A, Tier4B]
tier_names = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B']

def print_gradient_norms(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"Gradient norm for {name}: {grad_norm}")

for jdx, network in enumerate(networks):
    model = network().to(device)
    summary(model, (171,))
    for idx, tier in enumerate(tiers):

        train_dataset = tier(geos=geos, num_geom=NUM_GEOS)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 

        val_dataset = tier(geos=geos, num_geom=NUM_GEOS)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
   
        criterion = nn.MSELoss() # torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs.float())

                loss = criterion(outputs.float(), labels.float())
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs.float())

                    loss = criterion(outputs.float(), labels.float())
                    val_loss += loss.item()

            # Print epoch results
            train_loss = running_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        # Save the model
        torch.save(model.state_dict(), f"tier{idx}_{network_names[jdx]}.pt")
        plt.figure()
        # Finished epoch, plot loss
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss plot for tier {tier_names[idx]} {network_names[jdx]} Network')
        plt.savefig(f'loss_plot_tier_{tier_names[idx]}.png')