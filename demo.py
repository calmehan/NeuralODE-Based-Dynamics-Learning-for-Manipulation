# Prep

print("Prepping...")

import os
import sys
import time
import itertools
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from IPython.display import Image
from IPython.display import display
from numpngw import write_apng
# Display for Linux
import subprocess


from torchdiffeq import odeint_adjoint as odeint

from panda_pushing_env import PandaPushingEnv, TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE

from visualizers import GIFVisualizer, NotebookVisualizer

from learning_state_dynamics import (
    collect_data_random,
    process_data_single_step, SingleStepDynamicsDataset,
    process_data_multiple_step, MultiStepDynamicsDataset,
    NeuralODEModel, SE2PoseLoss, SingleStepLoss, MultiStepLoss,
    ResidualDynamicsModel,
    PushingController,
    free_pushing_cost_function,
    collision_detection,
    obstacle_avoidance_pushing_cost_function
)


collected_data = np.load('collected_data.npy', allow_pickle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda is available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda is available:", torch.cuda.is_available())

####### Single step rk4, lr = 0.005 on plateau #######

print("Training single-step RK4 model with lr starting 5e-3 on plateau")

BATCH_SIZE = 500

train_loader, val_loader = process_data_single_step(collected_data, BATCH_SIZE)

pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
step_loss = SingleStepLoss(pose_loss)


# 1) Grid settings
methods    = ['rk4']
lrs        = [5e-3]
NUM_EPOCHS = 1000
EVAL_INTERVAL = 20  # how often to record val loss

# Make a directory to save models
save_dir = "demo_models"
os.makedirs(save_dir, exist_ok=True)

# 2) Train+validate with ReduceLROnPlateau + tqdm
def train_validate_plateau(method, lr):
    model = NeuralODEModel(3, 3, [100, 100], method).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=0.8,
        patience=4
    )
    
    val_losses = []
    # wrap the epoch loop in tqdm
    epoch_iter = tqdm(
        range(1, NUM_EPOCHS + 1),
        desc=f"{method}, lr={lr}",
        unit="epoch"
    )

    for epoch in epoch_iter:
        # Training
        model.train()
        for b in train_loader:
            s, u, sn = b['state'].to(device), b['action'].to(device), b['next_state'].to(device)
            optim.zero_grad()
            loss = step_loss(model, s, u, sn)
            loss.backward()
            optim.step()
        
        # Validation & scheduler step at intervals
        if epoch % EVAL_INTERVAL == 0:
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for b in val_loader:
                    s, u, sn = b['state'].to(device), b['action'].to(device), b['next_state'].to(device)
                    tot += step_loss(model, s, u, sn).item()
            val = tot / len(val_loader)
            val_losses.append(val)
            scheduler.step(val)

            # update the tqdm postfix to show the latest validation loss
            epoch_iter.set_postfix(val_loss=f"{val:.4e}")
    
    # Save model after training
    model_path = os.path.join(save_dir, f"model_{method}_lr{lr}.pt")
    torch.save(model.state_dict(), model_path)
    print("Model saved to:", model_path)
    
    return val_losses

# 3) Run sweep
results = {}
for method in tqdm(methods, desc="Integrator"):
    for lr in tqdm(lrs, desc=f"LRs for {method}", leave=False):
        vals = train_validate_plateau(method, lr)
        results[(method, lr)] = vals

# 4) Final loss printout
for (method, lr), losses in results.items():
    print(f"{method:10s}  lr={lr:<7g}  -> final val loss = {losses[-1]:.4e}")


# 5) Plotting
epochs = list(range(EVAL_INTERVAL, NUM_EPOCHS+1, EVAL_INTERVAL))
for (method, lr), losses in results.items():
    plt.plot(epochs, losses, marker='o', label=f'{method}, lr={lr}')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Single-step RK4 Validation Loss with ReduceLROnPlateau')
plt.show()




################ Multi-step 4-step, rk4, lr = 3e-3 on plateau ####################

print("Training multi-step (4-step) RK4 model with lr starting 3e-3 on plateau")

# 1) Grid settings
methods     = ['rk4']
lrs         = [3e-3]
num_steps   = 4
NUM_EPOCHS  = 1500
EVAL_INTERVAL = 20   # record val loss every 20 epochs
BATCH_SIZE  = 500

pose_loss_fn = SE2PoseLoss(block_width=0.1, block_length=0.1)

# 2) Train+validate with ReduceLROnPlateau
def train_validate(method, lr):
    # build multi-step dataloaders
    train_loader, val_loader = process_data_multiple_step(
        collected_data,
        batch_size=BATCH_SIZE,
        num_steps=num_steps
    )

    # instantiate model, optimizer & scheduler
    model     = NeuralODEModel(3, 3, [100,100], method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=4
        # verbose=True
    )
    step_loss_fn = MultiStepLoss(pose_loss_fn, discount=0.9)

    val_losses = []
    # inner bar for epochs
    epoch_bar = tqdm(
        range(1, NUM_EPOCHS+1),
        desc=f"Epochs @ lr={lr:.1e}",
        leave=False,
        unit="epoch"
    )
    for epoch in epoch_bar:
        # --- train
        model.train()
        for b in train_loader:
            s, u, sn = (
                b['state'].to(device),
                b['action'].to(device),
                b['next_state'].to(device),
            )
            optimizer.zero_grad()
            loss = step_loss_fn(model, s, u, sn)
            loss.backward()
            optimizer.step()

        # --- validate & step scheduler at interval
        if epoch % EVAL_INTERVAL == 0:
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for b in val_loader:
                    s, u, sn = (
                        b['state'].to(device),
                        b['action'].to(device),
                        b['next_state'].to(device),
                    )
                    tot += step_loss_fn(model, s, u, sn).item()
            avg_val = tot / len(val_loader)
            val_losses.append(avg_val)

            # tell scheduler how we did
            scheduler.step(avg_val)

            # display train/val/lr
            current_lr = optimizer.param_groups[0]['lr']
            epoch_bar.set_postfix({
                'val': f"{avg_val:.2e}",
                'lr':  f"{current_lr:.1e}"
            })

    # --- Save model after training
    model_path = os.path.join(save_dir, f"model_{method}_steps{num_steps}_lr{lr}.pt")
    torch.save(model.state_dict(), model_path)

    return val_losses

# 3) Run sweep with tqdm
results = {}
for method in tqdm(methods, desc="Integrator"):
    for lr in tqdm(lrs, desc=f"LRs for {method}", leave=False):
        vals = train_validate(method, lr)
        results[(method, lr)] = vals

# --- after your grid‐search loop
print("\nFinal validation losses:")
for (method, lr), val_losses in results.items():
    final = val_losses[-1]
    print(f"{method:4s} | lr={lr:<7g} → final val loss = {final:.4e}")

# 4) Plotting
epochs = list(range(EVAL_INTERVAL, NUM_EPOCHS+1, EVAL_INTERVAL))
for (method, lr), losses in results.items():
    plt.plot(epochs, losses, marker='o', label=f'{method}, lr={lr}')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Multi-step (4-step) RK4 Validation Loss with ReduceLROnPlateau')
plt.show()






####################### Compare with Hw3 #######################

print('Comparing with hw3.')

# Load the models
# -- single-step residual
residual_single = ResidualDynamicsModel(3,3)
residual_single.load_state_dict(torch.load(
    "pushing_residual_dynamics_model.pt", map_location="cpu"))
residual_single.eval()

# -- multi-step residual (num_steps=4)
residual_multi = ResidualDynamicsModel(3,3)
residual_multi.load_state_dict(torch.load(
    "pushing_multi_step_residual_dynamics_model.pt", map_location="cpu"))
residual_multi.eval()

# -- single-step NeuralODE
ode_single = NeuralODEModel(3,3,[100,100],'rk4')
ode_single.load_state_dict(torch.load(
    "demo_models/model_rk4_lr0.005.pt", map_location="cpu"))
ode_single.eval()

# -- multi-step NeuralODE (num_steps=4)
ode_multi = NeuralODEModel(3,3,[100,100],'rk4')
ode_multi.load_state_dict(torch.load(
    "demo_models/model_rk4_steps4_lr0.003.pt", map_location="cpu"))
ode_multi.eval()

# single-step validation
val_ds1 = SingleStepDynamicsDataset(
    np.load('validation_data.npy', allow_pickle=True))
val_loader1 = torch.utils.data.DataLoader(val_ds1, batch_size=len(val_ds1))

# 4-step validation
num_steps = 4
val_ds4 = MultiStepDynamicsDataset(
    np.load('validation_data.npy', allow_pickle=True),
    num_steps=num_steps)
val_loader4 = torch.utils.data.DataLoader(val_ds4, batch_size=len(val_ds4))

pose_fn   = SE2PoseLoss(block_width=0.1, block_length=0.1)
ss_loss   = SingleStepLoss(pose_fn)
ms_loss   = MultiStepLoss(pose_fn, discount=1.0)

# Compute losses
# single-step
loss_res1 = 0.0
loss_ode1 = 0.0
for batch in val_loader1:
    s,u,sn = batch['state'], batch['action'], batch['next_state']
    loss_res1 += ss_loss(residual_single, s, u, sn).item()
    loss_ode1 += ss_loss(ode_single,      s, u, sn).item()

# multi-step (normalize by num_steps)
loss_res4 = 0.0
loss_ode4 = 0.0
for batch in val_loader4:
    s, ua, xs = batch['state'], batch['action'], batch['next_state']
    loss_res4 += ms_loss(residual_multi, s, ua, xs).item()
    loss_ode4 += ms_loss(ode_multi,      s, ua, xs).item()
    
loss_res4 /= num_steps
loss_ode4 /= num_steps

print(f"Single-step residual loss = {loss_res1:.4e}")
print(f"Single-step NeuralODE loss = {loss_ode1:.4e}")
print(f"4-step residual loss       = {loss_res4:.4e}")
print(f"4-step NeuralODE loss      = {loss_ode4:.4e}")

# Plot grouped bar chart
labels      = ['Single-step', '4-step']
res_losses  = [loss_res1, loss_res4]
ode_losses  = [loss_ode1, loss_ode4]

x     = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, res_losses, width, label='Residual (HW3)')
bars2 = ax.bar(x + width/2, ode_losses, width, label='NeuralODE')

# Log scale
ax.set_yscale('log')
ax.set_ylabel('Validation Loss')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Residual vs. NeuralODE Performance')
ax.legend()

# Annotate bar heights
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.2e}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize='small'
    )

plt.tight_layout()
plt.show()

#################################################### Simulation #####################################################

print('##################### Running simulations. #####################')
print('Loading models.')

# Load the models
# -- single-step residual
residual_single = ResidualDynamicsModel(3,3)
residual_single.load_state_dict(torch.load(
    "pushing_residual_dynamics_model.pt", map_location="cpu"))
residual_single.eval()

# -- multi-step residual (num_steps=4)
residual_multi = ResidualDynamicsModel(3,3)
residual_multi.load_state_dict(torch.load(
    "pushing_multi_step_residual_dynamics_model.pt", map_location="cpu"))
residual_multi.eval()

# -- single-step NeuralODE
ode_single = NeuralODEModel(3,3,[100,100],'rk4')
ode_single.load_state_dict(torch.load(
    "demo_models/model_rk4_lr0.005.pt", map_location="cpu"))
ode_single.eval()

# -- multi-step NeuralODE (num_steps=4)
ode_multi = NeuralODEModel(3,3,[100,100],'rk4')
ode_multi.load_state_dict(torch.load(
    "demo_models/model_rk4_steps4_lr0.003.pt", map_location="cpu"))
ode_multi.eval()

############## Obstacle-free

print('############ Starting obstacle-free cases. ############')

###### Residual single

print('### Obstacle-free, residual single ###')

# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()


env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
#controller = PushingController(env, single_model, free_pushing_cost_function, num_samples=100, horizon=10)

controller = PushingController(env, residual_single, free_pushing_cost_function, 100, 10)

# Move MPPI’s internal buffers to the same device as your NeuralODE
mp = controller.mppi
device = next(controller.model.parameters()).device

# # force both U and noise to GPU
# mp.U     = mp.U.to(device)
# mp.noise = mp.noise.to(device)


for name in ['U', 'noise', 'noise_sigma_inv', 'u_init', 'states', 'actions', 'cost_total']:
    if hasattr(mp, name):
        tensor = getattr(mp, name)
        if isinstance(tensor, torch.Tensor):
            setattr(mp, name, tensor.to(device))

# env.reset()

# state_0 = env.reset()
# state = state_0

state = env.reset()
# env.render_frame()                   # ensure we get the “before push” image


# num_steps_max = 100
num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break
 
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')

# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "obs_free_res_single.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux
        
        
# plt.close(fig)

###### NeuralODE single

print('### Obstacle-free, NeuralODE single ###')

# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()


env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
#controller = PushingController(env, single_model, free_pushing_cost_function, num_samples=100, horizon=10)

controller = PushingController(env, ode_single, free_pushing_cost_function, 100, 10)

# Move MPPI’s internal buffers to the same device as your NeuralODE
mp = controller.mppi
device = next(controller.model.parameters()).device


for name in ['U', 'noise', 'noise_sigma_inv', 'u_init', 'states', 'actions', 'cost_total']:
    if hasattr(mp, name):
        tensor = getattr(mp, name)
        if isinstance(tensor, torch.Tensor):
            setattr(mp, name, tensor.to(device))

# env.reset()

# state_0 = env.reset()
# state = state_0

state = env.reset()

# num_steps_max = 100
num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
        

# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "obs_free_ode_single.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux

# plt.close(fig)



###### Residual multi - 4-step

print('### Obstacle-free, residual multi (4) ###')

# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()


env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)

#controller = PushingController(env, single_model, free_pushing_cost_function, num_samples=100, horizon=10)

controller = PushingController(env, residual_multi, free_pushing_cost_function, 100, 10)

# Move MPPI’s internal buffers to the same device as your NeuralODE
mp = controller.mppi
device = next(controller.model.parameters()).device


for name in ['U', 'noise', 'noise_sigma_inv', 'u_init', 'states', 'actions', 'cost_total']:
    if hasattr(mp, name):
        tensor = getattr(mp, name)
        if isinstance(tensor, torch.Tensor):
            setattr(mp, name, tensor.to(device))

# env.reset()

# state_0 = env.reset()
# state = state_0

state = env.reset()

# num_steps_max = 100
num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')



# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "obs_free_res_multi.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux
        
# plt.close(fig)




###### NeuralODE multi - 4-step

print('### Obstacle-free, NeuralODE multi (4) ###')


# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()


env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
#controller = PushingController(env, single_model, free_pushing_cost_function, num_samples=100, horizon=10)

controller = PushingController(env, ode_multi, free_pushing_cost_function, 100, 10)

# Move MPPI’s internal buffers to the same device as your NeuralODE
mp = controller.mppi
device = next(controller.model.parameters()).device


for name in ['U', 'noise', 'noise_sigma_inv', 'u_init', 'states', 'actions', 'cost_total']:
    if hasattr(mp, name):
        tensor = getattr(mp, name)
        if isinstance(tensor, torch.Tensor):
            setattr(mp, name, tensor.to(device))

# env.reset()

# state_0 = env.reset()
# state = state_0

state = env.reset()

# num_steps_max = 100
num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')


# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "obs_free_ode_multi.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)
if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux
        
# plt.close(fig)




############## With-obstacle

print('############ Starting with-obstacle cases. ############')

###### Residual single

print('### With-obstacle, residual single ###')

# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

# set up controller and environment
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
controller = PushingController(env, residual_single,
                               obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
env.reset()

state_0 = env.reset()
state = state_0

num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_OBSTACLES
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
        
# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "with_obs_res_single.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux

# plt.close(fig)

###### NeuralODE single

print('### With-obstacle, NeuralODE single ###')


# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

# set up controller and environment
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
controller = PushingController(env, ode_single,
                               obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
env.reset()

state_0 = env.reset()
state = state_0

num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_OBSTACLES
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
        

# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "with_obs_ode_single.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux

# plt.close(fig)

###### Residual multi - 4-step

print('### With-obstacle, residual multi(4) ###')


# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

# set up controller and environment
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
controller = PushingController(env, residual_multi,
                               obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
env.reset()

state_0 = env.reset()
state = state_0

num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_OBSTACLES
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')

# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "with_obs_res_multi.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux
        
# plt.close(fig)

###### NeuralODE multi - 4-step

print('### With-obstacle, NeuralODE multi(4) ###')


# Control on an obstacle free environment

# fig = plt.figure(figsize=(8,8))
# hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

# set up controller and environment
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)
controller = PushingController(env, ode_multi,
                               obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
env.reset()

state_0 = env.reset()
state = state_0

num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_OBSTACLES
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
        
# save and display gif

gif_path = visualizer.get_gif()
# choose a new name (in the same folder, or anywhere else)
new_name = "with_obs_ode_multi.gif"
new_path = os.path.join(os.path.dirname(gif_path), new_name)
# rename/move the file
os.replace(gif_path, new_path)   # atomic on most OSes; use os.rename() if you prefer
gif_path = new_path

print("Animation saved to:", gif_path)

if sys.platform.startswith("win"):
    os.startfile(gif_path)
else:
    subprocess.run(["xdg-open", gif_path]) # linux

# plt.close(fig)


print ("########################## DONE ##########################")