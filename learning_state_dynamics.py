# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here

    collected_data = []
    
    state = env.reset()
    state_dim = np.array(state).shape[0]
    action = env.action_space.sample()
    action_dim = np.array(action).shape[0]


    for traj in range(num_trajectories):

        state = env.reset()
        
        states = np.zeros((trajectory_length+1, state_dim), dtype=np.float32)
        actions = np.zeros((trajectory_length, action_dim), dtype=np.float32)
    
        states[0] = np.array(state, dtype=np.float32)

        for i in range(trajectory_length):
            
            current_action = env.action_space.sample()
            new_state, reward, done, info = env.step(current_action)
            
            states[i+1] = np.array(new_state, dtype=np.float32)
            actions[i] = np.array(current_action, dtype=np.float32)

    
        
        collected_data.append({'states':states,
                           'actions':actions})

    # ---
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here

    dataset = SingleStepDynamicsDataset(collected_data)

    size = len(dataset)
    train_size = int(0.8 * size)
    val_size = size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here

    dataset = MultiStepDynamicsDataset(collected_data, num_steps = num_steps)

    size = len(dataset)
    train_size = int(0.8 * size)
    val_size = size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)


    # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here

        # item ranges from 0 to self.__len__()
        # self.__len__() is calculated by # of trajectories collected * # of steps in each trajectory
        traj_index = item // self.trajectory_length
        step_index = item % self.trajectory_length

        sample = {
            'state': self.data[traj_index]['states'][step_index],
            'action': self.data[traj_index]['actions'][step_index],
            'next_state': self.data[traj_index]['states'][step_index+1],
        }

        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here

        # item ranges from 0 to self.__len__()
        # self.__len__() is calculated by # of trajectories collected * # of steps in each trajectory
        traj_index = item // self.trajectory_length
        step_index = item % self.trajectory_length

        actions = self.data[traj_index]['actions'][step_index : step_index+self.num_steps]
        next_states = self.data[traj_index]['states'][step_index+1 : step_index+self.num_steps+1]
    

        sample = {
            # state is initial state
            'state': torch.tensor((self.data[traj_index]['states'][step_index]), dtype=torch.float32),
            'action': torch.tensor(actions, dtype=torch.float32),
            'next_state': torch.tensor(next_states, dtype=torch.float32),
        }

        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here

        rg = ((self.l**2 + self.w**2)/12)**(0.5)

        # if batch: use pose_pred[..., 0]. works for non batch too
        x_loss = F.mse_loss(pose_pred[..., 0], pose_target[..., 0])
        y_loss = F.mse_loss(pose_pred[..., 1], pose_target[..., 1])
        theta_loss = F.mse_loss(pose_pred[..., 2], pose_target[..., 2])

        se2_pose_loss = x_loss + y_loss + rg * theta_loss

        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here

        pred_state = model(state, action)
        single_step_loss = self.loss(pred_state, target_state)

        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        
        current_state = state
        total_loss = 0.0

        num_steps = actions.shape[1]

        # For each time step, apply the action, predict the next state, compute loss, and update state.
        for t in range(num_steps):
            # Get the action for the current step (shape: (batch_size, action_dim)).
            current_action = actions[:, t, :]
            # Predict the next state.
            predicted_next_state = model(current_state, current_action)
            # Compute the loss for the current step comparing predicted next state with the ground truth.
            step_loss = self.loss(predicted_next_state, target_states[:, t, :])
            # Discount the loss for the current step and add to total loss.
            total_loss += (self.discount ** t) * step_loss
            # Update current_state for the next iteration.
            current_state = predicted_next_state

        multi_step_loss = total_loss
        
        # ---
        return multi_step_loss



############################################################
############## NeuralODE Implemenation begins ##############

from torchdiffeq import odeint_adjoint as odeint



class ODEFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        layers = []
        inp_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(inp_dim, h), nn.ReLU()]
            inp_dim = h
        layers += [nn.Linear(inp_dim, state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, xa):
        """
        xa: (..., state_dim+action_dim)
        returns: (..., state_dim+action_dim)
        """
        # 1) compute dx/dt
        dx = self.net(xa)  # (..., state_dim)
        # 2) action derivative = 0
        du = torch.zeros_like(xa[..., self.state_dim:])  # (..., action_dim)
        # 3) concatenate to match the ODE’s full dimension
        return torch.cat([dx, du], dim=-1)



class NeuralODEModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, method='rk4'):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.func       = ODEFunc(state_dim, action_dim, hidden_sizes)
        self.method     = method

    def forward(self, state, action, dt=1.0):
        """
        state:  tensor of shape (B, state_dim)
        action: tensor of shape (B, action_dim)
        returns: next_state_pred of shape (B, state_dim)
        """
        # Initial condition for ODE: concatenate state and action
        xa0 = torch.cat([state, action], dim=-1)               # (B, state_dim+action_dim)
        t  = torch.tensor([0.0, dt], device=state.device, dtype=state.dtype)
        # Integrate ODE from t=0 → t=dt
        xa_t = odeint(self.func, xa0, t, method=self.method)   # (2, B, state_dim+action_dim)
        # extract the “state” part at final time
        xa_final   = xa_t[-1]                                   # (B, state_dim+action_dim)
        next_state = xa_final[..., :self.state_dim]            # (B, state_dim)
        return next_state








############## NeuralODE Implemenation ends  ################
#############################################################





# From HW3
class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here

        input_dim = state_dim + action_dim

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, state_dim)

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here

        x = torch.cat([state, action], dim=-1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        next_state = self.fc3(x)

        # ---
        return next_state

# From HW3
class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here

        input_dim = state_dim + action_dim

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, state_dim)


        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here

        x = torch.cat([state, action], dim=-1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        delta_x = self.fc3(x)

        next_state = state + delta_x

        # ---
        return next_state


# From HW3
def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """

    device, dtype = state.device, state.dtype
    
    target_pose = TARGET_POSE_FREE_TENSOR.to(device=device, dtype=dtype)  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here

    Q = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.1]], device=device, dtype=dtype)

    # shape: (B, state_size)
    error = state - target_pose.unsqueeze(0) # unsqueeze(0) => shape (1,3)

    cost = torch.sum(error@Q * error, dim=1)



    # ---
    return cost


# From HW3
def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    device, dtype = state.device, state.dtype
    
    obstacle_centre = OBSTACLE_CENTRE_TENSOR.to(device=device, dtype=dtype)  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None

    
    # --- Your code here
    
    B = state.shape[0]
    in_collision_list = []

    # Pre-compute obstacle corners (axis aligned rectangle)
    obs_cx, obs_cy = obstacle_centre[0], obstacle_centre[1]
    obs_half   = OBSTACLE_HALFDIMS_TENSOR.to(device=device, dtype=dtype)  # (half_width, half_length)
    obs_corners = torch.tensor([
        [obs_cx - obs_half[0], obs_cy - obs_half[1]],
        [obs_cx + obs_half[0], obs_cy - obs_half[1]],
        [obs_cx + obs_half[0], obs_cy + obs_half[1]],
        [obs_cx - obs_half[0], obs_cy + obs_half[1]]
    ], dtype=state.dtype, device=state.device)
    
    # Define obstacle axes (since obstacle is axis aligned)
    axes_obs = [
        torch.tensor([1.0, 0.0], dtype=state.dtype, device=state.device),
        torch.tensor([0.0, 1.0], dtype=state.dtype, device=state.device)
    ]

    ######    get_block_corners: See Helper funciton below    ######
    
    # For each state in the batch, check for collision.
    for i in range(B):
        s = state[i]  # [x, y, theta, ...] (we only use first 3 entries)
        block_center = s[:2]   # (x, y)
        theta = s[2]
        half_size = box_size / 2.0
        
        # Compute block corners (shape: (4,2))
        block_corners = get_block_corners(block_center, half_size, theta)
        
        # Block's axes: these are the normals to its edges (unit vectors)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        block_axis1 = torch.tensor([cos_t, sin_t], dtype=state.dtype, device=state.device)   # along one edge
        block_axis2 = torch.tensor([-sin_t, cos_t], dtype=state.dtype, device=state.device)  # perpendicular edge
        
        axes_block = [block_axis1, block_axis2]
        
        # Combine axes to test: obstacle's axes and block's axes.
        axes = axes_obs + axes_block
        
        collision = True  # assume collision until one separating axis is found
        for axis in axes:
            # Project both sets of corners onto this axis.
            proj_block = torch.matmul(block_corners, axis)
            proj_obs = torch.matmul(obs_corners, axis)
            # Get projection intervals.
            min_block, max_block = proj_block.min(), proj_block.max()
            min_obs, max_obs = proj_obs.min(), proj_obs.max()
            # If there is a gap, then there is no collision.
            if max_block < min_obs or max_obs < min_block:
                collision = False
                break
        in_collision_list.append(1.0 if collision else 0.0)
    
    in_collision = torch.tensor(in_collision_list, dtype=state.dtype, device=state.device)

    # ---
    return in_collision

# From HW3
def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR.to(device=state.device, dtype=state.dtype)  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here

    
    Q = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.1]], device=state.device, dtype=state.dtype)

    # shape: (B, state_size)
    error = state - target_pose.unsqueeze(0) # unsqueeze(0) => shape (1,3)

    state_cost = torch.sum((error @ Q) * error, dim=1)
    collision_cost = 100.0 * collision_detection(state)  # shape: (B,)

    cost = state_cost + collision_cost

    # ---
    return cost

# From HW3
class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model

        # unify device
        # self.device = next(model.parameters()).device
        device = next(model.parameters()).device
        
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low).to(device)
        u_max = torch.from_numpy(env.action_space.high).to(device)
        noise_sigma = (0.4 * torch.eye(env.action_space.shape[0])).to(device)
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         device=device,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here

        device = state.device
        action = action.to(device)

        next_state = self.model(state, action)


        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        
        state_tensor = torch.tensor(state, dtype=torch.float32)\
                              .unsqueeze(0)\
                              .to(next(self.model.parameters()).device)

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here

        action = action_tensor.squeeze(0).detach().cpu().numpy()

        # ---
        return action






# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here

# From HW3
# Helper: compute corners of a rotated square (block) given center, half-size, and theta.
def get_block_corners(center, half_size, theta):
    # Define the four corners in the block's local coordinate system
    local_corners = torch.tensor([
        [ half_size,  half_size],
        [ half_size, -half_size],
        [-half_size, -half_size],
        [-half_size,  half_size]
    ], dtype=center.dtype, device=center.device)  # shape: (4,2)
    # Rotation matrix
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([torch.stack([cos_t, -sin_t]),
                     torch.stack([sin_t,  cos_t])])  # shape: (2,2)
    # Rotate and translate local corners
    world_corners = (R @ local_corners.T).T + center  # shape: (4,2)
    return world_corners

# ---
# ============================================================
