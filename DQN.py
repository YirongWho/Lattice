import torch,random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import math 
from math import log
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
Replay Memory
'''
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
'''
DQN network
'''

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    


def main(args):

    DIMENSION=args.DIMENSION
    n_observation=DIMENSION*DIMENSION
    n_actions=DIMENSION*(DIMENSION-1)
    policy_net = DQN(n_observation, n_actions).to(device)
    target_net = DQN(n_observation, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=args.LR, amsgrad=True)
    memory = ReplayMemory(500)

    steps_done = 0
    train_loss=[]

    def select_action(state,steps_done):
        sample = random.random()
        eps_threshold = args.EPS_END + (args.EPS_START - args.EPS_END) * \
            math.exp(-1. * steps_done / args.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(-1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action= policy_net(state).max(-1).indices.view(1,1) 

        else:
            action=torch.randint(0, n_actions, (1,1), device=device, dtype=torch.long)
        return action


    def optimize_model():
        if len(memory) < args.BATCH_SIZE:
            return
        transitions = memory.sample(args.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_states_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(args.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values = target_net(next_states_batch).max(1).values.view(args.BATCH_SIZE, 1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * args.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        return loss.item()
            


    if torch.cuda.is_available():
        num_episodes = 6000
    else:
        num_episodes = 500

    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state = torch.rand(DIMENSION,DIMENSION,dtype=torch.float32)
        # calculate the hadamard ratio of state
        H_ratio=torch.norm(state,dim=1).prod().item()/(torch.abs(torch.det(state)).item())
        H_ratio=log(H_ratio) # a float number 
        lengths= torch.norm(state,dim=1)  # length of each row
        shortest_lengths =  [torch.min(lengths).item()]
        ratios=[H_ratio]
        for t in range(100):
            # Select and perform an action
            action= select_action(state.reshape(-1),steps_done)
            steps_done+=1
            i=(action//(DIMENSION-1)).item() 
            j=(action%(DIMENSION-1)).item()
            if j>=i: j+=1
            mu=torch.dot(state[i],state[j])/torch.dot(state[j],state[j])
            next_state=state.clone()
            next_state[i]-=torch.round(mu)*state[j]
            lengths[i] = torch.norm(next_state[i])  # update length of the i-th row
            shortest_lengths.append(torch.min(lengths).item())
            next_H_ratio = H_ratio-torch.log(torch.norm(state[i])).item()+torch.log(torch.norm(next_state[i])).item()
            ratios.append(next_H_ratio)
            reward = torch.tensor([[H_ratio-next_H_ratio]])

            # Store the transition in memory
            memory.push(state.reshape((1,DIMENSION**2)), action, next_state.reshape((1,DIMENSION**2)), reward)

            # Move to the next state
            state = next_state
            H_ratio = next_H_ratio

            # Perform one step of the optimization (on the policy network)
            train_loss.append(optimize_model())

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*args.TAU + target_net_state_dict[key]*(1-args.TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if (steps_done+1) % 1000 == 0:
            # save parameters of policy_net and target_net
                torch.save(policy_net.state_dict(), "new_experiment/policy_net.pth")
                torch.save(target_net.state_dict(), "new_experiment/target_net.pth")
        # plot ratios and shortest_length
        if (torch.cuda.is_available() and i_episode % 1000 == 0) or (not torch.cuda.is_available() and i_episode % 100 == 0):
            plt.plot(range(len(ratios)), ratios)
            plt.title("H_ratio of each step")
            plt.xlabel("steps")
            plt.ylabel("H_ratio")
            plt.xscale("log", base=10)
            plt.savefig(f"new_experiment/H_ratio_episode {i_episode}.png", dpi=150)
            plt.close()

            plt.plot(range(len(shortest_lengths)), shortest_lengths)
            plt.title("Shortest length of each step")
            plt.xlabel("steps")
            plt.ylabel("shortest length")
            plt.xscale("log", base=10)
            plt.savefig(f"new_experiment/shortest_length_episode {i_episode}.png", dpi=150)
            plt.close()

    plt.plot(range(len(train_loss)), train_loss)
    plt.title("Train loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.xscale("log", base=10)
    plt.savefig("new_experiment/train_loss.png", dpi=150)
                

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--DIMENSION",type=int,default=20)
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--LR", type=float, default=1e-4)
    parser.add_argument("--GAMMA", type=float, default=0.999) 
    parser.add_argument("--EPS_START", type=float, default=0.9)
    parser.add_argument("--EPS_END", type=float, default=0.05)
    parser.add_argument("--EPS_DECAY", type=int, default=200)
    parser.add_argument("--TAU", type=float, default=0.005)
    args = parser.parse_args()
    main(args)

    '''
Hyperparameters:
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
'''