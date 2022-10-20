from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class StepLSTM(nn.Module):
    ''' StepLSTM learns from the vol-price information and predicts the vol-price at the next step. Moreover, it gives us the episode-wise representation vector S '''
    def __init__(self, feature_size, hidden_dim, action_dim, layer_num, dropout, rep_dim, mlp_h_dim, step_out_dim, n_episode):
        super(StepLSTM, self).__init__() 
        # LSTM learns the step-wise caracteristiques
        self.lstm = torch.nn.LSTM(feature_size + rep_dim, hidden_dim, layer_num, batch_first=True, dropout=dropout)
        # Add a representation vector to represent the episode information
        self.representation = torch.nn.parameter.Parameter(torch.rand(n_episode, rep_dim))
        # MLP to return a volume and price prediction after step-wise and also episode-wise vectors
        self.fc1 = torch.nn.Linear(hidden_dim + action_dim, mlp_h_dim)
        self.fc2 = torch.nn.Linear(mlp_h_dim, step_out_dim)
    
    def forward(self, step_history):
        # Get the episode of each data in the batch
        episode_list = [int(step_history[i][0][-1].item()) for i in range(step_history.shape[0])]
        # Concatenate the representation vector of the current episode of dataset with the vol-price information
        batch_representation = torch.cat([self.representation[episode_list[i]].unsqueeze(0).clone() for i in range(step_history.shape[0])]) 
        batch_representation = torch.cat([batch_representation.unsqueeze(1).clone() for i in range(step_history.shape[1]-1)], 1)
        # check_print(self.representation, "Representation")
        # Full_state includes the step information from LSTM and episode information from representation vector
        full_state = torch.cat((step_history[:, :-1, :-1], batch_representation), -1)
        # MLP to return the vol-price prediction
        _, (hn, _) = self.lstm(full_state)
        batch_action = step_history[:, -1, 1].unsqueeze(1)
        h1 = self.fc1(torch.cat((hn[-1], batch_action), 1))
        nextstep = self.fc2(F.relu(h1))
        return nextstep
    
class EpisLSTM(nn.Module):
    ''' EpisLSTM learns from the episode-wise representation vector and predicts the representation of the next episode '''
    def __init__(self, feature_size, hidden_dim, layer_num, dropout, epis_out_dim):
        super(EpisLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(feature_size, hidden_dim, layer_num, batch_first=True, dropout=dropout)
        # Add a MLP after LSTM
        self.fc = torch.nn.Linear(hidden_dim, epis_out_dim)
        
    def forward(self, epis_history):
        self.lstm.flatten_parameters()
        _, (hn, _) = self.lstm(epis_history)
        out = self.fc(hn[-1])
        return out
    
class DP_Actor(nn.Module):
    def __init__(self, 
                 max_action, rep_dim, action_dim, # setting
                 state_dim, lstm_hidden_dim, lstm_layer_num, dropout, # LSTM
                 mlp_hidden_dim):
        super().__init__() 
        self.lstm = torch.nn.LSTM(state_dim + rep_dim - 1, lstm_hidden_dim, lstm_layer_num, batch_first=True, dropout=dropout)
        self.max_action = max_action
        self.fc1 = nn.Linear(lstm_hidden_dim + state_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, action_dim)
    
    def forward(self, state, rep):  # state: (RL_BATCH_SIZE, L, feature_size), rep: (RL_BATCH_SIZE, rep_dim)
        rep = torch.cat([rep.unsqueeze(1).clone() for _ in range(state.shape[1])], 1)
        c = torch.cat([state[:, :, :-1], rep], -1)
        _, (hn, _) = self.lstm(c)
        full_state = torch.cat([hn[-1], state[:, -1, :]], 1)
        h = torch.tanh(self.fc1(full_state))
        a = self.max_action * torch.tanh(self.fc2(h))
        return a

class DP_Critic(nn.Module):
    def __init__(self, 
                rep_dim, action_dim, # setting
                state_dim, lstm_hidden_dim, lstm_layer_num, dropout, # LSTM
                mlp_hidden_dim):
        super().__init__()
        
        # Q1 architecture
        self.lstm_1 = torch.nn.LSTM(state_dim + rep_dim - 1, lstm_hidden_dim, lstm_layer_num, batch_first=True, dropout=dropout)
        self.fc1_1 = nn.Linear(lstm_hidden_dim + state_dim + action_dim, mlp_hidden_dim)
        self.fc2_1 = nn.Linear(mlp_hidden_dim, 1)
        
        # Q2 architecture
        self.lstm_2 = torch.nn.LSTM(state_dim + rep_dim - 1, lstm_hidden_dim, lstm_layer_num, batch_first=True, dropout=dropout)
        self.fc1_2 = nn.Linear(lstm_hidden_dim + state_dim + action_dim, mlp_hidden_dim)
        self.fc2_2 = nn.Linear(mlp_hidden_dim, 1)
    
    def forward(self, state, rep, action):           
        rep = torch.cat([rep.unsqueeze(1).clone() for _ in range(state.shape[1])], 1)
        c = torch.cat([state[:, :, :-1], rep], -1)
        
        _, (hn_1, _) = self.lstm_1(c)
        full_state = torch.cat([hn_1[-1], state[:, -1, :], action ], -1)
        h_1 = F.relu(self.fc1_1(full_state))
        q_1 = self.fc2_1(h_1)
        
        _, (hn_2, _) = self.lstm_2(c)
        full_state = torch.cat([hn_2[-1], state[:, -1, :], action ], -1)
        h_2 = F.relu(self.fc1_2(full_state))
        q_2 = self.fc2_2(h_2)
        return q_1, q_2
    
    def Q1(self, state, rep, action):
        rep = torch.cat([rep.unsqueeze(1).clone() for _ in range(state.shape[1])], 1)
        c = torch.cat([state[:, :, :-1], rep], -1)
        
        _, (hn_1, _) = self.lstm_1(c)
        full_state = torch.cat([hn_1[-1], state[:, -1, :], action ], -1)
        h_1 = F.relu(self.fc1_1(full_state))
        q_1 = self.fc2_1(h_1)
        return q_1
        
class DP_TD3(nn.Module):
    def __init__(self, 
                 rep_dim, state_dim, action_dim, lstm_hidden_dim, lstm_layer_num, dropout,    # Common settings 
                 max_action, actor_mlp_hidden_dim, actor_lr,  # Actor
                 critic_mlp_hidden_dim, critic_lr  # Critic
                 ):
        super().__init__()
        
        # Initiate actor network
        self.actor = DP_Actor(max_action, rep_dim, action_dim, state_dim, lstm_hidden_dim, lstm_layer_num, dropout, actor_mlp_hidden_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Initiate critic network, composing of two identical deep Q-networks with independant parameters
        self.critic = DP_Critic(rep_dim, action_dim, state_dim, lstm_hidden_dim, lstm_layer_num, dropout, critic_mlp_hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
