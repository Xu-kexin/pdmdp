import math
from config import *
from env import SalesEnv
from utils import *
from data import *
from train import *
from test import *

if __name__ == "__main__":
    device = get_device()
    td_path = "DP_TD3_14[45314.62137002]"
    td_model = DP_TD3(REPRESENTATION_DIM, STATE_DIM, ACTION_DIM, RL_LSTM_HIDDEN_DIM, RL_LSTM_LAYER_NUM,
                           RL_DROPOUT, MAX_ACTION, ACTOR_MLP_HIDDEN_DIM, ACTOR_LR, CRITIC_MLP_HIDDEN_DIM, CRITIC_LR)
    td_model.load(td_path,map_location=torch.device('cpu'))
    td_model = td_model.to(device)
    td_model.eval()