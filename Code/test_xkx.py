import math
import argparse
from config import *
from env import SalesEnv
from utils import *
from data import *
from train import *
from test import *

if __name__ == "__main__":
    # Settings
    path_control()
    set_seed(SEED)
    device = get_device()

    # TD3 Training parameters
    parser = argparse.ArgumentParser()
    parser.description = 'please enter parameters of TD3'
    parser.add_argument("-d", "--dropout", help="rldropout", type=float, default="0.2")
    parser.add_argument("-a", "--actorlr", type=float, default=0.001)
    parser.add_argument("-c", "--criticlr", type=float, default=0.001)
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-b", "--batchsize", type=int, default=256)
    parser.add_argument("-s", "--startstep", type=int, default=200)
    args = parser.parse_args()
    RL_DROPOUT = args.dropout
    ACTOR_LR = args.actorlr
    CRITIC_LR = args.criticlr
    TRAIN_EPOCHS = args.epoch
    RL_BATCH_SIZE = args.batchsize
    START_TIMESTEP = args.startstep

    # Dataset
    sales_dataset = SalesDataset(csv_file_path=DATA_PATH, preprocessed=PREPROCESSED)  # DATA_PATH
    train_dataloader, test_dataloader = load_split_data(sales_dataset, STEP_LSTM_TRAIN_RATIO, STEP_BATCH_SIZE,
                                                        s="Step-LSTM")
    n_episode = math.ceil(len(sales_dataset) / EPIS_LENGTH) + 1


    step_lstm_model_path = "model/step_lstm_e_10_l_0.2252.pt"
    epis_lstm_model_path = "model/epis_lstm_e_25_l_0.0996.pt"
    step_lstm_model = torch.load(step_lstm_model_path)
    rep_array = step_lstm_model.representation

    # rep_dataset = RepDataset(rep_array)
    # train_dataloader, test_dataloader = load_split_data(rep_dataset, EPIS_LSTM_TRAIN_RATIO, EPIS_BATCH_SIZE, s="Epis-LSTM")
    # epis_lstm_model_path, epis_lstm_model = epis_lstm_train(device, train_dataloader, test_dataloader, EPIS_EPOCH, EPIS_LR)

    # TD3 train
    dic = {"RL_DROPOUT": RL_DROPOUT,
           "ACTOR_LR": ACTOR_LR,
           "CRITIC_LR": CRITIC_LR,
           "TRAIN_EPOCHS": TRAIN_EPOCHS,
           "RL_BATCH_SIZE": RL_BATCH_SIZE,
           "START_TIMESTEP": START_TIMESTEP}

    env = SalesEnv(INIT_PRICE, STATE_LEN)
    dp_actor_model_path = "DP_TD3_14[45314.62137002]"




    # TD3 evaluation

    TD3_evaluation(device, env, step_lstm_model_path, epis_lstm_model_path, dp_actor_model_path, TRAIN_EPIS_NUM,
                   TEST_EPIS_NUM, dic)