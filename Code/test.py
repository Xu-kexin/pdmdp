from networks import DP_TD3
from config import *
from env import *
from data import *
import wandb

import torch

def TD3_evaluation(device, env, step_path, epis_path, td_path, train_epis_num, test_epis_num, dic):
    f = open('log.txt', 'w')
    f.write(str(dic))
    f.write('\n')

    # Initialize figure generation tool
    wandb.init(project="dpmdp-project",config=dic)

    # Load Network parameters
    step_model = torch.load(step_path).to(device)
    epis_model = torch.load(epis_path).to(device)
    td_model = DP_TD3(REPRESENTATION_DIM, STATE_DIM, ACTION_DIM, RL_LSTM_HIDDEN_DIM, RL_LSTM_LAYER_NUM,
                       RL_DROPOUT, MAX_ACTION, ACTOR_MLP_HIDDEN_DIM, ACTOR_LR, CRITIC_MLP_HIDDEN_DIM, CRITIC_LR)
    td_model.load(td_path)
    td_model = td_model.to(device)
    td_model.eval()
    
    total_reward_list = []
    
    # Load representation array
    rep_array = step_model.representation.data
    for temp in range(N):
        total_reward = 0
        # print("Test Number: ", temp, end="   ")
        
        for episode in range(train_epis_num + 1, train_epis_num + test_epis_num + 1):
            
            # Learn the representation of next episode by epis-wise LSTM
            epis_rep = epis_model(rep_array[episode - L : episode].unsqueeze(0))
            
            # Initialization
            state, done = env.reset(INIT_PRICE), False
            episode_reward = 0
            step_history = np.zeros((EPIS_LENGTH + 1, STATE_DIM))
            step_history[0][1] = INIT_PRICE
            step_history[:, 2] = episode
            
            for t in range(EPIS_LENGTH):
                # action = td_model.actor(torch.FloatTensor(np.expand_dims(state, 0)).to(device), epis_rep).cpu().data.numpy()
                action = np.random.uniform(-MAX_ACTION, MAX_ACTION, 1)
                state, reward, done = env.step((episode - 1)* EPIS_LENGTH + t, action) 
                # step_history : [vol, price, epis] ; state: [inventory, action, remaining time]
                step_history[t+1][1] = step_history[t][1] + action
                step_history[t][0] = state[-2][0] - state[-1][0] 
                episode_reward += reward

                wandb.log({"Random action": action, "Random action reward": reward})
        
                if done:
                    # print("Random Policy Episode Reward", episode_reward, end = "     ")
                    total_reward += episode_reward
                    # sep_print("Reward")
                    # print("Quit State: {}".format(state[-1]))
                    # print("Epoch: {}, Reward: {:.2f}".format(episode - train_epis_num + 1, float(episode_reward)))

                    # Train the representation by step-wise LSTM
                    sales_dataset = EpsSaleDataset(step_history[:-1])
                    train_loader, _ = load_split_data(sales_dataset, SALES_TRAIN_RATIO, SALES_BATCH_SIZE, s="Sales")
                    
                    # -------------------------------------------------------------------------------
                    # Define optimizer and loss criterion
                    optimizer = torch.optim.Adam([step_model.representation], lr=TEST_INFERENCE_LR)
                    criterion = torch.nn.MSELoss()
                    # Backpropagation and update the parameters
                    for _ in range(EPOCH):
                        step_model.train()
                        for (train_data, train_label) in train_loader:   
                            train_data, train_label = train_data.to(device), train_label.to(device)  
                            output = step_model(train_data)
                            loss = criterion(output, train_label)
                            optimizer.zero_grad()             
                            loss.backward()                  
                            optimizer.step()
                    # -------------------------------------------------------------------------------
                    break
            
        print()
        total_reward_list.append(total_reward)
    
    print(total_reward_list)
    print("Random Policy Average Reward", sum(total_reward_list) / len(total_reward_list))
    f.write("Random Policy Average Reward")
    f.write(str(sum(total_reward_list) / len(total_reward_list)))
    f.write("\n")

    for temp in range(N):
        total_reward = 0
        # print("Test Number: ", temp, end="   ")

        for episode in range(train_epis_num + 1, train_epis_num + test_epis_num + 1):

            # Learn the representation of next episode by epis-wise LSTM
            epis_rep = epis_model(rep_array[episode - L: episode].unsqueeze(0))

            # Initialization
            state, done = env.reset(INIT_PRICE), False
            episode_reward = 0
            step_history = np.zeros((EPIS_LENGTH + 1, STATE_DIM))
            step_history[0][1] = INIT_PRICE
            step_history[:, 2] = episode

            for t in range(EPIS_LENGTH):
                action = td_model.actor(torch.FloatTensor(np.expand_dims(state, 0)).to(device), epis_rep).cpu().data.numpy()
                state, reward, done = env.step((episode - 1) * EPIS_LENGTH + t, action)
                # step_history : [vol, price, epis] ; state: [inventory, action, remaining time]
                step_history[t + 1][1] = step_history[t][1] + action
                step_history[t][0] = state[-2][0] - state[-1][0]
                episode_reward += reward
                print(state[0][0])
                wandb.log({"Test action": action, "Test reward": reward, "Check state[0][0]":state[0][0]})

                if done:
                    # print("Learned Policy Episode Reward", episode_reward, end="     ")
                    total_reward += episode_reward
                    # sep_print("Reward")
                    # print("Quit State: {}".format(state[-1]))
                    # print("Epoch: {}, Reward: {:.2f}".format(episode - train_epis_num + 1, float(episode_reward)))

                    # Train the representation by step-wise LSTM
                    sales_dataset = EpsSaleDataset(step_history[:-1])
                    train_loader, _ = load_split_data(sales_dataset, SALES_TRAIN_RATIO, SALES_BATCH_SIZE, s="Sales")

                    # -------------------------------------------------------------------------------
                    # Define optimizer and loss criterion
                    optimizer = torch.optim.Adam([step_model.representation], lr=TEST_INFERENCE_LR)
                    criterion = torch.nn.MSELoss()
                    # Backpropagation and update the parameters
                    for _ in range(EPOCH):
                        step_model.train()
                        for (train_data, train_label) in train_loader:
                            train_data, train_label = train_data.to(device), train_label.to(device)
                            output = step_model(train_data)
                            loss = criterion(output, train_label)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    # -------------------------------------------------------------------------------
                    break

        print()
        total_reward_list.append(total_reward)

    print(total_reward_list)
    print("Learned Policy Average Reward", sum(total_reward_list) / len(total_reward_list))
    f.write("Learned Policy Average Reward")
    f.write(str(sum(total_reward_list) / len(total_reward_list)))
    f.write("\n")
    f.close()
    wandb.finish()
