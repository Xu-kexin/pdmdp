from config import *
from networks import *
from env import *
from utils import ReplayBuffer
import wandb

import torch
import torch.nn.functional as F
import copy
import random
import pandas as pd

def step_lstm_train(device, train_loader, test_loader, epoch, lr, n_episode):
    stats = pd.read_csv(STATS_PATH)
    # Preparation
    sep_print("Step LSTM Training")
    clear_models("step_lstm")
    
    # Stop Training Criterion
    best_loss = float("inf")
    last_update = STEP_START_SAVE_EPOCH
    
    # Step-wise LSTM
    model = StepLSTM(STEP_LSTM_FEATURE_SIZE, STEP_LSTM_HIDDEN_DIM, ACTION_DIM, STEP_LSTM_LAYER_NUM, STEP_LSTM_DROPOUT, 
                     REPRESENTATION_DIM, MLP_H_DIM, STEP_OUTPUT_DIM, n_episode).to(device)
    model.to(device)
    model_path = "model/step_lstm.pt"
    best_model = model
    
    # Define optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Backpropagation and update the parameters
    for epoch in range(epoch):
        model.train()
        for (train_data, train_label) in train_loader: 
            train_data, train_label = train_data.to(device), train_label.to(device)  
            output = model(train_data)
            # check_print(output, "output")
            loss = criterion(output, train_label)  
            optimizer.zero_grad()             
            loss.backward()                  
            optimizer.step()
                         
        if epoch % STEP_LSTM_TRAIN_OUT_FREQ == 0:
            l1_loss, l2_loss = 0, 0
            testsize, batchsize = 0, 0
            model.eval()
            for (test_data, test_label) in test_loader:   
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_output = model(test_data)
                # Calculate average loss
                batchsize = test_output.shape[0]
                testsize += batchsize
                l2_loss += criterion(test_output, test_label) * batchsize
            l2_loss = torch.sqrt(l2_loss / testsize) * stats['vol_std'].item()
            print('epoch:{:<2d} | l2_loss:{:<4.2f}'.format(epoch, l2_loss))
            if (loss < best_loss) and (epoch >= STEP_START_SAVE_EPOCH):
                # Save the best perform model 
                last_update = epoch
                best_model = copy.deepcopy(model)
                model_path = "model/step_lstm_e_{}_l_{:<6.4f}.pt".format(epoch, loss) 
                torch.save(best_model, model_path)
                best_loss = loss
            if (last_update < epoch - STEP_WAIT_EPOCH):
                # Stop training if the loss does not decrease
                sep_print("Step LSTM Training Finished")
                print("Best Model saved in {}".format(model_path))
                return model_path, best_model
    sep_print("Step LSTM Training Finished")
    print("Best Model saved in {}".format(model_path))
    return model_path, best_model


def epis_lstm_train(device, train_loader, test_loader, epoch, lr):
    # Preparation
    sep_print("Episode LSTM Training")
    clear_models("epis_lstm")
    model_path = "model/epis_lstm.pt"
    
    # Stop Training Criterion
    best_loss = float("inf")
    last_update = EPIS_START_SAVE_EPOCH
    
    # Episode-wise LSTM
    model = EpisLSTM(EPIS_LSTM_FEATURE_SIZE, EPIS_LSTM_HIDDEN_DIM, EPIS_LSTM_LAYER_NUM, EPIS_LSTM_DROPOUT, REPRESENTATION_DIM)
    model = model.to(device)
    model_path = "model/epis_lstm.pt"
    best_model = model
    
    # Define optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    # Backpropagation and update the parameters
    for e in range(epoch):
        model.train()
        for (train_data, train_label) in train_loader:   
            train_data, train_label = train_data.to(device), train_label.to(device)  
            output = model(train_data)
            loss = criterion(output, train_label)  
            optimizer.zero_grad()             
            loss.backward()                  
            optimizer.step()
                          
        if e % EPIS_LSTM_TRAIN_OUT_FREQ == 0:
            loss = 0
            testsize, batchsize = 0, 0
            model.eval()
            for (test_data, test_label) in test_loader: 
                test_data, test_label = test_data.to(device), test_label.to(device)  
                test_output = model(test_data)
                # Calculate average loss
                batchsize = test_output.shape[0]
                testsize += batchsize
                loss += criterion(test_output, test_label) * batchsize 
            loss = loss / testsize
            print('epoch:{:<2d} | loss:{:<6.4f}'.format(e, loss))
            if (loss < best_loss) and (e > EPIS_START_SAVE_EPOCH):
                # Save the best perform model 
                last_update = e
                best_model = copy.deepcopy(model)
                model_path = "model/epis_lstm_e_{}_l_{:<6.4f}.pt".format(e, loss) 
                torch.save(best_model, model_path)
                best_loss = loss
            if (last_update < e - EPIS_WAIT_EPOCH):
                # Stop training if the loss does not decrease
                sep_print("Episode LSTM Training Finished")
                print("Best Model saved in {}".format(model_path))
                return model_path, best_model
    sep_print("Episode LSTM Training Finished")
    print("Best Model saved in {}".format(model_path))
    return model_path, best_model


def TD3_train(device, env, rep_array, n_episode,dic):
    # Initialize Replay buffer
    replay_buffer = ReplayBuffer(STATE_LEN, STATE_DIM, ACTION_DIM, REPRESENTATION_DIM, REPLAY_BUFFER_MAX_SIZE, device)
    
    # Create TD3 Network
    DP_TD3_model = DP_TD3(REPRESENTATION_DIM, STATE_DIM, ACTION_DIM, RL_LSTM_HIDDEN_DIM, RL_LSTM_LAYER_NUM,
                       RL_DROPOUT, MAX_ACTION, ACTOR_MLP_HIDDEN_DIM, ACTOR_LR, CRITIC_MLP_HIDDEN_DIM, CRITIC_LR)
    
    # Initialize counter for policy delay
    count = 0
    total_reward = 0

    # Initialize figure generation tool
    wandb.init(project="dpmdp-project",config=dic)
    
    # Training
    DP_TD3_model.to(device)
    DP_TD3_model.train()
    for epoch in range(TRAIN_EPOCHS):
        
        # Initialization
        state, done = env.reset(INIT_PRICE), False
        episode_reward = 0
        epis_index = random.randint(0, n_episode-1)
        epis_rep = rep_array[epis_index].unsqueeze(0)
        max_reward = 0

        for t in range(EPIS_LENGTH):
            # Random action for cold start || Action given by the policy network
            if replay_buffer.size < START_TIMESTEP:
                action = np.random.uniform(-MAX_ACTION, MAX_ACTION, 1)
                # print(" ", end="[Random Policy]")
            else:
                noise = np.random.normal(0, MAX_ACTION * ACTION_NOISE, ACTION_DIM)
                action = DP_TD3_model.actor(torch.FloatTensor(state).unsqueeze(0).to(device), epis_rep).cpu().data.numpy() + noise
                action = action.clip(-MAX_ACTION, MAX_ACTION)
                action = action.squeeze()
            
            # Perform selected action
            next_state, reward, done = env.step(epis_index * EPIS_LENGTH + t, action)
            done = float(done)
            replay_buffer.add(state, action, epis_rep.cpu().data.numpy(), next_state, reward, done)
            state = next_state
            episode_reward += reward
            
            if replay_buffer.size >= START_TIMESTEP:
                
                # Sample from Replay Buffer
                replay_state, replay_action, replay_epis_rep, replay_next_state, replay_reward, replay_done = replay_buffer.sample(RL_BATCH_SIZE)
                
                # Compute TD Target
                noise = (torch.randn(ACTION_DIM) * ACTION_NOISE).to(device)
                replay_next_action = (DP_TD3_model.actor_target(replay_state, replay_epis_rep) + noise).clamp(-MAX_ACTION, MAX_ACTION)
                target_Q1, target_Q2 = DP_TD3_model.critic_target(replay_next_state, replay_epis_rep, replay_next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                TD_target = replay_reward + (1. - replay_done) * DISCOUNT * target_Q
                
                # Compute Current Critic
                current_Q1, current_Q2 = DP_TD3_model.critic(replay_state, replay_epis_rep, replay_action)
                critic_loss = F.mse_loss(current_Q1, TD_target) + F.mse_loss(current_Q2, TD_target)

                # Monitor result
                wandb.log({"loss": critic_loss, "target Q": target_Q})

                # Optimize the critic
                DP_TD3_model.critic_optimizer.zero_grad()
                critic_loss.backward()
                DP_TD3_model.critic_optimizer.step()

                # Delayed policy updates
                if count % POLICY_FREQ == 0:
                    
                    # Compute actor loss
                    actor_loss = -DP_TD3_model.critic.Q1(replay_state, replay_epis_rep, DP_TD3_model.actor(replay_state, replay_epis_rep)).mean()

                    # Optimize the actor 
                    DP_TD3_model.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    DP_TD3_model.actor_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(DP_TD3_model.critic.parameters(), DP_TD3_model.critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    for param, target_param in zip(DP_TD3_model.actor.parameters(), DP_TD3_model.actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            if done:
                total_reward += episode_reward
                average_epis_reward = total_reward / (epoch+1)
                wandb.log({"Train episode reward":episode_reward,"Train average reward":average_epis_reward})
                # print()
                # print("Quit State: {}".format(state[-1]))
                print("Epoch: {}, Episode: {}, Historic Average Reward: {:.2f}".format(epoch+1, epis_index, float(average_epis_reward)))
                break

            # save best model
        if max_reward < average_epis_reward:
            max_reward = average_epis_reward
            filename = "model/DP_TD3_" + str(epoch+1) + str(average_epis_reward)
            DP_TD3_model.save(filename)

    wandb.finish()
    return DP_TD3_model.actor, filename
            
             