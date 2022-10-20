# Data Loading
# DATA_PATH = "data/processed_data.csv" # "data/raw_data.csv"
# PREPROCESSED = True  # Whether data in DATA_PATH is prepocessed
DATA_PATH = "data/raw_data.csv"
PREPROCESSED = False
PROCESSED_OUTPUT_PATH = "data/processed_data.csv"
STATS_PATH = "data/price_vol_stats.csv"
UNNORMALIZED_DATA_PATH = "data/unnormalized_data.csv"

# Data Extraction
POL_CDE = "YIK"
POD_CDE = "QZH"
EPIS_LENGTH = 25

# --------------------------------------------------------------------

# Environment Constants
EPISODE_DAYS = 14
STEP_PER_DAY = 6
K = STEP_PER_DAY * EPISODE_DAYS
ENV_STRIDE = 70
INIT_I = 60

# --------------------------------------------------------------------

# Training Parameters
SEED = 10

# -----------------------  STEP-LSTM  --------------------------------

# Step LSTM Parameters - Architecture
T = 15
STEP_LSTM_TRAIN_RATIO = 0.8
STEP_LSTM_FEATURE_SIZE = 2
STEP_LSTM_HIDDEN_DIM = 5
STEP_LSTM_LAYER_NUM = 2
STEP_LSTM_DROPOUT = 0.5
REPRESENTATION_DIM = 6
MLP_H_DIM = 4
STEP_OUTPUT_DIM = 1

# Step LSTM Parameters - Training parameters
STEP_LR = 0.001
STEP_EPOCH = 200
STEP_BATCH_SIZE = 64
STEP_START_SAVE_EPOCH = 10
STEP_WAIT_EPOCH = 3

# -----------------------  EPIS-LSTM  --------------------------------

# Episode LSTM Parameters - Architecture
L = 5
EPIS_LSTM_TRAIN_RATIO = 0.8
EPIS_LSTM_FEATURE_SIZE = REPRESENTATION_DIM
EPIS_LSTM_HIDDEN_DIM = 6
EPIS_LSTM_LAYER_NUM = 1
EPIS_LSTM_DROPOUT = 0.5
if EPIS_LSTM_LAYER_NUM == 1: EPIS_LSTM_DROPOUT = 0
EPIS_OUTPUT_DIM = REPRESENTATION_DIM

# Episode LSTM Parameters - Training parameters
EPIS_LR = 0.001
EPIS_EPOCH = 200
EPIS_BATCH_SIZE = 3
EPIS_START_SAVE_EPOCH = 10
EPIS_WAIT_EPOCH = 3

# ----------------------------- TD3 ----------------------------------

# TD3 Constants
INIT_PRICE = 3200
MAX_ACTION = 30
RL_LSTM_HIDDEN_DIM = 6
ACTION_DIM = 1
STATE_DIM = 3
STATE_LEN = T
DISCOUNT = 0.99

# TD3 Architechture parameters
RL_LSTM_LAYER_NUM = 2
ACTOR_MLP_HIDDEN_DIM = 6
CRITIC_MLP_HIDDEN_DIM = 6

# TD3 Training parameters
RL_DROPOUT = 0.2
ACTOR_LR = 0.001
CRITIC_LR = 0.001
TRAIN_EPOCHS = 100
ACTION_NOISE = 0.1
POLICY_FREQ = 2
TAU = 0.005

# Replay Buffer
REPLAY_BUFFER_MAX_SIZE = 10000
RL_BATCH_SIZE = 256
START_TIMESTEP = 200

# --------------------------------------------------------------------

# Verbose
STEP_LSTM_TRAIN_OUT_FREQ = 5
EPIS_LSTM_TRAIN_OUT_FREQ = 5

# --------------------------------------------------------------------

# Test config
TRAIN_EPIS_NUM = 50
TEST_EPIS_NUM = 3
N = 100

# Inference Parameter
SALES_TRAIN_RATIO = 1
SALES_BATCH_SIZE = 10
TEST_INFERENCE_LR = 0.001
EPOCH = 20
