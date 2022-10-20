import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize
import copy

from config import *
from utils import *

class SalesEnv():
    def __init__(self, p, state_len):
        self.data = pd.read_csv(UNNORMALIZED_DATA_PATH)
        self.state_len = state_len
        self.reset(p)
    
    def reset(self, p):
        self.state = np.zeros((self.state_len,STATE_DIM))
        self.state[-1] = [INIT_I, p, EPIS_LENGTH]
        self.k = 0
        return self.state
    
    def step(self, env_step, action):
        current_state = copy.deepcopy(self.state[-1])
        vol = self.price2vol(env_step, current_state[1] + action)
        if (vol > current_state[0]): vol = current_state[0]
        current_state[0] -= vol
        current_state[1] += action
        current_state[2] -= 1
        self.k += 1
        done =  (self.k == EPIS_LENGTH) or (current_state[0] == 0)
        reward = vol * current_state[1]
        self.state = np.concatenate((self.state[1:,:],[current_state]), axis = 0)
        return self.state, reward, done
    
    def price2vol(self, step, price):
        # Read the data of current window 
        low, high = max(0, step - ENV_STRIDE // 2), min(len(self.data), step + ENV_STRIDE // 2 + 1)
        df = self.data[low: high]
        df_nonzero = df.loc[df['NUM'] != 0]
        xdata, ydata = np.array(df_nonzero["PRICE"]), np.array(df_nonzero["NUM"])
        
        # The proportion of nonzero volume in the data
        p_nonzero = len(df_nonzero) / len(df)
        
        # Linearly fit the log(x) with Log(y), x - price, y - volume;
        # Actual price - volume formula, Log(y) = max(0, a * log(x) + b);
        # This formula is acquired with the constant elasticity assumption
        xdata_log = np.log(xdata).reshape(-1, 1)
        ydata_log = np.log(ydata).reshape(-1, 1)
        reg = LinearRegression().fit(xdata_log, ydata_log)
        a_lr, b_lr = reg.coef_[0], reg.intercept_

        # Maximise the likelihood
        def neg_log_likelihood_func(beta):
            assert beta > 0
            weighted_log_likelihood = 0
            for index, row in df_nonzero.iterrows():
                predict_exp = a_lr * np.log(row["PRICE"]) + b_lr   # Predicted expectation given by the fitted function
                if predict_exp <= 0: break
                alpha = beta * predict_exp   # Expectation of the gamma distribution is given by the formula alpha / beta
                likelihood = stats.gamma.logpdf(row["NUM"], alpha, 0, 1 / beta) * (1 / (np.abs(step - index) + 1))
                weighted_log_likelihood += likelihood
            return -weighted_log_likelihood
        res = minimize(neg_log_likelihood_func, x0=[5], bounds=[(2, None)])
        beta_opt = res.x
        
        # Estimate the volume given price
        zero_rand = np.random.rand()
        if zero_rand > p_nonzero:
            return 0
        else:
            exp = a_lr * np.log(price) + b_lr
            if exp <= 0: return 0
            num = np.ceil(np.random.gamma(shape=beta_opt*exp, scale=1/beta_opt))
            return num