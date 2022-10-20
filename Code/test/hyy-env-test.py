import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize

from config import *
from utils import *


CURRENT_STEP = 193
df = pd.read_csv(UNNORMALIZED_DATA_PATH)[CURRENT_STEP - ENV_STRIDE // 2:CURRENT_STEP + ENV_STRIDE // 2 + 1]
check_print(df, "df")

df_nonzero = df.loc[df['NUM'] != 0]
check_print(df_nonzero, "df_nonzero")

xdata, ydata = np.array(df_nonzero["PRICE"]), np.array(df_nonzero["NUM"])
check_print(xdata, "x")
check_print(ydata, "y")

p_nonzero = len(df_nonzero) / len(df)
check_print(p_nonzero, "p_nonzero")

# Linearly fit the log(x) with y, x - price, y - volume; Actual price - volume formula, y = max(0, a * log(x) + b); This formula is acquired with the constant elasticity assumption
xdata_log = np.log(xdata).reshape(-1, 1)
reg = LinearRegression().fit(xdata_log, ydata)
a_lr, b_lr = reg.coef_[0], reg.intercept_
check_print([a_lr, b_lr], "Regression parameters")

def neg_log_likelihood_func(beta):
    assert beta > 0
    weighted_log_likelihood = 0
    for index, row in df_nonzero.iterrows():
        predict_exp = a_lr * np.log(row["PRICE"]) + b_lr   # Predicted expectation given by the fitted function
        if predict_exp <= 0: break
        # print("For Price {:.2f}, Predict & Real: {:.2f} & {}".format(row["PRICE"], predict_exp, row["NUM"]))
        alpha = beta * predict_exp   # Expectation of the gamma distribution is given by the formula alpha / beta
        likelihood = stats.gamma.logpdf(row["NUM"], alpha, 0, 1 / beta) * (1 / (CURRENT_STEP - index + 1))
        # print("Weighted Likelihood for [index:{}, beta:{:.2f}, num:{}, price:{:.2f}] is {:.2f}".format(index, float(beta), row["NUM"], row["PRICE"], float(likelihood)))
        weighted_log_likelihood += likelihood
    return -weighted_log_likelihood

res = minimize(neg_log_likelihood_func, x0=[5], bounds=[(2, None)])
beta_opt = res.x
print("Estimated solution: ", beta_opt)

price = 2830
for _ in range(20):
    zero_rand = np.random.rand()
    if zero_rand > p_nonzero:
        print(0)
    else:
        exp = a_lr * np.log(price) + b_lr
        print("Exp: ", exp)
        if exp <= 0: print(0)
        num = np.ceil(np.random.gamma(shape=beta_opt*exp, scale=1/beta_opt))
        print("Num: ", num)