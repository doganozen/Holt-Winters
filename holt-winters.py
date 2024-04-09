################################################## 
# Holt-Winters 
##################################################
 
import itertools 
import warnings 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
from sklearn.metrics import mean_absolute_error 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.seasonal import seasonal_decompose 

# pd.set_option('display.float_format', lambda x: '%0.2' % x) 
# np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format}) warnings.filterwarnings('ignore') 

################################################## 
# Veri Seti 
################################################## 

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A. 
# Period of Record: March 1958 - December 2001 

data = sm.datasets.co2.load_pandas() 
y = data.data 
y.index.sort_values()

y = y['co2'].resample('MS').mean() 
y.shape 

y = y.fillna(y.bfill()) 

y.plot(figsize=(15, 6)) 
plt.show() 


# 1958'den 1997'sonuna kadar train set. 
train = y[:'1997-12-01'] 
len(train) 


# 1998'ilk ayından 2001'in sonuna kadar test set. 
test = y['1998-01-01':] 
len(test) 

################################# 
# Zaman Serisin Yapısal Analizi 
#################################
 
def ts_decompose(y, model="additive", stationary=False): 
    from statsmodels.tsa.seasonal import seasonal_decompose 
    result = seasonal_decompose(y, model=model) 
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False) 
    fig.set_figheight(10) 
    fig.set_figwidth(15) 
    
    axes[0].set_title("Decomposition for " + model + " model") 
    axes[0].plot(y, 'k', label='Original ' + model) 
    axes[0].legend(loc='upper left') 

    axes[1].plot(result.trend, label='Trend') 
    axes[1].legend(loc='upper left') 

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4))) 
    axes[2].legend(loc='upper left') 
    
    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4))) 
    axes[3].legend(loc='upper left') 
    plt.show() 
    
    if stationary: 
        print("HO: Seri Durağan değildir.") 
        print("H1: Seri Durağandır.") 
        p_value = sm.tsa.stattools.adfuller(y)[1] 
        if p_value < 0.05: 
            print(F"Sonuç: Seri Durağandır ({p_value}).") 
        else: 
            print(F"Sonuç: Seri Durağan Değildir ({p_value}).") 
            
for model in ["additive", "multiplicative"]: 
    ts_decompose(y, model, True) 

################################# 
# FORECASTING WITH SINGLE EXPONENTIAL SMOOTHING 
################################# 
    
# Durağan serilerde kullanılabilir. Trend ve mevsimsellik varsa kullanılamaz. 
# Kullanılabilir ama istenilen sonuçları vermez. 
    
################################# 
# Durağanlık Testi (Dickey-Fuller Testi) 
################################# 
    
def is_stationary(y): 
    print("HO: Seri Durağan değildir.") 
    print("H1: Seri Durağandır.") 
    p_value = sm.tsa.stattools.adfuller(y)[1] 
    if p_value < 0.05: 
        print(F"Sonuç: Seri Durağandır ({p_value}).") 
    else: 
        print(F"Sonuç: Seri Durağan Değildir ({p_value}).")

is_stationary(y)

################################# 
# Single Exponential Smoothing Model 
################################# 

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5) 
y_pred = ses_model.forecast(48) 
train.plot(title="Single Exponential Smoothing") 
test.plot() 
y_pred.plot() 
plt.show() 
mean_absolute_error(test, y_pred) 
# 5.70 

# DİKKAT! 
# Bu fonksiyonlar parametreleri mak. log-likelihood! 

################################# 
# Optimizing Single Exponential Smoothing 
#################################

def optimize_ses(train, alphas, step=48): 
    for alpha in alphas: 
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha) 
        y_pred = ses_model.forecast(step) 
        mae = mean_absolute_error(test, y_pred) 
        print("alpha:", round(alpha, 2), "mae:", round(mae, 4)) 
        

alphas = np.arange(0.01, 1, 0.10) 
optimize_ses(train, alphas) 
# alpha: 0.91 mae: 4.7012 

################################# 
# Final SES Model 
################################# 


ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.91) 
y_pred = ses_model.forecast(48) 
train["1985":].plot(title="Single Exponential Smoothing") 
test.plot() 
y_pred.plot() 
plt.show() 
mean_absolute_error(test, y_pred) 
# 4.70118 

################################# 
# FORECASTING WITH DOUBLE EXPONENTIAL SMOOTHING 
################################# 
# DES: Level + Trend 

################################# 
# DES Model 
################################# 

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5, smoothing_slope=0.5) 
y_pred = des_model.forecast(48) 
train["1985":].plot(title="Double Exponential Smoothing") 
test.plot() 
y_pred.plot() 
plt.show() 
mean_absolute_error(test, y_pred) 

################################# 
# Optimizing Double Exponential Smoothing 
#################################
 
def optimize_des(train, alphas, betas, step=48): 
    for alpha in alphas: 
        for beta in betas: 
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta) 
            
            y_pred = des_model.forecast(step) 
            mae = mean_absolute_error(test, y_pred) 
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4)) 

alphas = np.arange(0.01, 1, 0.10) 
betas = np.arange(0.01, 1, 0.10) 
optimize_des(train, alphas, betas) 

def optimize_des(train, alphas, betas, step=48): 
    print("Optimizing parameters...") 
    results = [] 
    for alpha in alphas: 
        for beta in betas: 
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta) 
            y_pred = des_model.forecast(step) 
            mae = mean_absolute_error(test, y_pred) 
            results.append([round(alpha, 2), round(beta, 2), round(mae, 2)]) 
        results = pd.DataFrame(results, columns=["alpha", "beta", "mae"]).sort_values("mae") 
        print(results) 

optimize_des(train, alphas, betas) 
# 0.01 0.51 1.73 

################################# 
# Final DES Model 
#################################
 
final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.01, smoothing_slope=0.51) 

y_pred = final_des_model.forecast(48) 
train["1985":].plot(title="Double Exponential Smoothing") 
test.plot() 
y_pred.plot() 
plt.show() 
mean_absolute_error(test, y_pred) 

################################# 
# FORECASTING WITH TRIPLE EXPONENTIAL SMOOTHING (HOLT-WINTERS) 
################################# 

# Holt-Winter yöntemi en gelişmiş exponential smoothing yöntemi. 
# Level + trend + mevsimsellik bileşenlerinin hepsini modelleyebiliyor.
 
################################# 
# TES Model 
################################# 

tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.5, smoothing_slope=0.5, smoothing_seasonal=0.5) 
y_pred = tes_model.forecast(48) 
train["1985":].plot(title="Triple Exponential Smoothing") 
test.plot() 
y_pred.plot() 
plt.show() 
mean_absolute_error(test, y_pred) 
# 4.65 

################################# 
# Optimizing Triple Exponential Smoothing 
#################################
 
alphas = betas = gammas = np.arange(0.20, 1, 0.10) 
abg = list(itertools.product(alphas, betas, gammas)) 
abg[0][2] 

def optimize_tes(train, abg, step=48): 
    print("Optimizing parameters...") 
    results = [] 
    for comb in abg: 
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2]) 
        y_pred = tes_model.forecast(step) 
        mae = mean_absolute_error(test, y_pred) 

        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

        results.append([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)]) 
    results = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "mae"]).sort_values("mae") 
    print(results) 
    
alphas = betas = gammas = np.arange(0.10, 1, 0.20) 
abg = list(itertools.product(alphas, betas, gammas))

optimize_tes(train, abg) 

################################# 
# Final TES Model 
################################# 

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.5, smoothing_slope=0.1, smoothing_seasonal=0.1) 
y_pred = final_tes_model.forecast(48) 
train["1985":].plot(title="Triple Exponential Smoothing") 
test.plot() 
y_pred.plot() 
plt.show() 
mean_absolute_error(test, y_pred)

################################# 
# Toplamsal Model mi Çarpımsal Model mi? 
################################# 

# Mevsimsellik ve artık bileşenleri trendden bağımsız ise toplamsal bir seri. 
# mevsimsellik ve artık bileşenleri trende bağımlıysa yani trende göre şekilleniyorsa çarpımsal bir seri.
 
# The additive model is Y[t] = T[t] + S[t] + e[t] 
# The multiplicative model is Y[t] = T[t] * S[t] * e[t] 

for model in ["additive", "multiplicative"]: 
    ts_decompose(y, model, True) 