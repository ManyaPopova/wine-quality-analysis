#####################################################################################################
#ввод данных и 1 часть задания
####################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.stats import chi2
from scipy.stats import  norm
from scipy.stats import kstest
from scipy.stats import chisquare
from scipy.stats import lognorm



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=';')

X = data['fixed acidity'] 
Y = data['volatile acidity']  

def estimate_params(data, dist_type='normal'):
    if dist_type == 'normal':
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        return mean, std
    elif dist_type == 'lognormal':
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data, ddof=0)
        return mu, sigma

params_X_normal = estimate_params(X, 'normal')
params_Y_normal = estimate_params(Y, 'normal')
params_X_lognormal = estimate_params(X, 'lognormal')
params_Y_lognormal = estimate_params(Y, 'lognormal')


plt.figure(figsize=(12, 6))

# График для X (фиксированная кислотность)
plt.subplot(1, 2, 1)
x_range = np.linspace(min(X), max(X), 100)
pdf_X_normal = stats.norm.pdf(x_range, params_X_normal[0], params_X_normal[1])
pdf_X_lognormal = stats.lognorm.pdf(x_range, s=params_X_lognormal[1], scale=np.exp(params_X_lognormal[0]))
plt.hist(X, bins=20, density=True, edgecolor='black', alpha=0.7, label='Данные')
plt.plot(x_range, pdf_X_normal, 'r-', label='Нормальное')
plt.plot(x_range, pdf_X_lognormal, 'g--', label='Логнормальное')
plt.title('Фиксированная кислотность (X)')
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.legend()

# График для Y (летучая кислотность)
plt.subplot(1, 2, 2)
y_range = np.linspace(min(Y), max(Y), 100)
pdf_Y_normal = stats.norm.pdf(y_range, params_Y_normal[0], params_Y_normal[1])
pdf_Y_lognormal = stats.lognorm.pdf(y_range, s=params_Y_lognormal[1], scale=np.exp(params_Y_lognormal[0]))
plt.hist(Y, bins=20, density=True, edgecolor='black', alpha=0.7, label='Данные')
plt.plot(y_range, pdf_Y_normal, 'r-', label='Нормальное')
plt.plot(y_range, pdf_Y_lognormal, 'g--', label='Логнормальное')
plt.title('Летучая кислотность (Y)')
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.legend()

plt.tight_layout()
plt.show()



print("Оценки параметров для X (фиксированная кислотность):")
print(f"Нормальное: μ = {params_X_normal[0]:.4f}, σ = {params_X_normal[1]:.4f}")
print(f"Логнормальное: μ_лог = {params_X_lognormal[0]:.4f}, σ_лог = {params_X_lognormal[1]:.4f}\n")

print("Оценки параметров для Y (летучая кислотность):")
print(f"Нормальное: μ = {params_Y_normal[0]:.4f}, σ = {params_Y_normal[1]:.4f}")
print(f"Логнормальное: μ_лог = {params_Y_lognormal[0]:.4f}, σ_лог = {params_Y_lognormal[1]:.4f}")
###############################################################################################################
# 2 часть задания
###############################################################################################################

#### для X
log_X = np.log(X)

def test_log_normal_mu(data_log, mu_hypothesis=2.09):
    """Тест отношения правдоподобия (LRT) для μ логнормального распределения."""
    mu_mle = np.mean(data_log)
    sigma_mle = np.std(data_log, ddof=0)
    
    ll_null = np.sum(norm.logpdf(data_log, loc=mu_hypothesis, scale=sigma_mle))
    
    ll_alt = np.sum(norm.logpdf(data_log, loc=mu_mle, scale=sigma_mle))
    
    # LRT-статистика и p-value (двусторонний тест)
    lrt_stat = -2 * (ll_null - ll_alt)
    p_value_two_sided = 1 - chi2.cdf(lrt_stat, df=1)
    
    # Односторонние тесты:
    if mu_mle > mu_hypothesis:
        p_value_right = 0.5 * p_value_two_sided  
        p_value_left = 1 - 0.5 * p_value_two_sided  
    else:
        p_value_right = 1 - 0.5 * p_value_two_sided  
        p_value_left = 0.5 * p_value_two_sided 
    
    return {
        'lrt_stat': lrt_stat,
        'p_value_two_sided': p_value_two_sided,
        'p_value_right': p_value_right,
        'p_value_left': p_value_left
    }

# Запуск теста для μ
mu_test = test_log_normal_mu(log_X, mu_hypothesis=2.09)
print("#2   ",f"Логнормальное μ:")
print(f"  LRT = {mu_test['lrt_stat']:.4f}")
print(f"  Двусторонний p-value = {mu_test['p_value_two_sided']:.4f}")
print(f"  Правосторонний p-value (μ > 2.09) = {mu_test['p_value_right']:.4f}")
print(f"  Левосторонний p-value (μ < 2.09) = {mu_test['p_value_left']:.4f}")


def test_log_normal_sigma(data_log, sigma_hypothesis=0.19):
    """Тест отношения правдоподобия (LRT) для σ² логнормального распределения."""
    mu_mle = np.mean(data_log)
    sigma_mle = np.std(data_log, ddof=0)
    
    ll_null = np.sum(norm.logpdf(data_log, loc=mu_mle, scale=sigma_hypothesis))

    ll_alt = np.sum(norm.logpdf(data_log, loc=mu_mle, scale=sigma_mle))
    
    # LRT-статистика и p-value (двусторонний тест)
    lrt_stat = -2 * (ll_null - ll_alt)
    p_value_two_sided = 1 - chi2.cdf(lrt_stat, df=1)
    
    # Односторонние тесты:
    if sigma_mle > sigma_hypothesis:
        p_value_right = 0.5 * p_value_two_sided  
        p_value_left = 1 - 0.5 * p_value_two_sided 
    else:
        p_value_right = 1 - 0.5 * p_value_two_sided 
        p_value_left = 0.5 * p_value_two_sided 
    
    return {
        'lrt_stat': lrt_stat,
        'p_value_two_sided': p_value_two_sided,
        'p_value_right': p_value_right,
        'p_value_left': p_value_left
    }

# Запуск теста для σ²
sigma_test = test_log_normal_sigma(log_X, sigma_hypothesis=0.19)
print("\nЛогнормальное σ²:")
print(f"  LRT = {sigma_test['lrt_stat']:.4f}")
print(f"  Двусторонний p-value = {sigma_test['p_value_two_sided']:.4f}")
print(f"  Правосторонний p-value (σ > 0.19) = {sigma_test['p_value_right']:.4f}")
print(f"  Левосторонний p-value (σ < 0.19) = {sigma_test['p_value_left']:.4f}")


#### для Y
log_Y = np.log(Y)

def test_log_normal_mu(data_log, mu_hypothesis=-0.69):
    """Тест отношения правдоподобия (LRT) для μ логнормального распределения."""
    mu_mle = np.mean(data_log)
    sigma_mle = np.std(data_log, ddof=0)
    
    ll_null = np.sum(norm.logpdf(data_log, loc=mu_hypothesis, scale=sigma_mle))
    
    ll_alt = np.sum(norm.logpdf(data_log, loc=mu_mle, scale=sigma_mle))
    
    # LRT-статистика и p-value (двусторонний тест)
    lrt_stat = -2 * (ll_null - ll_alt)
    p_value_two_sided = 1 - chi2.cdf(lrt_stat, df=1)
    
    # Односторонние тесты:
    if mu_mle > mu_hypothesis:
        p_value_right = 0.5 * p_value_two_sided  
        p_value_left = 1 - 0.5 * p_value_two_sided  
    else:
        p_value_right = 1 - 0.5 * p_value_two_sided  
        p_value_left = 0.5 * p_value_two_sided 
    
    return {
        'lrt_stat': lrt_stat,
        'p_value_two_sided': p_value_two_sided,
        'p_value_right': p_value_right,
        'p_value_left': p_value_left
    }

# Запуск теста для μ
mu_test = test_log_normal_mu(log_Y, mu_hypothesis=-0.69)
print(f"Логнормальное μ:")
print(f"  LRT = {mu_test['lrt_stat']:.4f}")
print(f"  Двусторонний p-value = {mu_test['p_value_two_sided']:.4f}")
print(f"  Правосторонний p-value (μ > -0.69) = {mu_test['p_value_right']:.4f}")
print(f"  Левосторонний p-value (μ < -0.69) = {mu_test['p_value_left']:.4f}")


def test_log_normal_sigma(data_log, sigma_hypothesis=0.35):
    """Тест отношения правдоподобия (LRT) для σ² логнормального распределения."""
    mu_mle = np.mean(data_log)
    sigma_mle = np.std(data_log, ddof=0)
    
    ll_null = np.sum(norm.logpdf(data_log, loc=mu_mle, scale=sigma_hypothesis))

    ll_alt = np.sum(norm.logpdf(data_log, loc=mu_mle, scale=sigma_mle))
    
    # LRT-статистика и p-value (двусторонний тест)
    lrt_stat = -2 * (ll_null - ll_alt)
    p_value_two_sided = 1 - chi2.cdf(lrt_stat, df=1)
    
    # Односторонние тесты:
    if sigma_mle > sigma_hypothesis:
        p_value_right = 0.5 * p_value_two_sided  
        p_value_left = 1 - 0.5 * p_value_two_sided 
    else:
        p_value_right = 1 - 0.5 * p_value_two_sided 
        p_value_left = 0.5 * p_value_two_sided 
    
    return {
        'lrt_stat': lrt_stat,
        'p_value_two_sided': p_value_two_sided,
        'p_value_right': p_value_right,
        'p_value_left': p_value_left
    }

# Запуск теста для σ²
sigma_test = test_log_normal_sigma(log_Y, sigma_hypothesis=0.35)
print("\nЛогнормальное σ²:")
print(f"  LRT = {sigma_test['lrt_stat']:.4f}")
print(f"  Двусторонний p-value = {sigma_test['p_value_two_sided']:.4f}")
print(f"  Правосторонний p-value (σ > 0.35) = {sigma_test['p_value_right']:.4f}")
print(f"  Левосторонний p-value (σ < 0.35) = {sigma_test['p_value_left']:.4f}")


##################################################################################################
# 3 часть задания
##################################################################################################

from scipy.stats import kstest
from scipy.stats import  chi2

# X

mu, sigma = 2.09, 0.19

ks_stat, ks_pvalue = kstest(X, 'lognorm', args=(sigma, 0, np.exp(mu)))
print("#3   ", f"K-S статистика: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")

percentiles = np.linspace(0, 100, 11)  
bins = np.percentile(X, percentiles)

observed, _ = np.histogram(X, bins=bins)

shape = sigma
scale = np.exp(mu)

cdf_values = lognorm.cdf(bins, shape, scale=scale)
expected_probs = np.diff(cdf_values)
expected = expected_probs * len(X) 

print("Сумма observed: {:.4f}".format(np.sum(observed)))
print("Сумма expected: {:.4f}".format(np.sum(expected)))

valid = expected >= 5
while not np.all(valid):

    idx = np.argmin(valid)
    
    if idx == 0:
        merge_with = 1
    elif idx == len(valid) - 1:
        merge_with = idx - 1
    else:
        merge_with = idx - 1  
    
    observed[merge_with] += observed[idx]
    observed = np.delete(observed, idx)

    expected[merge_with] += expected[idx]
    expected = np.delete(expected, idx)
    
    valid = expected >= 5

expected = expected * (np.sum(observed) / np.sum(expected))

chi2_stat, chi2_pvalue = chisquare(observed, expected)
print(f"χ² статистика: {chi2_stat:.4f}, p-value: {chi2_pvalue:.4f}")



#############################################################################
# Y
mu, sigma = -0.69, 0.35

ks_stat, ks_pvalue = kstest(Y, 'lognorm', args=(sigma, 0, np.exp(mu)))
print("#3   ", f"K-S статистика: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")

percentiles = np.linspace(0, 100, 11)  
bins = np.percentile(Y, percentiles)

observed, _ = np.histogram(Y, bins=bins)

shape = sigma
scale = np.exp(mu)

cdf_values = lognorm.cdf(bins, shape, scale=scale)
expected_probs = np.diff(cdf_values)
expected = expected_probs * len(Y) 

print("Сумма observed: {:.4f}".format(np.sum(observed)))
print("Сумма expected: {:.4f}".format(np.sum(expected)))

valid = expected >= 5
while not np.all(valid):

    idx = np.argmin(valid)
    
    if idx == 0:
        merge_with = 1
    elif idx == len(valid) - 1:
        merge_with = idx - 1
    else:
        merge_with = idx - 1  
    
    observed[merge_with] += observed[idx]
    observed = np.delete(observed, idx)

    expected[merge_with] += expected[idx]
    expected = np.delete(expected, idx)
    
    valid = expected >= 5

expected = expected * (np.sum(observed) / np.sum(expected))

chi2_stat, chi2_pvalue = chisquare(observed, expected)
print(f"χ² статистика: {chi2_stat:.4f}, p-value: {chi2_pvalue:.4f}")

######################################################################################################
# 4
######################################################################################################

corr, p_value = stats.spearmanr(X, Y)
print("#4  ", f"Коэффициент корреляции Спирмена: {corr:.4f}, p-value: {p_value:.4f}")

######################################################################################################
# 5
######################################################################################################

from scipy.stats import pearsonr

r, p_value = pearsonr(X, Y)

print("#5  ", f"Коэффициент корреляции Пирсона: r = {r:.4f}", f"p-value: {p_value:.4f}")




params = {
    'X': {'mu': 2.09, 'sigma': 0.19, 'n': 1599},
    'Y': {'mu': -0.69, 'sigma': 0.35, 'n': 1599}
}

alpha = 0.05
lrt_critical = chi2.ppf(1 - alpha, df=1)
print(f"Критическое значение LRT (α={alpha}): {lrt_critical:.4f}")

def ks_critical_value(n, alpha=0.05):
    return 1.36 / np.sqrt(n) 

def chi2_critical_value(k, p=2, alpha=0.05):
    df = k - 1 - p 
    return chi2.ppf(1 - alpha, df)

results = {}
for var in ['X', 'Y']:
    n = params[var]['n']
    results[var] = {
        'LRT_mu_crit': lrt_critical,
        'LRT_sigma_crit': lrt_critical,
        'KS_crit': ks_critical_value(n),
        'Chi2_crit': chi2_critical_value(8 if var == 'X' else 9)
    }


print("\nКритические значения для каждой переменной:")
for var in results:
    print(f"\n{var}:")
    for test in results[var]:
        print(f"{test}: {results[var][test]:.4f}")

observed_stats = {
    'X': {
        'LRT_mu': 2.7226,
        'LRT_sigma': 7.6215,
        'KS': 0.0601,
        'Chi2': 83.9857,
        'k': 8  
    },
    'Y': {
        'LRT_mu': 0.9277,
        'LRT_sigma': 0.3320,
        'KS': 0.3420,
        'Chi2': 0.3320,
        'k': 9 
    }
}


print("\nПроверка гипотез:")
for var in ['X', 'Y']:
    print(f"\n--- {var} ---")
    stats = observed_stats[var]
    crits = results[var]

    decision = "Отвергаем" if stats['LRT_mu'] > crits['LRT_mu_crit'] else "Не отвергаем"
    print(f"LRT для μ: {stats['LRT_mu']:.4f} > {crits['LRT_mu_crit']:.4f}? {decision}")

    decision = "Отвергаем" if stats['LRT_sigma'] > crits['LRT_sigma_crit'] else "Не отвергаем"
    print(f"LRT для σ²: {stats['LRT_sigma']:.4f} > {crits['LRT_sigma_crit']:.4f}? {decision}")

    decision = "Отвергаем" if stats['KS'] > crits['KS_crit'] else "Не отвергаем"
    print(f"K-S: {stats['KS']:.4f} > {crits['KS_crit']:.4f}? {decision}")

    chi2_crit = chi2_critical_value(stats['k'])
    decision = "Отвергаем" if stats['Chi2'] > chi2_crit else "Не отвергаем"
    print(f"χ²: {stats['Chi2']:.4f} > {chi2_crit:.4f}? {decision}")