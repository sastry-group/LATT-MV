import numpy as np
from scipy.optimize import differential_evolution
from numba import jit

@jit(nopython=True)
def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

@jit(nopython=True)
def log_likelihood_mixture(params, X):
    mu1, mu2, sigma = params
    log_likelihood_sum = 0.0
    for x, y in X:
        likelihood = max(normal_pdf(x, mu1, sigma) * normal_pdf(y, mu2, sigma),
                         normal_pdf(y, mu1, sigma) * normal_pdf(x, mu2, sigma))
        log_likelihood_sum += np.log(likelihood + 1e-10)
    return -log_likelihood_sum

@jit(nopython=True)
def log_likelihood_single(X, mu, sigma):
    log_likelihood_sum = 0.0
    for x, y in X:
        likelihood = normal_pdf(x, mu, sigma) * normal_pdf(y, mu, sigma)
        log_likelihood_sum += np.log(likelihood + 1e-10)
    return -log_likelihood_sum

def gaussian_mixture_analysis(X):
    # Remove any rows with NaN values
    mask = np.isnan(X).any(axis=1)
    X = X[~mask]

    # Optimize parameters for mixture model
    bounds = [(np.min(X), np.max(X)), (np.min(X), np.max(X)), (1e-6, np.std(X)*2)]
    result = differential_evolution(log_likelihood_mixture, bounds, args=(X,), strategy='best1bin', 
                                    popsize=20, tol=1e-7, mutation=(0.5, 1), recombination=0.7,
                                    maxiter=1000, disp=False)
    mixture_params = result.x

    # Calculate AIC for mixture model
    mixture_ll = -log_likelihood_mixture(mixture_params, X)
    mixture_aic = 2 * 3 - 2 * mixture_ll  # 3 parameters: mu1, mu2, sigma

    # Calculate AIC for single Gaussian model
    single_mu = np.mean(X)
    single_sigma = np.std(X)
    single_ll = -log_likelihood_single(X, single_mu, single_sigma)
    single_aic = 2 * 2 - 2 * single_ll  # 2 parameters: mu, sigma

    # Determine which model to use based on AIC difference
    if mixture_aic < single_aic - 40:
        return mixture_params[0], mixture_params[1]  # Return mu1 and mu2 from mixture model
    else:
        return single_mu, single_mu  # Return mu from single Gaussian model twice for consistency