import numpy as np

def multivariate_kl_divergence(mu_p, Cov_p, mu_q, Cov_q):
    d = len(mu_p)
    diff = mu_q - mu_p

    log_det = np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p))
    trace = np.trace(np.linalg.solve(Cov_q, Cov_p))
    mean_term = diff @ np.linalg.solve(Cov_q, diff)

    return float(0.5 * (log_det - d + trace + mean_term))