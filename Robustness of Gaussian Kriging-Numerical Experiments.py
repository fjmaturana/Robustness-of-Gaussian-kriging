###########################################################################
# Robustness of Gaussian Kriging: Numerical Experiments
#
# Companion simulation code for the paper:
#
# "Robustness of Gaussian Kriging under Non-Gaussian Random Fields"
#
# This script reproduces the numerical experiments reported in the paper,
# including:
#   1. Exact robustness under Gaussian and Student-t scale-mixture models.
#   2. Perturbation analysis using symmetric Gaussian mixtures.
#   3. Gaussian copula experiments comparing normal-score and naive
#      Gaussian prediction.
#   4. Automatic generation of the LaTeX tables reported in the manuscript.
#
# Author: Francisco Maturana
# Date: June 2025
###########################################################################

import numpy as np
from scipy.stats import norm

np.random.seed(123)

# ----------------------
# Common spatial setup
# ----------------------

n = 20
Delta = 1.0
x = np.arange(1, n + 1) * Delta
x0 = 0.0
sigma2 = 1.0
rho = 5.0

def cov_exponential(h, sigma2=1.0, rho=5.0):
    return sigma2 * np.exp(-np.abs(h) / rho)

locations = np.concatenate(([x0], x))
m = n + 1
Sigma = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        h = locations[i] - locations[j]
        Sigma[i, j] = cov_exponential(h, sigma2=sigma2, rho=rho)

L = np.linalg.cholesky(Sigma)

Sigma_0Z = Sigma[0, 1:].reshape(1, -1)
Sigma_ZZ = Sigma[1:, 1:]
Sigma_ZZ_inv = np.linalg.inv(Sigma_ZZ)
Sigma_0Z_Sigma_ZZ_inv = (Sigma_0Z @ Sigma_ZZ_inv).ravel()
a = Sigma_0Z_Sigma_ZZ_inv

def gaussian_kriging_predict(z_obs):
    return a @ z_obs

# ----------------------
# 1. Exact robustness: Gaussian and Student-t
# ----------------------

def simulate_gaussian(N):
    z = np.random.normal(size=(N, m))
    return z @ L.T

def simulate_student_t_scale_mixture(N, nu=5.0):
    z = np.random.normal(size=(N, m))
    Y = z @ L.T
    U = np.random.chisquare(df=nu, size=N)
    W = nu / U
    E_W = nu / (nu - 2.0)
    W_scaled = W / E_W
    sqrt_W = np.sqrt(W_scaled).reshape(-1, 1)
    return sqrt_W * Y

N = 40000

# Gaussian model
X_gauss = simulate_gaussian(N)
Z0_gauss = X_gauss[:, 0]
Z_gauss = X_gauss[:, 1:]
ZhatG_gauss = np.array([gaussian_kriging_predict(z) for z in Z_gauss])
R_Zstar_gauss = np.mean((ZhatG_gauss - Z0_gauss) ** 2)
R_ZhatG_gauss = R_Zstar_gauss

# Student-t scale mixture
X_t = simulate_student_t_scale_mixture(N, nu=5.0)
Z0_t = X_t[:, 0]
Z_t = X_t[:, 1:]
ZhatG_t = np.array([gaussian_kriging_predict(z) for z in Z_t])
R_Zstar_t = np.mean((ZhatG_t - Z0_t) ** 2)
R_ZhatG_t = R_Zstar_t

# ----------------------
# 2. Perturbation: symmetric Gaussian mixture, mild and strong
# ----------------------

Sigma_ZZ = Sigma[1:, 1:]

def gaussian_density(Z, mean, Sigma_ZZ):
    n_dim = Sigma_ZZ.shape[0]
    diff = Z - mean.reshape(1, -1)
    inv = np.linalg.inv(Sigma_ZZ)
    det = np.linalg.det(Sigma_ZZ)
    mah = np.einsum("ij,jk,ik->i", diff, inv, diff)
    coef = 1.0 / np.sqrt((2 * np.pi) ** n_dim * det)
    return coef * np.exp(-0.5 * mah)

def simulate_mixture(N, mu):
    comps = np.random.choice([-1, 1], size=N)
    means = comps.reshape(-1, 1) * mu.reshape(1, -1)
    z = np.random.normal(size=(N, m))
    X = means + z @ L.T
    return X

def conditional_mean_mixture(Z_samples, mu, Sigma_0Z_Sigma_ZZ_inv, Sigma_ZZ):
    mu0 = mu[0]
    muZ = mu[1:]
    Z = Z_samples
    m_plus = mu0 + (Sigma_0Z_Sigma_ZZ_inv @ (Z.T - muZ.reshape(-1, 1))).ravel()
    m_minus = -mu0 + (Sigma_0Z_Sigma_ZZ_inv @ (Z.T + muZ.reshape(-1, 1))).ravel()
    f_plus = gaussian_density(Z, muZ, Sigma_ZZ)
    f_minus = gaussian_density(Z, -muZ, Sigma_ZZ)
    w_plus = f_plus / (f_plus + f_minus)
    w_minus = 1.0 - w_plus
    Zstar = w_plus * m_plus + w_minus * m_minus
    return Zstar

# Strong perturbation
mu_strong = np.zeros(m)
mu_strong[0] = 1.0
mu_strong[5] = 0.5

X_mix_strong = simulate_mixture(N, mu_strong)
Z0_mix_strong = X_mix_strong[:, 0]
Z_mix_strong = X_mix_strong[:, 1:]
ZhatG_mix_strong = np.array([gaussian_kriging_predict(z) for z in Z_mix_strong])
Zstar_mix_strong = conditional_mean_mixture(Z_mix_strong, mu_strong, Sigma_0Z_Sigma_ZZ_inv, Sigma_ZZ)
R_Zstar_mix_strong = np.mean((Zstar_mix_strong - Z0_mix_strong) ** 2)
R_ZhatG_mix_strong = np.mean((ZhatG_mix_strong - Z0_mix_strong) ** 2)

# Mild perturbation: scaled down mu
scale = 0.25
mu_mild = scale * mu_strong

X_mix_mild = simulate_mixture(N, mu_mild)
Z0_mix_mild = X_mix_mild[:, 0]
Z_mix_mild = X_mix_mild[:, 1:]
ZhatG_mix_mild = np.array([gaussian_kriging_predict(z) for z in Z_mix_mild])
Zstar_mix_mild = conditional_mean_mixture(Z_mix_mild, mu_mild, Sigma_0Z_Sigma_ZZ_inv, Sigma_ZZ)
R_Zstar_mix_mild = np.mean((Zstar_mix_mild - Z0_mix_mild) ** 2)
R_ZhatG_mix_mild = np.mean((ZhatG_mix_mild - Z0_mix_mild) ** 2)

# ----------------------
# 3. Gaussian copula: normal-score vs naive predictor
# ----------------------

Sigma_Y = Sigma / sigma2
L_Y = np.linalg.cholesky(Sigma_Y)
mu_F = 0.0
sigma_F = 1.0

def simulate_gaussian_field(N):
    z = np.random.normal(size=(N, m))
    return z @ L_Y.T

def F_inv(u):
    return np.exp(mu_F + sigma_F * norm.ppf(u))

def simulate_copula(N):
    Y = simulate_gaussian_field(N)
    U = norm.cdf(Y)
    Z = F_inv(U)
    return Y, Z

Y, Z = simulate_copula(N)
Y0 = Y[:, 0]
Y_obs = Y[:, 1:]
Z0_cop = Z[:, 0]
Z_obs = Z[:, 1:]

YhatG = np.array([gaussian_kriging_predict(y) for y in Y_obs])
Z_NS = F_inv(norm.cdf(YhatG))
R_Z_NS = np.mean((Z_NS - Z0_cop) ** 2)

Zhat_naive = np.array([gaussian_kriging_predict(z) for z in Z_obs])
R_Z_naive = np.mean((Zhat_naive - Z0_cop) ** 2)

# ----------------------
# 4. LaTeX tables
# ----------------------

latex_table_1 = """\\begin{table}
	\\centering
	\\begin{tabular}{lcc}
		\\toprule
		Model & $R(Z^\\star;x_0)$ & $R(\\widehat Z_G;x_0)$ \\\\
		\\midrule
		Gaussian & %.3f & %.3f \\\\
		Student--$t$ scale mixture & %.3f & %.3f \\\\
		\\bottomrule
	\\end{tabular}
	\\label{tab:numerical}
\\end{table}""" % (
    R_Zstar_gauss,
    R_ZhatG_gauss,
    R_Zstar_t,
    R_ZhatG_t,
)

latex_table_2 = """\\begin{table}
	\\centering
	\\begin{tabular}{lccc}
		\\toprule
		Experiment & Predictor & $R(\\cdot;x_0)$ & Excess risk \\\\
		\\midrule
		Gaussian mixture (mild perturbation) & True conditional expectation $Z^\\star$ & %.3f & -- \\\\
		Gaussian mixture (mild perturbation) & Gaussian predictor $\\widehat Z_G$ & %.3f & %.3f \\\\
		Gaussian mixture (strong perturbation) & True conditional expectation $Z^\\star$ & %.3f & -- \\\\
		Gaussian mixture (strong perturbation) & Gaussian predictor $\\widehat Z_G$ & %.3f & %.3f \\\\
		Gaussian copula (lognormal marginals) & Normal-score predictor $Z_\\mathrm{NS}$ & %.3f & -- \\\\
		Gaussian copula (lognormal marginals) & Naive Gaussian predictor on $Z$ & %.3f & %.3f \\\\
		\\bottomrule
	\\end{tabular}
	\\label{tab:additional-numerical}
\\end{table}""" % (
    R_Zstar_mix_mild,
    R_ZhatG_mix_mild,
    R_ZhatG_mix_mild - R_Zstar_mix_mild,
    R_Zstar_mix_strong,
    R_ZhatG_mix_strong,
    R_ZhatG_mix_strong - R_Zstar_mix_strong,
    R_Z_NS,
    R_Z_naive,
    R_Z_naive - R_Z_NS,
)

print("Exact robustness (Gaussian, Student-t):")
print("  Gaussian      ", R_Zstar_gauss, R_ZhatG_gauss)
print("  Student-t     ", R_Zstar_t, R_ZhatG_t)

print("\nMixture perturbation (mild):")
print("  R(Z*; x0)   =", R_Zstar_mix_mild)
print("  R(Z_G; x0)  =", R_ZhatG_mix_mild)
print("  Excess      =", R_ZhatG_mix_mild - R_Zstar_mix_mild)

print("\nMixture perturbation (strong):")
print("  R(Z*; x0)   =", R_Zstar_mix_strong)
print("  R(Z_G; x0)  =", R_ZhatG_mix_strong)
print("  Excess      =", R_ZhatG_mix_strong - R_Zstar_mix_strong)

print("\nCopula (normal-score vs naive):")
print("  R(Z_NS; x0)     =", R_Z_NS)
print("  R(Z_naive; x0)  =", R_Z_naive)
print("  Excess naive    =", R_Z_naive - R_Z_NS)

print("\n\nLaTeX table 1:\n")
print(latex_table_1)
print("\n\nLaTeX table 2:\n")
print(latex_table_2)