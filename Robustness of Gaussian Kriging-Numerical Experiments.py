"""Numerical experiments for "Prediction Risk of Gaussian Kriging under Covariance-Preserving Non-Gaussian Dependence".

The script reproduces the five numerical tables used by the manuscript:

1. Gaussian and covariance-normalised Student-t exact robustness.
2. Covariance-preserving symmetric Gaussian-mixture perturbations.
3. Lognormal Gaussian-copula prediction.
4. Sensitivity to sampling geometry.
5. Sensitivity to correlation range and nugget ratio.

It also fits the descriptive, unweighted log-log regression used to check
the eighth-order mixture excess-risk rate. Results are printed as complete
LaTeX tables and written as CSV, LaTeX, and diagnostic text files to the
specified output directory, which defaults to ``results``. No empirical
data are required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import linregress, t as student_t_distribution


Array = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Default run parameters
# ---------------------------------------------------------------------------
# Running
#
#     python Robustness_of_Gaussian_Kriging_Numerical_Experiments.py
#
# is equivalent to running the script with:
#
#     --n-sim 40000 --seed 123 --output-dir results
#
# Command-line options may still be supplied to override these defaults.
DEFAULT_N_SIM = 40_000
DEFAULT_SEED = 123
DEFAULT_OUTPUT_DIR = Path("results")


@dataclass(frozen=True)
class RiskEstimate:
    risk: float
    mcse: float


@dataclass(frozen=True)
class ExcessEstimate:
    excess: float
    mcse: float
    relative: float
    relative_mcse: float


def covariance_matrix(locations: Array, rho: float, nugget_ratio: float = 0.0) -> Array:
    """Exponential covariance with unit total sill."""
    if rho <= 0:
        raise ValueError("rho must be positive")
    if not 0 <= nugget_ratio < 1:
        raise ValueError("nugget_ratio must lie in [0, 1)")
    distances = np.abs(locations[:, None] - locations[None, :])
    covariance = (1.0 - nugget_ratio) * np.exp(-distances / rho)
    covariance += nugget_ratio * np.eye(locations.size)
    return covariance


def kriging_coefficients(sigma: Array) -> tuple[Array, float]:
    """Simple-kriging coefficients and variance for target component zero."""
    sigma_zz = sigma[1:, 1:]
    sigma_z0 = sigma[1:, 0]
    coefficients = np.linalg.solve(sigma_zz, sigma_z0)
    variance = float(sigma[0, 0] - sigma[0, 1:] @ coefficients)
    return coefficients, variance


def risk_estimate(prediction: Array, target: Array) -> RiskEstimate:
    squared_error = np.square(prediction - target)
    return RiskEstimate(
        risk=float(np.mean(squared_error)),
        mcse=float(np.std(squared_error, ddof=1) / np.sqrt(squared_error.size)),
    )


def paired_excess_estimate(
    prediction: Array, benchmark: Array, target: Array
) -> ExcessEstimate:
    excess_components = np.square(prediction - benchmark)
    benchmark_components = np.square(benchmark - target)

    excess = float(np.mean(excess_components))
    benchmark_risk = float(np.mean(benchmark_components))
    relative = excess / benchmark_risk

    mcse = float(
        np.std(excess_components, ddof=1)
        / np.sqrt(excess_components.size)
    )

    ratio_influence = excess_components - relative * benchmark_components
    relative_mcse = float(
        np.std(ratio_influence, ddof=1)
        / (
            np.sqrt(ratio_influence.size)
            * benchmark_risk
        )
    )

    return ExcessEstimate(
        excess=excess,
        mcse=mcse,
        relative=relative,
        relative_mcse=relative_mcse,
    )


def relative_frobenius_error(estimate: Array, target: Array) -> float:
    """Return ||estimate-target||_F / ||target||_F."""
    return float(
        np.linalg.norm(estimate - target, ord="fro")
        / np.linalg.norm(target, ord="fro")
    )

def simulate_gaussian(rng: np.random.Generator, n_sim: int, sigma: Array) -> Array:
    chol = np.linalg.cholesky(sigma)
    return rng.normal(size=(n_sim, sigma.shape[0])) @ chol.T


def simulate_student_t(
    rng: np.random.Generator, n_sim: int, sigma: Array, nu: float
) -> Array:
    if nu <= 2:
        raise ValueError("nu must exceed two for a finite covariance")
    gaussian = simulate_gaussian(rng, n_sim, sigma)
    chi_square = rng.chisquare(df=nu, size=n_sim)
    scale = np.sqrt((nu - 2.0) / chi_square)
    return gaussian * scale[:, None]


def normalised_shift(sigma: Array) -> tuple[Array, float, float]:
    """Return the normalised shift, boundary, and experimental maximum."""
    shift = np.zeros(sigma.shape[0])
    shift[0] = 1.0
    shift[1] = -1.0

    b = shift / np.sqrt(shift @ np.linalg.solve(sigma, shift))

    eta_boundary = 1.0 / np.sqrt(
        b @ np.linalg.solve(sigma, b)
    )
    eta_max = 0.95 * eta_boundary

    return b, float(eta_boundary), float(eta_max)


def mixture_experiment(
    rng: np.random.Generator,
    n_sim: int,
    sigma: Array,
    eta_fraction: float,
) -> dict[str, float]:
    """Run one covariance-preserving symmetric Gaussian-mixture experiment."""
    b, eta_boundary, eta_max = normalised_shift(sigma)
    eta = eta_fraction * eta_max
    mu = eta * b
    omega = sigma - np.outer(mu, mu)
    min_eigenvalue = float(np.linalg.eigvalsh(omega)[0])
    if min_eigenvalue <= 0:
        raise ValueError(f"Omega is not positive definite: {min_eigenvalue}")

    components = rng.choice(np.array([-1.0, 1.0]), size=n_sim)
    errors = simulate_gaussian(rng, n_sim, omega)
    samples = components[:, None] * mu[None, :] + errors
    target = samples[:, 0]
    observations = samples[:, 1:]

    gaussian_weights, kriging_variance = kriging_coefficients(sigma)
    gaussian_prediction = observations @ gaussian_weights

    # Compute the exact conditional mean from Theorem 3 in the manuscript.
    # Partition Sigma and b according to target and observations.
    b0 = b[0]
    v = b[1:]
    c = sigma[1:, 0]
    b_matrix = sigma[1:, 1:]
    b_inv_v = np.linalg.solve(b_matrix, v)
    q = float(v @ b_inv_v)
    delta = float(b0 - c @ b_inv_v)
    linear_score = observations @ b_inv_v
    denominator = 1.0 - eta**2 * q
    if denominator <= 0.0:
        raise ValueError("Mixture conditional-mean denominator is not positive")

    exact_correction = (
        delta
        / denominator
        * (
            eta * np.tanh(eta * linear_score / denominator)
            - eta**2 * linear_score
        )
    )
    conditional_prediction = gaussian_prediction + exact_correction

    # Independently recompute the same conditional mean by averaging the
    # component-specific Gaussian conditional means using posterior weights.
    # The maximum difference is retained as a numerical validation diagnostic.
    omega_zz = omega[1:, 1:]
    omega_z0 = omega[1:, 0]
    component_weights = np.linalg.solve(omega_zz, omega_z0)
    mu0 = mu[0]
    muz = mu[1:]
    mean_plus = mu0 + (observations - muz) @ component_weights
    mean_minus = -mu0 + (observations + muz) @ component_weights
    discriminant = 2.0 * observations @ np.linalg.solve(omega_zz, muz)
    posterior_plus = expit(discriminant)
    component_conditional_prediction = (
        posterior_plus * mean_plus + (1.0 - posterior_plus) * mean_minus
    )
    maximum_predictor_discrepancy = float(
        np.max(np.abs(conditional_prediction - component_conditional_prediction))
    )

    risk_conditional = risk_estimate(conditional_prediction, target)
    risk_gaussian = risk_estimate(gaussian_prediction, target)
    excess = paired_excess_estimate(
        gaussian_prediction, conditional_prediction, target
    )

    sample_mean_norm = float(np.linalg.norm(np.mean(samples, axis=0)))
    covariance_error = relative_frobenius_error(
        np.cov(samples, rowvar=False), sigma
    )
    exact_mean = 0.5 * mu + 0.5 * (-mu)
    exact_covariance = omega + np.outer(mu, mu)
    return {
        "eta_fraction": eta_fraction,
        "eta": eta,
        "eta_boundary": eta_boundary,
        "eta_max": eta_max,
        "min_eigenvalue": min_eigenvalue,
        "kriging_variance": kriging_variance,
        "risk_conditional": risk_conditional.risk,
        "mcse_conditional": risk_conditional.mcse,
        "risk_gaussian": risk_gaussian.risk,
        "mcse_gaussian": risk_gaussian.mcse,
        "excess": excess.excess,
        "mcse_excess": excess.mcse,
        "relative_excess": excess.relative,
        "mcse_relative_excess": excess.relative_mcse,
        "sample_mean_norm": sample_mean_norm,
        "relative_covariance_error": covariance_error,
        "maximum_predictor_discrepancy": maximum_predictor_discrepancy,
        "exact_mean_norm": float(np.linalg.norm(exact_mean)),
        "exact_covariance_error": relative_frobenius_error(
            exact_covariance, sigma
        ),
    }


def exact_robustness_experiment(
    rng: np.random.Generator, n_sim: int, sigma: Array, nu: float = 5.0
) -> list[dict[str, float | str]]:
    weights, theoretical_risk = kriging_coefficients(sigma)
    output: list[dict[str, float | str]] = []
    for name, simulator in (
        ("Gaussian", lambda: simulate_gaussian(rng, n_sim, sigma)),
        (f"Student-t, nu={nu:g}", lambda: simulate_student_t(rng, n_sim, sigma, nu)),
    ):
        samples = simulator()
        estimate = risk_estimate(samples[:, 1:] @ weights, samples[:, 0])
        output.append(
            {
                "model": name,
                "theoretical_risk": theoretical_risk,
                "simulated_risk": estimate.risk,
                "mcse": estimate.mcse,
            }
        )
    return output


def copula_experiment(
    rng: np.random.Generator, n_sim: int, sigma: Array
) -> list[dict[str, float | str]]:
    latent = simulate_gaussian(rng, n_sim, sigma)
    y_observed = latent[:, 1:]
    z = np.exp(latent)
    target = z[:, 0]
    z_observed = z[:, 1:]

    latent_weights, conditional_variance = kriging_coefficients(sigma)
    latent_mean = y_observed @ latent_weights
    conditional_mean = np.exp(latent_mean + 0.5 * conditional_variance)
    plug_in_median = np.exp(latent_mean)

    original_covariance = np.e * (np.exp(sigma) - 1.0)
    original_weights, _ = kriging_coefficients(original_covariance)
    original_mean = np.exp(0.5)
    linear_prediction = original_mean + (z_observed - original_mean) @ original_weights

    predictors = (
        ("Conditional mean", conditional_mean),
        ("Plug-in conditional median", plug_in_median),
        ("Original-scale linear predictor", linear_prediction),
    )
    output: list[dict[str, float | str]] = []
    for name, prediction in predictors:
        estimate = risk_estimate(prediction, target)
        excess = paired_excess_estimate(prediction, conditional_mean, target)
        output.append(
            {
                "predictor": name,
                "risk": estimate.risk,
                "mcse": estimate.mcse,
                "excess": excess.excess,
                "mcse_excess": excess.mcse,
                "relative_excess": excess.relative,
                "mcse_relative_excess": excess.relative_mcse,            
            }
        )
    return output


def make_designs() -> dict[str, Array]:
    return {
        "Near-target": 0.25 * np.arange(1, 21, dtype=float),
        "Regular": np.arange(1, 21, dtype=float),
        "Clustered": np.array(
            [
                0.25, 0.50, 0.75, 1.00, 1.25,
                1.50, 1.75, 2.00, 2.25, 2.50,
                5.0, 6.5, 8.0, 9.5, 11.0,
                12.5, 14.0, 15.5, 17.0, 18.5,
            ]
        ),
        "Sparse": 2.0 * np.arange(1, 21, dtype=float),
    }


def geometry_experiments(
    rng: np.random.Generator, n_sim: int
) -> tuple[list[dict[str, float | str]], list[dict[str, float]]]:
    designs = make_designs()
    geometry: list[dict[str, float | str]] = []
    for name, observed_locations in designs.items():
        locations = np.concatenate(([0.0], observed_locations))
        sigma = covariance_matrix(locations, rho=5.0, nugget_ratio=0.1)
        result = mixture_experiment(rng, n_sim, sigma, eta_fraction=0.7)
        result.update(
            {
                "design": name,
                "nearest_distance": float(np.min(observed_locations)),
                "rho": 5.0,
                "nugget_ratio": 0.1,
            }
        )
        geometry.append(result)

    regular_locations = np.concatenate(([0.0], designs["Regular"]))
    parameter_pairs = [
        (2.0, 0.1),
        (5.0, 0.1),
        (10.0, 0.1),
        (5.0, 0.0),
        (5.0, 0.3),
    ]
    sensitivity: list[dict[str, float]] = []
    for range_value, nugget_value in parameter_pairs:
        sigma = covariance_matrix(regular_locations, range_value, nugget_value)
        result = mixture_experiment(rng, n_sim, sigma, eta_fraction=0.7)
        result.update({"rho": range_value, "nugget_ratio": nugget_value})
        sensitivity.append(result)
    return geometry, sensitivity


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_latex_integer(value: int) -> str:
    """Format an integer with LaTeX-safe thousands separators."""
    return f"{value:,}".replace(",", "{,}")


def fmt_percentage(proportion: float, digits: int = 2) -> str:
    """Format a proportion as a percentage without rounding small values to zero."""
    percentage = 100.0 * proportion
    display_threshold = 0.5 * 10.0 ** (-digits)
    if 0.0 < abs(percentage) < display_threshold:
        return rf"$<{display_threshold:.{digits + 1}f}\%$"
    return f"{percentage:.{digits}f}\\%"


def fmt_risk(row: dict[str, float | str], risk_key: str, mcse_key: str) -> str:
    return f"{float(row[risk_key]):.4f} ({float(row[mcse_key]):.4f})"


def latex_exact(rows: list[dict[str, float | str]]) -> str:
    body = []
    labels = ["Gaussian", r"Student--$t$, $\nu=5$"]
    for label, row in zip(labels, rows):
        body.append(
            f"{label} & {float(row['theoretical_risk']):.4f} & "
            f"{fmt_risk(row, 'simulated_risk', 'mcse')} \\\\"
        )
    return "\n".join(body)


def latex_mixture(rows: list[dict[str, float]]) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['eta_fraction']:.2f} & "
            f"{row['min_eigenvalue']:.4f} & "
            f"{row['risk_conditional']:.4f} "
            f"({row['mcse_conditional']:.4f}) & "
            f"{row['risk_gaussian']:.4f} "
            f"({row['mcse_gaussian']:.4f}) & "
            f"{fmt_percentage(row['relative_excess'])} "
            f"({fmt_percentage(row['mcse_relative_excess'])}) \\\\"
        )
    return "\n".join(body)


def latex_copula(rows: list[dict[str, float | str]]) -> str:
    labels = [
        r"Conditional mean $\widehat Z_{\mathrm{CM}}$",
        r"Plug-in conditional median $\widehat Z_{\mathrm{plug}}$",
        r"Original-scale linear predictor $\widehat Z_{\mathrm{lin}}$",
    ]
    body = []
    for label, row in zip(labels, rows):
        body.append(
            f"{label} & {fmt_risk(row, 'risk', 'mcse')} & "
            f"{float(row['excess']):.4f} ({float(row['mcse_excess']):.4f}) & "
            f"{100.0 * float(row['relative_excess']):.2f}\\% \\\\"
        )
    return "\n".join(body)


def latex_geometry(rows: list[dict[str, float | str]]) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['design']} & "
            f"{float(row['nearest_distance']):.2f} & "
            f"{float(row['kriging_variance']):.4f} & "
            f"{float(row['risk_conditional']):.4f} "
            f"({float(row['mcse_conditional']):.4f}) & "
            f"{float(row['risk_gaussian']):.4f} "
            f"({float(row['mcse_gaussian']):.4f}) & "
            f"{fmt_percentage(float(row['relative_excess']))} "
            f"({fmt_percentage(float(row['mcse_relative_excess']))}) \\\\"
        )
    return "\n".join(body)

def latex_sensitivity(rows: list[dict[str, float]]) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['rho']:.0f} & "
            f"{row['nugget_ratio']:.1f} & "
            f"{row['kriging_variance']:.4f} & "
            f"{row['risk_conditional']:.4f} "
            f"({row['mcse_conditional']:.4f}) & "
            f"{row['risk_gaussian']:.4f} "
            f"({row['mcse_gaussian']:.4f}) & "
            f"{fmt_percentage(row['relative_excess'])} "
            f"({fmt_percentage(row['mcse_relative_excess'])}) \\\\"
        )
    return "\n".join(body)

def complete_table(
    rows: str,
    caption: str,
    label: str,
    column_specification: str,
    header: str,
) -> str:
    """Wrap generated rows in a complete manuscript-ready LaTeX table."""
    number_of_columns = len(column_specification)

    if number_of_columns >= 4:
        size_commands = (
            "\\scriptsize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
        )
        column_specification = f"@{{}}{column_specification}@{{}}"
    else:
        size_commands = "\\small\n"

    tabular = (
        f"\\begin{{tabular}}{{{column_specification}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )

    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"{size_commands}"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{tabular}\n"
        "\\end{table}"
    )

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_local_slope(rows: list[dict[str, float]]) -> dict[str, float]:
    """Fit the manuscript's descriptive unweighted log-log regression.

    The returned standard error is the ordinary regression standard error. It
    does not incorporate Monte Carlo uncertainty in the estimated excess risks.
    """
    nonzero_rows = [
        row
        for row in rows
        if row["eta"] > 0
    ]
    if len(nonzero_rows) < 3:
        return {
            "n_points": float(len(nonzero_rows)),
            "slope": np.nan,
            "standard_error": np.nan,
            "r_squared": np.nan,
        }
    regression = linregress(
        np.log([row["eta"] for row in nonzero_rows]),
        np.log([row["excess"] for row in nonzero_rows]),
    )
    return {
        "n_points": float(len(nonzero_rows)),
        "slope": float(regression.slope),
        "standard_error": float(regression.stderr),
        "r_squared": float(regression.rvalue**2),
    }


def mixture_slope_figure(
    rows: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Create the manuscript's log-log excess-risk figure as a PDF.

    The fit uses all prespecified nonzero perturbation levels. Error bars are
    Monte Carlo standard errors calculated from replicate-level squared
    predictor differences. The shaded region is the nominal pointwise 95%
    ordinary-least-squares band for the mean fitted log excess risk. It is
    descriptive and does not incorporate the heterogeneous Monte Carlo
    uncertainty in the estimated excess risks.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonzero_rows = [
        row
        for row in rows
        if row["eta"] > 0
    ]
    if len(nonzero_rows) < 3:
        raise ValueError("At least three nonzero mixture points are required")

    fractions = np.asarray(
        [row["eta_fraction"] for row in nonzero_rows], dtype=float
    )
    excess = np.asarray([row["excess"] for row in nonzero_rows], dtype=float)
    excess_mcse = np.asarray(
        [row["mcse_excess"] for row in nonzero_rows], dtype=float
    )
    eta = np.asarray([row["eta"] for row in nonzero_rows], dtype=float)

    log_eta = np.log(eta)
    log_excess = np.log(excess)
    regression = linregress(log_eta, log_excess)

    fraction_grid = np.geomspace(fractions.min(), fractions.max(), 300)
    eta_max = float(nonzero_rows[0]["eta_max"])
    log_eta_grid = np.log(fraction_grid * eta_max)
    fitted_log = regression.intercept + regression.slope * log_eta_grid

    residuals = log_excess - (
        regression.intercept + regression.slope * log_eta
    )
    degrees_of_freedom = len(nonzero_rows) - 2
    residual_standard_error = float(
        np.sqrt(np.sum(residuals**2) / degrees_of_freedom)
    )
    centred_sum_squares = float(
        np.sum((log_eta - np.mean(log_eta)) ** 2)
    )
    mean_fit_standard_error = residual_standard_error * np.sqrt(
        1.0 / len(nonzero_rows)
        + (log_eta_grid - np.mean(log_eta)) ** 2 / centred_sum_squares
    )
    critical_value = float(
        student_t_distribution.ppf(0.975, degrees_of_freedom)
    )
    lower = np.exp(fitted_log - critical_value * mean_fit_standard_error)
    upper = np.exp(fitted_log + critical_value * mean_fit_standard_error)
    fitted = np.exp(fitted_log)

    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.fill_between(
        fraction_grid,
        lower,
        upper,
        color="#9ecae1",
        alpha=0.45,
        linewidth=0,
        label="Nominal 95% OLS band",
    )
    axis.plot(
        fraction_grid,
        fitted,
        color="#08519c",
        linewidth=1.8,
        label=f"OLS fit, slope = {regression.slope:.4f}",
    )
    axis.errorbar(
        fractions,
        excess,
        yerr=excess_mcse,
        fmt="o",
        markersize=4.5,
        color="black",
        ecolor="#525252",
        elinewidth=0.9,
        capsize=2.5,
        label="Simulated excess risk (MCSE)",
        zorder=3,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Perturbation level $\eta/\eta_{\max}$")
    axis.set_ylabel(r"Excess risk $\Delta R_\eta$")
    axis.grid(True, which="both", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def latex_mixture_figure(
    n_sim: int,
    slope: dict[str, float],
) -> str:
    """Return a manuscript-ready LaTeX block for ``mixture_slope.pdf``."""
    n_sim_latex = fmt_latex_integer(n_sim)
    caption = (
        "Log--log plot of excess risk "
        "$\\Delta R_\\eta = R_\\eta(\\widehat Z_G;x_0) "
        "- R_\\eta(Z^\\star_\\eta;x_0)$ against "
        "$\\eta/\\eta_{\\max}$ for the covariance-preserving "
        "Gaussian-mixture family with shift direction "
        "$\\widetilde b=(1,-1,0,\\ldots,0)^\\top$. Error bars are Monte "
        "Carlo standard errors calculated from replicate-level squared "
        f"predictor differences, based on $N={n_sim_latex}$ replicates. "
        "The descriptive unweighted log--log regression has "
        f"slope ${slope['slope']:.4f}$ (regression standard error "
        f"${slope['standard_error']:.4f}$, "
        f"$R^2={slope['r_squared']:.6f}$), consistent with the theoretical "
        "eighth-order rate of Theorem~\\ref{thm:mixture-eighth-order}. "
        "The shaded region is the nominal pointwise $95\\%$ "
        "ordinary-least-squares band for the mean fitted log excess risk. "
        "It is shown descriptively and does not incorporate the heterogeneous "
        "Monte Carlo uncertainty in the estimated excess risks."
    )
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width=0.75\\textwidth]{mixture_slope.pdf}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{fig:mixture-slope}\n"
        "\\end{figure}"
    )


def validation_report(
    mixture: list[dict[str, float]],
    geometry: list[dict[str, float | str]],
    sensitivity: list[dict[str, float]],
) -> str:
    """Create a concise reproducibility and identity-check report."""
    all_mixtures = mixture + geometry + sensitivity
    max_predictor_discrepancy = max(
        float(row["maximum_predictor_discrepancy"])
        for row in all_mixtures
    )
    max_exact_mean_norm = max(
        float(row["exact_mean_norm"])
        for row in all_mixtures
    )
    max_exact_covariance_error = max(
        float(row["exact_covariance_error"])
        for row in all_mixtures
    )
    max_eta_ratio_error = max(
        abs(
            float(row["eta_max"])
            / float(row["eta_boundary"])
            - 0.95
        )
        for row in all_mixtures
    )
    max_simulated_covariance_error = max(
        float(row["relative_covariance_error"])
        for row in all_mixtures
    )
    return "\n".join(
        [
            "Validation diagnostics",
            "======================",
            (
                "Maximum absolute difference between the theorem-based and "
                "component-posterior conditional predictors: "
                f"{max_predictor_discrepancy:.12g}"
            ),
            f"Maximum exact mixture mean norm: {max_exact_mean_norm:.12g}",
            (
                "Maximum absolute error in eta_max / eta_boundary = 0.95: "
                f"{max_eta_ratio_error:.12g}"
            ),
            (
                "Maximum exact relative covariance error: "
                f"{max_exact_covariance_error:.12g}"
            ),
            (
                "Maximum simulated relative covariance error: "
                f"{max_simulated_covariance_error:.12g}"
            ),
            (
                "The slope standard error is the unweighted OLS regression "
                "standard error and does not include Monte Carlo uncertainty."
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    n_sim_latex = fmt_latex_integer(args.n_sim)
    if args.n_sim < 100:
        raise ValueError("--n-sim must be at least 100")

    rng = np.random.default_rng(args.seed)
    regular_observations = np.arange(1, 21, dtype=float)
    regular_locations = np.concatenate(([0.0], regular_observations))
    sigma = covariance_matrix(regular_locations, rho=5.0, nugget_ratio=0.0)

    exact = exact_robustness_experiment(rng, args.n_sim, sigma)
    fractions = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.00]
    mixture = [mixture_experiment(rng, args.n_sim, sigma, f) for f in fractions]
    slope = estimate_local_slope(mixture)
    copula = copula_experiment(rng, args.n_sim, sigma)
    geometry, sensitivity = geometry_experiments(rng, args.n_sim)
    diagnostics = validation_report(mixture, geometry, sensitivity)

    tables = {
        "table_exact_robustness.tex": complete_table(
            latex_exact(exact),
            "Theoretical and simulated mean-square prediction risks under "
            "Gaussian and covariance-normalised Student--$t$ fields. Results "
            f"are based on $N={n_sim_latex}$ replicates. Monte Carlo standard "
            "errors are reported in parentheses.",
            "tab:numerical",
            "lcc",
            "Model & Theoretical risk & Simulated risk (MCSE)",
        ),
        "table_mixture_results.tex": complete_table(
            latex_mixture(mixture),
            "Prediction risks for the covariance-preserving Gaussian-mixture "
            "family with shift direction "
            "$\\widetilde b=(1,-1,0,\\ldots,0)^\\top$. At each perturbation "
            "level, results are based on "
            f"$N={n_sim_latex}$ common replicates used to evaluate both "
            "predictors. Monte Carlo standard errors are reported in parentheses. "
            "The final column reports relative excess risk and its Monte Carlo "
            "standard error, calculated from replicate-level squared predictor "
            "differences using the paired delta method. Values below "
            "$0.005\\%$ are reported as being below the displayed two-decimal "
            "precision.",
            "tab:mixture-results",
            "ccccc",
            "$\\eta/\\eta_{\\max}$ & "
            "$\\lambda_{\\min}(\\Omega_\\eta)$ & "
            "$R_\\eta(Z^\\star_\\eta)$ & "
            "$R_\\eta(\\widehat Z_G)$ & RER (MCSE)",
        ),
        "table_copula_results.tex": complete_table(
            latex_copula(copula),
            "Mean-square prediction risks under the lognormal Gaussian-copula "
            f"model. Results are based on $N={n_sim_latex}$ common replicates "
            "used to evaluate all three predictors. Monte Carlo standard errors "
            "are reported in parentheses. Excess "
            "risks are calculated relative to the conditional-mean predictor.",
            "tab:copula-results",
            "lccc",
            "Predictor & Prediction risk & Excess risk (MCSE) & Relative excess risk",
        ),
        "table_geometry_results.tex": complete_table(
            latex_geometry(geometry),
            "Effect of sampling geometry on prediction risk for $\\rho=5$, "
            "$\\lambda=0.1$, and $\\eta=0.7\\eta_{\\max}$. For each "
            f"design, results are based on $N={n_sim_latex}$ common "
            "replicates used to evaluate both predictors. Monte Carlo "
            "standard errors are reported in parentheses. Percentage MCSEs below "
            "$0.005\\%$ are reported as being below the displayed two-decimal "
            "precision.",
            "tab:geometry-results",
            "lccccc",
            "Design & Nearest distance & $\\sigma_K^2(x_0)$ & "
            "$R_\\eta(Z^\\star_\\eta)$ & "
            "$R_\\eta(\\widehat Z_G)$ & RER (MCSE)",
        ),
        "table_covariance_sensitivity.tex": complete_table(
            latex_sensitivity(sensitivity),
            "Sensitivity of prediction risk to the correlation range and nugget "
            "ratio under the regular sampling design and "
            "$\\eta=0.7\\eta_{\\max}$. For each covariance specification, "
            f"results are based on $N={n_sim_latex}$ common replicates used "
            "to evaluate both predictors. Monte Carlo standard "
            "errors are reported in parentheses. Percentage MCSEs below "
            "$0.005\\%$ are reported as being below the displayed two-decimal "
            "precision.",
            "tab:covariance-sensitivity",
            "lccccc",
            "$\\rho$ & $\\lambda$ & $\\sigma_K^2(x_0)$ & "
            "$R_\\eta(Z^\\star_\\eta)$ & "
            "$R_\\eta(\\widehat Z_G)$ & RER (MCSE)",
        ),
    }
    print(f"N={args.n_sim}, seed={args.seed}")
    for filename, table in tables.items():
        print(f"\n% {filename}\n{table}")
    print("\n% descriptive unweighted local mixture excess-risk slope")
    print(
        f"nonzero points={int(slope['n_points'])}, slope={slope['slope']:.6g}, "
        f"SE={slope['standard_error']:.6g}, R^2={slope['r_squared']:.6g}"
    )
    max_covariance_error = max(
        row["relative_covariance_error"]
        for row in mixture + geometry + sensitivity
    )  
    print(f"maximum simulated relative covariance error={max_covariance_error:.6g}")
    print("\n" + diagnostics)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        mixture_slope_figure(
            mixture,
            args.output_dir / "mixture_slope.pdf",
        )
        (args.output_dir / "figure_mixture_slope.tex").write_text(
            latex_mixture_figure(args.n_sim, slope) + "\n",
            encoding="utf-8",
        )
        write_csv(args.output_dir / "exact_robustness.csv", exact)
        write_csv(args.output_dir / "mixture_perturbations.csv", mixture)
        write_csv(args.output_dir / "copula_prediction.csv", copula)
        write_csv(args.output_dir / "sampling_geometry.csv", geometry)
        write_csv(args.output_dir / "covariance_sensitivity.csv", sensitivity)
        write_csv(args.output_dir / "mixture_slope.csv", [slope])
        (args.output_dir / "validation_diagnostics.txt").write_text(
            diagnostics + "\n", encoding="utf-8"
        )
        for filename, table in tables.items():
            (args.output_dir / filename).write_text(table + "\n", encoding="utf-8")
        all_tables = "\n\n".join(tables.values()) + "\n"
        (args.output_dir / "all_numerical_tables.tex").write_text(
            all_tables, encoding="utf-8"
        )


if __name__ == "__main__":
    main()
