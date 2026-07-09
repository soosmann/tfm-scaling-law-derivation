import glob
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def extract_nested(d: dict, keys: tuple) -> Any:
    """
    Based on the key structure in the key tuple, extract the value from the dict.

    Parameters:
        d (`dict`): The dict the data will be retrieved from.
        keys (`tuple`): Tuple containing the key hierarchy.

    Returns:
        Any: Extracted value, should be a number to continue without problems.

    """
    for k in keys:
        d = d[k]

    if isinstance(d, dict):
        raise TypeError(
            f"tuple indicating json path '{keys}' seems incomplete, got a dict returned."
        )
    return d


def load_scaling_data(
    data_path: str | Path,
    dataset_prefix: str | None = None,
    x_value_key: tuple = ("architecture", "param_count"),
    y_value_key: tuple = ("summary", "eval", "best_accuracy", "value"),
    color_value_key: tuple = ("architecture", "config", "max_units"),
):
    """
    Load and clean scaling law data from JSON payloads.

    Parameters:
        data_path (`str | Path`): Directory containing JSON files.
        dataset_prefix (`str | None`): If provided, only files starting with this prefix are used.
        x_value_key (`tuple`): Nested key path for x value of the scaling law fit, can lead for example to param count in json struct.
        y_value_key (`tuple`): Nested key path for y value of the scaling law fit, can lead for example to accuracy value in json struct.
        max_units_key (`tuple`): Nested key path for a coloring value for the plot (optional feature), can be metadata like hidden layer count.

    Returns:
        x_values, y_values, color_values : All `np.ndarray`s, `color_value_key` can stay None if not provided.
    """

    # retrieval of all json paths
    all_paths = sorted(glob.glob(str(Path(data_path) / "*.json")))

    param_counts = []
    best_accs = []
    max_units = []

    for path in all_paths:
        filename = Path(path).name

        # if sample for wanted dataset...
        if dataset_prefix is not None and not filename.startswith(dataset_prefix):
            continue

        try:
            # ... try to load data
            with open(path, "r") as handle:
                payload = json.load(handle)

                param_counts.append(extract_nested(payload, x_value_key))
                best_accs.append(extract_nested(payload, y_value_key))

                if color_value_key is not None:
                    max_units.append(extract_nested(payload, color_value_key))

        except (KeyError, TypeError):
            # Skip malformed entries
            continue

    # convert to arrays
    params = np.asarray(param_counts, dtype=np.float64)
    accuracy = np.asarray(best_accs, dtype=np.float64)

    max_units = np.asarray(max_units, dtype=np.float64) if len(max_units) > 0 else None

    # cleaning
    mask = np.isfinite(params) & np.isfinite(accuracy) & (params > 0)
    params = params[mask]
    accuracy = accuracy[mask]
    if max_units is not None:
        max_units = max_units[mask]

    # sanity check prints
    print(f"Loaded {len(params):,} valid samples")
    print(f"Parameter range : {params.min():.1e} → {params.max():.1e}")
    print(f"Accuracy range  : {accuracy.min():.2f} → {accuracy.max():.2f}")

    return params, accuracy, max_units


def get_pareto_frontier(x_values, y_values, n_bins):
    """
    Get the values for a possible Pareto Frontier Scaling Law Curve (the scaling law experiments with optimal values).

    Parameters:
        x_values: The x values for the scaling law estimation, the input, e.g. model param counts.
        y_values: The y values for the scaling law estimation, the output, e.g. accuracy values.
        n_bins: The amount of bins to be searched for a best value.

    Returns:
        tuple[np.array, np.array]: The x and y values with the best values of a bin.
    """
    # define the bins for optimal experiment search
    log_bins = np.logspace(
        np.log10(x_values.min()), np.log10(x_values.max()), n_bins + 1
    )
    bin_idx = np.digitize(x_values, log_bins) - 1  # 0-indexed bin per sample
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # init of the lists containing best x and y values
    frontier_x = []
    frontier_y = []

    # go through each bin
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() == 0:
            continue
        # find best value and append
        # TODO: Only for accuracy the method is max
        best_y = y_values[sel].max()
        best_x = x_values[sel][
            y_values[sel] == best_y
        ].mean()  # representative N in bin
        frontier_x.append(best_x)
        frontier_y.append(best_y)

    frontier_x = np.array(frontier_x)
    frontier_y = np.array(frontier_y)

    print(f"\nPareto frontier points: {len(frontier_x)}")

    return frontier_x, frontier_y


# TODO: This is for accuracy, i think loss needs something different
# def scaling_law(N, acc_inf, A, alpha):
#     """acc(N) = acc_inf - A * N^(-alpha)"""
#     return acc_inf - A * N ** (-alpha)
def scaling_law(N, asymptote, A, alpha, mode="max"):
    """
    Generic neural scaling law.

    Parameters:
        N (array-like): The x values fed used for generating the output, e.g. parameter count.
        asymptote (`float`): Limiting performance as N -> infinity.
        A (`float`): Scaling coefficient.
        alpha (`float`): Scaling exponent.
        mode (`str`): Whether larger values are better (accuracy) or worse (loss), possible values: {"max", "min"}

    Returns:
        Any: The scaling law estimation for input N
    """
    if mode == "max":
        return asymptote - A * N ** (-alpha)
    elif mode == "min":
        return asymptote + A * N ** (-alpha)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


def fit_scaling_law_least_squares(frontier_x, frontier_y, p0, bounds_lo, bounds_hi):
    """
    Function for the actual scaling law fit. Provides an output of form: (popt, perr, y_pred, r2, r).
    Explanations of outputs:
    popt: The found optimal parameters for the scaling law.
    perr: The error incorporated by the scaling law.
    y_pred: The predictions for y using `frontier_x`.
    r2: Coefficient of Determination, describes quality of fit, 1.0 is perfect fit, all below 0 worse than predicting mean.
    r: Pearson Correlation Coefficient, 1 is perfect positive linear relationship, -1 is perfect negative relationship, 0 is no correlation.

    Parameters:
        frontier_x: The x values that are used for scaling law derivation, e.g. param count.
        frontier_y: The y values that are used for scaling law derivation, e.g. accuracy.
        p0: Initial param guesses, may lead to faster convergence.
        bounds_lo: The lower bounds for the scaling law derived function.
        bounds_hi: The higher bounds for the scaling law derived function.

    Returns:
        Tuple of all relevant scaling law params and infos. Comes as: (popt, perr, y_pred, r2, r).
    """
    popt, pcov = curve_fit(
        scaling_law,
        frontier_x,
        frontier_y,
        p0=p0,
        bounds=(bounds_lo, bounds_hi),
        maxfev=20_000,
    )

    perr = np.sqrt(np.diag(pcov))

    y_pred = scaling_law(frontier_x, *popt)
    residuals = frontier_y - y_pred
    r2 = 1 - np.var(residuals) / np.var(frontier_y)
    r, _ = pearsonr(frontier_y, y_pred)

    return popt, perr, y_pred, r2, r


def get_frontier_slopes_and_curvature(frontier_x, frontier_y):
    """
    Calculate the log-based frontier slope at the beginning and at the end of the value curve
    and defines a function curvature value to show how strong the curve is bent.

    Parameters:
        frontier_x: The x values that are used for scaling law derivation, e.g. param count.
        frontier_y: The y values that are used for scaling law derivation, e.g. accuracy.

    Return:
        tuple: Tuple of three values, one for small value slope, one for large value slope and one curvature value
    """
    # get amount for 20% of the samples and put samples into log space
    k = max(3, len(frontier_x) // 5)
    logN = np.log10(frontier_x)

    # curve fitting for first and last 20% of values
    coef_small = np.polyfit(
        logN[:k],
        frontier_y[:k],
        deg=1,
    )
    coef_large = np.polyfit(
        logN[-k:],
        frontier_y[-k:],
        deg=1,
    )

    # slope retrieval
    frontier_slope_small = coef_small[0]
    frontier_slope_large = coef_large[0]

    # get the curvature of the function
    coef = np.polyfit(
        logN,
        frontier_y,
        deg=2,
    )

    frontier_curvature, _, _ = coef

    return frontier_slope_small, frontier_slope_large, frontier_curvature


def bootstrap_uncertainty_model(
    frontier_x, frontier_y, popt, n_boot, bounds_lo, bounds_hi
):
    """
    Fitting an uncertainty model using a bootstrapping approach (relying on the resources already available).
    Adapts the given x and y samples slightly for `n_boot` times to check the robustness of the scaling law and retrieve different fit curves.

    Parameters:
        frontier_x: The x values that are used for scaling law derivation, e.g. param count.
        frontier_y: The y values that are used for scaling law derivation, e.g. accuracy.
        popt: Initial param guesses, should be set to the optimal p values found in fitting the main curve, may lead to faster convergence.
        n_boot: The amount of bootstrap tries adapting the samples. Defines the length of the output.
        bounds_lo: The lower bounds for the scaling law derived function.
        bounds_hi: The higher bounds for the scaling law derived function.

    Returns:
        np.array: A list with curve params from the bootstrap tries.
    """
    # list for curve params from bootstraps
    boot_params = []

    for _ in range(n_boot):
        # change some values of the samples slightly
        idx = np.random.choice(len(frontier_x), len(frontier_x), replace=True)

        N_sample = frontier_x[idx] * (1 + 0.01 * np.random.randn(len(idx)))
        acc_sample = frontier_y[idx] + 0.1 * np.random.randn(len(idx))

        # do the scaling law fit with new samples
        try:
            popt_boot, _ = curve_fit(
                scaling_law,
                N_sample,  # frontier_N[idx],
                acc_sample,  # frontier_acc[idx],
                p0=popt,  # better than P0 → faster convergence
                bounds=(bounds_lo, bounds_hi),
                maxfev=5000,
            )
            boot_params.append(popt_boot)
        except RuntimeError:
            continue  # skip failed fits

    # for definite results, the amount of fits should be min. 20
    if len(boot_params) < 20:
        raise RuntimeError(
            f"Too few successful bootstrap fits ({len(boot_params)}). "
            "Scaling law likely unstable."
        )

    boot_params = np.array(boot_params)

    return boot_params


def sample_frontier(frontier_x, frontier_y, sample_count):
    """
    Sampling of frontier values. Includes normalization.

    Parameters:
        frontier_x: The x values that are used for scaling law derivation, e.g. param count.
        frontier_y: The y values that are used for scaling law derivation, e.g. accuracy.
        sample_count: The amount of samples that should be selected.

    Return:
        tuple: Two arrays with normalized sampled x and y values.
    """

    logN = np.log10(frontier_x)

    # x value sampling and normalization
    x_new = np.linspace(
        logN.min(),
        logN.max(),
        sample_count,
    )
    x_norm = (x_new - x_new.min()) / (x_new.max() - x_new.min())

    # y value sampling and normalization
    y_new = np.interp(
        x_new,
        logN,
        frontier_y,
    )
    y_norm = y_new / 100.0

    return x_norm, y_norm


def summarize_scaling_fit(
    y_inf_fit,
    A_fit,
    alpha_fit,
    perr,
    r2,
    r,
    y_inf_boot,
    A_boot,
    alpha_boot,
    boot_params,
):
    """
    Summarization print for the scaling laws behavior found.

    Parameters:
        y_inf_fit: The asymptotic value of the main fit scaling law curve.
        A_fit: Scaling coefficient of the main fit scaling law curve.
        alpha_fit: The scaling exponent of the main fit scaling law curve.
        perr: The error (inaccuracy) of the fit scaling law.
        r2: Coefficient of Determination
        r: Pearson Correlation Coefficient
        y_inf_boot: The bootstrapped asymptotic values.
        A_boot: The bootstrapped scaling coefficients.
        alpha_boot: The bootstrapped scaling exponents.
        boot_params: A list of the params belonging to all bootstrapped curves.

    Returns:
        None: Does only print information.
    """

    def summary_stats(x):
        return np.mean(x), np.std(x), np.percentile(x, 5), np.percentile(x, 95)

    acc_inf_mean, acc_inf_std, acc_inf_p5, acc_inf_p95 = summary_stats(y_inf_boot)
    A_mean, A_std, A_p5, A_p95 = summary_stats(A_boot)
    alpha_mean, alpha_std, alpha_p5, alpha_p95 = summary_stats(alpha_boot)

    print("\n── Fitted scaling law ──────────────────────────────")
    print(f"  y(N) = {y_inf_fit:.3f} - {A_fit:.3f} · N^(-{alpha_fit:.4f})")

    print("\nParameter estimates:")
    print(f"  y_inf = {y_inf_fit:.3f} ± {perr[0]:.3f} (fit) ± {acc_inf_std:.3f} (boot)")
    print(f"  A       = {A_fit:.3f} ± {perr[1]:.3f} (fit) ± {A_std:.3f} (boot)")
    # print(f"  alpha   = {alpha_fit:.4f} ± {perr[2]:.4f} (fit) ± {alpha_std:.4f} (boot)")
    print(f"  alpha = {alpha_fit:.4f} ± {perr[2]:.4f} (fit)")
    print(
        f"         bootstrap: mean={alpha_mean:.4f}, std={alpha_std:.4f}, "
        f"[{alpha_p5:.4f}, {alpha_p95:.4f}]"
    )

    print("\nFit quality:")
    print(f"  R²      = {r2:.4f}")
    print(f"  Pearson r = {r:.4f}")
    print(f"  Bootstrap samples used: {len(boot_params)}")


def plot_results(
    x_values,
    y_values,
    color_dim_values,
    frontier_x,
    frontier_y,
    popt,
    y_pred,
    r2,
    boot_params,
):
    """
    Function for plotting the results of the scaling law fit.

    Parameters:
        x_values: The x values used for the scaling law fit.
        y_values: The y values used for the scaling law fit.
        color_dim_values: Values used for coloring the samples in the plot.
        frontier_x: The x values belonging to the best values used for computing the optimal curve.
        frontier_y: The y values belonging to the best values used for computing the optimal curve.
        popt: The parameters of the optimal scaling law curve.
        y_pred: The predictions for y using `frontier_x`.
        r2: Coefficient of Determination.
        boot_params: The parameters belonging to the bootstrapped curves.

    Returns:
        None: Does only the plotting.


    """
    # popt extraction
    acc_inf_fit, A_fit, alpha_fit = popt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "LCBench — phoneme: neural scaling law", fontsize=14, fontweight="bold"
    )

    # Left: all points + frontier + fit
    ax = axes[0]

    if color_dim_values is not None:
        sc = ax.scatter(
            x_values,
            y_values,
            c=color_dim_values,
            cmap="viridis",
            s=8,
            alpha=0.35,
            linewidths=0,
            norm=mcolors.Normalize(
                vmin=color_dim_values.min(), vmax=color_dim_values.max()
            ),
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Max units")
    else:
        ax.scatter(x_values, y_values, s=8, alpha=0.35, color="steelblue")

    ax.scatter(
        frontier_x,
        frontier_y,
        s=50,
        color="red",
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        label="Pareto frontier",
    )

    N_smooth = np.logspace(np.log10(x_values.min()), np.log10(x_values.max()), 300)

    # Bootstrap prediction bands

    boot_curves = []

    for params_boot in boot_params:
        boot_curves.append(scaling_law(N_smooth, *params_boot))

    boot_curves = np.array(boot_curves)

    lower = np.percentile(boot_curves, 5, axis=0)
    upper = np.percentile(boot_curves, 95, axis=0)

    ax.plot(
        N_smooth,
        scaling_law(N_smooth, *popt),
        "r-",
        lw=2,
        label=f"Fit: $\\infty$={acc_inf_fit:.1f}, $\\alpha$={alpha_fit:.3f}",
    )
    ax.axhline(
        acc_inf_fit,
        color="red",
        lw=1,
        ls="--",
        alpha=0.5,
        label=f"acc_∞ = {acc_inf_fit:.2f}%",
    )

    ax.fill_between(
        N_smooth,
        lower,
        upper,
        color="red",
        alpha=0.15,
        label="90% bootstrap band",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("All runs + Pareto frontier + fit")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # --- Right: frontier only + fit + residuals ---
    ax2 = axes[1]
    ax2.scatter(
        frontier_x,
        frontier_y,
        s=60,
        color="red",
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        label="Pareto frontier",
    )
    ax2.plot(N_smooth, scaling_law(N_smooth, *popt), "k-", lw=2, label="Fit")
    ax2.axhline(
        acc_inf_fit,
        color="grey",
        lw=1,
        ls="--",
        alpha=0.7,
        label=f"acc_∞ = {acc_inf_fit:.2f}%",
    )

    # Residual stems
    for n, a, a_hat in zip(frontier_x, frontier_y, y_pred):
        ax2.plot([n, n], [a, a_hat], color="steelblue", lw=0.8, alpha=0.6)

    eq = (
        f"$acc(N) = {acc_inf_fit:.1f} - {A_fit:.2f} \\cdot N^{{-{alpha_fit:.3f}}}$\n"
        f"$R^2 = {r2:.3f}$"
    )
    ax2.text(
        0.05,
        0.07,
        eq,
        transform=ax2.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("Parameter count")
    ax2.set_ylabel("Best validation accuracy (%)")
    ax2.set_title("Frontier + fit (blue stems = residuals)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("scaling_law_fit.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved → scaling_law_fit.png")
    plt.show()


def save_scaling_law(
    features: dict,
    save_dir_name: str,
    dataset_name: str,
    input_path: str,
    output_path: str,
):
    """
    Function for saving a scaling law.
    Gets file name in format: `sl__dataset={dataset_name}__in={input_name}__out={output_name}.npz`.

    Parameters:
        features (`dict[str, Any]`): The scaling law information that should be saved in dict form with name->value.
        save_dir_name (`str`): Parent dir where the scaling law should be saved.
        dataset_name (`str`): Name of the dataset the curves constructing the scaling law belong to.
        input_path (`str`): The path in the json tree leading to the input value.
        output_path (`str`): The path in the json tree leading to the output value.

    Returns:
        None: Does only the curve saving.
    """
    input_name = ".".join(input_path)
    output_name = ".".join(output_path)
    file_name = f"sl__dataset={dataset_name}__in={input_name}__out={output_name}.npz"
    save_path = f"{save_dir_name}/{file_name}"

    # if not os.path.exists(f"{save_dir_name}/"):
    #     os.mkdir(save_dir_name)

    os.makedirs(save_dir_name, exist_ok=True)

    np.savez(
        save_path,
        **features,
        # popt=popt,  # [acc_inf, A, alpha]
        # boot_params=boot_params,  # (K, 3) bootstrap samples
    )

    print(f"Saved scaling law results → {save_path}")


def get_dataset_names(data_path: str) -> set[str]:
    """
    Using the path of the data, this function takes advantage of the naming structure of each sample
    and thus provides all dataset names of a learning curve dataset.

    Parameters:
        data_path (`str`): A string representing a path to extracted curves of datasets like LCBench or NATS.

    Returns:
        set[str]: The list of dataset names in the curve set.
    """
    paths = sorted(glob.glob(str(Path(data_path) / "*.json")))

    # first divide by "/" for disecting the path, then by "__" to obtain dataset name
    dataset_names = set(path.split("/")[-1].split("__")[0] for path in paths)

    return dataset_names
