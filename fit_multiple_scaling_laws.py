from utils import (
    bootstrap_uncertainty_model,
    fit_scaling_law_least_squares,
    get_dataset_names,
    get_frontier_slopes_and_curvature,
    get_pareto_frontier,
    load_scaling_data,
    sample_frontier,
    save_scaling_law,
    summarize_scaling_fit,
)

# CONFIG VALUES
USE_LCBENCH = False

LCBENCH_DATA_PATH = (
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/LCBench/data/exported"
)
NATS_DATA_PATH = (
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/NATS-Bench/exported"
)

# parameter count column
X_VALUE_KEY = ("architecture", "param_count")
# validation accuracy (0–100 scale assumed)
Y_VALUE_KEY = (
    "summary",
    "eval",
    "best_accuracy",
    "value",
)

# Pareto frontier binning
N_BINS = 100  # log-spaced bins across parameter range
N_BOOT = 500  # increase to ~500 for more stable results

# Fit bounds  [y_inf,  A,    alpha]
P0 = [80.0, 20.0, 0.1]  # initial guess
BOUNDS_LO = [50.0, 0.0, 0.0]
BOUNDS_HI = [100.0, 200.0, 2.0]

N_FRONTIER_SAMPLES = 32

# START OF SCRIPT

# retrieve dataset names
dataset_names = get_dataset_names(LCBENCH_DATA_PATH if USE_LCBENCH else NATS_DATA_PATH)

print(dataset_names)
print(len(dataset_names))

# iterate over them
for dataset_name in dataset_names:
    # data loading, leave color out because not relevant (no plots)
    x_values, y_values, _ = load_scaling_data(
        LCBENCH_DATA_PATH if USE_LCBENCH else NATS_DATA_PATH,
        dataset_name,
        x_value_key=X_VALUE_KEY,
        y_value_key=Y_VALUE_KEY,
    )

    # get the best values for the scaling law fit
    frontier_x, frontier_y = get_pareto_frontier(
        x_values=x_values, y_values=y_values, n_bins=N_BINS
    )

    # scaling law fit
    popt, perr, y_pred, r2, r = fit_scaling_law_least_squares(
        frontier_x, frontier_y, p0=P0, bounds_lo=BOUNDS_LO, bounds_hi=BOUNDS_HI
    )
    y_inf_fit, A_fit, alpha_fit = popt

    # uncertainty estimation
    boot_params = bootstrap_uncertainty_model(
        frontier_x=frontier_x,
        frontier_y=frontier_y,
        popt=popt,
        n_boot=N_BOOT,
        bounds_lo=BOUNDS_LO,
        bounds_hi=BOUNDS_HI,
    )

    # extract distributions
    y_inf_boot = boot_params[:, 0]
    A_boot = boot_params[:, 1]
    alpha_boot = boot_params[:, 2]

    frontier_slope_small, frontier_slope_large, frontier_curvature = (
        get_frontier_slopes_and_curvature(frontier_x, frontier_y)
    )

    x_samples, y_samples = sample_frontier(frontier_x, frontier_y, N_FRONTIER_SAMPLES)

    # dict with all features to be saved to npz file with scaling law data
    features = {
        "y_inf": y_inf_fit,
        "A": A_fit,
        "alpha": alpha_fit,
        "r2": r2,
        "param_min": float(x_values.min()),
        "param_max": float(x_values.max()),
        "max_y": float(y_values.max()),
        "mean_y": float(y_values.mean()),
        "n_models": len(x_values),
        "alpha_boot_mean": float(alpha_boot.mean()),
        "alpha_boot_std": float(alpha_boot.std()),
        "frontier_slope_small": float(frontier_slope_small),
        "frontier_slope_large": float(frontier_slope_large),
        "frontier_curvature": float(
            2 * frontier_curvature
        ),  # multiply by 2 for mathematical cleanliness
        "frontier_x": x_samples,
        "frontier_y": y_samples,
    }

    # summarization
    summarize_scaling_fit(
        y_inf_fit=y_inf_fit,
        A_fit=A_fit,
        alpha_fit=alpha_fit,
        perr=perr,
        r2=r2,
        r=r,
        y_inf_boot=y_inf_boot,
        A_boot=A_boot,
        alpha_boot=alpha_boot,
        boot_params=boot_params,
    )

    # save scaling law file
    save_scaling_law(
        # popt=popt,
        # boot_params=boot_params,
        features=features,
        save_dir_name="scaling_laws/lcbench" if USE_LCBENCH else "scaling_laws/nats",
        dataset_name=dataset_name,
        input_path=X_VALUE_KEY,
        output_path=Y_VALUE_KEY,
    )
