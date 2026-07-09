from utils import (
    bootstrap_uncertainty_model,
    fit_scaling_law_least_squares,
    get_pareto_frontier,
    load_scaling_data,
    plot_results,
    summarize_scaling_fit,
)

# CONFIG VALUES

DATA_PATH = (
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/LCBench/data/exported"
)
DATASET_NAME = "adult"

X_VALUE_KEY = ("architecture", "param_count")  # parameter count column
Y_VALUE_KEY = (
    "summary",
    "eval",
    "best_accuracy",
    "value",
)  # validation accuracy (0–100 scale assumed)
COLOR_VALUE_KEY = ("architecture", "config", "max_units")  # colour-coding column

# Pareto frontier binning
N_BINS = 100  # log-spaced bins across parameter range
N_BOOT = 500  # increase to ~500 for more stable results

# Fit bounds  [acc_inf,  A,    alpha]
P0 = [80.0, 20.0, 0.1]  # initial guess
BOUNDS_LO = [50.0, 0.0, 0.0]
BOUNDS_HI = [100.0, 200.0, 2.0]

# ACTUAL SCRIPT START

# data loading
x_values, y_values, color_values = load_scaling_data(
    DATA_PATH,
    DATASET_NAME,
    x_value_key=X_VALUE_KEY,
    y_value_key=Y_VALUE_KEY,
    color_value_key=COLOR_VALUE_KEY,
)

# best values for best values scaling law fit
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

# plotting
plot_results(
    x_values=x_values,
    y_values=y_values,
    color_dim_values=color_values,
    frontier_x=frontier_x,
    frontier_y=frontier_y,
    popt=popt,
    y_pred=y_pred,
    r2=r2,
    boot_params=boot_params,
)
