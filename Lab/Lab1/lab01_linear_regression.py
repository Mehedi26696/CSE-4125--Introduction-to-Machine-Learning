import numpy as np
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 42

SYNTHETIC_CONFIG = {
    "intercept": 3,
    "slope": 5,
    "n_samples": 100,
    "x_range": (1, 100),
    "learning_rate": 0.01,
    "iterations": 1000,
    "unscaled_learning_rate": 4e-4,
    "unscaled_iterations": 30_000,
    "feature_scaling": True,
}

REAL_DATA_CONFIG = {
    "file_path": os.path.join(os.path.dirname(__file__), "data_01.csv"),
    "learning_rate": 0.01,
    "iterations": 1000,
    "unscaled_learning_rate": 5e-4,
    "unscaled_iterations": 100_000,
    "feature_scaling": True,
}

def generate_synthetic_data(n, intercept, slope):
    random_generator = np.random.default_rng(RANDOM_SEED)
    x = np.arange(1, n + 1, dtype=np.float64)
    noise = random_generator.standard_normal(n)
    y = intercept + slope * x + noise
    return x, y


def save_data(filepath, x, y):
    data = np.column_stack((x, y))
    np.savetxt(filepath, data, delimiter=",", header="x,y", comments="")
    print(f"[INFO] Data saved to {filepath}")


def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=0)
    if np.isnan(data[0]).any():
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    print(f"[INFO] Loaded {x.shape[0]} samples from {filepath}")
    return x, y


def process_data(x_values, y_values, feature_scaling):
    num_samples = x_values.shape[0]
    scaling_params = {"mean": 0.0, "std": 1.0, "scaled": feature_scaling}

    processed_x = x_values.copy()

    if feature_scaling:
        feature_mean = np.mean(processed_x)
        feature_std = np.std(processed_x)
        processed_x = (processed_x - feature_mean) / feature_std
        scaling_params["mean"] = feature_mean
        scaling_params["std"] = feature_std
        print(
            f"[INFO] Feature scaling applied  "
            f"(mean={feature_mean:.4f}, std={feature_std:.4f})"
        )

    design_matrix = np.column_stack((np.ones(num_samples), processed_x))
    target_vector = y_values.reshape(-1, 1)

    return design_matrix, target_vector, scaling_params


def compute_cost(X, y, theta):
    m = X.shape[0]  
    predictions = X @ theta  
    errors = predictions - y   
    cost = (1.0 / (2 * m)) * np.sum(errors**2)   
    return cost 


def gradient_descent(X, y, theta, alpha, num_iters):
    m = X.shape[0]  
    cost_history = []  

    for i in range(num_iters):
        predictions = X @ theta   
        errors = predictions - y  
        gradient = (1.0 / m) * (X.T @ errors)  
        theta = theta - alpha * gradient 
        cost = compute_cost(X, y, theta) 
        cost_history.append(cost)

    return theta, cost_history   


def train(X, y, alpha, num_iters):
    n = X.shape[1]
    theta = np.zeros((n, 1))

    print(f"[INFO] Training  |  lr={alpha}, iterations={num_iters}")
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)
    print(f"[INFO] Training complete  |  Final cost = {cost_history[-1]:.6f}")

    return theta, cost_history


def evaluate(X, y, theta):
    evaluation_cost = compute_cost(X, y, theta)
    predicted_values = X @ theta
    residual_sum_of_squares = np.sum((y - predicted_values) ** 2)
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)

    mean_squared_error = residual_sum_of_squares / X.shape[0]
    r_squared_score = 1 - (residual_sum_of_squares / total_sum_of_squares)

    print(f"[EVAL] Cost (J)  = {evaluation_cost:.6f}")
    print(f"[EVAL] MSE = {mean_squared_error:.6f}")
    print(f"[EVAL] R^2 score = {r_squared_score:.6f}")
    return evaluation_cost,mean_squared_error , r_squared_score


def plot_data(x, y, title, xlabel, ylabel, save_as):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=15, alpha=0.6, label="Data points")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        print(f"[INFO] Plot saved to {save_as}")
    plt.show()


def plot_cost_history(cost_history, title, save_as):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_history) + 1), cost_history, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        print(f"[INFO] Plot saved to {save_as}")
    plt.show()


def plot_regression_line(
    x_original,
    y_original,
    theta,
    scale_params,
    title,
    xlabel,
    ylabel,
    save_as,
):
    plt.figure(figsize=(8, 5))
    plt.scatter(x_original, y_original, s=15, alpha=0.6, label="Data points")

    x_line = np.linspace(x_original.min(), x_original.max(), 300)

    if scale_params["scaled"]:
        x_line_scaled = (x_line - scale_params["mean"]) / scale_params["std"]
    else:
        x_line_scaled = x_line

    X_line = np.column_stack((np.ones(len(x_line)), x_line_scaled))
    y_line = X_line @ theta

    plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        print(f"[INFO] Plot saved to {save_as}")
    plt.show()


def print_parameters(model_parameters, scaling_params):
    print("\n" + "=" * 50)
    print("LEARNED MODEL PARAMETERS")
    print("=" * 50)
    print(f"  theta_0 (intercept) = {model_parameters[0, 0]:.6f}")
    print(f"  theta_1 (slope)     = {model_parameters[1, 0]:.6f}")

    if scaling_params["scaled"]:
        feature_mean = scaling_params["mean"]
        feature_std = scaling_params["std"]
        original_slope = model_parameters[1, 0] / feature_std
        original_intercept = (
            model_parameters[0, 0]
            - model_parameters[1, 0] * feature_mean / feature_std
        )
        print("\n  (Original feature space)")
        print(f"  intercept = {original_intercept:.6f}")
        print(f"  slope     = {original_slope:.6f}")
    print("=" * 50 + "\n")


def run_synthetic_data():
    config = SYNTHETIC_CONFIG
    print("\n" + "=" * 60)
    print("  PART 1: SYNTHETIC DATA  (y = 3 + 5x + noise)")
    print("=" * 60 + "\n")

    x, y = generate_synthetic_data(
        n=config["n_samples"],
        intercept=config["intercept"],
        slope=config["slope"],
    )

    csv_path = os.path.join(os.path.dirname(__file__), "lab01_data.csv")
    save_data(csv_path, x, y)

    plot_data(
        x,
        y,
        title="Synthetic Dataset: Generated Data Points",
        xlabel="x",
        ylabel="y",
        save_as=os.path.join(os.path.dirname(__file__), "synthetic_data.png"),
    )

    print("\n--- Without Feature Scaling ---")
    unscaled_design_matrix, unscaled_target_vector, unscaled_params = process_data(
        x, y, feature_scaling=False
    )
    unscaled_learning_rate = config["unscaled_learning_rate"]
    try:
        unscaled_theta, unscaled_cost_history = train(
            unscaled_design_matrix,
            unscaled_target_vector,
            alpha=unscaled_learning_rate,
            num_iters=config["unscaled_iterations"],
        )
        if np.isnan(unscaled_cost_history[-1]) or np.isinf(
            unscaled_cost_history[-1]
        ):
            raise ValueError("Cost diverged")

        plot_cost_history(
            unscaled_cost_history,
            title="Synthetic Dataset: Gradient Descent Cost (Unscaled)",
            save_as=os.path.join(
                os.path.dirname(__file__), "synthetic_unscaled_cost.png"
            ),
        )

        print_parameters(unscaled_theta, unscaled_params)
        evaluate(unscaled_design_matrix, unscaled_target_vector, unscaled_theta)

        plot_regression_line(
            x,
            y,
            unscaled_theta,
            unscaled_params,
            title="Synthetic Dataset: Fitted Regression Line (Unscaled)",
            xlabel="x",
            ylabel="y",
            save_as=os.path.join(
                os.path.dirname(__file__), "synthetic_unscaled_regression.png"
            ),
        )
    except (ValueError, FloatingPointError):
        print("  [WARNING] Gradient descent diverged without feature scaling.")

    print("\n--- With Feature Scaling ---")

    scaled_design_matrix, scaled_target_vector, scaling_params = process_data(
        x, y, feature_scaling=config["feature_scaling"]
    )

    scaled_theta, scaled_cost_history = train(
        scaled_design_matrix,
        scaled_target_vector,
        alpha=config["learning_rate"],
        num_iters=config["iterations"],
    )

    plot_cost_history(
        scaled_cost_history,
        title="Synthetic Dataset: Gradient Descent Cost (Scaled)",
        save_as=os.path.join(os.path.dirname(__file__), "synthetic_cost.png"),
    )

    print_parameters(scaled_theta, scaling_params)

    evaluate(scaled_design_matrix, scaled_target_vector, scaled_theta)

    plot_regression_line(
        x,
        y,
        scaled_theta,
        scaling_params,
        title="Synthetic Dataset: Fitted Linear Regression Line (Scaled)",
        xlabel="x",
        ylabel="y",
        save_as=os.path.join(os.path.dirname(__file__), "synthetic_regression.png"),
    )



def run_real_data():
    config = REAL_DATA_CONFIG
    print("\n" + "=" * 60)
    print("  PART 2: REAL DATA  (data_01.csv)")
    print("=" * 60 + "\n")

    x, y = load_data(config["file_path"])

    plot_data(
        x,
        y,
        title="Real Dataset: Observed Data Points",
        xlabel="x",
        ylabel="y",
        save_as=os.path.join(os.path.dirname(__file__), "real_data.png"),
    )

    print("\n--- Without Feature Scaling ---")
    unscaled_design_matrix, unscaled_target_vector, unscaled_params = process_data(
        x, y, feature_scaling=False
    )

    unscaled_learning_rate = config["unscaled_learning_rate"]
    try:
        unscaled_theta, unscaled_cost_history = train(
            unscaled_design_matrix,
            unscaled_target_vector,
            alpha=unscaled_learning_rate,
            num_iters=config["unscaled_iterations"],
        )
        if np.isnan(unscaled_cost_history[-1]) or np.isinf(
            unscaled_cost_history[-1]
        ):
            raise ValueError("Cost diverged")

        plot_cost_history(
            unscaled_cost_history,
            title="Real Dataset: Gradient Descent Cost (Unscaled)",
            save_as=os.path.join(
                os.path.dirname(__file__), "real_unscaled_cost.png"
            ),
        )

        print_parameters(unscaled_theta, unscaled_params)
        evaluate(unscaled_design_matrix, unscaled_target_vector, unscaled_theta)

        plot_regression_line(
            x,
            y,
            unscaled_theta,
            unscaled_params,
            title="Real Dataset: Fitted Regression Line (Unscaled)",
            xlabel="x",
            ylabel="y",
            save_as=os.path.join(
                os.path.dirname(__file__), "real_unscaled_regression.png"
            ),
        )
    except (ValueError, FloatingPointError):
        print("  [WARNING] Gradient descent diverged without feature scaling.")

    print("\n--- With Feature Scaling ---")

    scaled_design_matrix, scaled_target_vector, scaling_params = process_data(
        x, y, feature_scaling=config["feature_scaling"]
    )

    scaled_theta, scaled_cost_history = train(
        scaled_design_matrix,
        scaled_target_vector,
        alpha=config["learning_rate"],
        num_iters=config["iterations"],
    )

    plot_cost_history(
        scaled_cost_history,
        title="Real Dataset: Gradient Descent Cost (Scaled)",
        save_as=os.path.join(os.path.dirname(__file__), "real_cost.png"),
    )

    print_parameters(scaled_theta, scaling_params)

    evaluate(scaled_design_matrix, scaled_target_vector, scaled_theta)

    plot_regression_line(
        x,
        y,
        scaled_theta,
        scaling_params,
        title="Real Dataset: Fitted Linear Regression Line (Scaled)",
        xlabel="x",
        ylabel="y",
        save_as=os.path.join(os.path.dirname(__file__), "real_regression.png"),
    )


def choose_dataset():
    print("\nChoose the dataset to use:")
    print("  1. Real data (data_01.csv)")
    print("  2. Synthetic data (seed = 42)")
    print("  3. Both real and synthetic data")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1":
            run_real_data()
            return
        if choice == "2":
            run_synthetic_data()
            return
        if choice == "3":
            run_synthetic_data()
            run_real_data()
            return
        
        print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    choose_dataset()
