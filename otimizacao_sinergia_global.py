import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution

# ==========================================================
# EXPERIMENTAL DATA
# ==========================================================

t_data = np.array([0, 24, 48, 72])

T_DP1_25 = np.array([
    [1, 0.2685, 1.144833333, 0.659833333],
    [1, 0.4895, 1.1365, 0.825833333],
    [1, 0.381666667, 0.7335, 0.614166667]
])

T_DP2_5 = np.array([
    [1, 0.240833333, 0.792166667, 0.674],
    [1, 0.465333333, 0.968, 0.663166667],
    [1, 0.317, 0.631666667, 0.494166667]
])

T_DP5 = np.array([
    [1, 0.1865, 0.662166667, 0.6125],
    [1, 0.443666667, 0.897, 0.706166667],
    [1, 0.296666667, 0.4905, 0.386166667]
])

T_DP10 = np.array([
    [1, 0.2185, 0.5215, 0.421],
    [1, 0.480833333, 0.788666667, 0.663333333],
    [1, 0.424333333, 0.552333333, 0.4965]
])

# ==========================================================
# 24 h Dose-response (Dasatinib + PEITC)
# ==========================================================

T24_DP = [
    {"D0":1.25,"P0":1.5,"T":np.array([1.39616667,1.17516667,1.095])},
    {"D0":2.5,"P0":1.5,"T":np.array([1.31033333,1.06033333,1.08583333])},
    {"D0":5,"P0":1.5,"T":np.array([1.1525,0.88816667,0.97716667])},
    {"D0":10,"P0":1.5,"T":np.array([1.02033333,0.785,0.89])},
    {"D0":20,"P0":1.5,"T":np.array([0.81083333,0.51283333,0.66333333])},
    {"D0":40,"P0":1.5,"T":np.array([0.53366667,0.31116667,0.40616667])},
    {"D0":80,"P0":1.5,"T":np.array([0.23966667,0.12766667,0.19433333])},
    {"D0":1.25,"P0":3,"T":np.array([1.10033333,0.75233333,0.85083333])},
    {"D0":2.5,"P0":3,"T":np.array([1.02033333,0.70516667,0.80116667])},
    {"D0":5,"P0":3,"T":np.array([0.81416667,0.644,0.71083333])},
    {"D0":10,"P0":3,"T":np.array([0.721,0.56266667,0.62683333])},
    {"D0":20,"P0":3,"T":np.array([0.6285,0.40333333,0.46683333])},
    {"D0":40,"P0":3,"T":np.array([0.40816667,0.2625,0.297])},
    {"D0":80,"P0":3,"T":np.array([0.1915,0.14216667,0.15933333])},
    {"D0":1.25,"P0":6,"T":np.array([0.8865,0.63266667,0.65683333])},
    {"D0":2.5,"P0":6,"T":np.array([0.79766667,0.60133333,0.60716667])},
    {"D0":5,"P0":6,"T":np.array([0.70033333,0.52516667,0.5375])},
    {"D0":10,"P0":6,"T":np.array([0.66333333,0.48016667,0.49516667])},
    {"D0":20,"P0":6,"T":np.array([0.504,0.37883333,0.40233333])},
    {"D0":40,"P0":6,"T":np.array([0.41116667,0.27016667,0.287])},
    {"D0":80,"P0":6,"T":np.array([0.22316667,0.149,0.15])},
    {"D0":1.25,"P0":12,"T":np.array([0.47133333,0.3825,0.25433333])},
    {"D0":2.5,"P0":12,"T":np.array([0.38666667,0.34666667,0.18733333])},
    {"D0":5,"P0":12,"T":np.array([0.4125,0.34033333,0.16])},
    {"D0":10,"P0":12,"T":np.array([0.33933333,0.32,0.168])},
    {"D0":20,"P0":12,"T":np.array([0.32866667,0.28116667,0.17333333])},
    {"D0":40,"P0":12,"T":np.array([0.20166667,0.249,0.13783333])},
    {"D0":80,"P0":12,"T":np.array([0.05666667,0.04083333,0.00305])}
]

experiments = [
    {"name":"DP1_25","D0":1.25,"P0":12.0,"T":T_DP1_25},
    {"name":"DP2_5","D0":2.50,"P0":12.0,"T":T_DP2_5},
    {"name":"DP5","D0":5.00,"P0":12.0,"T":T_DP5},
    {"name":"DP10","D0":10.0,"P0":12.0,"T":T_DP10}
]

Aleatorio = 10

# ==========================================================
# COMPLETE MODEL (GAMMA)
# ==========================================================
#DP0
rT   = 1.538797
KT   = 2.754682
rF   = 3.809131
muT  = 0.457442
muF  = 8.351323
#D
muTD = 0.000956
muDF = 0.082835
muDT = 5.104440
muD  = 0.000014
#P
muTP = 0.000902
muFP = 0.000000
muPT = 5.865540
muP  = 0.000075

def complete_model(t, X, gamma):

    T, F, D, P = X

    dT = (
        rT * F * T * (1 - T / KT)
        - muTD * T * D - muTP * T * P
        - gamma * T * D * P
        - muT * T
    )

    dF = (
        rF * T
        - muFP * P * F
        - muF * F
    )

    dD = (
        -muDF * D * F
        -muDT * T * D
        -muD * D
    )

    dP = (
        -muPT * T * P
        -muP * P
    )

    return [dT, dF, dD, dP]

# ==========================================================
# SOLVER
# ==========================================================

def solve_model(params, t_eval, T0=1, F0=1, D0=0, P0=0):

    gamma = float(np.asarray(params).ravel()[0])

    X0 = [T0, F0, D0, P0]

    sol = solve_ivp(
        lambda t, X: complete_model(
            t, X, gamma
        ),
        (t_eval[0], t_eval[-1]),
        X0,
        t_eval=t_eval,
        method="RK45"
    )

    return sol.y

# ==========================================================
# OBJECTIVE FUNCTION FOR DIFFERENTIAL EVOLUTION
# ==========================================================

def objective_function(params):
    r = residuals(params)
    return np.sum(r**2)

# ==========================================================
# LEAST SQUARES OBJECTIVE FUNCTION
# ==========================================================

def residuals(params):

    residual_list = []

    for exp in experiments:

        sol = solve_model(
            params, t_data,
            T0=1, F0=1,
            D0=exp["D0"], P0=exp["P0"]
        )

        T_model = sol[0]

        for rep in exp["T"]:
            residual_list.extend(T_model - rep)

    for exp in T24_DP:

        sol = solve_model(
            params, np.array([0,24]),
            T0=1, F0=1,
            D0=exp["D0"], P0=exp["P0"]
        )

        T24_model = sol[0][1]
        residual_list.extend(T24_model - exp["T"])

    return np.array(residual_list)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

output_dir = "Results_synergy"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# PARAMETER ESTIMATION (GLOBAL)
# ==========================================================

def callback(xk, convergence):
    print(f"Generation finished | convergence = {convergence:.6e}")

print("Starting global optimization...")

result_global = differential_evolution(
    objective_function,
    bounds=[
        (-10,10), # gamma
    ],
    strategy="best1bin",
    maxiter=500,
    popsize=25,
    tol=1e-6,
    polish=False,
    seed=Aleatorio,
    disp=True,
    callback=callback
)

print("Global optimization finished")

# ==========================================================
# LOCAL REFINEMENT
# ==========================================================

print("Starting local optimization...")

lower_bounds = [-10]
upper_bounds = [10]

result = least_squares(
    residuals,
    result_global.x,
    bounds=(lower_bounds, upper_bounds)
)

print("Local optimization finished")

gamma_est = float(result.x[0])
gamma_global = float(result_global.x[0])

final_residuals = residuals(result.x)

# ==========================================================
# FINAL SOLUTION WITH ESTIMATED PARAMETERS
# ==========================================================

t_plot = np.linspace(t_data[0], t_data[-1], 500)
solutions = {}

for exp in experiments:
    solutions[exp["name"]] = solve_model(
        result.x, t_plot,
        D0=exp["D0"], P0=exp["P0"]
    )

# ==========================================================
# PLOTS
# ==========================================================

print("Generating plots...")

for exp in experiments:

    sol = solve_model(result.x, t_plot, D0=exp["D0"], P0=exp["P0"])
    T_fit = sol[0]

    plt.figure(figsize=(8,5))

    for i in range(3):
        plt.scatter(t_data, exp["T"][i], s=60, label=f"Experiment {i+1}")

    for dose in T24_DP:
        if dose["D0"] == exp["D0"] and dose["P0"] == exp["P0"]:
            plt.scatter(
                np.full(3,24), dose["T"],
                marker="s", color="red", s=70,
                label="24 h dose-response"
            )
            break

    plt.plot(t_plot, T_fit, linewidth=2, label="Model fit")
    plt.xlabel("Time (h)")
    plt.ylabel("Normalized tumor cell viability")
    plt.title(exp["name"])
    plt.ylim(0,2.5)
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(output_dir, f"T_fit_{exp['name']}_{Aleatorio}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

for exp in experiments:

    sol = solve_model(result.x, t_plot, D0=exp["D0"], P0=exp["P0"])
    F_fit = sol[1]

    plt.figure(figsize=(8,5))
    plt.plot(t_plot, F_fit, linewidth=2, label="Model prediction")
    plt.xlabel("Time (h)")
    plt.ylabel("Normalized pFAK activity")
    plt.title(exp["name"])
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(output_dir, f"F_fit_{exp['name']}_{Aleatorio}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

for exp in experiments:

    sol = solve_model(result.x, t_plot, D0=exp["D0"], P0=exp["P0"])
    D_fit = sol[2]

    plt.figure(figsize=(8,5))
    plt.plot(t_plot, D_fit, linewidth=2)
    plt.xlabel("Time (h)")
    plt.ylabel("Dasatinib")
    plt.grid(True)

    plt.savefig(
        os.path.join(output_dir, f"D_fit_{exp['name']}_{Aleatorio}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

for exp in experiments:

    sol = solve_model(result.x, t_plot, D0=exp["D0"], P0=exp["P0"])
    P_fit = sol[3]

    plt.figure(figsize=(8,5))
    plt.plot(t_plot, P_fit, linewidth=2)
    plt.xlabel("Time (h)")
    plt.ylabel("PEITC")
    plt.grid(True)

    plt.savefig(
        os.path.join(output_dir, f"P_fit_{exp['name']}_{Aleatorio}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

P_levels = sorted(set(exp["P0"] for exp in T24_DP))

for P_level in P_levels:

    plt.figure(figsize=(8,5))
    D_values = []
    T_model = []

    for exp in T24_DP:

        if exp["P0"] != P_level:
            continue

        sol = solve_model(
            result.x, np.array([0,24]),
            T0=1, F0=1,
            D0=exp["D0"], P0=exp["P0"]
        )

        D_values.append(exp["D0"])
        T_model.append(sol[0][1])

        plt.scatter(
            np.full(3, exp["D0"]),
            exp["T"],
            color="red", s=60
        )

    plt.plot(D_values, T_model, "-o", linewidth=2, label="Model fit")
    plt.xscale("log")
    plt.xlabel("Initial Dasatinib concentration")
    plt.ylabel("Normalized tumor cell viability (24 h)")
    plt.title(f"PEITC = {P_level}")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            output_dir,
            f"DoseResponse24h_DP_P{P_level}_{Aleatorio}.png"
        ),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

# ==========================================================
# SAVE RESULTS
# ==========================================================

with open(
    os.path.join(output_dir, f"results_global_synergy_{Aleatorio}.txt"),
    "w"
) as f:

    f.write("=====================================\n")
    f.write("GLOBAL SEARCH\n")
    f.write("=====================================\n\n")

    f.write(f"gamma = {gamma_global:.6f}\n")

    f.write("=====================================\n")
    f.write("GLOBAL OPTIMIZATION INFORMATION\n")
    f.write("=====================================\n\n")
    f.write(f"Global Nit  = {result_global.nit}\n")
    f.write(f"Global Nfev = {result_global.nfev}\n")
    f.write(f"Global cost = {result_global.fun}\n\n")

    f.write("=====================================\n")
    f.write("ESTIMATED PARAMETERS\n")
    f.write("=====================================\n\n")

    f.write(f"gamma = {gamma_est:.6f}\n")

    f.write("=====================================\n")
    f.write("LOCAL OPTIMIZATION INFORMATION\n")
    f.write("=====================================\n\n")
    f.write(f"Cost       = {2 * result.cost}\n")
    f.write(f"Optimality = {result.optimality}\n")
    f.write(f"Nfev       = {result.nfev}\n")
    f.write(f"Success    = {result.success}\n")
    f.write(f"Message    = {result.message}\n\n")

    f.write("=====================================\n")
    f.write("FINAL RESIDUALS\n")
    f.write("=====================================\n\n")

    for i, r in enumerate(final_residuals):
        f.write(f"Residual {i+1}: {r:.6f}\n")

    f.write("\n=====================================\n")
    f.write("MODEL SOLUTION\n")
    f.write("=====================================\n\n")

    for exp in experiments:

        sol = solve_model(
            result.x, t_plot,
            D0=exp["D0"], P0=exp["P0"]
        )

        for t,T,F,D,P in zip(t_plot, sol[0], sol[1], sol[2], sol[3]):
            f.write(
                f"{t:.4f}\t{T:.6f}\t{F:.6f}\t"
                f"{D:.6f}\t{P:.6f}\n"
            )

print(f"Parameters saved to {output_dir}")

