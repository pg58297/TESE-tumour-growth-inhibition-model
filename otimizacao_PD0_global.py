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

# Time points
t_data = np.array([0, 24, 48, 72])

# Tumor cell viability (example)
T_data1 = np.array([1, 0.6395, 1.372, 1.295833333])
T_data2 = np.array([1, 1.353666667, 2.064, 2.042833333])
T_data3 = np.array([1, 1.267333333, 1.950166667, 1.356333333])

#T_data = np.array([1, 1.086833333, 1.795388889, 1.565])

# pFAK / focal adhesion signal (example)
#F_data = np.array([]) - Not enough data

Aleatorio = 4

# ==========================================================
# REDUCED MODEL (NO DRUGS)
# ==========================================================
# D = 0
# P = 0

def reduced_model(t, X, rT, KT, rF, muT, muF):

    T, F = X

    dT = rT * F * T * (1 - T / KT) - muT * T

    dF = rF * T - muF * F

    return [dT, dF]

# ==========================================================
# SOLVER
# ==========================================================

def solve_model(params, t_eval, T0=1, F0=1):

    rT, KT, rF, muT, muF = params

    # Initial conditions from first experimental values
    X0 = [T0, F0]

    sol = solve_ivp(
        lambda t, X: reduced_model(t, X, rT, KT, rF, muT, muF),
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

    sol = solve_model(params, t_data)

    T_model = sol[0]
  
    res1 = T_model - T_data1
    res2 = T_model - T_data2
    res3 = T_model - T_data3

    return np.concatenate([res1, res2, res3])

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

output_dir = "Results_PD0"

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
        (0.0,10.0),         # rT
        (0.000001,10.0),    # KT
        (0.0,10.0),         # rF
        (0.0,10.0),         # muT
        (0.0,10.0)          # muF
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

result = least_squares(
    residuals,
    result_global.x,
    bounds=(
        [0.0, 0.000001, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0, 10.0]
    )
)


print("Local optimization finished")

# Estimated parameters
rT_est, KT_est, rF_est, muT_est, muF_est = result.x                             #(local)
rT_global, KT_global, rF_global, muT_global, muF_global = result_global.x       #(global)

# ==========================================================
# FINAL RESIDUALS
# ==========================================================

final_residuals = residuals(result.x)

# ==========================================================
# FINAL SOLUTION WITH ESTIMATED PARAMETERS
# ==========================================================

t_plot = np.linspace(t_data[0], t_data[-1], 500)

sol_final = solve_model(result.x, t_plot)

T_fit = sol_final[0]
F_fit = sol_final[1]

# ==========================================================
# PLOTS
# ==========================================================

print("Generating plots...")

# ----------------------------------------------------------
# T
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

# Ponto inicial comum
plt.scatter(
    [0],
    [1],
    s=80,
    marker="s",
    label="Initial condition"
)

# Experimental data - replicate 1
plt.scatter(
    t_data[1:],
    T_data1[1:],
    s=60,
    label="Experiment 1"
)

# Experimental data - replicate 2
plt.scatter(
    t_data[1:],
    T_data2[1:],
    s=60,
    label="Experiment 2"
)

# Experimental data - replicate 3
plt.scatter(
    t_data[1:],
    T_data3[1:],
    s=60,
    label="Experiment 3"
)

# Fitted solution
plt.plot(
    t_plot,
    T_fit,
    linewidth=2,
    label="Model fit"
)

plt.xlabel("Time (h)")
plt.ylabel("Normalized tumor cell viability")
plt.title("Least Squares Fit")

plt.ylim(0, 2.5)

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, f"T_fit_{Aleatorio}.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# F
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    t_plot,
    F_fit,
    linewidth=2,
    label="F(t)"
)

plt.xlabel("Time (h)")
plt.ylabel("Normalized FAK activity")
plt.title("Model-predicted FAK activity")

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, f"F_fit_{Aleatorio}.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SAVE RESULTS
# ==========================================================

with open(os.path.join(output_dir, f"results_PD0_global{Aleatorio}.txt"),"w") as f:

    f.write("=====================================\n")
    f.write("GLOBAL SEARCH\n")
    f.write("=====================================\n\n")

    f.write(f"rT   = {rT_global:.6f}\n")
    f.write(f"KT   = {KT_global:.6f}\n")
    f.write(f"rF   = {rF_global:.6f}\n")
    f.write(f"muT  = {muT_global:.6f}\n")
    f.write(f"muF  = {muF_global:.6f}\n\n")

    f.write("=====================================\n")
    f.write("GLOBAL OPTIMIZATION INFORMATION\n")
    f.write("=====================================\n\n")
   
    f.write(f"Global Nit  = {result_global.nit}\n")
    f.write(f"Global Nfev = {result_global.nfev}\n")
    f.write(f"Global cost = {result_global.fun}\n\n")

    f.write("=====================================\n")
    f.write("ESTIMATED PARAMETERS\n")
    f.write("=====================================\n\n")

    f.write(f"rT   = {rT_est:.6f}\n")
    f.write(f"KT   = {KT_est:.6f}\n")
    f.write(f"rF   = {rF_est:.6f}\n")
    f.write(f"muT  = {muT_est:.6f}\n")
    f.write(f"muF  = {muF_est:.6f}\n\n")

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

    final_residuals = residuals(result.x)

    for i, r in enumerate(final_residuals):
        f.write(f"Residual {i+1}: {r:.6f}\n")

    f.write("\n")

    f.write("=====================================\n")
    f.write("MODEL SOLUTION\n")
    f.write("=====================================\n\n")

    for t, T, F in zip(t_plot, T_fit, F_fit):

        f.write(
            f"{t:.4f}\t{T:.6f}\t{F:.6f}\n"
        )

print(f"Parameters saved to {output_dir}")































