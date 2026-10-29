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

# PEITC
T_P12 = np.array([
    [1, 0.529833333, 1.099, 1.051333333],            # P12_exp1
    [1, 0.857666667, 1.574, 1.111833333],            # P12_exp2
    [1, 0.672333333, 1.223666667, 0.802833333]       # P12_exp3
])

# ==========================================================
# 24 h Dose-response (PEITC)
# ==========================================================

T24_P = [

    {
        "P0": 1.25,
        "T": np.array([0.81433333, 0.68083333, 0.48433333])
                    #P1_25_exp1, P1_25_exp2, P1_25_exp3 
    },

    {
        "P0": 2.5,
        "T": np.array([0.8015, 0.66983333, 0.48916667])
                    #P2_5_exp1, P2_5_exp2, P2_5_exp3     
    },

    {
        "P0": 5.0,
        "T": np.array([0.82116667, 0.65833333, 0.48733333])
                    #P5_exp1, P5_exp2, P5_exp3                                            
    },

    {
        "P0": 10.0,
        "T": np.array([0.74666667, 0.53716667, 0.39283333])
                    #P10_exp1, P10_exp2, P10_exp3                                               
    },

    {
        "P0": 20.0,
        "T": np.array([0.12016667, 0.03733333, 0.04933333])
                    #P20_exp1, P20_exp2, P20_exp3
    },

    {
        "P0": 40.0,
        "T": np.array([0.104, 0.08816667, 0.0945])
                    #P40_exp1, P40_exp2, P40_exp3 
    },

    {
        "P0": 80.0,
        "T": np.array([0.02083333, 0.0076667, 0.012])
                    #P80_exp1, P80_exp2, P80_exp3
    }
    
]

# ==========================================================
# ALL EXPERIMENTS
# ==========================================================

experiments = [

    {
        "name": "P12",
        "P0": 12.0,
        "T": T_P12
    }

]

#Seed do metodo Global
Aleatorio=1

# ==========================================================
# PEITC MODEL
# ==========================================================

# Parameters estimated previously (control experiment)
rT   = 1.538797
KT   = 2.754682
rF   = 3.809131
muT  = 0.457442
muF  = 8.351323


def PEITC_model(t, X, muTP, muFP, muPT, muP):

    T, F, P = X

    dT = (
        rT * F * T * (1 - T / KT)
        - muTP * T * P
        - muT * T
    )

    dF = (
        rF * T
        - muFP * P * F
        - muF * F
    )

    dP = (
        - muPT * T * P
        - muP * P
    )

    return [dT, dF, dP]

# ==========================================================
# SOLVER
# ==========================================================

def solve_model(params, t_eval, T0=1, F0=1, P0=0):

    muTP, muFP, muPT, muP = params

    X0 = [T0, F0, P0]

    sol = solve_ivp(

        lambda t, X: PEITC_model(t, X, muTP, muFP, muPT, muP),

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
            params,
            t_data,
            T0=1,
            F0=1,
            P0=exp["P0"]
        )

        T_model = sol[0]

        for rep in exp["T"]:
            residual_list.extend(T_model - rep)

    for exp in T24_P:

        sol = solve_model(
            params,
            np.array([0,24]),
            T0=1,
            F0=1,
            P0=exp["P0"]
        )

        T24_model = sol[0][1]

        residual_list.extend(T24_model - exp["T"])

    return np.array(residual_list)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

output_dir = "Results_P"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# PARAMETER ESTIMATION (GLOBAL)
# ==========================================================

print("Starting Global Optimization...")

result_global = differential_evolution(

    objective_function,

    bounds=[

    (0,10),   # muTP
    (0,10),   # muFP
    (0,10),   # muPT
    (0,10)    # muP

    ],

    strategy="best1bin",

    maxiter=500,

    popsize=25,

    tol=1e-6,

    polish=False,

    seed=Aleatorio,

    disp=True
)

print("Finished Global...")

# ==========================================================
# LOCAL REFINEMENT
# ==========================================================

print("Starting Local Optimization...")

result = least_squares(

    residuals,

    result_global.x,

    bounds=(
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0]
    )

)

print("Finished Local...")

#===========================================================
# Estimated parameters
#===========================================================

muTP_est, muFP_est, muPT_est, muP_est = result.x                                #(local)
muTP_global, muFP_global, muPT_global, muP_global = result_global.x             #(global)

print("Estimated parameters:")
print(result.x)

# ==========================================================
# FINAL RESIDUALS
# ==========================================================

final_residuals = residuals(result.x)

# ==========================================================
# FINAL SOLUTION WITH ESTIMATED PARAMETERS
# ==========================================================

t_plot = np.linspace(t_data[0], t_data[-1], 500)

solutions = {}

for exp in experiments:

    solutions[exp["name"]] = solve_model(
        result.x,
        t_plot,
        P0=exp["P0"]
    )

# ==========================================================
# PLOTS
# ==========================================================

#------------------------------
# T
#------------------------------
for exp in experiments:

    sol = solve_model(
        result.x,
        t_plot,
        P0=exp["P0"]
    )

    T_fit = sol[0]
    F_fit = sol[1]
    P_fit = sol[2]

    plt.figure(figsize=(8,5))

    for i in range(3):

        plt.scatter(
            t_data,
            exp["T"][i],
            s=60,
            label=f"Experiment {i+1}"
        )

    for dose in T24_P:

        if dose["P0"] == exp["P0"]:

            plt.scatter(
                np.full(3, 24),
                dose["T"],
                marker="s",
                color="red",
                s=70,
                label="24 h dose-response"
            )

            break

    plt.plot(
        t_plot,
        T_fit,
        linewidth=2,
        label="Model fit"
    )

    plt.xlabel("Time (h)")
    plt.ylabel("Normalized tumor cell viability")
    plt.title(exp["name"])

    plt.ylim(0, 2.5)

    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            output_dir,
            f"T_fit_{exp['name']}_{Aleatorio}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------
# F
# ------------------------------
for exp in experiments:

    sol = solve_model(
        result.x,
        t_plot,
        P0=exp["P0"]
    )

    F_fit = sol[1]

    plt.figure(figsize=(8,5))

    plt.plot(
        t_plot,
        F_fit,
        linewidth=2,
        label="Model prediction"
    )

    plt.xlabel("Time (h)")
    plt.ylabel("Normalized FAK activity")
    plt.title(f"{exp['name']} - Model prediction")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            output_dir,
            f"F_fit_{exp['name']}_{Aleatorio}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------
# P
# ------------------------------
for exp in experiments:

    sol = solve_model(
        result.x,
        t_plot,
        P0=exp["P0"]
    )

    P_fit = sol[2]

    plt.figure(figsize=(8,5))

    plt.plot(
        t_plot,
        P_fit,
        linewidth=2,
        label="Model prediction"
    )

    plt.xlabel("Time (h)")
    plt.ylabel("Normalized PEITC")
    plt.title(f"{exp['name']} - Model prediction")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            output_dir,
            f"P_fit_{exp['name']}_{Aleatorio}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------
# P (24 h)
# ------------------------------

plt.figure(figsize=(8,5))

P_values = []
T_model = []

for exp in T24_P:

    sol = solve_model(
        result.x,
        np.array([0,24]),
        T0=1,
        F0=1,
        P0=exp["P0"]
    )

    P_values.append(exp["P0"])
    T_model.append(sol[0][1])

    plt.scatter(
        np.full(3, exp["P0"]),
        exp["T"],
        color="red",
        s=60
    )

plt.plot(
    P_values,
    T_model,
    "-o",
    linewidth=2,
    label="Model fit"
)

plt.xscale("log")

plt.xlabel("Initial PEITC concentration")
plt.ylabel("Normalized tumor cell viability (24 h)")

plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        output_dir,
        f"DoseResponse24h_P_{Aleatorio}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SAVE RESULTS
# ==========================================================

with open(os.path.join(output_dir, f"results_P_global{Aleatorio}.txt"),"w") as f:

    f.write("=====================================\n")
    f.write("GLOBAL SEARCH\n")
    f.write("=====================================\n\n")

    f.write(f"muTP = {muTP_global:.6f}\n")
    f.write(f"muFP = {muFP_global:.6f}\n")
    f.write(f"muPT = {muPT_global:.6f}\n")
    f.write(f"muP  = {muP_global:.6f}\n\n")

    f.write("=====================================\n")
    f.write("GLOBAL OPTIMIZATION INFORMATION\n")
    f.write("=====================================\n\n")
   
    f.write(f"Global Nit  = {result_global.nit}\n")
    f.write(f"Global Nfev = {result_global.nfev}\n")
    f.write(f"Global cost = {result_global.fun}\n\n")

    f.write("=====================================\n")
    f.write("ESTIMATED PARAMETERS\n")
    f.write("=====================================\n\n")

    f.write(f"muTP = {muTP_est:.6f}\n")
    f.write(f"muFP = {muFP_est:.6f}\n")
    f.write(f"muPT = {muPT_est:.6f}\n")
    f.write(f"muP  = {muP_est:.6f}\n\n")

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

    for exp in experiments:

        sol = solve_model(
            result.x,
            t_plot,
            P0=exp["P0"]
        )

        T_fit = sol[0]
        F_fit = sol[1]
        P_fit = sol[2]

        f.write(f"\n\n===== {exp['name']} =====\n\n")

        for t,T,F,P in zip(
            t_plot,
            T_fit,
            F_fit,
            P_fit
        ):

            f.write(
                f"{t:.4f}\t"
                f"{T:.6f}\t"
                f"{F:.6f}\t"
                f"{P:.6f}\n"
            )

print(f"Parameters saved to {output_dir}")


































