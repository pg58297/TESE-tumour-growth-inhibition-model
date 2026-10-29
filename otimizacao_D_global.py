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

# Dasatinib 
T_D1_25 = np.array([
    [1, 0.597166667, 1.352166667, 1.576166667],        #D1_25_exp1
    [1, 1.281166667, 2.200166667, 2.061],              #D1_25_exp2
    [1, 1.261166667, 2.224833333, 1.262]               #D1_25_exp3
])

T_D2_5 = np.array([
    [1, 0.525166667, 1.388833333, 1.392333333],        #D2_5_exp1
    [1, 1.232333333, 2.077833333, 2.263166667],        #D2_5_exp2
    [1, 1.206666667, 2.204166667, 1.2385]              #D2_5_exp3
])

T_D5 = np.array([
    [1, 0.471833333, 1.236666667, 1.348333333],        #D5_exp1
    [1, 1.034666667, 2.0205, 1.848333333],             #D5_exp2
    [1, 1.088, 1.696333333, 1.104833333]               #D5_exp3
])

T_D10 = np.array([
    [1, 0.5295, 0.935166667, 1.023166667],             #D10_exp1
    [1, 0.959833333, 1.739833333, 1.384333333],        #D10_exp2
    [1, 0.972, 1.393666667, 0.921333333]               #D10_exp3
])

# ==========================================================
# 24 h Dose-response (Dasatinib)
# ==========================================================

T24_D = [

    {
        "D0": 1.25,
        "T": np.array([0.814, 0.63633333, 0.58716667])
                    #D1_25_exp1, D1_25_exp2, D1_25_exp3 
    },

    {
        "D0": 2.5,
        "T": np.array([0.76933333, 0.6365, 0.52683333])
                    #D2_5_exp1, D2_5_exp2, D2_5_exp3     
    },

    {
        "D0": 5.0,
        "T": np.array([0.712, 0.5755, 0.48333333])
                    #D5_exp1, D5_exp2, D5_exp3                                            
    },

    {
        "D0": 10.0,
        "T": np.array([0.632, 0.47783333, 0.40583333])
                    #D10_exp1, D10_exp2, D10_exp3                                               
    },

    {
        "D0": 20.0,
        "T": np.array([0.46433333, 0.34766667, 0.25066667])
                    #D20_exp1, D20_exp2, D20_exp3
    },

    {
        "D0": 40.0,
        "T": np.array([0.32066667, 0.22433333, 0.1915])
                    #D40_exp1, D40_exp2, D40_exp3 
    },

    {
        "D0": 80.0,
        "T": np.array([0.15966667, 0.06583333, 0.0645])
                    #D80_exp1, D80_exp2, D80_exp3
    }

]

# ==========================================================
# ALL EXPERIMENTS
# ==========================================================

experiments = [

    {
        "name": "D1_25",
        "D0": 1.25,
        "T": T_D1_25
    },

    {
        "name": "D2_5",
        "D0": 2.5,
        "T": T_D2_5
    },

    {
        "name": "D5",
        "D0": 5.0,
        "T": T_D5
    },

    {
        "name": "D10",
        "D0": 10.0,
        "T": T_D10
    },

]

#Seed do metodo Global
Aleatorio=1

# ==========================================================
# Dasatinib MODEL
# ==========================================================

# Parameters estimated previously (control experiment)
rT   = 1.538797
KT   = 2.754682
rF   = 3.809131
muT  = 0.457442
muF  = 8.351323

def Dasatinib_model(t, X, muTD, muDF, muDT, muD):

    T, F, D = X

    dT = (
        rT * F * T * (1 - T / KT)
        - muTD * T * D
        - muT * T
    )

    dF = (
        rF * T
        - muF * F
    )

    dD = (
        - muDF * D * F
        - muDT * T * D
        - muD * D
    )

    return [dT, dF, dD]

# ==========================================================
# SOLVER
# ==========================================================

def solve_model(params, t_eval, T0=1, F0=1, D0=0):

    muTD, muDF, muDT, muD = params

    X0 = [T0, F0, D0]

    sol = solve_ivp(

        lambda t, X: Dasatinib_model(t, X, muTD, muDF, muDT, muD),

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
            D0=exp["D0"]
        )

        T_model = sol[0]

        for rep in exp["T"]:
            residual_list.extend(T_model - rep)

    for exp in T24_D:

        sol = solve_model(
            params,
            np.array([0,24]),
            T0=1,
            F0=1,
            D0=exp["D0"]
        )

        T24_model = sol[0][1]

        residual_list.extend(T24_model - exp["T"])

    return np.array(residual_list)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

output_dir = "Results_D"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# PARAMETER ESTIMATION (GLOBAL)
# ==========================================================

print("Starting Global Optimization...")

result_global = differential_evolution(

    objective_function,

    bounds=[

    (0,10),   # muTD
    (0,10),   # muDF
    (0,10),   # muDT
    (0,10),   # muD

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

muTD_est, muDF_est, muDT_est, muD_est = result.x                                #(local)
muTD_global, muDF_global, muDT_global, muD_global = result_global.x             #(global)

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
        D0=exp["D0"]
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
        D0=exp["D0"]
    )

    T_fit = sol[0]
    F_fit = sol[1]
    D_fit = sol[2]

    plt.figure(figsize=(8,5))

    for i in range(3):

        plt.scatter(
            t_data,
            exp["T"][i],
            s=60,
            label=f"Experiment {i+1}"
        )

    for dose in T24_D:

        if dose["D0"] == exp["D0"]:

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
        D0=exp["D0"]
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
# D
# ------------------------------
for exp in experiments:

    sol = solve_model(
        result.x,
        t_plot,
        D0=exp["D0"]
    )

    D_fit = sol[2]

    plt.figure(figsize=(8,5))

    plt.plot(
        t_plot,
        D_fit,
        linewidth=2,
        label="Model prediction"
    )

    plt.xlabel("Time (h)")
    plt.ylabel("Normalized Dasatinib")
    plt.title(f"{exp['name']} - Model prediction")

    plt.grid(True)
    plt.legend()

    plt.savefig(
        os.path.join(
            output_dir,
            f"D_fit_{exp['name']}_{Aleatorio}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ------------------------------
# D (24 h)
# ------------------------------

plt.figure(figsize=(8,5))

D_values = []
T_model = []

for exp in T24_D:

    sol = solve_model(
        result.x,
        np.array([0,24]),
        T0=1,
        F0=1,
        D0=exp["D0"]
    )

    D_values.append(exp["D0"])
    T_model.append(sol[0][1])

    plt.scatter(
        np.full(3, exp["D0"]),
        exp["T"],
        color="red",
        s=60
    )

plt.plot(
    D_values,
    T_model,
    "-o",
    linewidth=2,
    label="Model fit"
)

plt.xscale("log")

plt.xlabel("Initial Dasatinib concentration")
plt.ylabel("Normalized tumor cell viability (24 h)")

plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        output_dir,
        f"DoseResponse24h_D_{Aleatorio}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SAVE RESULTS
# ==========================================================

with open(os.path.join(output_dir, f"results_D_global{Aleatorio}.txt"),"w") as f:

    f.write("=====================================\n")
    f.write("GLOBAL SEARCH\n")
    f.write("=====================================\n\n")

    f.write(f"muTD = {muTD_global:.6f}\n")
    f.write(f"muDF = {muDF_global:.6f}\n")
    f.write(f"muDT = {muDT_global:.6f}\n")
    f.write(f"muD  = {muD_global:.6f}\n")

    f.write("=====================================\n")
    f.write("GLOBAL OPTIMIZATION INFORMATION\n")
    f.write("=====================================\n\n")
   
    f.write(f"Global Nit  = {result_global.nit}\n")
    f.write(f"Global Nfev = {result_global.nfev}\n")
    f.write(f"Global cost = {result_global.fun}\n\n")

    f.write("=====================================\n")
    f.write("ESTIMATED PARAMETERS\n")
    f.write("=====================================\n\n")

    f.write(f"muTD = {muTD_est:.6f}\n")
    f.write(f"muDF = {muDF_est:.6f}\n")
    f.write(f"muDT = {muDT_est:.6f}\n")
    f.write(f"muD  = {muD_est:.6f}\n")

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
            D0=exp["D0"]
        )

        T_fit = sol[0]
        F_fit = sol[1]
        D_fit = sol[2]

        f.write(f"\n\n===== {exp['name']} =====\n\n")

        for t,T,F,D in zip(
            t_plot,
            T_fit,
            F_fit,
            D_fit
        ):

            f.write(
                f"{t:.4f}\t"
                f"{T:.6f}\t"
                f"{F:.6f}\t"
                f"{D:.6f}\n"
            )

print(f"Parameters saved to {output_dir}")


































