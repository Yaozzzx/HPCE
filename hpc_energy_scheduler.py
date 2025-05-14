#!/usr/bin/env python3
"""
HPCEnergy Scheduler Script
Usage:
  python hpc_energy_scheduler.py --input_data FOLDER_NAME --clusters M --n_jobs N_JOBS --t_max T_MAX
This will load the pickled MAR2_hpl data under the specified subfolder, perform thermal-model fitting and ILP scheduling,
then combine results into a single CSV at ./results/<input_data>.csv.
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
import gurobipy as gp
from gurobipy import GRB

# suppress specific warnings
warnings.filterwarnings("ignore", message="Could not infer format, so each element")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def expand_pkg_to_core(pkg_df, n_cores):
    """
    Expand package-level power (rows × sockets) to core-level (rows × n_cores),
    by dividing each socket's power equally among its cores.
    Assumes equal number of cores per socket.
    """
    sockets = pkg_df.shape[1]
    cores_per_socket = n_cores // sockets
    core_power = pd.DataFrame(index=pkg_df.index,
                              columns=range(n_cores),
                              dtype=float)
    for sock in range(sockets):
        power_per_core = pkg_df.iloc[:, sock] / cores_per_socket
        start = sock * cores_per_socket
        end = start + cores_per_socket
        core_power.iloc[:, start:end] = power_per_core.values[:, None]
    return core_power


def main():
    parser = argparse.ArgumentParser(description='HPCEnergy Scheduler')
    parser.add_argument('--input_data', type=str, required=True,
                        help='Subfolder name under MAR2_hpl (e.g., 65k_64_56)')
    parser.add_argument('--clusters', type=int, default=5,
                        help='Number of frequency clusters (M)')
    parser.add_argument('--n_jobs', type=int, default=10,
                        help='Number of jobs to schedule')
    parser.add_argument('--t_max', type=float, default=100.0,
                        help='Maximum allowable temperature per core')
    args = parser.parse_args()

    # build path to pickle
    file_path = os.path.join(
        '.', 'data', 'MAR2_al', 'MAR2_hpl', args.input_data,
        'round0', 'preprocessing', 'round0.pkl'
    )
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    freq_df   = data['C0_norm_tsc_freq_core']
    temp_df   = data['temp_core']
    aperf_df  = data['aperf_core']
    mperf_df  = data['mperf_core']
    power_pkg = data['power_from_erg_pkg_cpu']

    N = freq_df.shape[1]
    M = args.clusters

    # Expand and normalize power
    power_core_raw = expand_pkg_to_core(power_pkg, N)
    util_ratio = aperf_df / mperf_df
    util_norm  = util_ratio.div(util_ratio.sum(axis=1), axis=0)
    power_core = util_norm.mul(power_core_raw)

    # Cluster frequencies
    freq_vals = freq_df.values.reshape(-1, 1)
    kmeans    = KMeans(n_clusters=M, random_state=0).fit(freq_vals)
    labels    = kmeans.labels_.reshape(freq_df.shape)

    f_ij = np.zeros((N, M))
    P_ij = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            mask = labels[:, i] == j
            if mask.any():
                f_ij[i, j] = freq_df.values[mask, i].mean()
                P_ij[i, j] = power_core.values[mask, i].mean()

    # Ambient & idle power
    c0_df     = data['C0_norm_tsc_core']
    idle_mask = c0_df <= 0.05
    p_bar     = power_core[idle_mask].mean().mean()

    # Build GS surrogate matrix
    T_a     = temp_df.min().values
    delta_T = (temp_df - T_a).values
    P_mat   = power_core.values

    GS = np.zeros((N, N))
    for ell in range(N):
        mlp = MLPRegressor(hidden_layer_sizes=(50,), activation='relu',
                           max_iter=1000, tol=1e-4, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mlp.fit(P_mat, delta_T[:, ell])
        W1 = mlp.coefs_[0]
        W2 = mlp.coefs_[1].flatten()
        GS[ell, :] = W1.dot(W2)

    mask_off = ~np.eye(N, dtype=bool)
    raw_vals = np.maximum(GS, 0.0)[mask_off]
    PHYS_MAX = np.percentile(raw_vals, 99)

    GS_cap = np.maximum(GS, 0.0).copy()
    GS_cap[mask_off] = np.minimum(GS_cap[mask_off], PHYS_MAX)
    GS_sym = 0.5 * (GS_cap + GS_cap.T)
    GS_sym[mask_off] = np.minimum(GS_sym[mask_off], PHYS_MAX)

    results_static   = []
    results_embedded = []

    # NN-static ILP
    model_s = gp.Model("NN_static")
    x_s = model_s.addVars(N, M, vtype=GRB.BINARY)
    y_s = model_s.addVars(N,     vtype=GRB.BINARY)
    model_s.setObjective(
        gp.quicksum(f_ij[i, j]*x_s[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE
    )
    model_s.addConstr(gp.quicksum(x_s[i, j] for i in range(N) for j in range(M)) == args.n_jobs)
    model_s.addConstr(gp.quicksum(y_s[i] for i in range(N)) == N - args.n_jobs)
    for i in range(N):
        model_s.addConstr(gp.quicksum(x_s[i, j] for j in range(M)) + y_s[i] == 1)
    for ell in range(N):
        expr = T_a[ell] + gp.quicksum(
            GS_sym[ell, i] * (p_bar*y_s[i] + gp.quicksum(P_ij[i, j]*x_s[i, j] for j in range(M)))
            for i in range(N)
        )
        model_s.addConstr(expr <= args.t_max)
    model_s.optimize()
    if model_s.status == GRB.OPTIMAL:
        x_sol = model_s.getAttr('x', x_s)
        y_sol = model_s.getAttr('x', y_s)
        core_p = {i: (p_bar if y_sol[i]>0.5 else next(P_ij[i, j] for j in range(M) if x_sol[i, j]>0.5)) for i in range(N)}
        temps = [T_a[ell] + sum(GS[ell, i]*core_p[i] for i in range(N)) for ell in range(N)]
        obj = model_s.ObjVal; max_t = max(temps)
        for i in range(N):
            st = 'idle' if y_sol[i]>0.5 else f"state_{[j for j in range(M) if x_sol[i,j]>0.5][0]}"
            results_static.append({'core':i, 'state':st, 'temp':temps[i],
                                   'obj_static':obj, 'max_t_static':max_t})

    # Embedded NN ILP
    mlp_full = MLPRegressor(hidden_layer_sizes=(50,), activation='relu',
                            max_iter=10000, tol=1e-4, random_state=0)
    mlp_full.fit(P_mat, delta_T)
    W1, b1 = mlp_full.coefs_[0], mlp_full.intercepts_[0]
    W2, b2 = mlp_full.coefs_[1], mlp_full.intercepts_[1]
    hsize = W1.shape[1]
    model_e = gp.Model("NN_embedded")
    x_e = model_e.addVars(N, M, vtype=GRB.BINARY)
    y_e = model_e.addVars(N,     vtype=GRB.BINARY)
    n_p = {i: p_bar*y_e[i] + gp.quicksum(P_ij[i,s]*x_e[i,s] for s in range(M)) for i in range(N)}
    model_e.addConstr(gp.quicksum(x_e[i,s] for i in range(N) for s in range(M))==args.n_jobs)
    model_e.addConstr(gp.quicksum(y_e[i] for i in range(N))==N-args.n_jobs)
    for i in range(N): model_e.addConstr(gp.quicksum(x_e[i,s] for s in range(M))+y_e[i]==1)
    model_e.setObjective(gp.quicksum(f_ij[i,s]*x_e[i,s] for i in range(N) for s in range(M)), GRB.MAXIMIZE)

    a = model_e.addVars(hsize, lb=-GRB.INFINITY)
    h = model_e.addVars(hsize, lb=0.0)
    for k in range(hsize):
        model_e.addConstr(a[k]==gp.quicksum(W1[i,k]*n_p[i] for i in range(N))+b1[k])
        model_e.addGenConstrMax(h[k],[a[k],0.0])
    Tvar = model_e.addVars(N, lb=-GRB.INFINITY)
    for ell in range(N):
        expr = T_a[ell]+b2[ell]+gp.quicksum(W2[k,ell]*h[k] for k in range(hsize))
        model_e.addConstr(Tvar[ell]==expr)
        model_e.addConstr(Tvar[ell]<=args.t_max)
    model_e.optimize()
    if model_e.status==GRB.OPTIMAL:
        x_sol = model_e.getAttr('x', x_e); y_sol = model_e.getAttr('x', y_e)
        solT   = model_e.getAttr('x', Tvar); obj_e = model_e.ObjVal; max_te = max(solT.values())
        for i in range(N):
            st = 'idle' if y_sol[i]>0.5 else f"state_{[j for j in range(M) if x_sol[i,j]>0.5][0]}"
            results_embedded.append({'core':i,'state':st,'temp':solT[i],
                                      'obj_embedded':obj_e,'max_t_embedded':max_te})

    # Combine via join with suffixes
    df_s = pd.DataFrame(results_static).set_index('core')
    df_e = pd.DataFrame(results_embedded).set_index('core')
    df    = df_s.join(df_e, how='inner', lsuffix='_static', rsuffix='_embedded')

    os.makedirs('results', exist_ok=True)
    out_csv = os.path.join('results', f'{args.input_data}_{args.clusters}_{args.n_jobs}_{args.t_max}.csv')
    df.to_csv(out_csv)
    print(f"Results saved to: {out_csv}")

if __name__ == '__main__':
    main()