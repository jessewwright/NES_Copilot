"""Ablation experiment for the NES norm-weight mechanism.

This script follows the specification provided in the user instructions. It
simulates synthetic datasets across a grid of conflict levels and norm weights,
fits two model variants (full vs. ablated) using Neural Posterior Estimation
(NPE) or ABC-SMC, computes evaluation metrics, and saves results and example
plots.
"""

from __future__ import annotations

import os
import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import SNPE, simulate_for_sbi
from torch import Tensor

try:
    import pyabc  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pyabc = None  # type: ignore

SEED = 20270606


def make_rng(seed: int | None = None) -> np.random.Generator:
    if seed is None:
        seed = SEED
    return np.random.default_rng(seed)


@dataclass
class SimulationParams:
    wn: float
    lam: float
    ws: float = 1.0
    a: float = 1.0
    noise: float = 1.0
    t0: float = 0.3
    dt: float = 0.01
    max_t: float = 5.0


class NESFull:
    def simulate(
        self, params: SimulationParams, n_trials: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        rows = [run_ddm_trial(params, rng) for _ in range(n_trials)]
        df = pd.DataFrame(rows)
        df["lam"] = params.lam
        df["wn_true"] = params.wn
        return df


class NESNoNorm:
    def simulate(
        self, params: SimulationParams, n_trials: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        fixed = SimulationParams(
            wn=0.0,
            lam=params.lam,
            ws=params.ws,
            a=params.a,
            noise=params.noise,
            t0=params.t0,
            dt=params.dt,
            max_t=params.max_t,
        )
        return NESFull().simulate(fixed, n_trials, rng)


def run_ddm_trial(p: SimulationParams, rng: np.random.Generator) -> dict:
    drift = p.ws - p.wn * p.lam
    evidence = 0.0
    t = 0.0
    while t < p.max_t:
        evidence += drift * p.dt + rng.normal(0.0, p.noise * np.sqrt(p.dt))
        t += p.dt
        if evidence >= p.a:
            return {"choice": 1, "rt": t + p.t0}
        if evidence <= -p.a:
            return {"choice": 0, "rt": t + p.t0}
    return {"choice": int(evidence > 0.0), "rt": p.max_t + p.t0}


def summary_stats(df: pd.DataFrame) -> np.ndarray:
    mean_rt = df["rt"].mean()
    var_rt = df["rt"].var(ddof=1)
    lam = df["lam"].iloc[0]
    wn = df["wn_true"].iloc[0]
    expected_choice = 1 if (1.0 - wn * lam) > 0 else 0
    err = np.mean(df["choice"] != expected_choice)
    return np.array([mean_rt, var_rt, err], dtype=np.float32)


def run_npe(model, data: pd.DataFrame, lam: float, seed: int = SEED):
    rng = make_rng(seed)
    prior = torch.distributions.LogNormal(torch.tensor(0.0), torch.tensor(0.5))

    def simulator(theta: Tensor) -> Tensor:
        wn_val = float(theta.item())
        p = SimulationParams(wn=wn_val, lam=lam)
        sim_df = model.simulate(p, n_trials=400, rng=rng)
        return torch.from_numpy(summary_stats(sim_df))

    theta, x = simulate_for_sbi(simulator, prior, num_simulations=30000)
    inference = SNPE(prior=prior, density_estimator="nsf")
    de = inference.append_simulations(theta, x).train(
        training_batch_size=128, max_num_epochs=30
    )
    posterior = inference.build_posterior(de)
    x_obs = torch.from_numpy(summary_stats(data))
    posterior.set_default_x(x_obs)
    return posterior


def run_abc_smc(model, data: pd.DataFrame, lam: float):
    if pyabc is None:
        raise RuntimeError("pyabc not installed")

    prior = pyabc.Distribution(wn=pyabc.RV("lognorm", 0.5, scale=np.exp(0)))

    def simulator(pars: dict) -> dict:
        p = SimulationParams(wn=pars["wn"], lam=lam)
        df = model.simulate(p, n_trials=400, rng=make_rng())
        s = summary_stats(df)
        return {"mrt": s[0], "vrt": s[1], "err": s[2]}

    def distance(x, x0):
        d = np.array([x["mrt"] - x0["mrt"], x["vrt"] - x0["vrt"], x["err"] - x0["err"]])
        return float(np.linalg.norm(d))

    obs = summary_stats(data)
    abc = pyabc.ABCSMC(simulator, prior, distance)
    abc.new("sqlite://", {"mrt": obs[0], "vrt": obs[1], "err": obs[2]})
    history = abc.run(max_nr_populations=6)
    df, _ = history.get_distribution()
    return df["wn"].to_numpy()


def waic_metric(model, data: pd.DataFrame, samples: Iterable[float]) -> float:
    x_obs = summary_stats(data)
    log_liks = []
    for wn in samples:
        p = SimulationParams(wn=wn, lam=data["lam"].iloc[0])
        sim_df = model.simulate(p, n_trials=400, rng=make_rng())
        x_sim = summary_stats(sim_df)
        log_lik = -0.5 * ((x_obs - x_sim) ** 2)
        log_liks.append(log_lik)
    log_liks = np.stack(log_liks)
    return float(az.waic(log_liks).waic)


def rmse_rt(model, data: pd.DataFrame, samples: Iterable[float]) -> float:
    lam = data["lam"].iloc[0]
    obs = data["rt"].mean()
    preds = []
    for wn in samples:
        p = SimulationParams(wn=wn, lam=lam)
        sim_df = model.simulate(p, n_trials=400, rng=make_rng())
        preds.append(sim_df["rt"].mean())
    return float(np.sqrt(np.mean((np.array(preds) - obs) ** 2)))


def run_experiment(use_abc: bool = False) -> pd.DataFrame:
    datasets: List[Tuple[pd.DataFrame, float, float]] = []
    rng = make_rng()
    for wn_true in [0.8, 1.2, 1.6]:
        for lam in np.linspace(0, 1, 6):
            p = SimulationParams(wn=wn_true, lam=lam)
            df = NESFull().simulate(p, n_trials=400, rng=rng)
            datasets.append((df, wn_true, lam))

    results = []
    for sim_id, (df, wn_true, lam) in enumerate(datasets):
        for model_obj, label in [(NESFull(), "NES-full"), (NESNoNorm(), "NES-ablated")]:
            if label == "NES-full":
                if use_abc:
                    samples = run_abc_smc(model_obj, df, lam)
                else:
                    post = run_npe(model_obj, df, lam)
                    samples = post.sample((1000,), x=summary_stats(df)).numpy()
            else:
                samples = np.zeros(1000)
            res = {
                "model": label,
                "sim_id": sim_id,
                "true_wn": wn_true,
                "post_mean_wn": float(samples.mean()),
                "mae": float(abs(samples.mean() - wn_true)),
                "waic": waic_metric(model_obj, df, samples),
                "rmse_rt": rmse_rt(model_obj, df, samples),
            }
            results.append(res)

    return pd.DataFrame(results)


def calibration_plot(df: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, sim_id in enumerate(sorted(df["sim_id"].unique())):
        ax = axes.flat[idx]
        subset = df[df["sim_id"] == sim_id]
        wn_true = subset["true_wn"].iloc[0]
        samples = subset[subset["model"] == "NES-full"]["post_mean_wn"]
        ax.hist(samples, bins=20, range=(0, 2), color="C0")
        ax.set_title(f"sim {sim_id}, wn={wn_true}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    df = run_experiment(use_abc=False)
    date = dt.date.today().isoformat()
    res_path = os.path.join("results", f"ablation_experiment_{date}.csv")
    df.to_csv(res_path, index=False)

    plot_path = os.path.join("figures", "calibration.png")
    calibration_plot(df[df["model"] == "NES-full"], plot_path)

    print(f"Results saved to {res_path}")
    print(f"Calibration plot saved to {plot_path}")
