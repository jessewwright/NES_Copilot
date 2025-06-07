import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import arviz as az


def load_subject_data(csv_path: str) -> pd.DataFrame:
    """Load a single subject CSV file."""
    return pd.read_csv(csv_path)


def _basic_summary_stats(df: pd.DataFrame) -> np.ndarray:
    """Compute simple summary statistics expected by the NES NPE model."""
    return np.array(
        [
            df["rt"].mean(),
            df["rt"].std(),
            df["response"].mean(),
            df["rt"].quantile(0.1),
            df["rt"].quantile(0.9),
        ]
    )


def _load_npe_posterior(checkpoint_path: str):
    """Load an sbi posterior from checkpoint."""
    from sbi.utils.io import load

    ckpt = load(checkpoint_path)
    return ckpt["posterior"]


def _sample_nes_posterior(
    posterior,
    x_obs: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Sample parameters from the NES posterior."""
    import torch  # local import

    samples = posterior.sample(
        (num_samples,), x=torch.as_tensor(x_obs, dtype=torch.float32)
    )
    return samples.detach().cpu().numpy()


def _wiener_like(
    rt: float,
    response: int,
    v: float,
    a: float,
    t: float,
    z: float = 0.5,
    sv: float = 0.0,
    sz: float = 0.0,
    st: float = 0.0,
) -> float:
    """Simple and crude Wiener log-likelihood approximation."""
    rt_adj = max(rt - t, 1e-6)
    # very rough approximation using an inverse Gaussian
    mu = (a * z) / max(v, 1e-6)
    lam = a**2
    pdf = np.sqrt(lam / (2 * np.pi * rt_adj**3)) * np.exp(
        -lam * (rt_adj - mu) ** 2 / (2 * mu**2 * rt_adj)
    )
    return np.log(max(pdf, 1e-300))


def fit_nes_model(
    subject_csv: str,
    npe_checkpoint_path: str,
    num_samples: int = 500,
) -> az.InferenceData:
    """Fit NES model and return InferenceData with log-likelihoods."""
    import torch  # local import to avoid hard dependency when unused

    df = load_subject_data(subject_csv)
    x_obs = _basic_summary_stats(df)
    posterior = _load_npe_posterior(npe_checkpoint_path)
    samples = _sample_nes_posterior(posterior, x_obs, num_samples)

    num_trials = len(df)
    log_lik = np.zeros((num_samples, num_trials))

    for i, params in enumerate(samples):
        # parameters order is determined by posterior
        v, a, t, *rest = params.tolist()
        for j, row in df.iterrows():
            log_lik[i, j] = _wiener_like(row["rt"], row["response"], v, a, t)

    posterior_dict = {f"param_{k}": samples[:, k] for k in range(samples.shape[1])}

    idata = az.from_dict(
        posterior=posterior_dict,
        log_likelihood={"nes": log_lik},
        observed_data={"rt": df["rt"].values, "response": df["response"].values},
    )
    return idata


def fit_hddm_model(
    subject_csv: str,
    num_samples: int = 1000,
    burn: int = 500,
) -> az.InferenceData:
    """Fit a basic HDDM model and return InferenceData with log-likelihoods."""
    import hddm  # local import

    df = load_subject_data(subject_csv)
    model = hddm.HDDM(df, include=["v", "a", "t"])
    model.sample(num_samples, burn=burn)

    trace = model.get_traces()
    posterior = {k: np.expand_dims(v.values, 0) for k, v in trace.items()}

    num_draws = trace.shape[0]
    num_trials = len(df)
    log_lik = np.zeros((num_draws, num_trials))

    for d in range(num_draws):
        params = trace.iloc[d]
        v, a, t = params["v"], params["a"], params["t"]
        for j, row in df.iterrows():
            log_lik[d, j] = _wiener_like(row["rt"], row["response"], v, a, t)

    idata = az.from_dict(
        posterior=posterior,
        log_likelihood={"hddm": log_lik},
        observed_data={"rt": df["rt"].values, "response": df["response"].values},
    )
    return idata


def compare_models(
    nes_idata: az.InferenceData,
    hddm_idata: az.InferenceData,
) -> Dict[str, pd.DataFrame]:
    """Return WAIC and LOO comparison tables."""
    cmp_dict = {"NES": nes_idata, "HDDM": hddm_idata}
    waic = az.compare(cmp_dict, ic="waic", scale="deviance")
    loo = az.compare(cmp_dict, ic="loo", scale="deviance")
    return {"waic": waic, "loo": loo}
