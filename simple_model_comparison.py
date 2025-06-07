"""Example script using the model_comparison_pipeline utilities."""

import argparse
import arviz as az

from nes_copilot.model_comparison_pipeline import (
    fit_nes_model,
    fit_hddm_model,
    compare_models,
)


def main(args: argparse.Namespace) -> None:
    nes_idata = fit_nes_model(args.subject_csv, args.nes_checkpoint, args.nes_samples)
    hddm_idata = fit_hddm_model(args.subject_csv, args.hddm_samples, args.hddm_burn)

    nes_idata.to_netcdf("nes_idata.nc")
    hddm_idata.to_netcdf("hddm_idata.nc")

    comps = compare_models(nes_idata, hddm_idata)
    print("WAIC comparison:\n", comps["waic"])
    print("LOO comparison:\n", comps["loo"])

    comps["waic"].to_csv("waic_comparison.csv")
    comps["loo"].to_csv("loo_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model comparison pipeline")
    parser.add_argument("subject_csv")
    parser.add_argument("nes_checkpoint")
    parser.add_argument("--nes_samples", type=int, default=500)
    parser.add_argument("--hddm_samples", type=int, default=1000)
    parser.add_argument("--hddm_burn", type=int, default=500)
    main(parser.parse_args())
