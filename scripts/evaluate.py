import os 
import shutil
import argparse
import yaml
import hydra
import omegaconf as oc
import multiprocessing

import xarray as xr

import matplotlib.pyplot as plt

from evaluation.evaluators import EvaluatorBase


def evaluate_forecast(
        overrides: list,
        model_name: str,
        ax: plt.axis = None,
        evaluator: EvaluatorBase = None,
        verification_da: xr.DataArray = None,
        climatology_da: xr.DataArray = None,
        ) -> (plt.axis, str, dict):
    """
    Perform an analysis for a single forecast and variable. Optionally, default evaluation configurations are
    overridden.

    :param overrides: List of strings to override the default evaluation hydra configs
    :param model_name: The name of the model that is evaluated
    :param ax: A matplotlib axis instance to plot the results to
    :param evaluator: The evaluator instance performing the analyses
    :param verification_da: The verification data array, if known
    :param climatology_da: The climatology data array, if known
    :return: A tuple containing (1) the axis instance supplemented with the evaluations results, (2) a string of the
        directory where results for this evaluation are written to, and (3) a dictionary containing the verification
        and climatology data arrays used in this evaluation.
    """
    # Initialize the hydra configurations for this forecast
    with hydra.initialize(version_base=None, config_path="../evaluation/configs/", job_name=model_name):
        cfg = hydra.compose(config_name="hydra_config", overrides=overrides)
        cfg.model_name = model_name

    analysis = cfg.analysis
    if analysis.skip: return (ax, None, evaluator)
    print(f"### Performing '{analysis.name}' analysis on variable '{cfg.eval_variable}' with model "
          f"'{model_name}' ###")

    # Initialize evaluator
    if evaluator is None: evaluator = hydra.utils.instantiate(cfg.evaluator, forecast_path=cfg.paths.forecast)#, times=times)

    # Verification and climatology are not transfered (shared) from previous to current analysis by default
    if not cfg.transfer_das:
        verification_da = None
        climatology_da = None
    elif climatology_da is not None:
        evaluator.set_climatology(climatology_da=climatology_da)

    # Set or generate verification data array
    if verification_da is not None:
        evaluator.set_verification(verification_da=verification_da)
    else:
        if analysis.on_latlon:
            vname = evaluator.variable_metas[cfg.eval_variable]["fname_era5"]
            verification_path = os.path.join(cfg.paths.verification_ll + vname + ".nc")
        else:
            verification_path = os.path.join(cfg.paths.verification + cfg.eval_variable + ".nc")
        evaluator.generate_verification(verification_path=verification_path)

    if cfg.rescale_das:
        scale_verif = False if analysis.on_latlon else True
        evaluator.scale_das(scale_verif=scale_verif, scale_file_path=cfg.paths.scale_file)

    if "acc" in analysis.name.lower() and climatology_da is None and not analysis.on_latlon:
        # Precomputed climatology (specified as climatology_path) only exists for LatLon analysis
        analysis.arguments.climatology_path = None

    # Perform analysis
    result = getattr(evaluator, analysis.method)(**analysis.arguments)

    # Plot result
    if "plot_method" in analysis.keys():
        ax = getattr(evaluator, analysis.plot_method)(result, ax, model_name)
    
    # Write configuration settings for this analysis to file
    dst_path = os.path.join(cfg.paths.eval_directory, "configs", analysis.name.lower() + "_config.yaml")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as file:
        oc.OmegaConf.save(config=cfg, f=file)

    return (ax, cfg.paths.eval_directory, evaluator)


def process_variable(
        config: dict,
        vname: str
    ):
    """
    Process a single variable, e.g., z500, by performing all specified analyses. Results and are plotted to file.

    :param config: The main config specifying the analyses, variables, forecasts and overides
    :param vname: The variable considered here
    """

    # Dictionary to store and share data (e.g., forecast, verification, climatology) among analyses and models
    model_library = {}
    evaluator = verification_da = climatology_da = None

    for analysis_name in config["analyses"]:
        ax = None

        for model_name in config["forecasts"]["models"]:

            # Extract forecast model information
            on_latlon = True
            fc_dict = config["forecasts"]["models"][model_name]
            fc_path = os.path.join(config["forecasts"]["src_path"], fc_dict["forecast_path"])
            overrides = [f"+paths.forecast={fc_path}",
                         f"+paths.eval_directory={config['evaluation_directory']}",
                         f"eval_variable={vname}",
                         f"analysis={analysis_name}"]
            if "forecast_overrides" in fc_dict.keys():
                overrides += fc_dict["forecast_overrides"] 
            if "analysis_overrides" in fc_dict.keys() and analysis_name in fc_dict["analysis_overrides"].keys():
                overrides += fc_dict['analysis_overrides'][analysis_name]
            if "global_overrides" in config["forecasts"].keys() and config["forecasts"]["global_overrides"] is not None:
                overrides += config["forecasts"]["global_overrides"]

            evaluator = model_library[model_name] if model_name in model_library.keys() else None

            # Perform analysis
            ax, eval_directory, evaluator = evaluate_forecast(
                overrides=overrides,
                model_name=model_name,
                ax=ax,
                evaluator=evaluator,
                verification_da=verification_da,
                climatology_da=climatology_da
                )

            # Update
            if evaluator is not None:
                model_library[model_name] = evaluator
                verification_da = evaluator.verification_da
                climatology_da = evaluator.climatology_da

        if ax is not None:
            # Finalize plot and write it to file
            ax.legend(fontsize="small")
            dst_path = os.path.join(eval_directory, "plots", analysis_name)
            os.makedirs(dst_path, exist_ok=True)
            plt.savefig(f"{os.path.join(dst_path, vname + '_' + analysis_name)}.pdf", format="pdf")
            plt.close()
    

def main(config: dict):
    """
    Performs all analyses on the forecast files specified in the config. Results are plotted and written to file along
    with the according config file.

    :param config: The main config specifying the analyses, variables, forecasts and overides
    """
    if config["evaluate_variables_in_parallel"]:
        print("Initializing and starting separate processes for each variable")
        processes = []
        for vname in config["variables"]:
            process = multiprocessing.Process(target=process_variable, args=(config, vname, ))
            process.start()
            processes.append(process)
        # Wait for each process to finish
        for process in processes:
            process.join()
    else:
        print("Processing variables iteratively")
        for vname in config["variables"]:
            process_variable(config=config, vname=vname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Forecast[s]")
    parser.add_argument('-config-path', type=str, default="evaluation/configs/main_config.yaml")
    run_args = parser.parse_args()

    with open(run_args.config_path, 'r') as stream: 
        config = yaml.safe_load(stream)
    main(config=config)
    print("Done.")    
