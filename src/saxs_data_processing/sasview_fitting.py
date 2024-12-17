from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
from bumps.names import *


def fit_power_law(data):
    """
    Fit a sasview power law model to data

    :param data: sasdata data object
    :type data: sasdata data object
    :return results: dictionary with parameters and values
    :type results: dict
    :param result: sasmodels/bumps result object
    :type result: ^^
    :param problem: sasmodels/bumps problem object
    :type problem: ^^
    """

    kernel = load_model("power_law")

    pars = dict(scale=1, background=0.001, power=4.0)
    model = Model(kernel, **pars)
    model.power.range(0, 10)
    model.scale.range(0, 5)

    M = Experiment(data=data, model=model)
    problem = FitProblem(M)

    result = fit(problem, method="amoeba")

    results = {}
    for l, v, dv in zip(problem.labels(), result.x, result.dx):
        results[l] = v
        results[l + "_uncertainty"] = dv
    results["chisq"] = problem.chisq_str()

    return results, result, problem


def fit_sphere(
    data, polydispersity=True, r_init=600, pd_init=0.15, pd_distribution="lognormal"
):
    """
    Fit a sphere model to data. Works in angstroms

    :param data: q/I/sig data in Brenden df format
    :type data: pandas dataframe
    :param polydispersity: whether or not to fit lognormal polydispersity parameters
    :type polydispersity: bool
    :param r_init: initial radius guess, in angstroms. default 600 (120nm diameter particle)
    :type r_init: int
    :return results: dict of result values
    :rtype results: dict
    :return result: fit(problem) object
    :rtype results: bumps problem result
    :return problem: bumps problem
    :rtype problem: bumps problem
    """

    kernel = load_model("sphere")

    # set up model
    pars = dict(scale=1, background=0.001, sld=1, radius=r_init, radius_pd=pd_init)
    model = Model(kernel, **pars)
    model.radius.range(10, 5000)
    model.scale.range(0, 5)
    if polydispersity:
        model.radius_pd.range(0, 1)
        model.radius_pd_type = pd_distribution

    M = Experiment(data=data, model=model)
    problem = FitProblem(M)

    result = fit(problem, method="amoeba")

    results = {}

    for l, v, dv in zip(problem.labels(), result.x, result.dx):
        results[l] = v
        results[l + "_uncertainty"] = dv

    results["chisq"] = problem.chisq_str()

    return results, result, problem
