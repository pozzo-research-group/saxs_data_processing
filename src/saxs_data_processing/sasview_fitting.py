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
