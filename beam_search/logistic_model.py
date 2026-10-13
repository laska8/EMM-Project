import gc

import pyfixest as pf


# reciprocity measure of the main analysis: at least one reciprocal partner
TARGET = "recip_binary"


# same model as logistic_resgression/pooled_logistic.ipynb:
# logit with month and subreddit fixed effects,
# standard errors clustered by user
def model_formula(target=TARGET):
    return (
        f"returned_next_month ~ {target} + log_sent + log_received"
        " | month + subreddit"
    )


def fit_logistic_model(df, target=TARGET):

    # no reciprocity effect to estimate when it does not vary
    # in the subgroup (e.g. users who received no replies)
    if df[target].nunique() < 2:
        return None

    try:
        model = pf.feglm(
            model_formula(target),
            data=df,
            family="logit",
            vcov={"CRV1": "user"},
            copy_data=False,
            fixef_rm="none",
            iwls_maxiter=100,
        )

    # raised when every variable is collinear with the fixed effects
    except ValueError:
        return None

    converged = model.convergence
    estimates = model.tidy()

    # the fitted model keeps several copies of the data and is only
    # freed by the garbage collector, so collect before the next fit
    del model
    gc.collect()

    if not converged:
        return None

    # reciprocity itself was dropped as collinear
    if target not in estimates.index:
        return None

    row = estimates.loc[target]

    return {
        "beta": row["Estimate"],
        "se": row["Std. Error"],
        "ci_low": row["2.5%"],
        "ci_high": row["97.5%"],
        "p_value": row["Pr(>|t|)"],
    }
