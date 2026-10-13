import gc

import pyfixest as pf


# same model as logistic_resgression/pooled_logistic.ipynb:
# logit with month and subreddit fixed effects,
# standard errors clustered by user
FORMULA = (
    "returned_next_month ~ recip_binary + log_sent + log_received"
    " | month + subreddit"
)

TARGET = "recip_binary"


def fit_logistic_model(df, formula=FORMULA):

    # no reciprocity effect to estimate when it does not vary
    # in the subgroup (e.g. users who received no replies)
    if df[TARGET].nunique() < 2:
        return None

    try:
        model = pf.feglm(
            formula,
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
    if TARGET not in estimates.index:
        return None

    row = estimates.loc[TARGET]

    return {
        "beta": row["Estimate"],
        "se": row["Std. Error"],
        "ci_low": row["2.5%"],
        "ci_high": row["97.5%"],
        "p_value": row["Pr(>|t|)"],
    }
