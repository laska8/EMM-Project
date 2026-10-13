import time

import pandas as pd

from logistic_model import TARGET, fit_logistic_model
from emm_selectors import (
    apply_description,
    generate_local_selectors,
    refine_description,
)


class BeamSearchEMM:

    def __init__(
        self,
        df,
        selector_columns,
        n_bins=8,
        beam_width=10,
        max_depth=3,
        min_support=5000,
        min_cell=10,
        alpha=0.5,
        target=TARGET
    ):
        self.df = df
        self.selector_columns = selector_columns
        self.n_bins = n_bins
        self.min_cell = min_cell

        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_support = min_support
        self.alpha = alpha

        # column whose coefficient is compared with the reference model
        self.target = target

        self.reference_result = None


    def fit_reference_model(self):

        self.reference_result = fit_logistic_model(
            self.df,
            self.target
        )

        if self.reference_result is None:
            raise RuntimeError(
                "Reference model did not converge."
            )

        return self.reference_result


    def evaluate_subgroup(self, description, subgroup):

        n = len(subgroup)

        if n < self.min_support:
            return None

        # users with/without a reciprocal partner who did/did not return.
        # With an empty cell the logit is separated: the coefficient runs
        # off to +-infinity (around -14 in practice) with a small standard
        # error, which would fill the beam with artefacts. This happens in
        # very active subgroups, where almost everyone has a partner.
        # The check always uses the binary indicator, also when another
        # reciprocity measure is the target, so the same subgroups are kept.
        cells = subgroup.groupby(
            ["recip_binary", "returned_next_month"]
        ).size()

        if len(cells) < 4 or cells.min() < self.min_cell:
            return None

        model_result = fit_logistic_model(
            subgroup,
            self.target
        )

        if model_result is None:
            return None

        beta_ref = self.reference_result["beta"]
        beta_sub = model_result["beta"]

        delta = abs(
            beta_sub - beta_ref
        )

        quality = (
            n ** self.alpha
        ) * delta

        return {
            "description": " AND ".join(
                str(selector)
                for selector in description
            ),

            "depth": len(description),

            "n": n,

            "return_rate":
                subgroup["returned_next_month"].mean(),

            "beta":
                beta_sub,

            "se":
                model_result["se"],

            "ci_low":
                model_result["ci_low"],

            "ci_high":
                model_result["ci_high"],

            "p_value":
                model_result["p_value"],

            "reference_beta":
                beta_ref,

            "delta":
                delta,

            "quality":
                quality,

            "_selectors":
                description,
        }


    def search(self):

        if self.reference_result is None:
            self.fit_reference_model()

        beam = [tuple()]
        all_results = []

        # "A AND B" and "B AND A" are the same subgroup
        evaluated = set()

        for depth in range(
            1,
            self.max_depth + 1
        ):

            candidates = []
            start = time.time()

            for description in beam:

                # the whole dataset at depth 1
                if description:
                    parent = self.df[
                        apply_description(self.df, description)
                    ]
                else:
                    parent = self.df

                # local discretization: the cut points come from the
                # subgroup being refined, so they get finer as it shrinks
                selectors = generate_local_selectors(
                    parent,
                    self.selector_columns,
                    self.n_bins
                )

                for selector in selectors:

                    new_description = refine_description(
                        description,
                        selector
                    )

                    if frozenset(new_description) in evaluated:
                        continue

                    evaluated.add(
                        frozenset(new_description)
                    )

                    subgroup = parent[selector.apply(parent)]

                    # the selector must remove something from the parent
                    if len(subgroup) == len(parent):
                        continue

                    result = self.evaluate_subgroup(
                        new_description,
                        subgroup
                    )

                    if result is not None:
                        candidates.append(result)

            candidates.sort(
                key=lambda x: x["quality"],
                reverse=True
            )

            all_results.extend(candidates)

            beam = [
                result["_selectors"]
                for result in candidates[
                    :self.beam_width
                ]
            ]

            print(
                f"Depth {depth}: "
                f"{len(candidates)} valid candidates "
                f"({(time.time() - start) / 60:.1f} min)",
                flush=True
            )

            if not beam:
                break

        results_df = pd.DataFrame(
            all_results
        )

        if results_df.empty:
            return results_df

        results_df = (
            results_df
            .drop(columns=["_selectors"])
            .sort_values(
                "quality",
                ascending=False
            )
            .reset_index(drop=True)
        )

        return results_df