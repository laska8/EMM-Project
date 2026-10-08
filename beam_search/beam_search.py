import pandas as pd

from logistic_model import fit_logistic_model
from emm_selectors import apply_description


class BeamSearchEMM:

    def __init__(
        self,
        df,
        selectors,
        beam_width=10,
        max_depth=2,
        min_support=5000,
        alpha=0.5
    ):
        self.df = df
        self.selectors = selectors

        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_support = min_support
        self.alpha = alpha

        self.reference_result = None


    def fit_reference_model(self):

        self.reference_result = fit_logistic_model(
            self.df
        )

        if self.reference_result is None:
            raise RuntimeError(
                "Reference model did not converge."
            )

        return self.reference_result


    def evaluate_subgroup(self, description):

        mask = apply_description(
            self.df,
            description
        )

        subgroup = self.df[mask]

        n = len(subgroup)

        if n < self.min_support:
            return None

        if subgroup["returned_next_month"].nunique() < 2:
            return None

        model_result = fit_logistic_model(
            subgroup
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

            for description in beam:

                used_columns = {
                    selector.column
                    for selector in description
                }

                for selector in self.selectors:

                    # avoid using same attribute twice
                    if selector.column in used_columns:
                        continue

                    new_description = (
                        description + (selector,)
                    )

                    if frozenset(new_description) in evaluated:
                        continue

                    evaluated.add(
                        frozenset(new_description)
                    )

                    result = self.evaluate_subgroup(
                        new_description
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
                f"{len(candidates)} valid candidates"
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