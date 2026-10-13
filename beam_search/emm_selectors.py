from dataclasses import dataclass

@dataclass(frozen =True)
class Selector:
    column:str
    operator: str
    value: float

    def apply(self,df):
        if self.operator == "<=":
            return df[self.column] <= self.value
        if self.operator == ">":
            return df[self.column] > self.value
        if self.operator == "==":
            return df[self.column] == self.value
        raise ValueError("Unknown operator")

    def __str__(self):
        return f"{self.column}{self.operator}{self.value}"


# LBCA strategy of Meeng & Knobbe (2021): Local, Binary, Coarse, All.
# The cut points are the equal-frequency bin boundaries of the subgroup
# that is being refined (not of the whole dataset), every cut point gives
# a "<=" and a ">" selector, and all of them are returned.
# Difference from the paper (strategy 3-lbca): it uses "<=" and ">=" with
# cut points computed separately for each operator, here both selectors
# share one cut point and split the subgroup into two complementary parts.
def generate_local_selectors(subgroup, col, n_bins=8):
    quantiles = [i / n_bins for i in range(1, n_bins)]

    selectors=[]
    for cols in col:
        # "lower" keeps the cut points at values that occur in the subgroup
        thresholds = (subgroup[cols].dropna().quantile(quantiles, interpolation="lower").unique())

        for threshold in thresholds:
            selectors.append(Selector(cols, "<=", float(threshold)))
            selectors.append(Selector(cols, ">", float(threshold)))

    return selectors


def refine_description(description, selector):
    # a local cut point lies inside the subgroup, so a new bound on the
    # same side of the same column is tighter and replaces the old one
    # (replies_sent>1 refined with replies_sent>3 is just replies_sent>3)
    kept = tuple(
        old for old in description
        if (old.column, old.operator) != (selector.column, selector.operator)
    )

    return kept + (selector,)


def apply_description(df, description):
    mask = True
    for selector in description:
        mask = mask & selector.apply(df)

    return mask
