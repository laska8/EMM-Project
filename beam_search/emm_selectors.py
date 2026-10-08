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


def generate_numeric_selectors(df, col, quantiles=(0.25, 0.5, 0.75)):
    selectors=[]
    for cols in col:
        thresholds = (df[cols].dropna().quantile(quantiles).unique())

        for threshold in thresholds:
            selectors.append(Selector(cols, "<=", float(threshold)))
            selectors.append(Selector(cols, ">", float(threshold)))

    return selectors


def apply_description(df, description):
    mask = True
    for selector in description:
        mask = mask & selector.apply(df)

    return mask
