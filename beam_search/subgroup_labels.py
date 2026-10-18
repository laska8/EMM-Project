import re


def readable(description):
    """'replies_sent>1.0 AND replies_sent<=14.0' -> '1 < sent ≤ 14' for figure labels."""

    bounds = {}

    for part in description.split(" AND "):
        column, operator, value = re.match(
            r"replies_(\w+?)(<=|>)(.+)", part
        ).groups()
        bounds.setdefault(column, {})[operator] = f"{float(value):g}"

    conditions = []

    # always "sent" before "received"
    for column in ["sent", "received"]:
        bound = bounds.get(column, {})

        # a lower and an upper bound on one attribute form an interval
        if ">" in bound and "<=" in bound:
            conditions.append(f"{bound['>']} < {column} ≤ {bound['<=']}")
        elif ">" in bound:
            conditions.append(f"{column} > {bound['>']}")
        elif "<=" in bound:
            conditions.append(f"{column} ≤ {bound['<=']}")

    return ", ".join(conditions)
