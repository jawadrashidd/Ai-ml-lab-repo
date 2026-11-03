"""
Helper functions for the EEE 4710 labs.

Each function is short and commented, so you can read exactly what it does.
Use help(lab_utils.function_name) to see how to call a function.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_missing(df):
    """Show how many values are missing in each column, and what percentage of the rows that is."""
    count = df.isnull().sum()               # number of empty cells in each column
    percent = count / len(df) * 100         # as a percentage of all rows
    table = pd.DataFrame({"missing": count, "percent": percent.round(1)})
    return table.sort_values(by="missing", ascending=False)


def target_rate(df, column, target):
    """For each value of `column`, show the average of `target` and how many rows have that value.

    Example: target_rate(df, "Sex", "Survived") gives the survival rate of men and of women.
    """
    table = df.groupby(column)[target].agg(["mean", "count"])
    table = table.rename(columns={"mean": target + "_rate"})
    return table.sort_values(by=target + "_rate", ascending=False)


def plot_target_rate(df, column, target):
    """Bar chart of the average `target` for each value of `column`."""
    sns.barplot(data=df, x=column, y=target, errorbar=None)
    plt.ylabel(target + " rate")
    plt.title(target + " rate by " + column)
    plt.show()
