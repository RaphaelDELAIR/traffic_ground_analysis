import pandas as pd


def within_Xmin(real, pred, m=1):
    assert len(real) == len(pred)
    c = 0
    for r, p in zip(real, pred):
        if abs(r - p) < m:
            c += 1
    return c / len(real) * 100


def daterange(start_date, end_date):
    """
    Utils function to make the list of days for parallelization
    """
    trange = end_date - start_date
    for n in range(trange.days):
        yield start_date + n * pd.Timedelta("1d")
