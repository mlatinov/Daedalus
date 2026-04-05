import numpy as np
import pandas as pd
from scipy import stats


def column_transform(data: pd.DataFrame, column: str, transformation: str) -> pd.DataFrame:
    """
    Apply a numeric transformation to a column in a DataFrame.

    Supported transformations:
        - "log"          : Natural log (log1p to handle zeros safely)
        - "boxcox"       : Box-Cox power transform (requires positive values)
        - "yeojohnson"   : Yeo-Johnson power transform (handles negatives and zeros)
        - "sqrt"         : Square-root transform (clips negatives to zero first)

    Returns a copy of data with the specified column replaced by its transformed values.
    """
    data = data.copy()
    col_values = data[column].astype(float)

    if transformation == "log":
        data[column] = np.log1p(col_values)

    elif transformation == "boxcox":
        # Box-Cox requires strictly positive values — shift if necessary
        shift = max(0, -col_values.min() + 1e-6)
        transformed, _ = stats.boxcox(col_values + shift)
        data[column] = transformed

    elif transformation == "yeojohnson":
        transformed, _ = stats.yeojohnson(col_values)
        data[column] = transformed

    elif transformation == "sqrt":
        data[column] = np.sqrt(col_values.clip(lower=0))

    else:
        raise ValueError(
            f"Unknown transformation '{transformation}'. "
            "Choose from: 'log', 'boxcox', 'yeojohnson', 'sqrt'."
        )

    return data
