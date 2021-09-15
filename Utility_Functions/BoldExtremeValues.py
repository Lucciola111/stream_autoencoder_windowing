import numpy as np


def bold_extreme_values(data, col=True, format_string="%.2f", bolded_value=True, nearest_value=100):
    """

    Parameters
    ----------
    data: Data frame which should be transformed
    col: Define whether a row or a column be should examined
    format_string: Round of strings
    bolded_value: Value which should be highlighted
    nearest_value: Value if a value which is nearest to a defined value should be highlighted

    Returns a transformed data frame
    -------

    """

    def find_nearest_value(array):
        array = np.asarray(array)
        idx = (np.abs(array - nearest_value)).argmin()
        return float(array[idx])

    # If maximum value should be bold
    if bolded_value == 'max':
        # Apply columnwise
        if col:
            extrema = data != data.max()
            extrema = extrema.iloc[:, 0]
        # Apply rowwise
        else:
            extrema = data.apply(lambda x: x != data.max(axis=1))
            extrema = extrema.iloc[0]
    # If minimum value should be bold
    elif bolded_value == 'min':
        # Apply columnwise
        if col:
            extrema = data != data.min()
            extrema = extrema.iloc[:, 0]
        # Apply rowwise
        else:
            extrema = data.apply(lambda x: x != data.min(axis=1))
            extrema = extrema.iloc[0]
    # If value closest to individually defined value should be bold
    else:
        # Apply columnwise
        if col:
            extrema = data != find_nearest_value(data)
            extrema = extrema.iloc[:, 0]
        # Apply rowwise
        else:
            extrema = data.apply(lambda x: x != find_nearest_value(data))
            extrema = extrema.iloc[0]

    # Adapt latex
    axis = 1 if col else 0
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x, axis=axis)
    formatted = data.apply(lambda x: format_string % x, axis=axis)
    return formatted.where(extrema, bolded)
