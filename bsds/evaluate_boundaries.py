import numpy as np
from . import thin, correspond_pixels


def evaluate_boundaries_bin(predicted_boundaries_bin, gt_boundaries,
                            max_dist=0.0075, apply_thinning=True):
    """
    Evaluate the accuracy of a predicted boundary.

    :param predicted_boundaries_bin: the predicted boundaries as a (H,W)
    binary array
    :param gt_boundaries: a list of ground truth boundaries, as returned
    by the `load_boundaries` or `boundaries` methods
    :param max_dist: (default=0.0075) maximum distance parameter
    used for determining pixel matches. This value is multiplied by the
    length of the diagonal of the image to get the threshold used
    for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
    thinning to the predicted boundaries before evaluation
    :return: tuple `(count_r, sum_r, count_p, sum_p)` where each of
    the four entries are float values that can be used to compute
    recall and precision with:
    ```
    recall = count_r / (sum_r + (sum_r == 0))
    precision = count_p / (sum_p + (sum_p == 0))
    ```
    """
    acc_prec = np.zeros(predicted_boundaries_bin.shape, dtype=bool)
    predicted_boundaries_bin = predicted_boundaries_bin != 0

    if apply_thinning:
        predicted_boundaries_bin = thin.binary_thin(predicted_boundaries_bin)

    sum_r = 0
    count_r = 0
    for gt in gt_boundaries:
        match1, match2, cost, oc = correspond_pixels.correspond_pixels(
            predicted_boundaries_bin, gt, max_dist=max_dist
        )
        match1 = match1 > 0
        match2 = match2 > 0
        # Precision accumulator
        acc_prec = acc_prec | match1
        # Recall
        sum_r += gt.sum()
        count_r += match2.sum()

    # Precision
    sum_p = predicted_boundaries_bin.sum()
    count_p = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p


def evaluate_boundaries(predicted_boundaries, gt_boundaries,
                        thresholds=99, max_dist=0.0075, apply_thinning=True,
                        progress=None):
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    :param predicted_boundaries: the predicted boundaries as a (H,W)
    floating point array where each pixel represents the strength of the
    predicted boundary
    :param gt_boundaries: a list of ground truth boundaries, as returned
    by the `load_boundaries` or `boundaries` methods
    :param thresholds: either an integer specifying the number of thresholds
    to use or a 1D array specifying the thresholds
    :param max_dist: (default=0.0075) maximum distance parameter
    used for determining pixel matches. This value is multiplied by the
    length of the diagonal of the image to get the threshold used
    for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
    thinning to the predicted boundaries before evaluation
    :param progress: a function that can be used to monitor progress;
    use `tqdm.tqdm` or `tdqm.tqdm_notebook` from the `tqdm` package
    to generate a progress bar.
    :return: tuple `(count_r, sum_r, count_p, sum_p, thresholds)` where each
    of the first four entries are arrays that can be used to compute
    recall and precision at each threshold with:
    ```
    recall = count_r / (sum_r + (sum_r == 0))
    precision = count_p / (sum_p + (sum_p == 0))
    ```
    The thresholds are also returned.
    """
    if progress is None:
        progress = lambda x, *args, **kwargs: x

    # Handle thresholds
    if isinstance(thresholds, int):
        thresholds = np.linspace(1.0 / (thresholds + 1),
                                 1.0 - 1.0 / (thresholds + 1), thresholds)
    elif isinstance(thresholds, np.ndarray):
        if thresholds.ndim != 1:
            raise ValueError('thresholds array should have 1 dimension, not {}'.format(thresholds.ndim))
        pass
    else:
        raise ValueError('thresholds should be an int or a NumPy array, not a {}'.format(type(thresholds)))

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    for i_t, thresh in enumerate(progress(list(thresholds))):
        predicted_boundaries_bin = predicted_boundaries >= thresh

        acc_prec = np.zeros(predicted_boundaries_bin.shape, dtype=bool)

        if apply_thinning:
            predicted_boundaries_bin = thin.binary_thin(predicted_boundaries_bin)

        for gt in gt_boundaries:

            match1, match2, cost, oc = correspond_pixels.correspond_pixels(
                predicted_boundaries_bin, gt, max_dist=max_dist
            )
            match1 = match1 > 0
            match2 = match2 > 0
            # Precision accumulator
            acc_prec = acc_prec | match1
            # Recall
            sum_r[i_t] += gt.sum()
            count_r[i_t] += match2.sum()

        # Precision
        sum_p[i_t] = predicted_boundaries_bin.sum()
        count_p[i_t] = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p, thresholds
