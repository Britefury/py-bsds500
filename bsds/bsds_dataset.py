import os
import numpy as np
from skimage.util import img_as_float
from skimage.io import imread
from scipy.io import loadmat
from . import correspond_pixels, thin


def _sample_names(dir, subset):
    names = []
    files = os.listdir(os.path.join(dir, subset))
    for fn in files:
        dir, filename = os.path.split(fn)
        name, ext = os.path.splitext(filename)
        if ext.lower() == '.jpg':
            names.append(os.path.join(subset, name))
    return names

class BSDSDataset (object):
    """
    BSDS dataset wrapper

    Given the path to the root of the BSDS dataset, this class provides
    methods for loading images, ground truths and evaluating predictions

    Attribtes:

    bsds_path - the root path of the dataset
    data_path - the path of the data directory within the root
    images_path - the path of the images directory within the data dir
    gt_path - the path of the groundTruth directory within the data dir
    train_sample_names - a list of names of training images
    val_sample_names - a list of names of validation images
    test_sample_names - a list of names of test images
    """
    def __init__(self, bsds_path):
        """
        Constructor

        :param bsds_path: the path to the root of the BSDS dataset
        """
        self.bsds_path = bsds_path
        self.data_path = os.path.join(bsds_path, 'BSDS500', 'data')
        self.images_path = os.path.join(self.data_path, 'images')
        self.gt_path = os.path.join(self.data_path, 'groundTruth')

        self.train_sample_names = _sample_names(self.images_path, 'train')
        self.val_sample_names = _sample_names(self.images_path, 'val')
        self.test_sample_names = _sample_names(self.images_path, 'test')

    def read_image(self, name):
        """
        Load the image identified by the sample name (you can get the names
        from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a (H,W,3) array containing the image, scaled to range [0,1]
        """
        path = os.path.join(self.images_path, name + '.jpg')
        return img_as_float(imread(path))

    def ground_truth_mat(self, name):
        """
        Load the ground truth Matlab file identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: the `groundTruth` entry from the Matlab file
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_ground_truth_mat(path)

    def segmentations(self, name):
        """
        Load the ground truth segmentations identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_segmentations(path)

    def boundaries(self, name):
        """
        Load the ground truth boundaries identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        path = os.path.join(self.gt_path, name + '.mat')
        return self.load_boundaries(path)

    @staticmethod
    def load_ground_truth_mat(path):
        """
        Load the ground truth Matlab file at the specified path
        and return the `groundTruth` entry.
        :param path: path
        :return: the 'groundTruth' entry from the Matlab file
        """
        gt = loadmat(path)
        return gt['groundTruth']

    @staticmethod
    def load_segmentations(path):
        """
        Load the ground truth segmentations from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        gt = BSDSDataset.load_ground_truth_mat(path)
        num_gts = gt.shape[1]
        return [gt[0,i]['Segmentation'][0,0] for i in range(num_gts)]

    @staticmethod
    def load_boundaries(path):
        """
        Load the ground truth boundaries from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        gt = BSDSDataset.load_ground_truth_mat(path)
        num_gts = gt.shape[1]
        return [gt[0,i]['Boundaries'][0,0] for i in range(num_gts)]

    @staticmethod
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


    @staticmethod
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

