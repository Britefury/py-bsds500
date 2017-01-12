import os, argparse

import numpy as np
from bsds.bsds_dataset import BSDSDataset
from skimage.util import img_as_float
from skimage.io import imread

SAMPLE_NAMES = ['2018', '3063', '5096', '6046', '8068']
N_THRESHOLDS = 5

parser = argparse.ArgumentParser(description='Verify the BSDS-500 boundary evaluation suite')
parser.add_argument('bsds_path', type=str,
                    help='the root path of teh BSDS-500 dataset')

args = parser.parse_args()

bsds_path = args.bsds_path
bench_dir_path = os.path.join(bsds_path, 'bench', 'data')

def compute_rec_prec_f1(count_r, sum_r, count_p, sum_p):
    rec = count_r / (sum_r + (sum_r == 0))
    prec = count_p / (sum_p + (sum_p == 0))
    f1_denom = (prec + rec + ((prec+rec) == 0))
    f1 = 2.0 * prec * rec / f1_denom
    return rec, prec, f1


count_r_overall = np.zeros((N_THRESHOLDS,))
sum_r_overall = np.zeros((N_THRESHOLDS,))
count_p_overall = np.zeros((N_THRESHOLDS,))
sum_p_overall = np.zeros((N_THRESHOLDS,))

count_r_best = 0
sum_r_best = 0
count_p_best = 0
sum_p_best = 0

thresholds = None

print('Per image:')
for sample_index, sample_name in enumerate(SAMPLE_NAMES):
    # Get the paths for the ground truth and predicted boundaries
    gt_path = os.path.join(bench_dir_path, 'groundTruth', '{}.mat'.format(sample_name))
    pred_path = os.path.join(bench_dir_path, 'png', '{}.png'.format(sample_name))

    # Load them
    pred = img_as_float(imread(pred_path))
    gt_b = BSDSDataset.load_boundaries(gt_path)

    # Evaluate predictions
    count_r, sum_r, count_p, sum_p, thresholds = BSDSDataset.evaluate_boundaries(
        pred, gt_b, thresholds=N_THRESHOLDS, apply_thinning=True)

    count_r_overall += count_r
    sum_r_overall += sum_r
    count_p_overall += count_p
    sum_p_overall += sum_p

    # Compute precision, recall and F1
    rec, prec, f1 = compute_rec_prec_f1(count_r, sum_r, count_p, sum_p)

    # Find best F1 score
    best_ndx = np.argmax(f1)

    count_r_best += count_r[best_ndx]
    sum_r_best += sum_r[best_ndx]
    count_p_best += count_p[best_ndx]
    sum_p_best += sum_p[best_ndx]

    print('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        sample_index + 1, thresholds[best_ndx], rec[best_ndx], prec[best_ndx], f1[best_ndx]))

# Computer overall precision, recall and F1
rec_overall, prec_overall, f1_overall = compute_rec_prec_f1(
    count_r_overall, sum_r_overall, count_p_overall, sum_p_overall)

# Find best F1 score
best_i_ovr = np.argmax(f1_overall)

print('')
print('Overall:')
for thresh_i in range(N_THRESHOLDS):
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        thresholds[thresh_i], rec_overall[thresh_i], prec_overall[thresh_i], f1_overall[thresh_i]))

print('')
print('Summary:')
rec_best, prec_best, f1_best = compute_rec_prec_f1(
    float(count_r_best), float(sum_r_best), float(count_p_best), float(sum_p_best)
)

rec_unique, rec_unique_ndx = np.unique(rec_overall, return_index=True)
prec_unique = prec_overall[rec_unique_ndx]
if rec_unique.shape[0] > 1:
    prec_interp = np.interp(np.arange(0, 1, 0.01), rec_unique, prec_unique, left=0.0, right=0.0)
    area_pr = prec_interp.sum() * 0.01
else:
    area_pr = 0.0

print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
    thresholds[best_i_ovr], rec_overall[best_i_ovr], prec_overall[best_i_ovr], f1_overall[best_i_ovr],
    rec_best, prec_best, f1_best, area_pr))
