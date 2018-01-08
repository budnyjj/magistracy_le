#!/usr/bin/env python

import os.path
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import stats.methods as methods


################
# Declarations #
################

SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_ALPHA, SYM_BETA = SYM_PARAMS = sp.symbols('a b')

PREDICT_ALPHA = 0
PREDICT_BETA = 5.7 # |PRECISE_BETA| + 0.7

# linear function
SYM_EXPR = sp.sympify('a + b*x')

EPS = 0.005

def threshold(array, eps):
    return np.nonzero((array > -eps) & (array < eps))

################
# Program code #
################

DESCRIPTION = 'Use this script to approximate zero-level line'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-r', '--read-from', metavar='PATH',
                    type=str, required=True,
                    help='file to read data from')
parser.add_argument('-w', '--write-to', metavar='PATH',
                    type=str, help='file to write plot in')
args = parser.parse_args()

# print predicted values
print('Predict. alpha: {}'.format(PREDICT_ALPHA))
print('Predict. beta:  {}'.format(PREDICT_BETA))

# load source data
file_name, _ = os.path.splitext(args.read_from)
err_stds_x = np.load('{}_err-stds-x.npy'.format(file_name))
err_stds_y = np.load('{}_err-stds-y.npy'.format(file_name))
accs_diff = np.load('{}_param-accs-diff.npy'.format(file_name))

# plot contour
contour = plt.contour(
    err_stds_x, err_stds_y, accs_diff)
plt.clabel(contour, inline=True, fontsize=10)
plt.title('$ d_{param_{LSE}} - d_{param_{Sa}} $')
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')
plt.grid(True)

# find indexes of zero differences
ref_idxs = threshold(accs_diff, EPS)
print('Number of ref. values: {}'.format(len(ref_idxs[0])))

# extract values correspondinf to zero differences
ref_vals_x = err_stds_x[ref_idxs]
ref_vals_y = err_stds_y[ref_idxs]
plt.plot(
    ref_vals_x, ref_vals_y,
    color='r', linestyle=' ',
    marker='.', markersize=5,
    mfc='r', label="ref. values")

# approximate coordinates with linear function
approx_alpha, approx_beta = methods.linear_sa(ref_vals_x, ref_vals_y)
print('Approx. alpha: {}'.format(approx_alpha))
print('Approx. beta:  {}'.format(approx_beta))

# plot results of approximation
approx_lambda = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: approx_alpha, SYM_BETA: approx_beta}),
    'numpy')
approx_vectorized = np.vectorize(approx_lambda)
approx_vals_y = approx_vectorized(ref_vals_x)
plt.plot(
    ref_vals_x, approx_vals_y,
    color='r', linestyle='-',
    marker='.', markersize=5,
    mfc='r', label="approx.")
plt.text(
    ref_vals_x[-1], approx_vals_y[-1],
    '$ \sigma_{\delta} = ' + str(approx_beta) + '\sigma_{\epsilon} + ' + str(approx_alpha) + ' $')

# plot predicted divider
predict_lambda = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: PREDICT_ALPHA, SYM_BETA: PREDICT_BETA}),
    'numpy')
predict_vectorized = np.vectorize(predict_lambda)
predict_vals_y = predict_vectorized(ref_vals_x)
plt.plot(
    ref_vals_x, predict_vals_y,
    color='b', linestyle='-',
    marker='.', markersize=5,
    mfc='b', label="predict.")
plt.text(
    ref_vals_x[-1], predict_vals_y[-1],
    '$ \sigma_{\delta} = ' + str(PREDICT_BETA) + '\sigma_{\epsilon} + ' + str(PREDICT_ALPHA) + ' $')

plt.legend(loc=2)

if args.write_to:
    file_name, file_ext = os.path.splitext(args.write_to)
    plt.savefig('{}_param-accs-diff-approx{}'.format(file_name, file_ext),
                dpi=200)

# plt.show()
