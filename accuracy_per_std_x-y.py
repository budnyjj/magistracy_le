#!/usr/bin/env python

import os.path
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import stats.estimators as estimators
import stats.methods as methods
import stats.accuracy as accuracy


################
# Declarations #
################

SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_ALPHA, SYM_BETA = SYM_PARAMS = sp.symbols('a b')

# linear function
SYM_EXPR = sp.sympify('a + b*x')

MIN_X = 0
MAX_X = 10
NUM_VALS = 100              # number of source values

PRECISE_ALPHA = 0           # real 'alpha' value of source distribution
PRECISE_BETA = 5            # real 'beta' value of source distiribution

ERR_NUM_STD_ITER = 20       # number of stds iterations

ERR_MIN_STD_X = 0.000       # minimal std of X error values
ERR_MAX_STD_X = 2.000       # maximal std of X error values
ERR_STD_STEP_X = (ERR_MAX_STD_X - ERR_MIN_STD_X) / ERR_NUM_STD_ITER

ERR_MIN_STD_Y = 0.000       # minimal std of Y error values
ERR_MAX_STD_Y = 2.000       # maximal std of Y error values
ERR_STD_STEP_Y = (ERR_MAX_STD_Y - ERR_MIN_STD_Y) / ERR_NUM_STD_ITER

NUM_ITER = 100              # number of realizations


################
# Program code #
################

DESCRIPTION = 'Use this script to determine estimates accuracy'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-w', '--write-to', metavar='PATH',
                    type=str, help='file to write plot in')
args = parser.parse_args()

print('Expression:           {}'.format(SYM_EXPR))
print('Real ALPHA:           {}'.format(PRECISE_ALPHA))
print('Real BETA:            {}'.format(PRECISE_BETA))
print('Real X:               {}..{}'.format(MIN_X, MAX_X))
print('STD X:                {}..{}'.format(ERR_MIN_STD_X, ERR_MAX_STD_X))
print('STD X step:           {}'.format(ERR_STD_STEP_X))
print('STD Y:                {}..{}'.format(ERR_MIN_STD_Y, ERR_MAX_STD_Y))
print('STD Y step:           {}'.format(ERR_STD_STEP_Y))
print('Number of iterations: {}'.format(NUM_ITER))

# build precise values
precise_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: PRECISE_ALPHA, SYM_BETA: PRECISE_BETA}),
    'numpy')
precise_vectorized = np.vectorize(precise_expr)
# get precise values
precise_vals_x, precise_vals_y = estimators.precise(
    precise_expr, NUM_VALS,
    MIN_X, MAX_X)

# generate array of X error stds
err_stds_x = np.linspace(ERR_MIN_STD_X, ERR_MAX_STD_X, ERR_NUM_STD_ITER)
# generate array of Y error stds
err_stds_y = np.linspace(ERR_MIN_STD_Y, ERR_MAX_STD_Y, ERR_NUM_STD_ITER)
# create meshgrid
err_stds_x, err_stds_y = np.meshgrid(err_stds_x, err_stds_y)
# collect accuracies of estimates
lse_param_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
lse_predict_measured_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
lse_predict_precise_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
pearson_param_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
pearson_predict_measured_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
pearson_predict_precise_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))

num_std_iter = ERR_NUM_STD_ITER**2
std_iter = 0
for std_i, err_std_row in enumerate(np.dstack((err_stds_x, err_stds_y))):
    for std_j, (err_std_x, err_std_y) in enumerate(err_std_row):
        std_iter += 1
        print("Iteration {}/{}: std X: {:.2f}, std Y: {:.2f}".format(
              std_iter, num_std_iter, err_std_x, err_std_y))

        # current accuracies for this std
        cur_lse_param_acc = 0
        cur_lse_predict_measured_acc = 0
        cur_lse_predict_precise_acc = 0
        cur_pearson_param_acc = 0
        cur_pearson_predict_measured_acc = 0
        cur_pearson_predict_precise_acc = 0

        # iterate by error standart derivation values
        for iter_i in range(NUM_ITER):
            # get mesured values with errors
            measured_vals_x, measured_vals_y = estimators.uniform(
                precise_expr, NUM_VALS,
                MIN_X, MAX_X,
                err_std_x, err_std_y)
            # get control values with errors
            control_measured_vals_x, control_measured_vals_y = estimators.uniform(
                precise_expr, NUM_VALS,
                MIN_X, MAX_X,
                err_std_x, err_std_y)
            # get precise control output values
            control_precise_vals_y = precise_vectorized(control_measured_vals_x)

            # compute LSE parameter estimations
            lse_alpha, lse_beta = methods.linear_lse(measured_vals_x, measured_vals_y)
            lse_lambda = sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_ALPHA: lse_alpha, SYM_BETA: lse_beta}),
                'numpy')
            # compute LSE parameter accuracy
            lse_param_acc = accuracy.avg_euclidean_dst(
                np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
                np.array([lse_alpha, lse_beta]))
            cur_lse_param_acc += lse_param_acc
            # compute LSE predicted values by control input
            lse_vectorized = np.vectorize(lse_lambda)
            lse_control_vals_y = lse_vectorized(control_measured_vals_x)
            # compute LSE prediction accuracy against measured values
            lse_predict_measured_acc = accuracy.avg_euclidean_dst(
                control_measured_vals_y,
                lse_control_vals_y)
            cur_lse_predict_measured_acc += lse_predict_measured_acc
            # compute LSE prediction accuracy against precise values
            lse_predict_precise_acc = accuracy.avg_euclidean_dst(
                control_precise_vals_y,
                lse_control_vals_y)
            cur_lse_predict_precise_acc += lse_predict_precise_acc

            # compute Pearson's parameter estimations
            pearson_alpha, pearson_beta = methods.linear_pearson(measured_vals_x, measured_vals_y)
            pearson_lambda = sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_ALPHA: pearson_alpha, SYM_BETA: pearson_beta}),
                'numpy')
            # compute Pearson's parameter accuracy
            pearson_param_acc = accuracy.avg_euclidean_dst(
                np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
                np.array([pearson_alpha, pearson_beta]))
            cur_pearson_param_acc += pearson_param_acc
            # compute LSE predicted values by control input
            pearson_vectorized = np.vectorize(pearson_lambda)
            pearson_control_vals_y = pearson_vectorized(control_measured_vals_x)
            # compute Pearson prediction accuracy against measured values
            pearson_predict_measured_acc = accuracy.avg_euclidean_dst(
                control_measured_vals_y,
                pearson_control_vals_y)
            cur_pearson_predict_measured_acc += pearson_predict_measured_acc
            # compute Pearson prediction accuracy against precise values
            pearson_predict_precise_acc = accuracy.avg_euclidean_dst(
                control_precise_vals_y,
                pearson_control_vals_y)
            cur_pearson_predict_precise_acc += pearson_predict_precise_acc

        lse_param_accs[std_i, std_j] = cur_lse_param_acc
        lse_predict_measured_accs[std_i, std_j] = cur_lse_predict_measured_acc
        lse_predict_precise_accs[std_i, std_j] = cur_lse_predict_precise_acc
        pearson_param_accs[std_i, std_j] = cur_pearson_param_acc
        pearson_predict_measured_accs[std_i, std_j] = cur_pearson_predict_measured_acc
        pearson_predict_precise_accs[std_i, std_j] = cur_pearson_predict_precise_acc

# get averages by number of iterations
lse_param_accs /= NUM_ITER
lse_predict_measured_accs /= NUM_ITER
lse_predict_precise_accs /= NUM_ITER
pearson_param_accs /= NUM_ITER
pearson_predict_measured_accs /= NUM_ITER
pearson_predict_precise_accs /= NUM_ITER

# compute differences between accuracies
param_accs_diff = lse_param_accs - pearson_param_accs
predict_precise_accs_diff = lse_predict_precise_accs - pearson_predict_precise_accs
predict_measured_accs_diff = lse_predict_measured_accs - pearson_predict_measured_accs

fig = plt.figure(0)
contour_param = plt.contour(
    err_stds_x, err_stds_y, param_accs_diff)
plt.title('$ d_{param_{LSE}} - d_{param_{Pearson}} $')
plt.clabel(contour_param, inline=True, fontsize=10)
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')

fig = plt.figure(1)
contour_predict_precise = plt.contour(
    err_stds_x, err_stds_y, predict_precise_accs_diff)
plt.clabel(contour_predict_precise, inline=True, fontsize=10)
plt.title('$ d_{predict-precise_{LSE}} - d_{predict-precise_{Pearson}} $')
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')

fig = plt.figure(2)
contour_predict_measured = plt.contour(
    err_stds_x, err_stds_y, predict_measured_accs_diff)
plt.clabel(contour_predict_measured, inline=True, fontsize=10)
plt.title('$ d_{predict-measured_{LSE}} - d_{predict-measured_{Pearson}} $')
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')

if args.write_to:
    file_name, file_ext = os.path.splitext(args.write_to)

    np.save('{}_err-stds-x.npy'.format(file_name), err_stds_x)
    np.save('{}_err-stds-y.npy'.format(file_name), err_stds_y)
    np.save('{}_param-accs-diff.npy'.format(file_name), param_accs_diff)
    np.save('{}_predict-precise-accs-diff.npy'.format(file_name), predict_precise_accs_diff)
    np.save('{}_predict-measured-accs-diff.npy'.format(file_name), predict_measured_accs_diff)

    plt.figure(0)
    plt.savefig('{}_param{}'.format(file_name, file_ext),
                dpi=200)

    plt.figure(1)
    plt.savefig('{}_predict-precise{}'.format(file_name, file_ext),
                dpi=200)

    plt.figure(2)
    plt.savefig('{}_predict-measured{}'.format(file_name, file_ext),
                dpi=200)
