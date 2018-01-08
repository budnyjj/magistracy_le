#!/usr/bin/env python

import os.path
import argparse
import numpy as np
import sympy as sp

import stats.estimators as estimators
import stats.methods as methods
import stats.accuracy as accuracy


################
# Constants    #
################

DESCRIPTION = 'Determines estimate accuracies'

SYM_X, SYM_Y = SYM_VALUES = sp.symbols('x y')
SYM_ALPHA, SYM_BETA = SYM_PARAMS = sp.symbols('a b')

# linear function
SYM_EXPR = sp.sympify('a + b*x')

MIN_X = 0
MAX_X = 10
NUM_VALS = 100              # number of source values

PRECISE_ALPHA = 0           # real 'alpha' value of source distribution
PRECISE_BETA = 0            # real 'beta' value of source distiribution

ERR_NUM_STD_ITER = 20       # number of stds iterations

ERR_MIN_STD_X = 0.000       # minimal std of X error values
ERR_MAX_STD_X = 2.000       # maximal std of X error values
ERR_STEP_STD_X = (ERR_MAX_STD_X - ERR_MIN_STD_X) / ERR_NUM_STD_ITER

ERR_MIN_STD_Y = 0.000       # minimal std of Y error values
ERR_MAX_STD_Y = 2.000       # maximal std of Y error values
ERR_STEP_STD_Y = (ERR_MAX_STD_Y - ERR_MIN_STD_Y) / ERR_NUM_STD_ITER

NUM_ITER = 100              # number of realizations
NUM_CONTROL_VALS = 100      # number of control values


################
# Program code #
################

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '-o', '--output', metavar='PATH',
    type=str, required=True,
    help='base path to write data')
args = parser.parse_args()
output_path, _ = os.path.splitext(args.output)

print('Expression:               {}'.format(SYM_EXPR))
print('Precise ALPHA:            {}'.format(PRECISE_ALPHA))
print('Precise BETA:             {}'.format(PRECISE_BETA))
print('Real X:                   {}..{}'.format(MIN_X, MAX_X))
print('STD X:                    {}..{}'.format(ERR_MIN_STD_X, ERR_MAX_STD_X))
print('STD X step:               {}'.format(ERR_STEP_STD_X))
print('STD Y:                    {}..{}'.format(ERR_MIN_STD_Y, ERR_MAX_STD_Y))
print('STD Y step:               {}'.format(ERR_STEP_STD_Y))
print('Number of iterations:     {}'.format(NUM_ITER))
print('Number of control values: {}'.format(NUM_CONTROL_VALS))
print('Output path:              {}'.format(output_path))

# build precise values
precise_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: PRECISE_ALPHA, SYM_BETA: PRECISE_BETA}),
    'numpy')
precise_vectorized = np.vectorize(precise_expr)
# get precise values
precise_vals_x, precise_vals_y = estimators.precise(
    precise_vectorized, NUM_VALS,
    MIN_X, MAX_X)

# generate array of X error stds
err_stds_x = np.linspace(ERR_MIN_STD_X, ERR_MAX_STD_X, ERR_NUM_STD_ITER)
# generate array of Y error stds
err_stds_y = np.linspace(ERR_MIN_STD_Y, ERR_MAX_STD_Y, ERR_NUM_STD_ITER)
# create meshgrid
err_stds_x, err_stds_y = np.meshgrid(err_stds_x, err_stds_y)

# get precise control values without errors
control_precise_vals_x, control_precise_vals_y = estimators.uniform(
    precise_vectorized, NUM_CONTROL_VALS,
    MIN_X, MAX_X)

# collect accuracies of estimates
lse_param_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
lse_predict_measured_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
lse_predict_precise_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
sa_param_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
sa_predict_measured_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
sa_predict_precise_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))

num_std_iter = ERR_NUM_STD_ITER**2
std_iter = 0
for std_i, err_std_row in enumerate(np.dstack((err_stds_x, err_stds_y))):
    for std_j, (err_std_x, err_std_y) in enumerate(err_std_row):
        std_iter += 1
        print("Iteration {}/{}: std X: {:.2f}, std Y: {:.2f}".format(
              std_iter, num_std_iter, err_std_x, err_std_y))

        # iterate by error standart derivation values
        for iter_i in range(NUM_ITER):
            # get mesured values with errors
            measured_vals_x, measured_vals_y = estimators.uniform(
                precise_vectorized, NUM_VALS,
                MIN_X, MAX_X,
                err_std_x, err_std_y)

            # get control values
            control_measured_vals_x, control_measured_vals_y = estimators.with_errors(
                control_precise_vals_x, control_precise_vals_y,
                err_std_x, err_std_y)

            # compute LSE parameter estimations
            lse_alpha, lse_beta = methods.linear_lse(
                measured_vals_x, measured_vals_y)
            lse_lambda = sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_ALPHA: lse_alpha, SYM_BETA: lse_beta}),
                'numpy')
            # compute LSE parameter accuracy
            lse_param_acc = accuracy.avg_euclidean_dst(
                np.array(((PRECISE_ALPHA), (PRECISE_BETA))),
                np.array((lse_alpha, lse_beta)))
            lse_param_accs[std_i, std_j] += lse_param_acc
            # compute LSE predicted values by control input
            lse_vectorized = np.vectorize(lse_lambda)
            lse_control_vals_y = lse_vectorized(control_measured_vals_x)
            # compute LSE prediction accuracy against measured values
            lse_predict_measured_acc = accuracy.avg_euclidean_dst(
                control_measured_vals_y,
                lse_control_vals_y)
            lse_predict_measured_accs[std_i, std_j] += lse_predict_measured_acc
            # compute LSE prediction accuracy against precise values
            lse_predict_precise_acc = accuracy.avg_euclidean_dst(
                control_precise_vals_y,
                lse_control_vals_y)
            lse_predict_precise_accs[std_i, std_j] += lse_predict_precise_acc

            # compute Sa's parameter estimations
            sa_alpha, sa_beta = methods.linear_sa(
                measured_vals_x, measured_vals_y)
            sa_lambda = sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_ALPHA: sa_alpha, SYM_BETA: sa_beta}),
                'numpy')
            # compute Sa's parameter accuracy
            sa_param_acc = accuracy.avg_euclidean_dst(
                np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
                np.array([sa_alpha, sa_beta]))
            sa_param_accs[std_i, std_j] += sa_param_acc
            # compute LSE predicted values by control input
            sa_vectorized = np.vectorize(sa_lambda)
            sa_control_vals_y = sa_vectorized(control_measured_vals_x)
            # compute Sa prediction accuracy against measured values
            sa_predict_measured_acc = accuracy.avg_euclidean_dst(
                control_measured_vals_y,
                sa_control_vals_y)
            sa_predict_measured_accs[std_i, std_j] += sa_predict_measured_acc
            # compute Sa prediction accuracy against precise values
            sa_predict_precise_acc = accuracy.avg_euclidean_dst(
                control_precise_vals_y,
                sa_control_vals_y)
            sa_predict_precise_accs[std_i, std_j] += sa_predict_precise_acc

# get averages by number of iterations
lse_param_accs /= NUM_ITER
lse_predict_measured_accs /= NUM_ITER
lse_predict_precise_accs /= NUM_ITER
sa_param_accs /= NUM_ITER
sa_predict_measured_accs /= NUM_ITER
sa_predict_precise_accs /= NUM_ITER

np.save(
    '{}_err-stds-x.npy'.format(output_path),
    err_stds_x)
np.save(
    '{}_err-stds-y.npy'.format(output_path),
    err_stds_y)
np.save(
    '{}_lse-param-accs.npy'.format(output_path),
    lse_param_accs)
np.save(
    '{}_lse-predict-precise-accs.npy'.format(output_path),
    lse_predict_precise_accs)
np.save(
    '{}_lse-predict-measured-accs.npy'.format(output_path),
    lse_predict_measured_accs)
np.save(
    '{}_sa-param-accs.npy'.format(output_path),
    sa_param_accs)
np.save(
    '{}_sa-predict-precise-accs.npy'.format(output_path),
    sa_predict_precise_accs)
np.save(
    '{}_sa-predict-measured-accs.npy'.format(output_path),
    sa_predict_measured_accs)
