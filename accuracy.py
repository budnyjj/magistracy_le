#!/usr/bin/env python

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

PRECISE_ALPHA = 0              # real 'alpha' value of source distribution
PRECISE_BETA = 5               # real 'beta' value of source distiribution

ERR_STD_X = 1               # std of X error values
ERR_STD_Y = 1               # std of Y error values


################
# Program code #
################

DESCRIPTION = 'Use this script to determine estimates accuracy'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-w', '--write-to', metavar='PATH',
                    type=str, help='file to write plot in')
args = parser.parse_args()

print('Expression:    {}'.format(SYM_EXPR))
print('Precise ALPHA: {}'.format(PRECISE_ALPHA))
print('Precise BETA:  {}'.format(PRECISE_BETA))
print('Error X std:   {}'.format(ERR_STD_X))
print('Error Y std:   {}\n'.format(ERR_STD_Y))

plt.figure(0)
plt.xlabel('$ x $')
plt.ylabel('$ y $')
plt.grid(True)

# build precise values
precise_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: PRECISE_ALPHA, SYM_BETA: PRECISE_BETA}),
    'numpy')
precise_vals_x, precise_vals_y = estimators.precise(
    precise_expr, NUM_VALS,
    MIN_X, MAX_X)

# get values with errors
estimated_vals_x, estimated_vals_y = estimators.uniform(
    precise_expr, NUM_VALS,
    MIN_X, MAX_X,
    ERR_STD_X, ERR_STD_Y)
plt.plot(
    estimated_vals_x, estimated_vals_y,
    color='r', linestyle=' ',
    marker='.', markersize=10,
    mfc='r')

# get theoretically predicted values
plt.plot(
    precise_vals_x, precise_vals_y,
    color='r', linestyle='-',
    marker='.', markersize=5,
    mfc='r', label="values")

# compute LSE parameter estimations
lse_alpha, lse_beta = methods.linear_lse(estimated_vals_x, estimated_vals_y)
param_accuracy_lse = accuracy.avg_euclidean_dst(
    np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
    np.array([lse_alpha, lse_beta]))
lse_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: lse_alpha, SYM_BETA: lse_beta}),
    'numpy')
estimated_lse_vals_y = np.vectorize(lse_expr)(estimated_vals_x)
predict_estimated_accuracy_lse = accuracy.avg_euclidean_dst(
    estimated_vals_y,
    estimated_lse_vals_y)
precise_lse_vals_y = np.vectorize(lse_expr)(precise_vals_x)
predict_precise_accuracy_lse = accuracy.avg_euclidean_dst(
    precise_vals_y,
    precise_lse_vals_y)

print('LSE ALPHA:                        {}'.format(lse_alpha))
print('LSE BETA:                         {}'.format(lse_beta))
print('LSE param accuracy:               {}'.format(param_accuracy_lse))
print('LSE predict accuracy (estimated): {}'.format(predict_estimated_accuracy_lse))
print('LSE predict accuracy (precise):   {}'.format(predict_precise_accuracy_lse))

plt.plot(
    precise_vals_x, precise_lse_vals_y,
    color='g', linestyle='-',
    marker='.', markersize=5,
    mfc='g', label="LSE")

# compute Pearson's parameter estimations
# test code
# pearson_alpha, pearson_beta = methods.linear_pearson(
#     np.array([[2, 16], [2, 26], [4,16], [4,26]]).transpose(),
#     np.array([219, 261, 127, 231]))
pearson_alpha, pearson_beta = methods.linear_pearson(estimated_vals_x, estimated_vals_y)
param_accuracy_pearson = accuracy.avg_euclidean_dst(
    np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
    np.array([pearson_alpha, pearson_beta]))
pearson_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: pearson_alpha, SYM_BETA: pearson_beta}),
    'numpy')
estimated_pearson_vals_y = np.vectorize(pearson_expr)(estimated_vals_x)
predict_estimated_accuracy_pearson = accuracy.avg_euclidean_dst(
    estimated_vals_y,
    estimated_pearson_vals_y)
precise_pearson_vals_y = np.vectorize(pearson_expr)(precise_vals_x)
predict_precise_accuracy_pearson = accuracy.avg_euclidean_dst(
    precise_vals_y,
    precise_pearson_vals_y)

print('Pearson ALPHA:                        {}'.format(pearson_alpha))
print('Pearson BETA:                         {}'.format(pearson_beta))
print('Pearson param accuracy:               {}'.format(param_accuracy_pearson))
print('Pearson predict accuracy (estimated): {}'.format(predict_estimated_accuracy_pearson))
print('Pearson predict accuracy (precise):   {}'.format(predict_precise_accuracy_pearson))

plt.plot(
    precise_vals_x, precise_pearson_vals_y,
    color='b', linestyle='-',
    marker='.', markersize=5,
    mfc='b', label="Pearson")
plt.legend(loc=2)

if args.write_to:
    plt.savefig(args.write_to, dpi=100)
plt.show()
