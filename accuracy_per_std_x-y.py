#!/usr/bin/env python

import os.path
import argparse
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
PRECISE_BETA = 0.5          # real 'beta' value of source distiribution

ERR_MIN_STD_X = 0.000       # minimal std of X error values
ERR_MAX_STD_X = 2.000       # maximal std of X error values

ERR_MIN_STD_Y = 0.000       # minimal std of Y error values
ERR_MAX_STD_Y = 2.000       # maximal std of Y error values

ERR_NUM_STD_ITER = 10       # number of stds iterations

NUM_ITER = 10               # number of realizations


################
# Program code #
################

DESCRIPTION = 'Use this script to determine estimates accuracy'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-w', '--write-to', metavar='PATH',
                    type=str, help='file to write plot in')
args = parser.parse_args()

print('Expression:    {}'.format(SYM_EXPR))
print('Real ALPHA:    {}'.format(PRECISE_ALPHA))
print('Real BETA:     {}'.format(PRECISE_BETA))
print('Real X:        {}..{}'.format(MIN_X, MAX_X))
print('STD X:         {}..{}'.format(ERR_MIN_STD_X, ERR_MAX_STD_X))
print('STD Y:         {}..{}'.format(ERR_MIN_STD_Y, ERR_MAX_STD_Y))
print('Number of iterations: {}'.format(ERR_NUM_STD_ITER * NUM_ITER))

# build precise values
precise_expr = sp.lambdify(
    SYM_X,
    SYM_EXPR.subs({SYM_ALPHA: PRECISE_ALPHA, SYM_BETA: PRECISE_BETA}),
    'numpy')

# generate array of X error stds
err_stds_x = np.linspace(ERR_MIN_STD_X, ERR_MAX_STD_X, ERR_NUM_STD_ITER)
# generate array of Y error stds
err_stds_y = np.linspace(ERR_MIN_STD_Y, ERR_MAX_STD_Y, ERR_NUM_STD_ITER)
# create meshgrid
err_stds_x, err_stds_y = np.meshgrid(err_stds_x, err_stds_y)
# collect accuracies of estimates
lse_param_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
lse_predict_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
pearson_param_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))
pearson_predict_accs = np.zeros((ERR_NUM_STD_ITER, ERR_NUM_STD_ITER))

for std_i, err_std_row in enumerate(np.dstack((err_stds_x, err_stds_y))):
    row_lse_param_accs = np.zeros(ERR_NUM_STD_ITER)
    row_lse_predict_accs = np.zeros(ERR_NUM_STD_ITER)
    row_pearson_predict_accs = np.zeros(ERR_NUM_STD_ITER)
    row_pearson_param_accs = np.zeros(ERR_NUM_STD_ITER)

    for std_j, (err_std_x, err_std_y) in enumerate(err_std_row):
        # current accuracies for this std
        cur_lse_param_acc = 0
        cur_lse_predict_acc = 0
        cur_pearson_param_acc = 0
        cur_pearson_predict_acc = 0

        # iterate by error standart derivation values
        for iter_i in range(NUM_ITER):
            # get precise values
            precise_vals_x, precise_vals_y = estimators.precise(
                precise_expr, NUM_VALS,
                MIN_X, MAX_X)
            # get values with errors
            estimated_vals_x, estimated_vals_y = estimators.uniform(
                precise_expr, NUM_VALS,
                MIN_X, MAX_X,
                err_std_x, err_std_y)
            # get control values with errors
            control_vals_x, control_vals_y = estimators.uniform(
                precise_expr, NUM_VALS,
                MIN_X, MAX_X,
                err_std_x, err_std_y)

            # compute LSE parameter estimations
            lse_alpha, lse_beta = methods.linear_lse(estimated_vals_x, estimated_vals_y)
            # compute LSE parameter accuracy
            lse_param_accuracy = accuracy.avg_euclidean_dst(
                np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
                np.array([lse_alpha, lse_beta]))
            cur_lse_param_acc += lse_param_accuracy
            # compute LSE prediction accuracy
            lse_expr = sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_ALPHA: lse_alpha, SYM_BETA: lse_beta}),
                'numpy')
            lse_predict_accuracy = accuracy.avg_euclidean_dst(
                control_vals_y,
                np.vectorize(lse_expr)(control_vals_x))
            cur_lse_predict_acc += lse_predict_accuracy

            # compute Pearson's parameter estimations
            pearson_alpha, pearson_beta = methods.linear_pearson(estimated_vals_x, estimated_vals_y)
            # compute Pearson's parameter accuracy
            pearson_param_accuracy = accuracy.avg_euclidean_dst(
                np.array([[PRECISE_ALPHA], [PRECISE_BETA]]),
                np.array([pearson_alpha, pearson_beta]))
            cur_pearson_param_acc += pearson_param_accuracy
            # compute Pearson prediction accuracy
            pearson_expr = sp.lambdify(
                SYM_X,
                SYM_EXPR.subs({SYM_ALPHA: pearson_alpha, SYM_BETA: pearson_beta}),
                'numpy')
            pearson_predict_accuracy = accuracy.avg_euclidean_dst(
                control_vals_y,
                np.vectorize(pearson_expr)(control_vals_x))
            cur_pearson_predict_acc += pearson_predict_accuracy

        row_lse_param_accs[std_j] = cur_lse_param_acc / NUM_ITER
        row_lse_predict_accs[std_j] = cur_lse_predict_acc / NUM_ITER
        row_pearson_param_accs[std_j] = cur_pearson_param_acc / NUM_ITER
        row_pearson_predict_accs[std_j] = cur_pearson_predict_acc / NUM_ITER

    print(row_lse_param_accs)

    lse_param_accs[std_i] = row_lse_param_accs
    lse_predict_accs[std_i] = row_lse_predict_accs
    pearson_param_accs[std_i] = row_pearson_param_accs
    pearson_predict_accs[std_i] = row_pearson_predict_accs

# print(err_stds_x)
# print(err_stds_y)
# print(lse_accs)
# print(pearson_accs)

fig = plt.figure(0)
ax = fig.gca(projection='3d')
ax.view_init(elev=10., azim=-140)
ax.set_xlabel('$ \\sigma_x $')
ax.set_ylabel('$ \\sigma_y $')
ax.set_zlabel('$ \\rho_{param} $')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surface_lse = ax.plot_surface(
    err_stds_x, err_stds_y, lse_param_accs,
    rstride=1, cstride=1,
    cmap=cm.Greens,
    label='LSE'
)
surface_pearson = ax.plot_surface(
    err_stds_x, err_stds_y, pearson_param_accs,
    rstride=1, cstride=1,
    cmap=cm.Blues,
    label='Pearson'
)
bar_lse = plt.Rectangle((0, 0), 0.1, 0.1, fc='g')
bar_pearson = plt.Rectangle((0, 0), 0.1, 0.1, fc='b')
ax.legend((bar_lse, bar_pearson), ("LSE", "Pearson"))

# fig = plt.figure(2)
# ax = fig.gca(projection='3d')
# ax.view_init(elev=10., azim=-140)
# ax.set_xlabel('$ \\sigma_x $')
# ax.set_ylabel('$ \\sigma_y $')
# ax.set_zlabel('$ \\rho_{param_{LSE}} - \\rho_{param_{Pearson}} $')
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.plot_surface(
#     err_stds_x, err_stds_y, lse_param_accs - pearson_param_accs,
#     rstride=1, cstride=1,
#     cmap=cm.coolwarm,
#     label='parameter: LSE-Pearson'
# )

# fig = plt.figure(3)
# ax = fig.gca(projection='3d')
# ax.view_init(elev=10., azim=-140)
# ax.set_xlabel('$ \\sigma_x $')
# ax.set_ylabel('$ \\sigma_y $')
# ax.set_zlabel('$ \\rho_{predict_{LSE}} $')
# accs_surf = ax.plot_surface(
#     err_stds_x, err_stds_y, lse_predict_accs,
#     rstride=1, cstride=1,
#     cmap=cm.coolwarm,
#     label='predict: LSE'
# )
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig = plt.figure(4)
# ax = fig.gca(projection='3d')
# ax.view_init(elev=10., azim=-140)
# ax.set_xlabel('$ \\sigma_x $')
# ax.set_ylabel('$ \\sigma_y $')
# ax.set_zlabel('$ \\rho_{param_{Pearson}} $')
# accs_surf = ax.plot_surface(
#     err_stds_x, err_stds_y, pearson_predict_accs,
#     rstride=1, cstride=1,
#     cmap=cm.coolwarm,
#     label='predict: Pearson'
# )
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig = plt.figure(5)
# ax = fig.gca(projection='3d')
# ax.view_init(elev=10., azim=-140)
# ax.set_xlabel('$ \\sigma_x $')
# ax.set_ylabel('$ \\sigma_y $')
# ax.set_zlabel('$ \\rho_{predict_{LSE}} - \\rho_{_predict{Pearson}} $')
# accs_surf = ax.plot_surface(
#     err_stds_x, err_stds_y, lse_predict_accs - pearson_predict_accs,
#     rstride=1, cstride=1,
#     cmap=cm.coolwarm,
#     label='predict: LSE-Pearson'
# )
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

if args.write_to:
    file_name, file_ext = os.path.splitext(args.write_to)

    plt.figure(0)
    plt.savefig('{}_param{}'.format(file_name, file_ext),
                dpi=200)

    # plt.figure(1)
    # plt.savefig('{}_pearson-param{}'.format(file_name, file_ext),
    #             dpi=200)

    # plt.figure(2)
    # plt.savefig('{}_param-diff{}'.format(file_name, file_ext),
    #             dpi=200)

    # plt.figure(3)
    # plt.savefig('{}_lse-predict{}'.format(file_name, file_ext),
    #             dpi=200)

    # plt.figure(4)
    # plt.savefig('{}_pearson-predict{}'.format(file_name, file_ext),
    #             dpi=200)

    # plt.figure(5)
    # plt.savefig('{}_predict-diff{}'.format(file_name, file_ext),
    #             dpi=200)

plt.show()
