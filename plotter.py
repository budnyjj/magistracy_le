#!/usr/bin/env python

import os.path
import argparse
import numpy as np

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


################
# Constants    #
################

DESCRIPTION = 'Plots estimate accuracies'

################
# Program code #
################

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '-i', '--input', metavar='PATH',
    type=str, required=True,
    help='base path to read data')
parser.add_argument(
    '-o', '--output', metavar='PATH',
    type=str, help='base path to write plots')
parser.add_argument(
    '-s', '--show',
    dest='show', action='store_true',
    help='show plots')
parser.set_defaults(show=False)
args = parser.parse_args()
input_path, _ = os.path.splitext(args.input)
output_path, output_ext = None, None
if args.output:
    output_path, output_ext = os.path.splitext(args.output)

print('Input path:    {}'.format(input_path))
print('Output path:   {}'.format(output_path))

# load data
err_stds_x = np.load(
    '{}_err-stds-x.npy'.format(input_path))
err_stds_y = np.load(
    '{}_err-stds-y.npy'.format(input_path))
lse_param_accs = np.load(
    '{}_lse-param-accs.npy'.format(input_path))
lse_predict_precise_accs = np.load(
    '{}_lse-predict-precise-accs.npy'.format(input_path))
lse_predict_measured_accs = np.load(
    '{}_lse-predict-measured-accs.npy'.format(input_path))
sa_param_accs = np.load(
    '{}_sa-param-accs.npy'.format(input_path))
sa_predict_precise_accs = np.load(
    '{}_sa-predict-precise-accs.npy'.format(input_path))
sa_predict_measured_accs = np.load(
    '{}_sa-predict-measured-accs.npy'.format(input_path))

# compute differences between accuracies
param_accs_diff = lse_param_accs - sa_param_accs
predict_precise_accs_diff = lse_predict_precise_accs - sa_predict_precise_accs
predict_measured_accs_diff = lse_predict_measured_accs - sa_predict_measured_accs

plt.figure(0)
contour_param = plt.contour(
    err_stds_x, err_stds_y, param_accs_diff)
plt.title('$ d_{param_{LSE}} - d_{param_{SA}} $')
plt.clabel(contour_param, inline=True, fontsize=10)
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')
plt.savefig(
    '{}_param{}'.format(output_path, output_ext),
    dpi=200)

plt.figure(1)
contour_predict_precise = plt.contour(
    err_stds_x, err_stds_y, predict_precise_accs_diff)
plt.clabel(contour_predict_precise, inline=True, fontsize=10)
plt.title('$ d_{predict-precise_{LSE}} - d_{predict-precise_{SA}} $')
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')
plt.savefig(
    '{}_predict-precise{}'.format(output_path, output_ext),
    dpi=200)

plt.figure(2)
contour_predict_measured = plt.contour(
    err_stds_x, err_stds_y, predict_measured_accs_diff)
plt.clabel(contour_predict_measured, inline=True, fontsize=10)
plt.title('$ d_{predict-measured_{LSE}} - d_{predict-measured_{SA}} $')
plt.xlabel('$ \sigma_{\epsilon} $')
plt.ylabel('$ \sigma_{\delta} $')
plt.savefig(
    '{}_predict-measured{}'.format(output_path, output_ext),
    dpi=200)

if args.show:
    plt.show()
