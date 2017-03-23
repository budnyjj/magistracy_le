import math
import random
import numpy as np
import sympy as sp


def precise(
        f_expression,
        num_vals,
        min_x,
        max_x):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        y = expression(x)

    Parameters:
        f_expression --- functional relation between free and dependent variable
        num_vals     --- number of values to generate
        min_x        --- minimal value of free variable
        max_x        --- maximal value of free variable
        err_std_x    --- standart deviation of errors applied to the free variable
        err_std_y    --- standart deviation of errors applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    x = np.linspace(min_x, max_x, num_vals, dtype=np.float)
    y = np.vectorize(f_expression)(x)
    return x, y

def determined(
        f_expression,
        num_vals,
        min_x,
        max_x,
        err_std_x,
        err_std_y):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        x* in [min_x, max_x], x*_i - x*_{i-1} = (max_x - min_x) / num_vals
        y* = expression(x*)
        x = x* + gauss(err_std_x)
        y = y* + gauss(err_std_y)

    Parameters:
        f_expression --- functional relation between free and dependent variable
        num_vals     --- number of values to generate
        min_x        --- minimal value of free variable
        max_x        --- maximal value of free variable
        err_std_x    --- standart deviation of errors applied to the free variable
        err_std_y    --- standart deviation of errors applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    # real X values without errors
    real_x, real_y = precise(f_expression, num_vals, min_x, max_x)
    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(0, err_std_x)
    )(real_x)
    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(0, err_std_y)
    )(real_y)
    return x, y

def uniform(
        f_expression,
        num_vals,
        min_x,
        max_x,
        err_std_x,
        err_std_y):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        x* is a uniform value from [min_x, max_x)
        y* = expression(x*)
        x = x* + gauss(err_std_x)
        y = y* + gauss(err_std_y)

    Parameters:
        f_expression --- functional relation between free and dependent variable
        num_vals     --- number of values to generate
        min_x        --- minimal value of free variable
        max_x        --- maximal value of free variable
        err_std_x    --- standart deviation of errors applied to the free variable
        err_std_y    --- standart deviation of errors applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    # real X values without errors
    real_x = np.random.uniform(min_x, max_x, num_vals)
    real_y = np.vectorize(f_expression)(real_x)
    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(0, err_std_x)
    )(real_x)
    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(0, err_std_y)
    )(real_y)
    return x, y

def normal(
        f_expression,
        num_vals,
        min_x,
        max_x,
        err_std_x,
        err_std_y):
    """Provides specified number of values (x, y),
       linked with specified relation, using the following scheme:
        x* is a normal value from [min_x, max_x)
        y* = expression(x*)
        x = x* + gauss(err_std_x)
        y = y* + gauss(err_std_y)

    Parameters:
        f_expression --- functional relation between free and dependent variable
        num_vals     --- number of values to generate
        min_x        --- minimal value of free variable
        max_x        --- maximal value of free variable
        err_std_x    --- standart deviation of errors applied to the free variable
        err_std_y    --- standart deviation of errors applied to the result values

    Returns:
         a pair of numpy arrays with measurements of the free and dependent variables.
    """
    # real X values without errors
    avg_x = (min_x + max_x) / 2
    std_x = (max_x - min_x) / 6
    real_x = np.random.normal(avg_x, std_x, num_vals)
    real_y = np.vectorize(f_expression)(real_x)
    # add X errors with current normal distribution
    x = np.vectorize(
        lambda v: v + random.gauss(0, err_std_x)
    )(real_x)
    # add Y errors with current normal distribution
    y = np.vectorize(
        lambda v: v + random.gauss(0, err_std_y)
    )(real_y)
    return x, y
