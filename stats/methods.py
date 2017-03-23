import numpy as np

def linear_lse(vals_x, vals_y):
    """Computes coefficients of the linear model using LSE method.

    Parameters:
        vals_x --- array of free values
        vals_y --- array of dependent values

    Return:
        a pair of computed parameters of the linear model.
    """
    num_rows_x = 1 if len(vals_x.shape) == 1 else vals_x.shape[0]
    vals = np.vstack((vals_x, vals_y))
    avg_vals = np.mean(vals, axis=1)
    cov_vals = np.cov(vals)
    est_x0 = avg_vals[:num_rows_x]
    est_alpha = avg_vals[num_rows_x:]
    var_x = cov_vals[:num_rows_x, :num_rows_x]
    cf_correlation = cov_vals[:num_rows_x, num_rows_x:]
    est_beta = np.dot(cf_correlation, np.linalg.inv(var_x))
    return est_alpha - np.dot(est_beta, est_x0), est_beta.flatten()

def linear_pearson(vals_x, vals_y):
    """Computes coefficients of the linear model using Pearson's method.

    Parameters:
        vals_x --- array of free values
        vals_y --- array of dependent values

    Return:
        a pair of computed parameters of the linear model.
    """
    num_rows_x = 1 if len(vals_x.shape) == 1 else vals_x.shape[0]
    vals = np.vstack((vals_x, vals_y))
    avg_vals = np.mean(vals, axis=1)
    cov_vals = np.cov(vals)
    eig_vals, eig_vectors = np.linalg.eig(cov_vals)
    # sort eigenvectors using the order of corresponding eigenvectors:
    # from biggest to smallest and take the num_rows_x largest ones
    eig_vals_order = np.argsort(-eig_vals)
    c = eig_vectors.transpose()[eig_vals_order][:num_rows_x].transpose()
    est_x0 = avg_vals[:num_rows_x]
    est_alpha = avg_vals[num_rows_x:]
    c_top = c[:num_rows_x]
    c_bottom = c[num_rows_x:]
    est_beta = np.dot(c_bottom, np.linalg.inv(c_top))
    return est_alpha - np.dot(est_beta, est_x0), est_beta.flatten()
