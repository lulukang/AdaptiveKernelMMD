import numpy as np
import sklearn.metrics.pairwise as sk_metric

# p_sample = model.sample(5000)
# square_mmd_fine(p_samples, q_samples, 'gaussian')

def square_mmd_fine(p_samples, q_samples, kernel_type):
    """
    n_p: number of samples from true distribution p
    assume n_p >> n_q
    """

    kernel_dict = {
        'gaussian': sk_metric.rbf_kernel,
        'laplacian': sk_metric.laplacian_kernel,
        'sigmoid': sk_metric.sigmoid_kernel,
        'polynomial': sk_metric.polynomial_kernel,
        'cosine': sk_metric.cosine_similarity,
    }

    kernel = kernel_dict[kernel_type]

    # p_samples = np.array(p_samples)
    # q_samples = np.array(q_samples)

    k_xi_xj = kernel(p_samples, p_samples)
    k_yi_yj = kernel(q_samples, q_samples)
    k_xi_yj = kernel(p_samples, q_samples)

    n_p = p_samples.shape[0]
    n_q = q_samples.shape[0]

    off_diag_k_xi_xj = (np.sum(k_xi_xj) - np.sum(np.diag(k_xi_xj))) / n_p / (n_p - 1)
    off_diag_k_yi_yj = (np.sum(k_yi_yj) - np.sum(np.diag(k_yi_yj))) / n_q / (n_q - 1)
    sum_k_xi_yj = np.sum(k_xi_yj) * 2 / n_p / n_q

    return off_diag_k_xi_xj + off_diag_k_yi_yj - sum_k_xi_yj
