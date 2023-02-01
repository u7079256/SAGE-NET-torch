import torch


def saliency_loss(name, mse_beta=None):
    """
    Returns loss for the saliency task.
    :param name: string identifier of loss function.
    :param mse_beta: regularizer for weighted mse.
    :return: the loss function.
    noticed that full_frame_loss = 'kld'
    crop_loss = 'kld' in other words, all loss used in training are kld loss
    """
    assert name in ['mse', 'sse', 'nss', 'simo', 'kld'], f'Unknown loss function: {name}'

    def mean_squared_error(y_true, y_pred):
        """
        Mean squared error loss.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value.
        """
        return torch.mean((y_pred - y_true) ** 2)

    def weighted_mean_squared_error(y_true, y_pred):
        """
        Regularized mean squared error loss.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value.
        """
        return torch.mean((y_pred - y_true) ** 2 / (255 - y_true + mse_beta))  # TODO does 255-y_true make sense?

    def sum_squared_errors(y_true, y_pred):
        """
        Sum of squared errors loss.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value.
        """
        return torch.sum((y_pred - y_true) ** 2)

    def kullback_leibler_divergence(y_true, y_pred, eps=1e-9):
        """
        Kullback-Leiber divergence.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :param eps: regularization epsilon.
        :return: loss value.
        """
        P = y_pred
        P = P / (eps + P.sum(dim=(1, 2, 3), keepdim=True))
        Q = y_true
        Q = Q / (eps + Q.sum(dim=(1, 2, 3), keepdim=True))

        kld = (Q * (Q / (eps + P)).log()).sum(dim=(1, 2, 3))

        return kld

    def information_gain(y_true, y_pred, y_base, eps=1e-8):
        """
        Information gain (sec 4.1.3 of [1]). Assumes shape (b, 1, h, w) for all tensors.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :param y_base: baseline.
        :param eps: regularization epsilon.
        :return: loss value (one symbolic value per batch element).
        """
        P = y_pred / (eps + y_pred.max(dim=(1, 2, 3), keepdim=True)[0])
        Q = y_true
        B = y_base

        Qb = torch.round(Q)  # discretize at 0.5
        N = Qb.sum(dim=(1, 2, 3), keepdim=True)

        ig = (Qb * (torch.log(eps + P) / torch.log(2) - torch.log(eps + B) / torch.log(2))).sum(dim=(1, 2, 3)) / (
                eps + N)

        return ig.mean()

    def simo_loss(y_true, y_pred):
        """
        Loss defined by simo. Assumes shape (b, 2, h, w) for all tensors.
        y[:, 0, :, :] is saliency, we want KLD for saliency.
        y[:, 1, :, :] is fixation, we want IG for fixation using saliency groundtruth as baseline.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value (one symbolic value per batch element).
        """
        eps = 1e-7
        y_true_sal = y_true[:, 0:1, :, :]
        y_true_fix = y_true[:, 1:, :, :]

        y_pred_sal = y_pred[:, 0:1, :, :]
        y_pred_fix = y_pred[:, 1:, :, :]

        # Normalize the ground truth saliency and prediction saliency
        P = y_pred_sal
        P = P / (eps + torch.sum(P, dim=[1, 2, 3], keepdim=True))
        Q = y_true_sal
        Q = Q / (eps + torch.sum(Q, dim=[1, 2, 3], keepdim=True))

        # Kullback-Leiber divergence
        kld = torch.sum(Q * torch.log(eps + Q / (eps + P)), dim=[1, 2, 3])

        # Information gain
        B = y_true_sal
        Qb = torch.round(Q)  # discretize at 0.5
        N = torch.sum(Qb, dim=[1, 2, 3], keepdim=True)

        ig = torch.sum(Qb * (torch.log(eps + P) / torch.log(2) - torch.log(eps + B) / torch.log(2)), dim=[1, 2, 3]) / (
                eps + N)

        return kld - ig

    def normalized_scanpath_saliency(y_true, y_pred):
        """
        Normalized Scanpath Saliency (sec 4.1.2 of [1]). Assumes shape (b, 1, h, w) for all tensors.
        :param y_true: groundtruth.
        :param y_pred: prediction.
        :return: loss value (one symbolic value per batch element).
        """

        P = y_pred
        P = P / (torch.tensor(1e-6, dtype=P.dtype) + P.max(dim=(1, 2, 3), keepdim=True).values)
        Q = y_true

        Qb = torch.round(Q)  # discretize at 0.5
        N = Qb.sum(dim=(1, 2, 3), keepdim=True)

        mu_P = P.mean(dim=(1, 2, 3), keepdim=True)
        std_P = P.std(dim=(1, 2, 3), keepdim=True)
        P_sign = (P - mu_P) / (torch.tensor(1e-6, dtype=P.dtype) + std_P)

        nss = (P_sign * Qb) / (torch.tensor(1e-6, dtype=P.dtype) + N)
        nss = nss.sum(dim=(1, 2, 3))

        return -nss  # maximize nss

    if name == 'mse' and mse_beta is not None:
        return weighted_mean_squared_error
    elif name == 'mse' and mse_beta is None:
        return mean_squared_error
    elif name == 'sse':
        return sum_squared_errors
    elif name == 'nss':
        return normalized_scanpath_saliency
    elif name == 'simo':
        return simo_loss
    elif name == 'kld':
        return kullback_leibler_divergence

    """
    REFERENCES:
    [1] @article{salMetrics_Bylinskii,
      title     = {What do different evaluation metrics tell us about saliency models?},
      author    = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
      journal   = {arXiv preprint arXiv:1604.03605},
      year      = {2016}
    }
    [2] https://github.com/marcellacornia/mlnet/blob/master/model.py
    """
