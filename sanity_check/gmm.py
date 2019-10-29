import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import cv2
import scipy.stats
import scipy.special
from util import general_utils


def gmm(U, V, N, x, eta):
    S2 = np.clip(V - U ** 2, a_min=0.01, a_max=None)
    S = np.sqrt(S2)

    XdU = x - U
    XdUoS = XdU / S
    XdUoS2 = XdUoS ** 2

    nTotal = 1 / eta - 1  # fixed number
    N = nTotal * N / np.sum(N, axis=1,
                            keepdims=True)  # D x K: at each feature i, the total of points along the mixtures is nTotal
    P = N / nTotal  # D x K: where P[i, j] is the weight of mixture j for feature i
    prob = np.sum(P * scipy.stats.norm(0, 1).cdf(np.abs(XdUoS)), axis=1,
                  keepdims=True)  # D x 1: the probability of feature i belonging to foreground (being a outlier)

    log_prob = np.log(N) + -0.5 * XdUoS2 - np.log(S)
    Gamma = scipy.special.softmax(log_prob, axis=1)  # the mixture probability of the new point

    N = N + Gamma  # Change the point density of each mixtures (according to the mixture probability of the new point)
    Eta = Gamma / N
    U = U + Eta * (x - U)
    V = V + Eta * (x ** 2 - V)

    return U, V, N, prob


def gmm_numpy_tensor(x, U, V, N, eta):
    """
    x: B x C x H x W
    U: B x (C x K) x H x W
    V: B x (C x K) x H x W
    N: B x (C x K) x H x W
    """

    B, C, H, W = x.shape
    B, CK, H, W = U.shape
    K = int(CK / C)

    S2 = np.clip(V - np.power(U, 2), a_min=0.01, a_max=None)
    S = np.sqrt(S2)

    # X_cat: B x CK x H x W
    X_cat = np.concatenate([
        np.concatenate([x[:, i:i + 1, :, :] for _ in range(K)], axis=1)
        for i in range(C)
    ], axis=1)

    XdU = X_cat - U  # B x CK x H x W
    XdUoS = XdU / S  # B x CK x H x W
    XdUoS2 = np.power(XdUoS, 2)  # B x CK x H x W

    nTotal = 1 / eta - 1  # scalar

    N = np.concatenate([
        nTotal * N[:, i * K:(i + 1) * K, :, :] / np.sum(N[:, i * K:(i + 1) * K, :, :], axis=1, keepdims=True)
        for i in range(C)
    ], axis=1)
    assert N.shape == (B, CK, H, W)

    P = N / nTotal  # P: B x CK x H x W
    assert P.shape == (B, CK, H, W)

    # cdf: B x CK x H x W
    cdf = np.concatenate([
        scipy.stats.norm(0, 1).cdf(np.abs(XdUoS[:, i:i + 1, :, :]))
        for i in range(CK)
    ], axis=1)
    assert cdf.shape == (B, CK, H, W)

    # prob: B x CK x H x W
    prob = np.concatenate([
        np.sum(P[:, i * K:(i + 1) * K, :, :] * cdf[:, i * K:(i + 1) * K, :, :], axis=1, keepdims=True)
        for i in range(C)
    ], axis=1)
    assert prob.shape == (B, C, H, W)

    log_prob = np.log(N) + -0.5 * XdUoS2 - np.log(S)
    Gamma = scipy.special.softmax(log_prob, axis=1)

    N = N + Gamma
    Eta = Gamma / N
    U = U + Eta * (X_cat - U)
    V = V + Eta * (np.power(X_cat, 2) - V)

    return U, V, N, prob


class GMMBlock(nn.Module):
    def __init__(self, eta=0.01):
        super(GMMBlock, self).__init__()

        self.eta = eta

    def forward(self, x, U, V, N, eta):
        """
        x: B x C x H x W
        U: B x (C x K) x H x W
        V: B x (C x K) x H x W
        N: B x (C x K) x H x W
        """
        B, C, H, W = x.shape
        B, CK, H, W = U.shape
        K = int(CK / C)

        S2 = torch.clamp(V - torch.pow(U, 2), min=0.01)
        S = torch.sqrt(S2)

        # X_cat: B x CK x H x W
        X_cat = torch.cat([
            torch.cat([x[:, i:i + 1, :, :] for _ in range(K)], dim=1)
            # X_cat[:, i*K:(i+1)*K, :, :] corresponds to a feature map with K mixtures
            for i in range(C)
        ], dim=1)

        XdU = X_cat - U  # B x CK x H x W
        XdUoS = XdU / S  # B x CK x H x W
        XdUoS2 = torch.pow(XdUoS, 2)  # B x CK x H x W

        nTotal = 1 / eta - 1  # scalar

        N = torch.cat([
            nTotal * N[:, i * K:(i + 1) * K, :, :] / torch.sum(N[:, i * K:(i + 1) * K, :, :], dim=1, keepdim=True)
            for i in range(C)
        ], dim=1)
        assert N.shape == torch.Size(np.array([B, CK, H, W]))

        P = N / nTotal  # P: B x CK x H x W
        assert P.shape == torch.Size(np.array([B, CK, H, W]))

        # cdf: B x CK x H x W
        cdf = torch.cat([
            Normal(0, 1).cdf(torch.abs(XdUoS[:, i:i + 1, :, :]))
            for i in range(CK)
        ], dim=1)
        assert cdf.shape == torch.Size(np.array([B, CK, H, W]))

        # prob: B x CK x H x W
        prob = torch.cat([
            torch.sum(P[:, i * K:(i + 1) * K, :, :] * cdf[:, i * K:(i + 1) * K, :, :], dim=1, keepdim=True)
            for i in range(C)
        ], dim=1)
        assert prob.shape == torch.Size(np.array([B, C, H, W]))

        log_prob = torch.log(N) + -0.5 * XdUoS2 - torch.log(S)

        # Gamma = nn.Softmax(dim=1)(log_prob)

        Gamma = torch.cat([
            nn.Softmax(dim=1)(log_prob[:, i * K:(i + 1) * K, :, :])
            for i in range(C)
        ], dim=1)

        N = N + Gamma
        Eta = Gamma / N
        U = U + Eta * (X_cat - U)
        V = V + Eta * (torch.pow(X_cat, 2) - V)

        return U, V, N, prob


if __name__ == '__main__':
    eta = 0.01
    k = 3
    c = 3

    """
    baseline/highway
    """
    frame_dir = "/Users/lekhang/Desktop/Khang/data/highway/input"
    frame_files = general_utils.get_all_files(f"{frame_dir}", keep_dir=True)
    frame_files = sorted(frame_files)

    frame_0 = cv2.imread(frame_files[0], 0)
    h, w = frame_0.shape

    U = np.array([np.array(cv2.imread(frame_files[i], 0).flatten()) / 255. for i in range(k)]).T
    U = np.random.rand(*U.shape)  # TODO: set this make the result look very good - why?
    assert U.shape == (h * w, k)
    V = U ** 2
    N = np.ones((h * w, k))

    U2 = torch.from_numpy(np.random.rand(1, c * k, h, w)).float()
    V2 = torch.pow(U2, 2)
    N2 = torch.ones((1, c * k, h, w)).float()

    gmm_tensor = GMMBlock()

    for frame_file in frame_files:
        frame = cv2.imread(frame_file, 0)
        frame_rgb = cv2.imread(frame_file)

        frame = frame / 255.
        frame_rgb = frame_rgb / 255.

        U, V, N, prob = gmm(U, V, N, np.expand_dims(frame.flatten(), axis=-1), eta)

        frame_tensor = torch.from_numpy(np.expand_dims(np.moveaxis(frame_rgb[:, :, :], -1, 0), axis=0)).float()
        U2, V2, N2, prob2 = gmm_tensor(frame_tensor, U2, V2, N2, eta)
        prob2 = prob2.numpy()

        foreground = (prob > 0.95).reshape((h, w)).astype("uint8") * 255
        foreground21 = (prob2[:, 0, :, :] > 0.95).reshape((h, w)).astype("uint8") * 255
        foreground22 = (prob2[:, 1, :, :] > 0.95).reshape((h, w)).astype("uint8") * 255
        foreground23 = (prob2[:, 2, :, :] > 0.95).reshape((h, w)).astype("uint8") * 255

        cv2.imshow("Video", frame)
        # cv2.imshow("Foreground", foreground)
        cv2.imshow("Foreground 1", foreground21)
        # cv2.imshow("Foreground 2", foreground22)
        # cv2.imshow("Foreground 3", foreground23)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
