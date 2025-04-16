import numpy as np

from clarkson_coreset_algorithm.clarkson import computeClarksonCoreset
from .strategy import Strategy


class ClarksonCoreset(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ClarksonCoreset, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        embedding = embedding.numpy()

        q_ixds, _ = computeClarksonCoreset(embedding)

        np.random.shuffle(q_ixds)
        q_ixds = q_ixds[:n]

        return idxs_unlabeled[q_idxs]
