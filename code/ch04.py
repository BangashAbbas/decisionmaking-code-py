import networkx as nx
import numpy as np

from scipy.stats import norm
from typing import Callable

from ch02 import Variable


def statistics(variables: list[Variable], graph: nx.DiGraph, data: np.ndarray) -> list[np.ndarray]:
    """
    A function for extracting the statistics, or counts, from a discrete data set `data`,
    assuming a Bayesian network with `variables` and structure `graph`.
    The data set `data` is an n x m matrix, where n is the number of variables and
    m is the number of data points. This function returns an array M of length n.
    The ith component consists of a q_i x r_i matrix of counts.

    ASSUMES DATA IS ZERO-INDEXED: instead of the entries begin classes {1, 2, 3}, they must be {0, 1, 2}

    Note: `np.ravel_multi_index` indexes the parental instantiations differently than the textbook's `sub2ind`
    """
    n = len(variables)
    r = np.array([var.r for var in variables])
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    M = [np.zeros((q[i], r[i])) for i in range(n)]
    for o in data.T:
        for i in range(n):
            k = o[i]
            parents = list(graph.predecessors(i))
            j = 0
            if len(parents) != 0:
                j = np.ravel_multi_index(o[parents], r[parents])
            M[i][j, k] += 1.0
    return M


def prior(variables: list[Variable], graph: nx.DiGraph) -> list[np.ndarray]:
    """
    A function for generating a prior alpha_{ijk} where all entries are 1.
    The array of matrices that this function returns takes the same form
    as the counts generated by `statistics`.
    To determine the appropriate dimensions, the function takes as input the
    list of variables `variables` and structure `graph`.
    """
    n = len(variables)
    r = [var.r for var in variables]
    q = np.array([int(np.prod([r[j] for j in graph.predecessors(i)])) for i in range(n)])
    return [np.ones((q[i], r[i])) for i in range(n)]


def gaussian_kernel(b: float) -> Callable[[float], float]:
    """Returns a zero-mean Gaussian kernel φ(x) with bandwidth `b`"""
    return lambda x: norm.pdf(x, loc=0, scale=b)


def kernel_density_estimate(kernel: Callable[[float | np.ndarray], float], observations: np.ndarray) -> Callable:
    """Implementation of kernel density estimation for a kernel and list of observations"""
    return lambda x: np.mean([kernel(x - o) for o in observations])
