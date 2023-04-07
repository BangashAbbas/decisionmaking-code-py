import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal

from ch02 import Assignment, Factor, FactorTable, BayesianNetwork
from ch02 import marginalize, condition_multiple


class InferenceMethod(ABC):
    @abstractmethod
    def infer(self, *args, **kwargs):
        pass


class DiscreteInferenceMethod(InferenceMethod):
    """
    I introduce the DiscreteInferenceMethod superclass to allow `infer` to be called with different inference methods,
    as shall be seen in the rest of this chapter.
    """
    @abstractmethod
    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        pass


class ExactInference(DiscreteInferenceMethod):
    """
    A naive exact inference algorithm for a discrete Bayesian network `bn`,
    which takes as input a set of query variable names query and evidence associating values with observed variables.
    The algorithm computes a joint distribution over the query variables in the form of a factor.
    """
    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        phi = Factor.prod(bn.factors)
        phi = condition_multiple(phi, evidence)
        for name in (set(phi.variable_names) - set(query)):
            phi = marginalize(phi, name)
        phi.normalize()
        return phi


class VariableElimination(DiscreteInferenceMethod):
    """
    An implementation of the sum-product variable elimination algorithm,
    which takes in a Bayesian network `bn`, a list of query variables `query`, and evidence `evidence`.
    The variables are processed in the order given by `ordering`.
    """
    def __init__(self, ordering: list[int]):
        self.ordering = ordering

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        factors = [condition_multiple(phi, evidence) for phi in bn.factors]
        for i in self.ordering:
            name = bn.variables[i].name
            if name not in query:
                indices = [j for j in range(len(factors)) if factors[j].in_scope(name)]
                if len(indices) != 0:
                    phi = Factor.prod([factors[j] for j in indices])
                    for j in sorted(indices, reverse=True):
                        del factors[j]
                    phi = marginalize(phi, name)
                    factors.append(phi)
        phi = Factor.prod(factors)
        phi.normalize()
        return phi


class DirectSampling(DiscreteInferenceMethod):
    """
    The direct sampling inference method, which takes a Bayesian network `bn`,
    a list of query variables `query`, and evidence `evidence`.

    The method draws `m` samples from the Bayesian network and retains those samples
    that are consistent with the evidence. A factor over the query variables is returned.
    This method can fail if no samples that satisfy the evidence are found.
    """
    def __init__(self, m):
        self.m = m

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        table = FactorTable()
        for _ in range(self.m):
            a = bn.sample()
            if all(a[k] == v for (k, v) in evidence.items()):
                b = a.select(query)
                table[b] = table.get(b, default_val=0.0) + 1
        variables = [var for var in bn.variables if var.name in query]
        phi = Factor(variables, table)
        phi.normalize()
        return phi


class LikelihoodWeightedSampling(DiscreteInferenceMethod):
    """
    The likelihood weighted sampling inference method, which takes a Bayesian network `bn`,
    a list of query variables `query`, and evidence `evidence`.

    The method draws `m` samples from the Bayesian network but sets values from evidence when possible,
    keeping track of the conditional probability when doing so. These probabilities are used to weight
    the samples such that the final inference estimate is accurate. A factor over the query variables is returned.
    """
    def __init__(self, m):
        self.m = m

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        table = FactorTable()
        ordering = list(nx.topological_sort(bn.graph))
        for _ in range(self.m):
            a, w = Assignment(), 1.0
            for j in ordering:
                name, phi = bn.variables[j].name, bn.factors[j]
                if name in evidence:
                    a[name] = evidence[name]
                    w *= phi.table[a.select(phi.variable_names)]
                else:
                    a[name] = condition_multiple(phi, a).sample()[name]
            b = a.select(query)
            table[b] = table.get(b, default_val=0.0) + w
        variables = [var for var in bn.variables if var.name in query]
        phi = Factor(variables, table)
        phi.normalize()
        return phi


def blanket(bn: BayesianNetwork, a: Assignment, i: int) -> Factor:
    """
    A method for obtaining P(X_i | x_{-i}) for a Bayesian network `bn` given a current assignment `a`.
    """
    name = bn.variables[i].name
    value = a[name]  # TODO - F841 local variable 'value' is assigned to but never used (Talk to Mykel & Tim about this)
    a_prime = a.copy()
    del a_prime[name]
    factors = [phi for phi in bn.factors if phi.in_scope(name)]
    phi = Factor.prod([condition_multiple(factor, a_prime) for factor in factors])
    phi.normalize()
    return phi


class GibbsSampling(DiscreteInferenceMethod):
    """
    Gibbs sampling implemented for a Bayesian network `bn` with evidence `evidence` and an ordering `ordering`.
    The method iteratively updates the assignment `a` for `m` iterations.
    """
    def __init__(self, m_samples: int, m_burnin: int, m_skip: int, ordering: list[int]):
        self.m_samples = m_samples
        self.m_burnin = m_burnin
        self.m_skip = m_skip
        self.ordering = ordering

    def infer(self, bn: BayesianNetwork, query: list[str], evidence: Assignment) -> Factor:
        table = FactorTable()
        a = Assignment(bn.sample() | evidence)
        self.gibbs_sample(a, bn, evidence, self.ordering, self.m_burnin)
        for i in range(self.m_samples):
            self.gibbs_sample(a, bn, evidence, self.ordering, self.m_skip)
            b = a.select(query)
            table[b] = table.get(b, default_val=0) + 1
        variables = [var for var in bn.variables if var.name in query]
        phi = Factor(variables, table)
        phi.normalize()
        return phi

    @staticmethod
    def gibbs_sample(a: Assignment, bn: BayesianNetwork, evidence: Assignment, ordering: list[int], m: int):
        for _ in range(m):
            GibbsSampling.update_gibbs_sample(a, bn, evidence, ordering)

    @staticmethod
    def update_gibbs_sample(a: Assignment, bn: BayesianNetwork, evidence: Assignment, ordering: list[int]):
        for i in ordering:
            name = bn.variables[i].name
            if name not in evidence:
                b = blanket(bn, a, i)
                a[name] = b.sample()[name]


class MultivariateGaussianInference(InferenceMethod):
    """
        Inference in a multivariate Gaussian distribution D.

        D: multivariate_normal object defined by scipy.stats
        query: NumPy array of integers specifying the query variables
        evidence_vars: NumPy array of integers specifying the evidence variables
        evidence: NumPy array containing the values of the evidence variables
    """
    def infer(self,
              D: multivariate_normal,
              query: np.ndarray,
              evidence_vars: np.ndarray,
              evidence: np.ndarray) -> multivariate_normal:
        mu, Sigma = D.mean, D.cov
        b, mu_a, mu_b = evidence, mu[query], mu[evidence_vars]
        A = Sigma[query][:, query]
        B = Sigma[evidence_vars][:, evidence_vars]
        C = Sigma[query][:, evidence_vars]
        mu = mu_a + (C @ np.linalg.solve(B, b - mu_b))
        Sigma = A - (C @ (np.linalg.inv(B) @ C.T))
        return multivariate_normal(mu, Sigma)
