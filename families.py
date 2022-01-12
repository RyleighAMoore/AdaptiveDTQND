import numpy as np
from scipy import special as sp
from opoly1d import OrthogonalPolynomialBasis1D

'''
Credit: Akil Narayan - Pyopoly
'''

class JacobiPolynomials(OrthogonalPolynomialBasis1D):
    def __init__(self, alpha=0., beta=0.):
        OrthogonalPolynomialBasis1D.__init__(self)
        assert alpha > -1., beta > -1.
        self.alpha, self.beta = alpha, beta

    def recurrence_driver(self,N):
        # Returns the first N+1 recurrence coefficient pairs for the Jacobi
        # polynomial family.

        if N < 1:
            return np.ones((0,2))

        ab = np.ones((N+1,2)) * np.array([self.beta**2.- self.alpha**2., 1.])

        # Special cases
        ab[0,0] = 0.
        ab[1,0] = (self.beta - self.alpha) / (self.alpha + self.beta + 2.)
        ab[0,1] = np.exp( (self.alpha + self.beta + 1.) * np.log(2.) +
                          sp.gammaln(self.alpha + 1.) + sp.gammaln(self.beta + 1.) -
                          sp.gammaln(self.alpha + self.beta + 2.)
                        )

        if N > 1:
            ab[1,1] = 4. * (self.alpha + 1.) * (self.beta + 1.) / (
                       (self.alpha + self.beta + 2.)**2 * (self.alpha + self.beta + 3.) )

            if N > 2:
                ab[2,0] /= (2. + self.alpha + self.beta) * (4. + self.alpha + self.beta)

                inds = np.arange(2.,N+1)
                ab[3:,0] /= (2. * inds[:-1] + self.alpha + self.beta) * (2 * inds[:-1] + self.alpha + self.beta + 2.)
                ab[2:,1] = 4 * inds * (inds + self.alpha) * (inds + self.beta) * (inds + self.alpha + self.beta)
                ab[2:,1] /= (2. * inds + self.alpha + self.beta)**2 * (2. * inds + self.alpha + self.beta + 1.) * (2. * inds + self.alpha + self.beta - 1)

        ab[:,1] = np.sqrt(ab[:,1])

        if self.probability_measure:
            ab[0,1] = 1.

        return ab

class HermitePolynomials(OrthogonalPolynomialBasis1D):
    def __init__(self, rho=0.):
        OrthogonalPolynomialBasis1D.__init__(self)
        assert rho > -1.
        self.rho = rho

    def recurrence_driver(self, N):
        # Returns the first N+1 recurrence coefficient pairs for the Hermite
        # polynomial family.

        if N < 1:
            return np.ones((0,2))

        ab = np.zeros((N,2))
        ab[0,1] = sp.gamma(self.rho+0.5)

        ab[1:,1] = 0.5*np.arange(1., N)
        ab[np.arange(N) % 2 == 1,1] += self.rho

        ab[:,1] = np.sqrt(ab[:,1])

        if self.probability_measure:
            ab[0,1] = 1.

        return ab

