import numpy as np
from scipy import special as sp

class OrthogonalPolynomialBasis1D:
    def __init__(self, recurrence=[]):
        self.probability_measure=True
        self.ab = np.zeros([0,2])
        self.lambdas = []
        pass

    def recurrence(self,N):
        """
        Returns the first N+1 orthogonal polynomial recurrence pairs.
        The orthonormal polynomial family satisfies the recurrence

          p_{-1}(x) = 0
          p_0(x) = 1/ab[0,1]

          x p_n(x) = ab[n+1,1] p_{n+1}(x) + ab[n+1,0] p_n(x) + ab[n,1] p_{n-1}(x)
           (n >= 0)

        The value ab[0,0] is ignored and never used.

        Recurrence coefficients ab, once computed, are stored as an
        instance variable. On subsequent calls to this function, the
        stored values are returned if the instance variable already
        contains the desired coefficients. If the instance variable does
        not contain enough coefficients, then a call to
        recurrence_driver is performed to compute the desired
        coefficients, and the output is stored in the instance variable.

        Parameters
        ----------
        N: positive integer
            Maximum polynomial degree for desired recurrence coefficients

        Returns
        -------
        ab: ndarray
            (N+1) x 2 array of recurrence coefficients.
        """

        if N+1 > self.ab.shape[0]:
            self.ab = self.recurrence_driver(N)
            return self.ab
        else:
            return self.ab[:(N+1),:]

    # Recurrence coefficient functions should be defined as follows:
    # The returned array has size (N+1) x 2. The [0,0] entry is not used
    # and set to 0. If the array is ab, then the orthonormal version of
    # the three-term recurrence with the degree-n polynomial p_n at x is:
    #
    #   ab[n+1, 1] * p_{n+1} = (x - ab[n+1,0]) * p_n - ab[n, 1] p_{n-1}
    def recurrence_driver(self,N):
        raise ValueError('Define this')
        return

    def eval(self, x, n, d=0):
        # Evaluates univariate orthonormal polynomials given their
        # three-term recurrence coefficients ab.
        #
        # Evaluates the d'th derivative. (Default = 0)
        #
        # Returns a numel(x) x numel(n) x numel(d) array.

        n = np.asarray(n)
        if isinstance(x, int) or isinstance(x, float):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        if n.size < 1 or x.size < 1:
            return np.zeros(0)

        nmax = np.max(n)
        ab = self.recurrence(nmax+1)

        assert nmax < ab.shape[0]
        assert np.min(n) > -1
        assert np.all(d >= 0)

        p = np.zeros( x.shape + (nmax+1,) )
        xf = x.flatten()

        p[:,0] = 1/ab[0,1]

        if nmax > 0:
            p[:,1] = 1/ab[1,1] * ( (xf - ab[1,0])*p[:,0] )

        for j in range(2, nmax+1):
            p[:,j] = 1/ab[j,1] * ( (xf - ab[j,0])*p[:,j-1] - ab[j-1,1]*p[:,j-2] )

        if type(d) == int:
            if d == 0:
                return p[:,n.flatten()]
            else:
                d = [d]

        preturn = np.zeros([p.shape[0], n.size, len(d)])

        # Parse the list d to find which indices contain which
        # derivative orders

        indlocations = [i for i,val in enumerate(d) if val==0]
        for i in indlocations:
            preturn[:,:,i] = p[:,n.flatten()]

        for qd in range(1, max(d)+1):

            pd = np.zeros(p.shape)

            for qn in range(qd,nmax+1):
                if qn == qd:
                    # The following is an over/underflow-resistant way to
                    # compute ( qd! * kappa_{qd} ), where qd is the
                    # derivative order and kappa_{qd} is the leading-order
                    # coefficient of the degree-qd orthogonal polynomial.
                    # The explicit formula for the lading coefficient of the
                    # degree-qd orthonormal polynomial is prod(1/b[j]) for
                    # j=0...qd.
                    pd[:,qn] = np.exp( sp.gammaln(qd+1) - np.sum( np.log( ab[:(qd+1),1] ) ) )
                else:
                    pd[:,qn] = 1/ab[qn,1] * ( ( xf - ab[qn,0] ) * pd[:,qn-1] - ab[qn-1,1] * pd[:,qn-2] + qd*p[:,qn-1] )

            # Assign pd to proper locations
            indlocations = [i for i,val in enumerate(d) if val==qd]
            for i in indlocations:
                preturn[:,:,i] = pd[:,n.flatten()]

            p = pd

        if len(d) == 1:
            return preturn.squeeze(axis=2)
        else:
            return preturn

    def jacobi_matrix(self, N):
        """
        Returns the N x N jacobi matrix associated to the polynomial family.
        """
        ab = self.recurrence(N+1)
        J = np.diag(ab[1:N,1], k=1) + np.diag(ab[1:(N+1),0],k=0) + np.diag(ab[1:N,1], k=-1)
        return J

    def apply_jacobi_matrix(self, v):
        """
        Premultiplies the input array by the Jacobi matrix of the
        appropriate size for the polynomial family. Applies the Jacobi
        matrix across the first dimension of v.

        Parameters
        ----------
        v: ndarray
            Input vector or array

        Returns
        -------
        Jv: ndarray
            J*v, where J is the Jacobi matrix of size v.shape[0].
        """

        N = v.shape[0]
        ab = self.recurrence(N+1)

        # Rebroadcast v so we can take advantage of numpy's
        # multiplication
        v = np.moveaxis(v, 0, -1)

        #J = np.diag(ab[1:N,1], k=1) + np.diag(ab[1:(N+1),0],k=0) + np.diag(ab[1:N,1], k=-1)
        Jv = v*ab[1:(N+1),0]
        Jv[...,:-1] += v[...,1:]*ab[1:N,1]
        Jv[...,1:] += v[...,:-1]*ab[1:N,1]
        Jv = np.moveaxis(Jv, -1, 0)

        return Jv

    def gauss_quadrature(self, N):
        """
        Computes the N-point Gauss quadrature rule associated to the
        recurrence coefficients ab.
        """

        from numpy.linalg import eigh

        lamb,v = eigh(self.jacobi_matrix(N))

        return lamb, v[0,:]**2

    def gauss_radau_quadrature(self, N, anchor=0.):
        """
        Computes the N-point Gauss quadrature rule associated to the
        polynomial family, with a node at the specified anchor.
        """

        ### Note: the quadrature weight underflows for anchor far
        ### outside the support interval. This causes imprecise quadrature 
        ### results for large-degree polynomials evaluated far outside 
        ### the support interval.


        from numpy.linalg import eigh

        ab = self.recurrence(N+1)
        c = self.r_eval(anchor, N)

        cd = ab.copy()
        cd[N,0] += c*cd[N,1]

        J = np.diag(cd[1:N,1], k=1) + np.diag(cd[1:(N+1),0],k=0) + np.diag(cd[1:N,1], k=-1)
        lamb,v = eigh(J)

        return lamb, v[0,:]**2

    def leading_coefficient(self, N):
        """
        Returns the leading coefficients for the first N polynomial basis elements.
        """
        assert N > 0
        return np.cumprod( self.recurrence(N)[:,1] )

    def canonical_connection(self, N):
        """
        Returns the N x N matrix C, where row n of C contains expansion
        coefficients for p_n in the monomial basis. I.e.,

         p_n(x) = sum_{j=0}^{n} C[n,j] x**j,

        for n = 0, ..., N-1.
        """

        ab = self.recurrence(N)
        C = np.zeros([N,N])

        if N < 1:
            return C

        C[0,0] = 1/ab[0,1]
        if N == 1:
            return C

        C[1,1] = C[0,0]/ab[1,1]
        C[1,0] = -ab[1,0]*C[0,0]/ab[1,1]

        for n in range(1, N-1):
            C[n+1,0] = -ab[n+1,0]*C[n,0] - ab[n,1]*C[n-1,0]
            C[n+1,n] = C[n,n-1] - ab[n+1,0]*C[n,n]
            C[n+1,n+1] = C[n,n]

            js = np.arange(1,n)
            C[n+1,js] = C[n,js-1] - ab[n+1,0]*C[n,js] - ab[n,1]*C[n-1,js]

            C[n+1,:] /= ab[n+1,1]

        return C

    def canonical_connection_inverse(self, N):
        """
        Returns the N x N matrix C, where row n of C contains expansion
        coefficients for x^n in the orthonormal basis . I.e.,

         x^n = sum_{j=0}^{n} C[n,j] p_j(x)

        for n = 0, ..., N-1.
        """

        ab = self.recurrence(N)
        C = np.zeros([N,N])

        if N < 1:
            return C

        C[0,0] = ab[0,1]
        if N == 1:
            return C

        C[1,1] = ab[0,1]*ab[1,1]
        C[1,0] = ab[1,0]*ab[0,1]

        for n in range(1,N-1):
            C[n+1,:] = self.apply_jacobi_matrix(C[n,:])

        return C

    def tuple_product_generator(self, IC, ab=None):
        """
        Helper function that increments indices for a polynomial product expansion.

        IC is a vector with entries

          IC[j] = < p_j, p_alpha >, j \in range(N),

        where N = IC.size. The notation < ., .> is the inner product
        under which the polynomial family is orthonormal. alpha is a
        multi-index of arbitrary shape with a polynomial defined by

          p_alpha = \prod_{j \in range(alpha.size)} p_{alpha[j]}(x).

        The value of alpha is not needed by this function.

        This function returns an N x N matrix C with entries

            C[n,j] = < p_n p_j, p_alpha >, j, n \in range(N)

        Parameters
        ----------
        IC: vector (1d array)
            Values of input inner products

        ab: ndarray, optional
            Recurrence coefficients

        Returns
        -------
        C: ndarray
            Output coefficient expansion vector for beta
        """

        N = IC.size
        ab = self.recurrence(N+1)
        C = np.zeros((N,N))
        C[0,:] = IC

        for n in range(N-1):
            C[n+1,:] = self.apply_jacobi_matrix(C[n,:]) - ab[n+1,0]*C[n,:]
            if n > 0:
                C[n+1,:] -= ab[n,1]*C[n-1,:]
            C[n+1,:] /= ab[n+1,1]

        return C

    def tuple_product(self, N, alpha):
        """
        Computes integrals of polynomial products. Returns an N x N matrix C with entries

            C[n,m] = < p_n p_m, p_alpha >,

        where alpha is a vector of integers and p_alpha is defined

            p_alpha = prod_{j=1}^{alpha.size} p_[alpha[j]](x),

        The notation <., .> denotes the inner product under which the
        polynomial family is orthogonal.

        Parameters
        ----------
        N: integer
            Size of matrix to return

        alpha: ndarray (1d)
            Multi-index defining a polynomial product

        Returns
        -------
        C: ndarray
            Output N x N matrix containing integral values
        """

        M = alpha.size
        if M == 0:
            return np.eye(N)

        #Nmax = max(N, np.max(alpha)+1)
        Nmax = N + np.sum(alpha) + 1
        C = np.zeros((Nmax, Nmax))

        ab = self.recurrence(Nmax+1)

        # C[j,k] = delta_{j,k}
        # Initial condition: IC[j] = < p_0 p_j, p_{alpha[0]} >
        #                          = C[alpha[0],:]
        C[alpha[0],alpha[0]] = 1.

        for j in range(M):
            IC = C[alpha[j],:]/ab[0,1]
            C = self.tuple_product_generator(IC, ab=ab)

        return C[:N,:N]

    def derivative_expansion(self, N, d):
        """
        Computes an N x N matrix with expansion coefficients for
        derivatives of orthogonal polynomials. I.e., computes the numbers C[n,j] for

          p_n^{(d)}(x) = sum_{j=0}^n C[n,j] p_j(x),

        for j,n = 0, ..., N-1.

        Parameters
        ----------
        N: integer
            Size of matrix to return

        d: integer
            Derivative order

        Returns
        -------
        C: ndarray
            Output N x N matrix containing expansion coefficients
        """

        from scipy.special import gammaln

        assert N >= 0
        assert d >= 0

        if N == 0:
            return np.zeros(0)

        if d == 0:
            return np.eye(N)

        ab = self.recurrence(N+1)
        C = np.eye(N+d+1)[:N,:]

        for dj in range(1,d+1):
            Cprev = C[:,:-1].copy()
            C = np.zeros([N,N+d+1-dj])
            C[dj,0] = np.exp(gammaln(dj+1) - np.sum(np.log(ab[1:(dj+1),1])))

            for n in range(dj,N-1):
                C[n+1,:] = 1/ab[n+1,1] * ( self.apply_jacobi_matrix(C[n,:]) - ab[n+1,0]*C[n,:] - ab[n,1]*C[n-1,:] + dj*Cprev[n,:] )

        return C[:,:-1]

    def r_eval(self,x, n, d=0):
        """
        Evalutes ratios of orthonormal polynomials. These are given by

          r_n(x) = p_n(x) / p_{n-1}(x),  n >= 1

        The output is a x.size x n.size array.
        """

        n = np.asarray(n)

        if isinstance(x, float) or isinstance(x, int):
            x = np.asarray([x])
        else:
            x = np.asarray(x)

        if n.size < 1 or x.size < 1:
            return np.zeros(0)

        nmax = np.max(n)
        ab = self.recurrence(nmax+1)

        assert nmax < ab.shape[0]
        assert np.min(n) > -1
        assert np.all(d >= 0) and np.all(d < 1)

        r = np.zeros( x.shape + (nmax+1,) )
        xf = x.flatten()

        r[:,0] = 1/ab[0,1]
        if nmax > 0:
            r[:,1] = 1/ab[1,1] * ( x - ab[1,0] ) / ab[0,1]

        for j in range(2, nmax+1):
            r[:,j] = 1/ab[j,1] * ( (xf - ab[j,0]) - ab[j-1,1]/r[:,j-1] )

        r = r[:,n.flatten()]

        if type(d) == int:
            if d == 0:
                return r
            else:
                d = [d]

    def qpoly1d_eval(self, x, n, d=0):
        """
        Evalutes Christoffel-function normalized polynomials. These are
        given by

          q_k(x) = p_k(x) / sqrt( sum_{j=0}^{n-1} p_j^2 ), k = 0, ..., n-1

        The output is a x.size x n array
        """

        assert n > 0

        ab = self.recurrence(n-1)

        q = np.zeros((x.size, n))
        q[:,0] = 1.
        qt = np.zeros(x.size)

        if n > 1:
            qt = 1/ab[1,1] * (x - ab[1,0]) * q[:,0]
            q[:,1] = qt / np.sqrt(1 + qt**2)

        for j in range(1, n-1):
            qt = 1/ab[j+1,1] * ( (x - ab[j+1,0])*q[:,j] - ab[j,1] * q[:,j-1] / np.sqrt(1 + qt**2) )
            q[:,j+1] = qt / np.sqrt(1 + qt**2)

        if type(d) == int:
            if d == 0:
                return q
            else:
                d = [d]

        assert False
        qreturn = np.zeros([q.shape[0], q.shape[1], len(d)])
        for (qi,qval) in enumerate(d):
            if qval == 0:
                qreturn[:,:,qi] = q

        for qd in range(1, max(d)+1):
            assert False

        return qreturn

    def christoffel_function(self, x, k):
        """
        Computes the normalized (inverse) Christoffel function lambda, defined as

          lambda**2 = k / sum(p**2, axi=1),

        where p is a matrix containing evaluations of an orthonormal
        polynomial family up to degree k-1, defined by the recurrence
        coefficients ab.
        """

        assert k > 0

        p = self.eval(x, range(k))
        return np.sqrt(float(k) / np.sum(p**2, axis=1))
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from families import JacobiPolynomials, HermitePolynomials
    import variableTransformations as VT
    from scipy.stats import multivariate_normal
    from math import pi
    
    def fun(x):
        return 2*x**2
        # return np.ones(len(x))
    
    '''
    Example 1: $\int_{-1}^1 x^2 dx$
    Use Legendre polynomials, w(x)=1, x in [-1,1], 0 otherwise
    '''
    J = JacobiPolynomials(alpha=0, beta=0)
    N = 30
    k = 30
    x, w = J.gauss_quadrature(N)
    print(np.dot(fun(x),w)*2)
    # >> outputs 1/3 but the solution is 2/3

    '''
    Example 2: $\int_{-inf}^inf e^{-x^2} x^2 dx$
    Use Hermite polynomials, w(x)=e^{-x^2}
    '''
    H = HermitePolynomials(rho=0)
    N = 10
    k = 10
    x, w = H.gauss_quadrature(N)
    print(np.dot(fun(x),w)*np.sqrt(pi))
    
    sigma = .1
    mu = 0
    H = HermitePolynomials(rho=0)
    
    x, w = H.gauss_quadrature(N)
    scaling = np.asarray([[mu, sigma]])
    xCan=VT.map_to_canonical_space(x, 1, scaling) #math to   
    print(np.dot(fun(xCan),w))
    
    # x = np.linspace(-1, 1, 40)

    # Compute interpolant
    c = np.dot(H.eval(xCan, range(N)).T, w*fun(xCan))
    print('c=',c[0])
    
    vals = np.dot(H.eval(xCan, range(N)), c)
    
    plt.figure()
    plt.plot(x, vals, '.')
    # plt.plot(x,fun(x))
    plt.show()
    # plt.plot(x, V[:,:4])

    # plt.show()
    
    # for i in range(k):
    #     for j in range(k):
    #         if i ==j:
    #             print(np.dot(V[:,i]*V[:,j],w))
            # if i !=j:
                # print(np.dot(V[:,i]*V[:,j],w)*(1/np.sqrt(pi)*0.1))
                # assert np.isclose((np.dot(V[:,i]*V[:,j]*(1/np.sqrt(pi)*sigma),w)),0)
    
    
    # # H = HermitePolynomials()
    # # N = 100
    # # k = 15

    # # x, w = J.gauss_quadrature(N)
    # # V = H.eval(x, range(k))
    
    # # plt.figure()
    # # plt.plot(x, V[:,:k])
    # # plt.show()
    
    
    
