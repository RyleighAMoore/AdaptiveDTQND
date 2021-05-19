import numpy as np

def get_lu_leja_samples(poly, generate_basis_matrix,
                        candidate_samples,num_leja_samples,
                        preconditioning_function=None,initial_samples=None):
    """
    Generate Leja samples using LU factorization. 

    Parameters
    ----------
    generate_basis_matrix : callable
        basis_matrix = generate_basis_matrix(candidate_samples)
        Function to evaluate a basis at a set of samples

    generate_candidate_samples : callable
        candidate_samples = generate_candidate_samples(num_candidate_samples)
        Function to generate candidate samples. This can siginficantly effect
        the fekete samples generated

    num_candidate_samples : integer
        The number of candidate_samples

    preconditioning_function : callable
        basis_matrix = preconditioning_function(basis_matrix)
        precondition a basis matrix to improve stability
        samples are the samples used to build the basis matrix. They must
        be in the same order as they were used to create the rows of the basis 
        matrix.

    TODO unfortunately some preconditioing_functions need only basis matrix
    or samples, but cant think of a better way to generically pass in function
    here other than to require functions that use both arguments

    num_leja_samples : integer
        The number of desired leja samples. Must be <= num_indices

    initial_samples : np.ndarray (num_vars,num_initial_samples)
       Enforce that the initial samples are chosen (in the order specified)
       before any other candidate sampels are chosen. This can lead to
       ill conditioning and leja sequence terminating early

    Returns
    -------
    leja_samples : np.ndarray (num_vars, num_indices)
        The samples of the Leja sequence

    data_structures : tuple
        (Q,R,p) the QR factors and pivots. This can be useful for
        quickly building an interpolant from the samples
    """
    # candidate_samples = generate_candidate_samples(num_candidate_samples)
    if initial_samples is not None:
        candidate_samples = np.hstack((initial_samples,candidate_samples))
        num_initial_rows = initial_samples.shape[1]
    else:
        num_initial_rows=0
    
    numSamples = len(candidate_samples.T)
    basis_matrix = generate_basis_matrix(candidate_samples.T, poly.lambdas[:num_leja_samples,:], poly.ab, poly)

    assert num_leja_samples <= basis_matrix.shape[1]
    if preconditioning_function is not None:
        weights = preconditioning_function(basis_matrix, candidate_samples)
        basis_matrix = (basis_matrix.T*weights).T
    else:
        weights = None
    import scipy as sp
    import matplotlib.pyplot as plt
    
    L,U,p, successBoolean = truncated_pivoted_lu_factorization(
        basis_matrix,num_leja_samples,num_initial_rows)
    if (p.shape[0]!=num_leja_samples):
        successBoolean = False
    else:
        successBoolean == True
    p = p[:num_leja_samples]
    leja_samples = candidate_samples[:,p]
    # Ignore basis functions (columns) that were not considered during the
    # incomplete LU factorization
    L = L[:,:num_leja_samples]
    U = U[:num_leja_samples,:num_leja_samples]
    data_structures=[L,U,p]
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        print(('N:', basis_matrix.shape[1]))
        plt.plot(leja_samples[0,0],leja_samples[1,0],'*')
        plt.plot(leja_samples[0,:],leja_samples[1,:],'ro',zorder=10)
        plt.scatter(candidate_samples[0,:],candidate_samples[1,:],s=weights*100,color='b')
        plt.show()
    return leja_samples, data_structures, successBoolean

def truncated_pivoted_lu_factorization(A,max_iters,num_initial_rows=0,
                                       truncate_L_factor=True):
    """
    Compute a incomplete pivoted LU decompostion of a matrix.

    Parameters
    ----------
    A np.ndarray (num_rows,num_cols)
        The matrix to be factored

    max_iters : integer
        The maximum number of pivots to perform. Internally max)iters will be 
        set such that max_iters = min(max_iters,K), K=min(num_rows,num_cols)

    num_initial_rows: integer or np.ndarray()
        The number of the top rows of A to be chosen as pivots before
        any remaining rows can be chosen.
        If object is an array then entries are raw pivots which
        will be used in order.
    

    Returns
    -------
    L_factor : np.ndarray (max_iters,K)
        The lower triangular factor with a unit diagonal. 
        K=min(num_rows,num_cols)

    U_factor : np.ndarray (K,num_cols)
        The upper triangular factor

    raw_pivots : np.ndarray (num_rows)
        The sequential pivots used to during algorithm to swap rows of A. 
        pivots can be obtained from raw_pivots using 
        get_final_pivots_from_sequential_pivots(raw_pivots)

    pivots : np.ndarray (max_iters)
        The index of the chosen rows in the original matrix A chosen as pivots
    """
    num_rows,num_cols = A.shape
    min_num_rows_cols = min(num_rows, num_cols)
    max_iters = min(max_iters, min_num_rows_cols)
    if ( A.shape[1] < max_iters ):
        msg = "truncated_pivoted_lu_factorization: "
        msg += " A is inconsistent with max_iters. Try deceasing max_iters or "
        msg += " increasing the number of columns of A"
        raise Exception(msg)

    # Use L to store both L and U during factoriation then copy out U in post
    # processing
    LU_factor = A.copy()
    raw_pivots = np.arange(num_rows)#np.empty(num_rows,dtype=int)
    LU_factor,raw_pivots,it, successBoolean = continue_pivoted_lu_factorization(
        LU_factor,raw_pivots,0,max_iters,num_initial_rows)
        
    if not truncate_L_factor:
        return LU_factor, raw_pivots
    else:
        pivots = get_final_pivots_from_sequential_pivots(
            raw_pivots)[:it+1]
        L_factor, U_factor = split_lu_factorization_matrix(LU_factor,it+1)
        L_factor = L_factor[:it+1,:it+1]
        U_factor = U_factor[:it+1,:it+1]
        return L_factor, U_factor, pivots, successBoolean
    
def add_columns_to_pivoted_lu_factorization(LU_factor,new_cols,raw_pivots):
    """
    Given factorization PA=LU add new columns to A in unpermuted order and update
    LU factorization
    
    raw_pivots : np.ndarray (num_pivots)
    The pivots applied at each iteration of pivoted LU factorization.
    If desired one can use get_final_pivots_from_sequential_pivots to 
    compute final position of rows after all pivots have been applied.
    """
    assert LU_factor.shape[0]==new_cols.shape[0]
    assert raw_pivots.shape[0]<=new_cols.shape[0]
    num_new_cols = new_cols.shape[1]
    num_pivots = raw_pivots.shape[0]
    for it in range(num_pivots):
        pivot = raw_pivots[it]
        swap_rows(new_cols,it,pivot)

        # update U_factor
        # recover state of col vector from permuted LU factor
        # Let  (jj,kk) represent iteration and pivot pairs
        # then if lu factorization produced sequence of pairs
        # (0,4),(1,2),(2,4) then LU_factor[:,0] here will be col_vector
        # in LU algorithm with the second and third permutations
        # so undo these permutations in reverse order
        col_vector = LU_factor[it+1:,it].copy()
        for ii in range(num_pivots-it-1):
            # (it+1) necessary in two lines below because only dealing
            # with compressed col vector which starts at row it in LU_factor
            jj=raw_pivots[num_pivots-1-ii]-(it+1)
            kk=num_pivots-ii-1-(it+1)
            swap_rows(col_vector,jj,kk)
        row_vector = new_cols[it,:]

        update = np.outer(col_vector,row_vector)
        new_cols[it+1:,:] -= update

        #new_cols = add_rows_to_pivoted_lu_factorization(
        #    new_cols[:it+1,:],new_cols[it+1:,:],num_pivots)

    LU_factor = np.hstack((LU_factor,new_cols))
    return LU_factor

def add_rows_to_pivoted_lu_factorization(LU_factor,new_rows,num_pivots):
    assert LU_factor.shape[1]==new_rows.shape[1]
    num_new_rows = new_rows.shape[0]
    LU_factor_extra = new_rows.copy()
    for it in range(num_pivots):
        LU_factor_extra[:,it]/=LU_factor[it,it]
        col_vector = LU_factor_extra[:,it]
        row_vector = LU_factor[it,it+1:]
        update = np.outer(col_vector,row_vector)
        LU_factor_extra[:,it+1:] -= update
        
    return np.vstack([LU_factor,LU_factor_extra])

def swap_rows(matrix,ii,jj):
    temp = matrix[ii].copy()
    matrix[ii]=matrix[jj]
    matrix[jj]=temp

def pivot_rows(pivots,matrix,in_place=True):
    if not in_place:
        matrix = matrix.copy()
    num_pivots = pivots.shape[0]
    assert num_pivots <= matrix.shape[0]
    for ii in range(num_pivots):
        swap_rows(matrix,ii,pivots[ii])
    return matrix

def get_final_pivots_from_sequential_pivots(sequential_pivots,num_pivots=None):
    if num_pivots is None:
        num_pivots = sequential_pivots.shape[0]
    assert num_pivots >= sequential_pivots.shape[0]
    pivots = np.arange(num_pivots)
    return pivot_rows(sequential_pivots,pivots,False)

def christoffel_weights(basis_matrix):
    """
    Evaluate the 1/K(x),from a basis matrix, where K(x) is the 
    Christoffel function.
    """
    return 1./np.sum(basis_matrix**2,axis=1)

def sqrtNormal_weights(candidate_samples):
    x = candidate_samples[0,:]
    y= candidate_samples[1,:]
    # z =(1/np.sqrt(2*np.pi))*np.exp(-(x**2/(2)+ y**2/(2)))
    z =(1/np.pi)*np.exp(-(x**2/(1)+ y**2/(1)))
    # z =np.exp(-(x**2/(1)+ y**2/(1)))
    return np.sqrt(z)


def continue_pivoted_lu_factorization(LU_factor,raw_pivots,current_iter,
                                      max_iters,num_initial_rows=0):
    it = current_iter
    for it in range(current_iter,max_iters):
        # find best pivot
        if np.isscalar(num_initial_rows) and (it<num_initial_rows):
            pivot = np.argmax(np.absolute(LU_factor[it:num_initial_rows,it]))+it
            # pivot = it
        elif (not np.isscalar(num_initial_rows) and
              (it<num_initial_rows.shape[0])):
            pivot=num_initial_rows[it]
        else:
            pivot = np.argmax(np.absolute(LU_factor[it:,it]))+it


        # update pivots vector
        #swap_rows(pivots,it,pivot)
        raw_pivots[it]=pivot
      
        # apply pivots(swap rows) in L factorization
        swap_rows(LU_factor,it,pivot)

        # check for singularity
        successBoolean = True
        if abs(LU_factor[it,it])<np.finfo(float).eps:
            msg = "pivot %1.2e"%abs(LU_factor[it,it])
            msg += " is to small. Stopping factorization."
            successBoolean = False
            # return LU_factor, raw_pivots, it, successBoolean
            # print (msg)

        # update L_factor
        LU_factor[it+1:,it] /= LU_factor[it,it];

        # udpate U_factor
        col_vector = LU_factor[it+1:,it]
        row_vector = LU_factor[it,it+1:]
        
        update = np.outer(col_vector,row_vector)
        LU_factor[it+1:,it+1:]-= update
    return LU_factor, raw_pivots, it, successBoolean

def split_lu_factorization_matrix(LU_factor,num_pivots=None):
    """
    Return the L and U factors of an inplace LU factorization

    Parameters
    ----------
    num_pivots : integer
        The number of pivots performed. This allows LU in place matrix
        to be split during evolution of LU algorithm
    """
    if num_pivots is None:
        num_pivots = np.min(LU_factor.shape)
    L_factor = np.tril(LU_factor)
    if L_factor.shape[1]<L_factor.shape[0]:
        # if matrix over-determined ensure L is a square matrix
        n0 = L_factor.shape[0]-L_factor.shape[1]
        L_factor=np.hstack([L_factor,np.zeros((L_factor.shape[0],n0))])
    if num_pivots<np.min(L_factor.shape):
        n1 = L_factor.shape[0]-num_pivots
        n2 = L_factor.shape[1]-num_pivots
        L_factor[num_pivots:,num_pivots:] = np.eye(n1,n2)
    np.fill_diagonal(L_factor,1.)
    U_factor = np.triu(LU_factor)
    U_factor[num_pivots:,num_pivots:] = LU_factor[num_pivots:,num_pivots:]
    return L_factor, U_factor