import numpy as np
from scipy import sparse as sp
from functools import reduce

class hts:
    def __init__(self, C, S, Ut, n, na, nb):
        self.C = C
        self.S = S
        self.Ut = Ut
        self.n = n
        self.na = na
        self.nb = nb

class thf:
    def __init__(self, K, R, Zt, kset, m, p, ks, kt):
        self.K = K
        self.R = R
        self.Zt = Zt
        self.kset = kset
        self.m = m
        self.p = p
        self.ks = ks
        self.kt = kt


def hts_tools(C: np.ndarray=np.empty(0), 
              h: int=1, 
              Ut: np.ndarray=np.empty(0), 
              nb: int=0,
              out: str = "obj"):
    """
    Some useful tools for the cross-sectional forecast reconciliation of a linearly constrained (e.g., 
    hierarchical/grouped) multiple time series.

    Parameters
    ----------
    C : numpy.ndarray
        (n_a x n_b) cross-sectional (contemporaneous) matrix mapping the bottom level series into the higher
        level ones.
    h : int
        Forecast horizon (default is 1).
    Ut : numpy.ndarray
        Zero constraints cross-sectional (contemporaneous) kernel matrix (U'y = 0) spanning the null space 
        valid for the reconciled forecasts. It can be used instead of parameter C, but nb is needed if U' != 
        [I - C]. If the hierarchy admits a structural rapresentation, U' has dimension (n_a x n).
    nb : int
        Number of bottom time series; if C is present, nb and Ut are not used.
    out : str
        out ="obj" will return a single object (default), out = "list" will return a list of variables

    Returns
    -------
    The following variables will be returned as attributes of a hts object or as a list depending on the 
    parameter out:
    C : sparse.csc_matrix
        (n x n_b) cross-sectional (contemporaneous) aggregation matrix.
    S : sparse.csc_matrix
        (n x n_b) cross-sectional (contemporaneous) summing matrix
    Ut : sparse.csc_matrix
        (n_a x n) Zero constraints cross-sectional (contemporaneous) kernel matrix. If the hierarchy admits a 
        structural representation U' = [I - C].
    n : int
        number of variables n_a + n_b
    na : int
        Number of upper levels variables
    nb : int
        Number of bottom level variables
    """
    out = match_arg(out, ["obj", "list"])
    if C.size == 0:
        if Ut.size == 0:
            raise ValueError("Please, give C or Ut")
        else:
            if not sp.isspmatrix_csc(Ut): #trasforma Ut in sparsa se non lo è
                Ut = sp.csc_matrix(Ut)
            if not (sp.identity(Ut.shape[0]) != Ut[:,0:Ut.shape[0]]).todense().any(): #condizione accettata
            # if (sp.identity(Ut.shape[0]) == Ut[:,0:Ut.shape[0]]).todense().all():   # meno efficiente (a detta di scipy)
                C = -Ut[:,Ut.shape[0]:]
                if (abs(C).sum(axis=1)==0).any(): # rimuove righe nulle di C se ce ne sono
                    C = C[np.where(abs(C).sum(axis=1)!=0)[0],:]
                    print("Removed a zeros row in C matrix")
                    Ut = sp.hstack([sp.identity(C.shape[0]), -C])
                n = Ut.shape[1]
                na = Ut.shape[0]
                nb = n-na
                S = sp.vstack([C, sp.identity(nb)], format="csc")
            elif nb==0:
                raise ValueError("Ut is not in form [I -C], give also nb")   
            else:
                n = Ut.shape[1]
                na = n-nb
                C = np.empty(0)
                S = np.empty(0)
    else:
        if not sp.isspmatrix_csc(C):
            C = sp.csc_matrix(C)
        if (abs(C).sum(axis=1)==0).any():
            C = C[np.where(abs(C).sum(axis=1)!=0)[0],:]
            print("Removed a zeros row in C matrix")
        nb = C.shape[1]
        na = C.shape[0]
        n = nb+na
        S = sp.vstack([C, sp.identity(nb)], format="csc")
        Ut = sp.hstack([sp.identity(C.shape[0]), -C])
    if n <= nb:
        raise ValueError("n <= nb, total number of TS is less (or equal) than the number of bottom TS.")
    if C.size !=0:
        if (abs(C).sum(axis=1)==1).any():
            print("There is only one non-zero value in " + str((abs(C).sum(axis=1)==1).sum()) + " row(s) of C.\n" +
                  "Remember that Foreco can also work with unbalanced hierarchies (recommended).")
        if h > 1:
            C = sp.kron(C, sp.identity(h), format="csc")
            S = sp.kron(S, sp.identity(h), format="csc")
    Ut = sp.kron(Ut, sp.identity(h), format="csc")
    if out == "obj":
        return hts(C, S, Ut, n, na, nb)
    else:
        return C, S, Ut, n, na, nb

def thf_tools(m: list, 
             h: int=1,
             out: str = "obj"):
    """
    Some useful tools for forecast reconciliation through temporal hierarchies.

    Parameters
    ----------
    m : list
        Highest available sampling frequency per seasonal cycle (max. order of temporal aggregation, m), or a 
        subset of the p factors of m.
    h : integer
        Forecast horizon for the lowest frequency (most temporally aggregated) time series (default is 1).
    out : string
        out ="obj" will return a single object (default), out = "list" will return a list of variables

    Returns
    -------
    The following variables will be returned as attributes of a hts object or as a list depending on the 
    parameter out:
    K : sparse.coo_matrix
        Temporal aggregation matrix
    R : sparse.coo_matrix
        Temporal summing matrix
    Zt : sparse.coo_matrix
        Zero constraints temporal kernel matrix
    kset : list
        List of factors (p) of m in descending order (from m to 1).
    m : integer
        Highest available sampling frequency per seasonal cycle (max. order of temporal aggregation).
    p : integer
        Number of elements of kset.
    ks : integer
        Sum of p-1 factors of m (out of m itself), k*.
    kt : integer
        Sum of all factors of m (kt = ks + m).
    """
    out = match_arg(out, ["obj", "list"])
    if isinstance(m, int):
        m = [m]
    if len(m) == 0:
        raise ValueError("List of sampling frequency must not be empty")
    if len(m)>1:
        kset = sorted(m, reverse=True)
        m = kset[0]
        if m < 2: 
            raise ValueError("m must be > 1")
        if kset[len(kset)-1] != 1:
            kset.append(1)
        if not all(x in list(divisors(m)) for x in kset):
            raise ValueError("%s is not a subset of %s" %(str(kset),str(sorted(list(divisors(m)), reverse=True))))
    else:
        m = m[0]
        if m < 2: 
            raise ValueError("m must be > 1")
        kset = sorted(divisors(m), reverse = True)
    p = len(kset)
    ks = int(sum([m/x for x in kset[:p-1]]))
    kt = int(sum([m/x for x in kset]))
    K = sp.vstack(list(map(sp.kron, 
                           map(sp.eye, (m/x*h for x in kset[0:p-1])), 
                           map(lambda x: np.repeat(1,x), kset[0:p-1]))))
    Zt = sp.hstack([sp.eye(h*ks), -K])
    R = sp.vstack([K, sp.eye(m*h)])
    if out == "obj":
        return thf(K, R, Zt, kset, m, p, ks, kt)
    else:
        return K, R, Zt, kset, m, p, ks, kt

def cov2cor(cov):
    """
    Scales a covariance matrix into the corresponding correlation matrix

    Parameters
    cov : numpy.ndarray
        a covariance matrix

    Returns
    -------
    cor : numpy.ndarray
        a correlation matrix
    """
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    cor = cov / outer_v
    cor[cov == 0] = 0
    return cor

def shrink_estim(x: np.ndarray):
    """
    Shrinkage of the covariance matrix according to Schäfer and Strimmer (2005).

    Parameters
    ----------
    x : numpy.ndarray
        residual matrix

    Returns
    -------
    A list with two objects: the first is the shrunk covariance matrix and the second ($lambda) is 
    the shrinkage intensity coefficient.
    """
    p1 = x.shape[1]
    n2 = x.shape[0]
    covm = x.transpose().dot(x)/n2
    tar = np.diag(np.diag(covm))
    corm = cov2cor(covm)
    xs = x/np.sqrt(np.diag(covm))
    xs2 = np.power(xs,2)
    v = (1 / (n2 * (n2 - 1)))*(xs2.transpose().dot(xs2) - 1 / n2 * np.power(xs.transpose().dot(xs),2))
    np.fill_diagonal(v, 0)
    corapn = cov2cor(tar)
    d = np.power(corm-corapn,2)
    lam = v.sum()/d.sum()
    lam = max(min(lam,1),0)
    shrink_cov = lam*tar + (1-lam)*covm
    return sp.csc_matrix(shrink_cov), lam

def match_arg(x: str, lst):
    """
    matches x against a list of candidate values.

    Parameters
    ----------
    x : str
        a string variable
    lst : list
        a list of strings if candidate values

    Returns
    -------
    The unabbreviated version of the exact or unique partial match if there is one; otherwise, an error a 
    ValueError is raised.
    """
    d = [el for el in lst if x in el]
    if len(d)==0:
        raise ValueError("Invalid argument: %s" %x)
    return d[0]

def divisors(n):  
    """
    Given an integer, return a set of all its divisors
    """
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def Dmat(h: int,
         m: list, 
         n: int):
    """
    This function returns the (hn(k*+m) x hn(k*+m)) permutation matrix to rearrange the values in a desired 
    shape.

    Parameters
    ----------
    h : integer
        forecast horizon for the lowest frequency (most temporally agregated) time series
    m : list
        Highest available sampling frequency per seasonal cycle (max. order of temporal aggregation, m), or 
        a subset of p factors m
    n : integer
        number of the cross-sectional variables n = n_a + n_b

    Returns
    -------
    A sparse matrix
    """
    if len(m)==1:
        kset=1
    else:
        kset = sorted(m, reverse=True)
        m = kset[0]
    I = sp.csr_matrix(sp.eye(h*sum([m/x for x in kset])*n))
    fir = np.tile(range(1,h+1), len(kset))
    sec = np.repeat([int(m/x) for x in kset], h)
    i= np.tile(np.concatenate([np.repeat(fir[i],sec[i]) for i in range(len(fir))]), n)
    I = I[np.argsort(i, kind="stable"),:]
    return(I)

def my_crossprod(res):
    return np.dot(res.transpose(), res)/res.shape[0]

def acf1(res):
    n = len(res)
    data = np.asarray(res)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)
    acf1 = ((data[:n - 1] - mean) * (data[1:] - mean)).sum() / float(n) / c0
    return acf1