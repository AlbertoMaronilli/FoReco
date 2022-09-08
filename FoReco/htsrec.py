import numpy as np
import pandas as pd
from scipy import sparse as sp
from tools import hts_tools, match_arg, shrink_estim
from reco import recoM, recoS

class htsrec_obj_out:
    def __init__(self, 
                 recf, 
                 nn_check, 
                 rec_check,
                 W = None,
                 varf = None, 
                 M = None, 
                 G = None, 
                 S = None,
                 info = None):
        self.recf = recf
        self.nn_check = nn_check
        self.rec_check = rec_check
        if not W is None:
            self.W = W
        if not varf is None:
            self.varf = varf
        if not M is None:
            self.M = M
        if not G is None:
            self.G = G
        if not S is None:
            self.S = S  
        if not info is None:
            self.info = info 

def htsrec(basef: np.ndarray, 
           comb: str, 
           C: np.ndarray=np.empty(0), 
           res: np.ndarray=np.empty(0), 
           Ut: np.ndarray=np.empty(0), 
           nb: int = 0, 
           Type: str = "M", 
           sol: str = "direct", 
           keep: str = "obj", 
           nn: bool = False, 
           nn_type: str = "osqp", 
           linsolv: chr = "sps",
           bounds: np.ndarray = np.empty(0), 
           W: np.ndarray = np.empty(0),
           ncores: int = 1,
           **settings
          ):
    """
    Cross-sectional (contemporaneous) forecast reconciliation of a linearly constrained 
    (e.g.,hierarchical/grouped) multiple time series. 
    The reconciled forecasts are calculated either through a projection approach (Byron, 1978, 
    see also van Erven and Cugliari, 2015, and Wickramasuriya et al., 2019), or the equivalent 
    structural approach by Hyndman et al. (2011). Moreover, the classic bottom-up approach is 
    available.
    
    Parameters
    ----------
    basef : numpy.ndarray
        (h x n) matrix of base forecasts to be reconciled; is the forecast horizon and  
        is the total number of time series.
    comb : string 
        Type of the reconciliation. Except for Bottom-up, each option corresponds to a specific
        (n x n) covariance matrix:
            "bu" = bottom-up
            "ols" = identity
            "struc" = structural variances
            "wls" = series variances (uses res)
            "shr" = shrunk covariance matrix (uses res)
            "sam" = sample covariance matrix (uses res)
            "w" = use your personale matrix W in param W
    C : numpy.ndarray
        (n_a x n_b) cross-sectional (contemporaneous) matrix mapping the bottom level series 
        into the higher level ones.
    res : numpy.ndarray
        (N x n) in-sample residuals matrix needed when comb = {"wls", "shr", "sam"}. The columns 
        must be in the same order as basef.
    Ut : numpy.ndarray
        zero constraints cross-sectional (contemporaneous) kernel matrix (U'y = 0) spanning the 
        null space valid for the reconciled forecasts. It can be used instead of parameter C, but 
        nb (n = n_a + n_b) is needed if U' != [I - C]. If the hierarchy admits a structural 
        rapresentation, U' has dimension (n_a x n)
    nb : int 
        Number of bottom time series; if C is present, nb and Ut are not used.
    Type : string
        Approach used to compute the reconciled forecasts: "M" for the projection approach with
        matrix M (default), or "S" for the structural approach with summing matrix S.
    sol : string
        Solution technique for the reconciliation problem: either "direct" (default) for the 
        closed-form matrix solution, or "osqp" for the numerical solution (solving a linearly 
        constrained quadratic program using osqp.solve()).
    keep : string
        Return an object of the reconciled forecasts at all levels (if keep = "obj") or only
        the reconciled forecasts matrix (if keep = "recf").
    nn : boolean
        Logical value: True if non-negative reconciled forecasts are wished.
    linsolv: string
        Solver to be used for resolving the linear sistems.
            "sps": scipy.sparse.linalg.spsolve, linear system solver based on LU factorization, 
                may struggle when dealing with very large hierarchies
            "gmr": scipy.sparse.linalg.gmres, linear system solver based on generalised minimal 
                residual method, faster than spsolve, especially with large hierachies, but less accurate
    bounds : numpy.ndarray 
        (n x 2) matrix containing the cross-sectional bounds: the first column is the lower 
        bound, and the second column is the upper bound.
    W : numpy.ndarray
        This option permits to directly enter the covariance matrix. W must be a numpy.matrix (n x n).
        If comb is different from "w", W is not used
    ncores: integer
        Number of processes to be spawned for the computation of the reconciled forecasts using OSQP or 
        gmres, effective only when h>1. If ncores=1 (dafault) multiprocessing is not used.
    settings : Settings for osqp (object osqpSettings). The default options are: verbose = FALSE, 
        eps_abs = 1e-5, eps_rel = 1e-5, polish_refine_iter = 100 and polish = TRUE. For details, see
        the osqp documentation (Stellato et al., 2019).
        
    Returns
    -------
    If the parameter keep is equal to "recf", then the function returns only the (h x n) reconciled 
    forecasts matrix, otherwise (keep="obj") it returns an object whose attributes depend on what type of 
    representation (type) and solution technique (sol) have been used:
        recf : numpy.ndarray
            (h x n) reconciled forecasts matrix
        W : numpy.ndarray
            Covariance matrix used for forecast reconciliation
        nn_check : int
            number of negative values
        rec_check : bool
            Logical value: has the hierarchy been respected?
        varf : numpy.array
            (n x 1) reconciled forecasts variance vector for h = 1, diag(M*W) (only if type = "direct")
        M : numpy.matrix
            Projection matrix, M (projection approach) (only if sol = "direct")
        G : numpy.matrix
            Projection matrix, G (structural approach) (only if sol = "direct" and Type = "S")
        S : Cross-sectional summing matrix, S (only if sol = "direct" and Type = "S")
        info : numpy.ndarray
            matrix with information in columns for each forecast horizon h (rows): run time (run_time), 
            number of iteration (iter), norm of primal residual (pri_res), status of osqp's solution 
            (status) and polish's status (status_polish). It will also be returned with nn = TRUE if 
            OSQP will be used.
            
        
    """
    if not isinstance(basef, np.ndarray):
        raise TypeError("basef must be a numpy.ndarray")
    if not isinstance(C, (np.ndarray, sp.csc_matrix)):
        raise TypeError("C must be a numpy.ndarray or a scipy.sparse.csc_matrix")
    if not isinstance(Ut, (np.ndarray, sp.csc_matrix)):
        raise TypeError("Ut must be a numpy.ndarray or a scipy.sparse.csc_matrix")
    if not isinstance(comb, str):
        raise TypeError("comb must be a string")
    if not isinstance(Type, str):
        raise TypeError("Type must be a string")
    if not isinstance(sol, str):
        raise TypeError("sol must be a string")
    if not isinstance(keep, str):
        raise TypeError("keep must be a string")
    if not isinstance(nb, int):
        raise TypeError("nb must be an integer")
    if not isinstance(nn, bool):
        raise TypeError("nn must be a boolean")
    if not isinstance(nn_type, str):
        raise TypeError("nn_type must be a string")
    if not isinstance(res, np.ndarray):
        raise TypeError("res must be a numpy.ndarray")
    if not isinstance(bounds, np.ndarray):
        raise TypeError("bounds must be a numpy.ndarray")
    if not isinstance(W, np.ndarray):
        raise TypeError("W must be a numpy.ndarray")
    if ncores:
        if not isinstance(ncores, int):
            raise TypeError("ncores must be an integer")

    comb = match_arg(comb, ["bu", "ols", "struc", "w", "shr", "sam", "wls"])
    Type = match_arg(Type, ["M", "S"])
    keep = match_arg(keep, ["obj", "recf"])
    sol = match_arg(sol, ["direct", "osqp"])
    nn_type = match_arg(nn_type, ["osqp", "sntz"])
    linsolv = match_arg(linsolv, ["sps", "gmr", "cg"])
    if len(basef.shape) == 1:
        basef = basef.reshape((1,-1))
    n= basef.shape[1]
    if C.size == 0:
        if Ut.size == 0:
            raise ValueError("Please, give C or Ut")
        elif nb == 0:
            C, S, Ut, n, na, nb = hts_tools(Ut = Ut, h = 1, out="list")
        else:
            C, S, Ut, n, na, nb = hts_tools(Ut = Ut, nb = nb, h = 1, out="list")
    else:
        C, S, Ut, n, na, nb = hts_tools(C = C, h = 1, out="list")
    if basef.shape[1] != n:
        raise ValueError("Incorrect dimension of Ut or basef (they don't have same columns).")
    if comb in ["wls", "shr", "sam"]:
        if res.size == 0:
            raise ValueError("Don't forget residuals!")
        elif res.shape[1] != n:
            raise ValueError("The number of columns of res must be %d" %n)
        elif (comb == "sam" and res.shape[0] < n):
            raise ValueError("The number of rows of res is less than columns: \n it could lead to singularity problems if comb == 'sam'.")
     # per ora uso solo l'implementazione di default per il calcolo di wls sam shr
    if comb == "bu":
        if basef.shape[1] != nb:
            basef_bu = basef[:, na:n]
        else:
            basef_bu = basef
        if nn:
            basef_bu = np.multiply(basef_bu, (basef_bu>0))
        outf = sp.coo_matrix.dot(basef_bu, S.transpose()) #ritorna una matrice densa
        if keep == "recf":
            return outf
        else:
            nn_check = np.sum(outf<0)
            rec_check = np.all(Ut.dot(outf.transpose()) < 1e-6)
            out = htsrec_obj_out(outf, nn_check, rec_check)
            return out
    elif comb == "ols":
        W = sp.identity(n)
    elif comb == "struc":
        W = sp.spdiags(S.sum(1).transpose(),0,n,n)
    elif comb == "wls":
        W = sp.csr_matrix(np.dot(res.transpose(),res)/res.shape[0])
        W = sp.diags(W.diagonal())
    elif comb == "shr":
        W = shrink_estim(res)[0]
    elif comb == "sam":
        W = sp.csr_matrix(np.dot(res.transpose(),res)/res.shape[0])
    elif comb == "w":
        print(len(W.shape))
        if W.size == 0:
            raise ValueError("Please, put in option W your covariance matrix")
        if not isinstance(W, np.ndarray) or len(W.shape) != 2:
            raise ValueError("W must be a matrix " + str(n) + "x" + str(n))
        if W.shape[1] != n or W.shape[0] != n:
            raise ValueError("W must be a matrix " + str(n) + "x" + str(n))
        if np.count_nonzero(W - np.diag(np.diagonal(W)))==0:
            W = sp.dia_matrix(W)
        else:
            W = sp.csr_matrix(W)
    b_pos = np.concatenate([np.zeros(na), np.ones(nb)])
    #rec_sol = recoM(basef = basef, W = W, Ht = Ut, sol = sol, nn = nn, 
     #               keep = keep, S = S, settings = settings, b_pos = b_pos, 
      #              bounds = bounds, nn_type = nn_type)
    if Type == "S":
        if keep == "recf":
            out = recoS(basef = basef, W = W, sol=sol, S=S, nn=nn, nn_type=nn_type,
                            b_pos=b_pos, keep=keep, bounds=bounds, settings=settings, ncores=ncores)
        else:
            if sol == "osqp": 
                out, info = recoS(basef = basef, W = W, sol=sol, S=S, nn=nn, nn_type=nn_type,
                            b_pos=b_pos, keep=keep, bounds=bounds, settings=settings, ncores=ncores)
                nn_check = np.sum(out<0)
                rec_check = np.all(Ut.dot(out.transpose()) < 1e-6)
                out = htsrec_obj_out(out, nn_check=nn_check, rec_check=rec_check, W=W, info = info)
            else:
                rec_sol, varf, M, G, S, info = recoS(basef = basef, W = W, sol=sol, S=S, nn=nn,  nn_type=nn_type,
                                           b_pos=b_pos, keep=keep, bounds=bounds, settings=settings, ncores=ncores)
                nn_check = np.sum(rec_sol<0)
                rec_check = np.all(Ut.dot(rec_sol.transpose()) < 1e-6)
                out = htsrec_obj_out(rec_sol, nn_check=nn_check, rec_check=rec_check, W=W, varf=varf, M=M, G=G, S=S, info=info)
    else:
        if keep == "recf":
            out = recoM(basef = basef, W = W, Ht = Ut, sol=sol, S=S, nn=nn, nn_type=nn_type,
                            b_pos=b_pos, keep=keep, bounds=bounds, linsolv=linsolv, settings=settings, ncores=ncores)
        else:
            if sol == "osqp": 
                out, info = recoM(basef = basef, W = W, Ht = Ut, sol=sol, S=S, nn=nn,  nn_type=nn_type,
                            b_pos=b_pos, keep=keep, linsolv=linsolv, bounds=bounds, settings=settings, ncores=ncores)
                nn_check = np.sum(out<0)
                rec_check = np.all(Ut.dot(out.transpose()) < 1e-6)
                out = htsrec_obj_out(out, nn_check=nn_check, rec_check=rec_check, W=W, info=info)
            else:
                rec_sol, varf, M, info = recoM(basef = basef, W = W, Ht = Ut, sol=sol, S=S, nn=nn, nn_type=nn_type,
                                     b_pos=b_pos, keep=keep, linsolv=linsolv, bounds=bounds, settings=settings, ncores=ncores)
                nn_check = np.sum(rec_sol<0)
                rec_check = np.all(Ut.dot(rec_sol.transpose()) < 1e-6)
                out = htsrec_obj_out(rec_sol, nn_check=nn_check, rec_check=rec_check, W=W, varf=varf, M=M, info=info)
    return out

def pddf_htsrec(basef: pd.DataFrame, 
                comb: str, 
                C: np.ndarray=np.empty(0), 
                res: np.ndarray=np.empty(0), 
                Ut: np.ndarray=np.empty(0), 
                nb: int = 0, 
                Type: str = "M", 
                sol: str = "direct", 
                keep: str = "obj", 
                nn: bool = False,  
                nn_type: str = "osqp",
                linsolv: chr = "sps",
                bounds: np.ndarray = np.empty(0), 
                W: np.ndarray = np.empty(0),
                ncores=1,
                **settings
                ):
    """
    Same as htsrec but the input variable basef needs to be a pandas.DataFrame. 
    Also the output variables recf and info will be a pandas.DataFrame
    """
    if not isinstance(basef, pd.DataFrame):
        raise TypeError("basef must be a pandas.dataframe")
    if not isinstance(keep, str):
        raise TypeError("keep must be a string")
    colnames = basef.columns.values
    basef = basef.to_numpy()
    keep = match_arg(keep, ["obj", "recf"])
    out = htsrec(basef, comb, C, res, Ut, nb, Type, sol, keep, nn, nn_type, linsolv, bounds, W, ncores=ncores, **settings)
    if keep == "recf":
        out = pd.DataFrame(data=out, columns=colnames)
    else:
        out.recf = pd.DataFrame(data=out.recf, columns=colnames)
        if hasattr(out, 'info'):
            out.info = pd.DataFrame(data=out.info, columns=["obj_val", "run_time", "iter", "pri_res",
                                                        "status", "status_polish"])
    return out


