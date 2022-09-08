import numpy as np
import pandas as pd
from scipy import sparse as sp
from tools import thf_tools, match_arg, shrink_estim, Dmat, my_crossprod, acf1
from reco import recoM, recoS
from scipy.linalg import toeplitz

class thfrec_obj_out:
    def __init__(self, 
                 recf, 
                 nn_check, 
                 rec_check,
                 Omega = None,
                 varf = None, 
                 M = None, 
                 G = None, 
                 S = None,
                 info = None):
        self.recf = recf
        self.nn_check = nn_check
        self.rec_check = rec_check
        if not Omega is None:
            self.Omega = Omega
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

def thfrec(basef: np.ndarray, 
          m: int, 
          comb: str, 
          res: np.ndarray=np.empty(0),
          Type: str = "M", 
          sol: str = "direct", 
          keep: str = "obj", 
          nn: bool = False, 
          nn_type: str = "osqp",
          linsolv: chr = "sps",
          bounds: np.ndarray=np.empty(0), 
          Omega: np.ndarray = np.empty(0),
          ncores=1, 
          **settings):
    """
    Forecast reconciliation of one time series through temporal hierarchies (Athanasopoulos et al., 2017). 
    The reconciled forecasts are calculated either through a projection approach (Byron, 1978), or the 
    equivalent structural approach by Hyndman et al. (2011). Moreover, the classic bottom-up approach is 
    available.

    Parameters
    ----------
    basef : numpy.ndarray
        (h(k*+m) x 1) vector of base forecasts to be reconciled, containing the forecasts at all the needed 
        temporal frequencies ordered as [lowest_freq' ... highest_freq']'.
    m : integer
        Highest available sampling frequency per seasonal cycle (max. order of temporal aggregation, m), 
        or a subset of p factors of m.
    comb : string
        Type of the reconciliation. Except for bottom up, all other options correspond to a different 
        ((k*+m) x (k*+m)) covariance matrix, k* is the sum of (p-1) factors of m (excluding m):
            "bu" (bottom-up);
            "ols" (Identity);
            "struc" (Structural variances);
            "wlsv" (Series variances);
            "wlsh" (Hierarchy variances);
            "acov" (Auto-covariance matrix);
            "strar1" (Structural Markov);
            "sar1" (Series Markov);
            "har1" (Hierarchy Markov);
            "shr" (Shrunk cross-variance matrix);
            "sam" (Sample cross-variance matrix);
            "omega" use your personal matrix Omega in param Omega.
    res : numpy.ndarray
        vector containing the in-sample residuals at all the temporal frequencies ordered as basef, i.e. 
        [lowest_freq' ... highest_freq']', needed to estimate the covariance matrix when comb = {"wlsv", 
        "wlsh", "acov", "strar1", "sar1", "har1", "shr", "sam"}.
    Type : string
        Approach used to compute the reconciled forecasts: "M" for the projection approach with matrix M 
        (default), or "S" for the structural approach with temporal summing matrix R.
    sol : string
        Solution technique for the reconciliation problem: either "direct" (default) for the closed-form 
        matrix solution, or "osqp" for the numerical solution (solving a linearly constrained quadratic 
        program using osqp.solve()).
    keep : string
        Return a list object of the reconciled forecasts at all levels (if keep = "list") or only the 
        reconciled forecasts matrix (if keep = "recf").
    nn : boolean
        Logical value: TRUE if non-negative reconciled forecasts are wished.
    linsolv: string
        Solver to be used for resolving the linear sistems.
            "sps": scipy.sparse.linalg.spsolve, linear system solver based on LU factorization, may struggle
                when dealing with very large hierarchies (default);
            "gmr": scipy.sparse.linalg.gmres, linear system solver based on generalised minimal residual
                method, faster than spsolve, especially with large hierachies, but less accurate;
    bounds : numpy.ndarray
        ((k*+m) x 2) matrix with temporal bounds: the first column is the lower bound, and the second column 
        is the upper bound.
    Omega : numpy.ndarray
        This option permits to directly enter the covariance matrix:
            1. Omega must be a p.d. ((k*+m) x (k*+m)) matrix or a list of matrix (one for each forecast horizon);
            2. if comb is different from "omega", Omega is not used.
    ncores: integer
        Number of processes to be spawned for the computation of the reconciled forecasts using OSQP or gmres, 
        effective only when h>1. If ncores=1 (dafault) multiprocessing is not used.
    settings : Settings for osqp (object osqpSettings). The default options are: verbose = FALSE, 
        eps_abs = 1e-5, eps_rel = 1e-5, polish_refine_iter = 100 and polish = TRUE. For details, see
        the osqp documentation (Stellato et al., 2019).
    
    Returns
    -------
    If the parameter keep is equal to "recf", then the function returns a tuple containing the (h(k*+m)x1) 
    reconciled forecasts vector and a list containing the name of each variable, otherwise (keep="all") the 
    same tuple is returned but recf is an object whose content depends on what type of representation (type) 
    and solution technique (sol) have been used:
        recf : numpy.ndarray
            (h(k*+m)x1) reconciled forecasts matrix
        Omega : numpy.ndarray
            Covariance matrix used for forecast reconciliation
        nn_check : int
            number of negative values
        rec_check : bool
            Logical value: has the hierarchy been respected?
        varf : numpy.array
            ((k*+m)x1) reconciled forecasts variance vector for h = 1, diag(M*W) (only if type = "direct")
        M : numpy.ndarray
            Projection matrix, M (projection approach) (only if sol = "direct")
        G : numpy.ndarray
            Projection matrix, G (structural approach) (only if sol = "direct" and Type = "S")
        S : Temporal summing matrix, R (only if sol = "direct" and Type = "S")
        info : numpy.ndarray
            matrix with information in columns for each forecast horizon h (rows): run time (run_time), 
            number of iteration (iter), norm of primal residual (pri_res), status of osqp's solution 
            (status) and polish's status (status_polish). It will also be returned with nn = TRUE if 
            OSQP will be used.
    

    """
    if not isinstance(basef, np.ndarray):
        raise TypeError("basef must be a numpy.ndarray")
    if not isinstance(comb, str):
        raise TypeError("comb must be a string")
    if not isinstance(Type, str):
        raise TypeError("Type must be a string")
    if not isinstance(sol, str):
        raise TypeError("sol must be a string")
    if not isinstance(keep, str):
        raise TypeError("keep must be a string")
    if not isinstance(m, int):
        raise TypeError("m must be an integer")
    if not isinstance(nn, bool):
        raise TypeError("nn must be a boolean")
    if not isinstance(nn_type, str):
        raise TypeError("nn_type must be a string")
    if not isinstance(res, np.ndarray):
        raise TypeError("res must be a numpy.ndarray")
    if not isinstance(bounds, np.ndarray):
        raise TypeError("bounds must be a numpy.ndarray")
    if not isinstance(Omega, np.ndarray):
        raise TypeError("Omega must be a numpy.ndarray")
    if ncores:
        if not isinstance(ncores, int):
            raise TypeError("ncores must be an integer")
    K, R, Zt, kset, m, p, ks, kt = thf_tools(m, out="list")
    comb = match_arg(comb, ["bu", "ols", "struc", "wlsv", "wlsh", "acov",
                            "strar1", "sar1", "har1", "shr", "sam", "omega"])
    Type = match_arg(Type, ["M", "S"])
    keep = match_arg(keep, ["obj", "recf"])
    nn_type = match_arg(nn_type, ["osqp", "sntz"])
    linsolv = match_arg(linsolv, ["sps", "gmr", "cg"])
    if not basef.shape[1] == 1:
        raise ValueError("basef must be a vector")
    if comb == "bu" and len(basef) % m == 0:
        h = int(len(basef)/m)
        Dh = Dmat(h=h, m=kset, n=1)
        BASEF = np.reshape(basef, (h,m))
    elif not len(basef) % kt == 0:
        raise ValueError("basef vector has a number of elements not in line with the frequency of the series")
    else:
        h = int(len(basef)/kt)
        Dh = Dmat(h=h, m=kset, n=1)
        BASEF = np.reshape(Dh.dot(basef), (h, -1))  
    if comb in ["wlsv", "wlsh", "acov", "strar1", "sar1", "har1", "sGlasso", "hGlasso", "shr", "sam"]:
        if res.size == 0:
            raise ValueError("Don't forget residuals!")
        if not res.shape[1] == 1:
            raise ValueError("res must be a vector")
        if not len(res) % kt == 0:
            raise ValueError("res vector has a number of row not in line with frequency of the series")
        N = int(len(res)/kt)
        DN = Dmat(h=N, m = kset, n=1)
        RES = np.reshape(DN.dot(res), (N,-1))
        if comb == "sam" and N < kt:
            raise ValueError("N < (k* + m): it could lead to singularity problems if comb == sam")
        if comb == "acov" and N < m:
            raise ValueError("N < m: it could lead to singularity problems if comb == acov")
    if comb == "bu":
        if not BASEF.shape[1] == m:
            BASEF = BASEF[:, ks:kt]
        if nn:
            BASEF = np.multiply(BASEF, (BASEF>0))
        OUTF = BASEF.dot(R.transpose().todense())
        outf = Dh.transpose().dot(OUTF.flatten().transpose()).A1
        if keep == "obj":
            M = R.dot(sp.hstack([sp.csr_matrix((m,ks)), sp.eye(m).todense()]).todense())
            nn_check = np.sum(outf<0)
            rec_check = np.all(OUTF.dot(Zt.transpose().todense()) < 1e-6)
            outf = thfrec_obj_out(outf, nn_check=nn_check, rec_check=rec_check, M = M, S = R)
        aggror=np.concatenate([np.tile(x, int(m/x*h)) for x in kset], dtype="str")
        valnum=np.concatenate([np.arange(1, int((m/x)*h+1)) for x in kset], dtype="str")
        varnames=["".join(("k", aggror[i], "h", valnum[i])) for i in range(kt*h)]
        return outf, varnames
    if comb == "ols":
        Omega = sp.identity(kt)
    if comb == "struc":
        Omega = sp.spdiags(R.sum(1).transpose(),0,kt,kt)
    if comb == "wlsv":
        var_freq = list(map(my_crossprod, (res[np.concatenate([np.tile(x, int(m/x*N)) for x in kset]) == y,:] for y in kset)))
        Omega = sp.diags(np.concatenate([np.tile(list(var_freq)[i], int(m/kset[i])) for i in range(len(kset))], axis=None))
    if comb == "wlsh":
        diagO = my_crossprod(RES).diagonal()
        Omega = sp.diags(diagO)
    if comb == "acov":
        var_freq = list(map(my_crossprod, (RES[:,np.concatenate([np.tile(x, int(m/x)) for x in kset]) == y] for y in kset)))
        Omega = sp.block_diag(var_freq)
    if comb == "strar1":
        rho = list(map(acf1, (res[np.concatenate([np.tile(x, int(m/x*N)) for x in kset]) == y,:] for y in kset)))
        expo = list(map(toeplitz, [np.arange(m/x) for x in kset]))
        Gam = sp.block_diag(list(map(np.power, rho, expo)))
        ostr2 = sp.diags(np.sqrt(R.sum(axis=1)).A1)
        Omega = ostr2.dot(Gam).dot(ostr2)
    if comb == "sar1":
        rho = list(map(acf1, (res[np.concatenate([np.tile(x, int(m/x*N)) for x in kset]) == y,:] for y in kset)))
        expo = list(map(toeplitz, [np.arange(m/x) for x in kset]))
        Gam = sp.block_diag(list(map(np.power, rho, expo)))
        var_freq = list(map(my_crossprod, (res[np.concatenate([np.tile(x, int(m/x*N)) for x in kset]) == y,:] for y in kset)))
        Os2 = sp.diags(np.sqrt(np.concatenate([np.tile(list(var_freq)[i], int(m/kset[i])) for i in range(len(kset))], axis=None)))
        Omega = Os2.dot(Gam).dot(Os2)
    if comb == "har1":
        rho = list(map(acf1, (res[np.concatenate([np.tile(x, int(m/x*N)) for x in kset]) == y,:] for y in kset)))
        expo = list(map(toeplitz, [np.arange(m/x) for x in kset]))
        Gam = sp.block_diag(list(map(np.power, rho, expo)))
        diagO = my_crossprod(RES).diagonal()
        Oh2 = sp.diags(np.sqrt(diagO))
        Omega = Oh2.dot(Gam).dot(Oh2)
    if comb == "shr":
        Omega = shrink_estim(RES)[0]
    if comb == "sam":
        Omega = my_crossprod(RES)
    if comb == "omega":
        if omega.size == 0:
            raise ValueError("Please, put in option Omega your covariance matrix")
    b_pos = np.concatenate([np.zeros(kt-m), np.ones(m)])
    #obs_names = 
    if Type == "S":
        if keep == "recf":
            out = recoS(basef=BASEF, W=Omega, S=R, sol=sol, nn=nn, keep=keep, settings=settings,
                       b_pos=b_pos, bounds=bounds, ncores=ncores, nn_type=nn_type)
            out = Dh.transpose().dot(out.flatten().transpose())
            if not len(out.shape) == 1:
                out=out.A1
        else:
            if sol == "osqp":
                rec_sol, info = recoS(basef=BASEF, W=Omega, S=R, sol=sol, nn=nn, keep=keep, settings=settings,
                                  b_pos=b_pos, bounds=bounds, ncores=ncores, nn_type=nn_type)
                nn_check = np.sum(rec_sol<0)
                rec_check = np.all(rec_sol.dot(Zt.transpose().todense()) < 1e-6)
                rec_sol = Dh.transpose().dot(rec_sol.flatten().transpose())
                if not len(rec_sol.shape) == 1:
                    rec_sol=rec_sol.A1
                out = thfrec_obj_out(rec_sol, nn_check=nn_check, rec_check=rec_check, Omega=Omega, info = info)
            else: 
                rec_sol, varf, M, G, S, info = recoS(basef=BASEF, W=Omega, S=R, sol=sol, nn=nn, keep=keep, , nn_type=nn_type
                                                     settings=settings, b_pos=b_pos, bounds=bounds, ncores=ncores)
                nn_check = np.sum(rec_sol<0)
                rec_check = np.all(rec_sol.dot(Zt.transpose().todense()) < 1e-6)
                rec_sol = Dh.transpose().dot(rec_sol.flatten().transpose())
                if not len(rec_sol.shape) == 1:
                    rec_sol=rec_sol.A1
                out = thfrec_obj_out(rec_sol, nn_check=nn_check, rec_check=rec_check, Omega=Omega, varf=varf,
                                     M=M, G=G, S=S, info=info)
    else:
        if keep == "recf":
            out = recoM(basef=BASEF, W=Omega, S=R, Ht=Zt, sol=sol, nn=nn, linsolv=linsolv, keep=keep, settings=settings,
                       b_pos=b_pos, bounds=bounds, ncores=ncores, nn_type=nn_type)
            out = Dh.transpose().dot(out.flatten().transpose())
            if not len(out.shape) == 1:
                out=out.A1
        else:
            if sol == "osqp":
                rec_sol, info = recoM(basef=BASEF, W=Omega, S=R, Ht=Zt, sol=sol, nn=nn, linsolv=linsolv, keep=keep, settings=settings,
                                  b_pos=b_pos, bounds=bounds, ncores=ncores, nn_type=nn_type)
                nn_check = np.sum(rec_sol<0)
                rec_check = np.all(rec_sol.dot(Zt.transpose().todense()) < 1e-6)
                rec_sol = Dh.transpose().dot(rec_sol.flatten().transpose())
                if not len(rec_sol.shape) == 1:
                    rec_sol=rec_sol.A1
                out = thfrec_obj_out(rec_sol, nn_check=nn_check, rec_check=rec_check, Omega=Omega, info = info)
            else: 
                rec_sol, varf, M, info = recoM(basef=BASEF, W=Omega, S=R, Ht=Zt, sol=sol, nn=nn, linsolv=linsolv, keep=keep, 
                                                     settings=settings, b_pos=b_pos, bounds=bounds, ncores=ncores, nn_type=nn_type)
                nn_check = np.sum(rec_sol<0)
                rec_check = np.all(rec_sol.dot(Zt.transpose().todense()) < 1e-6)
                rec_sol = Dh.transpose().dot(rec_sol.flatten().transpose())
                if not len(rec_sol.shape) == 1:
                    rec_sol=rec_sol.A1
                out = thfrec_obj_out(rec_sol, nn_check=nn_check, rec_check=rec_check, Omega=Omega, varf=varf,
                                     M=M, info=info)
    aggror=np.concatenate([np.tile(x, int(m/x*h)) for x in kset], dtype="str")
    valnum=np.concatenate([np.arange(1, int((m/x)*h+1)) for x in kset], dtype="str")
    varnames=["".join(("k", aggror[i], "h", valnum[i])) for i in range(kt*h)]
    return out, varnames
    
        

def pddf_thfrec(basef: pd.DataFrame, 
                m: int, 
                comb: str, 
                res: np.ndarray=np.empty(0),
                Type: str = "M", 
                sol: str = "direct", 
                keep: str = "obj", 
                nn: bool = False, 
                nn_type: str = "osqp",
                linsolv: chr = "sps",
                bounds: np.ndarray=np.empty(0), 
                Omega: np.ndarray = np.empty(0),
                ncores=None, 
                **settings):
    """
    Same as htsrec but the input variable basef needs to be a pandas.DataFrame. 
    Also the output variables recf and info will be a pandas.DataFrame
    """
    if not isinstance(basef, pd.DataFrame):
        raise TypeError("basef must be a pandas.dataframe")
    if not isinstance(keep, str):
        raise TypeError("keep must be a string")
    basef = basef.to_numpy()
    keep = match_arg(keep, ["obj", "recf"])
    out, varnames = thfrec(basef, m, comb, res, Type, sol, keep, nn, nn_type, linsolv, bounds, Omega, ncores, **settings)
    if keep=="recf":
        out = pd.DataFrame(data=np.reshape(out, (1,-1)), columns=varnames)
    else:
        out.recf = pd.DataFrame(data=np.reshape(out.recf, (1,-1)), columns=varnames)
        if hasattr(out, 'info'):
            out.info = pd.DataFrame(data=out.info, columns=["obj_val", "run_time", "iter", "pri_res",
                                                            "status", "status_polish"])
    return out
