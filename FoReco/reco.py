import numpy as np
import copy
import osqp
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, gmres, splu, cg, spilu, LinearOperator
from numpy.linalg import solve
from scipy.linalg import cho_factor, cho_solve
from tools import match_arg
from functools import partial
from multiprocessing.pool import Pool
import time

def recoM(basef: np.ndarray, 
         W: sp.csr_matrix, 
         Ht: sp.bsr_matrix,
         S: sp.coo_matrix,
         sol: str = "direct", 
         nn: bool = False, 
         linsolv: str = "sps",
         nn_type: str = "osqp",
         b_pos: np.ndarray=np.empty(0), 
         keep: str = "recf",
         bounds: np.ndarray=np.empty(0),
         ncores = None,
         settings={}):
    n=basef.shape[1]
    info=None
    if not bounds.size==0:
        if not isinstance(bounds, np.ndarray) or bounds.shape[1]!=2 or bounds.shape[0] != n:
            raise ValueError("bounds must be a matrix (%dx2)" %n)
        else: 
            sol="osqp"
    if sol == "direct":
        baseT=basef.transpose()
        HtT=Ht.transpose()
        
        try: # gmres/spsolve
            lm_dx = Ht.dot(baseT)
            lm_sx = sp.csc_matrix(Ht.dot(W).dot(HtT))
            # start = time.time()
            if linsolv == "gmr":
                # ILUfact = spilu(lm_sx)
                if lm_dx.shape[1] == 1:
                    # M = LinearOperator(
                    #     shape = lm_sx.shape,
                    #     matvec = lambda b: ILUfact.solve(lm_dx)
                    # )
                    s_lm, andata = gmres(lm_sx, lm_dx, tol=1e-4)
                    if not andata == 0:
                        print("Warning! linear system may be not solved\n")
                elif ncores == 1: ### loop gmres
                    s_lm = np.empty(lm_dx.shape)
                    for i in range(lm_dx.shape[1]): 
                        s_lm[:,i], andata = gmres(lm_sx, lm_dx[:,i], tol=1e-4)
                        if not andata == 0:
                            print("Warning! a linear system may be not solved\n")
                else: ### gmres + MP
                    Llm_dx=list(np.transpose(lm_dx))
                    Llm_dx = [vec.reshape((-1,1)) for vec in Llm_dx]
                    pool = Pool(ncores)
                    Ls_lm=pool.map(partial(gmres_core, lm_sx=lm_sx), Llm_dx)
                    pool.close()
                    pool.join()
                    s_lm=np.hstack(list(Ls_lm))
            elif linsolv=="cg":
                if lm_dx.shape[1] == 1:
                    s_lm, andata = cg(lm_sx, lm_dx, tol=1e-4)
                    if not andata == 0:
                        print("Warning! linear system may be not solved\n")
                elif ncores == 1: ### loop cg
                    s_lm = np.empty(lm_dx.shape)
                    for i in range(lm_dx.shape[1]): 
                        s_lm[:,i], andata = cg(lm_sx, lm_dx[:,i], tol=1e-4)
                        if not andata == 0:
                            print("Warning! a linear system may be not solved\n")
                else: ### cg + MP
                    Llm_dx=list(np.transpose(lm_dx))
                    Llm_dx = [vec.reshape((-1,1)) for vec in Llm_dx]
                    pool = Pool(ncores)
                    Ls_lm=pool.map(partial(cg_core, lm_sx=lm_sx), Llm_dx)
                    pool.close()
                    pool.join()
                    s_lm=np.hstack(list(Ls_lm))

            else:
                LU = splu(lm_sx.tocsc(), permc_spec="MMD_AT_PLUS_A")
                s_lm = LU.solve(lm_dx)
                # s_lm = cho_solve(cho_factor(lm_sx.todense(), lower=True), lm_dx)


            # end = time.time()
            # print("tempo LinSis:", np.round(end-start,4),"s")
            if basef.shape[0]==1:
                recf=(sp.identity(W.shape[1]).dot(baseT)-W.dot(HtT).dot(s_lm).reshape([basef.shape[1], 1])).transpose()
            else:
                recf=(sp.identity(W.shape[1]).dot(baseT)-W.dot(HtT).dot(s_lm)).transpose()

        except NameError as e: # solve
            print(f"What went wrong is {e}")
            print("scipy.sparse.linalg.spsolve raised an error\nUsing numpy.linalg.solve...")
            lm_dx = Ht.dot(baseT)
            lm_sx = Ht.dot(W).dot(HtT).todense()
            if basef.shape[0]==1:
                recf=(sp.identity(W.shape[1]).dot(baseT)-W.dot(HtT).dot(solve(lm_sx, lm_dx)).reshape([basef.shape[1], 1])).transpose()
            else:
                recf=(sp.identity(W.shape[1]).dot(baseT)-W.dot(HtT).dot(solve(lm_sx, lm_dx))).transpose()

        if nn and np.any(recf<-np.finfo(float).eps):
            if nn_type=="osqp":
                if sp.isspmatrix_dia(W):
                    P = sp.spdiags(1/W.diagonal(),0,n,n)
                else:
                    P = cho_solve(cho_factor(W.todense(), lower=True), np.eye(n))
                Id = np.unique(np.where(recf<-np.finfo(float).eps)[0])
                if keep == "recf":
                    recf[Id, :] = M_osqp(basef[Id,:], Ht=Ht, P=P, nn=nn, bounds=bounds, 
                                         settings=settings, b_pos=b_pos, ncores=ncores, keep=keep)
                else:
                    recf = np.hstack([recf, np.zeros([recf.shape[0],6])])
                    recf[Id, :] = M_osqp(basef[Id,:], Ht=Ht, P=P, nn=nn, bounds=bounds, 
                                         settings=settings, b_pos=b_pos, ncores=ncores, keep=keep)
                    recf, info = np.hsplit(recf, [recf.shape[1]-6])
            else:
                recf_nb=recf[:, (S.shape[0]-S.shape[1]):S.shape[0]]
                recf_nb[recf_nb<0]=0
                recf = sp.csc_matrix.dot(recf_nb, S.transpose())
        if keep == "recf":
            return recf
        else: 
            try:
                M = sp.identity(W.shape[1]) - W.dot(HtT).dot(spsolve(Ht.dot(W).dot(HtT).tocsc(), Ht.tocsc()))
            except:
                M = sp.identity(W.shape[1]) - W.dot(HtT).dot(solve(Ht.dot(W).dot(HtT).todense(), Ht.todense()))
            varf = np.diagonal(M.dot(W).todense())
            return recf, varf, M, info
    else:
        if sp.isspmatrix_dia(W):
            P = sp.spdiags(1/W.diagonal(),0,n,n)
        else:
            P = cho_solve(cho_factor(W.todense(), lower=True), np.eye(n))
        if keep =="recf":
            recf = M_osqp(basef, Ht=Ht, P=P, nn=nn, bounds=bounds, settings=settings, b_pos=b_pos, 
                          ncores=ncores, keep=keep)
            return recf
        else:
            recf = M_osqp(basef, Ht=Ht, P=P, nn=nn, bounds=bounds, settings=settings, b_pos=b_pos, 
                          ncores=ncores, keep=keep)
            recf, info = np.hsplit(recf, [recf.shape[1]-6])
            return recf, info
        

def recoS(basef: np.ndarray, 
         W: sp.csr_matrix, 
         S: sp.coo_matrix,
         sol: str = "direct", 
         nn: bool = False, 
         b_pos: np.ndarray=np.empty(0), 
         keep: str = "recf",
         bounds: np.ndarray=np.empty(0),
         ncores = None,
         settings={}):
    sol = match_arg(sol, ["direct", "osqp"])
    n=basef.shape[1]
    info = None
    St = S.transpose()
    if not bounds.size==0:
        if not isinstance(bounds, np.ndarray) or bounds.shape[1]!=2 or bounds.shape[0] != b_pos.sum():
            raise ValueError("bounds must be a matrix (%dx2)" %b_pos.sum())
        else:
            sol = "osqp"
    if sol == "direct":   
        if sp.isspmatrix_dia(W):
            Wm1 = sp.spdiags(1/W.diagonal(),0,n,n)
            try: # spsolve
                lm_dx1 = sp.csc_matrix(St.dot(Wm1).dot(basef.transpose()))
                lm_sx1 = sp.csc_matrix(St.dot(Wm1).dot(S))
                recf = S.dot(spsolve(lm_sx1, lm_dx1)).transpose()
                if not basef.shape[0] == 1:
                    recf= recf.todense()
                else:
                    recf = recf.reshape((1,n))
            except: # solve
                print("scipy.sparse.linalg.spsolve raised an error\nUsing numpy.linalg.solve...")
                lm_dx1 = St.dot(Wm1).dot(basef.transpose())
                lm_sx1 = St.dot(Wm1).dot(S).todense()
                recf = S.dot(solve(lm_sx1, lm_dx1)).transpose()

            if nn and np.any(recf<-np.finfo(float).eps):
                if nn_type=="osqp":
                    P = St.dot(Wm1).dot(S)
                    q = - Wm1.dot(S).transpose()
                    Id = np.unique(np.where(recf<-np.finfo(float).eps)[0])
                    if keep == "recf":
                        recf[Id, :] = S_osqp(basef[Id,:], S=S, P=P, q=q, nn=nn, bounds=bounds, settings=settings, 
                                             b_pos=b_pos, ncores=ncores, keep=keep)
                    else:
                        recf = np.hstack([recf, np.zeros([recf.shape[0],6])])
                        recf[Id, :] = S_osqp(basef[Id,:], S=S, P=P, q=q, nn=nn, bounds=bounds, settings=settings, 
                                             b_pos=b_pos, ncores=ncores, keep=keep)
                        recf, info = np.hsplit(recf, [recf.shape[1]-6])
                else:
                    recf_nb=recf[:, (S.shape[0]-S.shape[1]):S.shape[0]]
                    recf_nb[recf_nb<0]=0
                    recf = sp.csc_matrix.dot(recf_nb, St)
            if keep == "obj":
                lm_dx2 = St.dot(Wm1).todense()
                try:
                    G = spsolve(lm_sx1, lm_dx2)
                except:
                    G = solve(lm_sx1.todense(), lm_dx2.todense())
                M = S.dot(G)
                varf = np.diagonal(M.dot(W.todense()))
        else:
            try: # spsolve
                Q = spsolve(W.tocsc(),S.tocsc())
                lm_dx1 = sp.csc_matrix(Q.transpose().dot(basef.transpose()))
                lm_sx1 = sp.csc_matrix(St.dot(Q))
                recf = S.dot(spsolve(lm_sx1, lm_dx1)).transpose()
                if not basef.shape[0] == 1:
                    recf= recf.todense()
                else:
                    recf = recf.reshape((1,n))
            except: # solve
                print("scipy.sparse.linalg.spsolve raised an error\nUsing numpy.linalg.solve...")
                Q = solve(W.todense(),S.todense())
                lm_dx1 = Q.transpose().dot(basef.transpose())
                lm_sx1 = St.dot(Q)
                recf = S.dot(solve(lm_sx1, lm_dx1)).transpose()
            if nn and np.any(recf<-np.finfo(float).eps):
                P = St.dot(Q)
                q = - Q.transpose()
                Id = np.unique(np.where(recf<-np.finfo(float).eps)[0])
                if keep == "recf":
                    recf[Id, :] = S_osqp(basef[Id,:], S=S, P=P, q=q, nn=nn, bounds=bounds, settings=settings, 
                                         b_pos=b_pos, ncores=ncores, keep=keep)
                else:
                    recf = np.hstack([recf, np.zeros([recf.shape[0],6])])
                    recf[Id, :] = S_osqp(basef[Id,:], S=S, P=P, q=q, nn=nn, bounds=bounds, settings=settings, 
                                         b_pos=b_pos, ncores=ncores, keep=keep)
                    recf, info = np.hsplit(recf, [recf.shape[1]-6])
            if keep == "obj":
                try:
                    G = spsolve(lm_sx1, sp.csc_matrix(Q.transpose()))
                except:
                    G = solve(lm_sx1.todense(), Q.transpose().todense())
                M = S.dot(G)
                varf = np.diagonal(M.dot(W.todense()))
        if keep == "recf":
            return recf
        else:
            return recf, varf, M, G, S, info
    else:
        if sp.isspmatrix_dia(W):
            Q = sp.spdiags(1/W.diagonal(),0,n,n)
            P = St.dot(Q).dot(S)
            q = - Q.dot(S).transpose()
        else:
            try: 
                Q = spsolve(W.tocsc(),S.tocsc())
            except:
                print("scipy.sparse.linalg.spsolve raised an error\nUsing numpy.linalg.solve...")
                Q = solve(W.todense(),S.todense())
            P = St.dot(Q)
            q = - Q.transpose()
        if keep == "recf":
            recf = S_osqp(basef, S=S, P=P, q=q, nn=nn, bounds=bounds, settings=settings, b_pos=b_pos, 
                          ncores=ncores, keep=keep)
            return recf
        else:
            recf = S_osqp(basef, S=S, P=P, q=q, nn=nn, bounds=bounds, settings=settings, b_pos=b_pos, 
                          ncores=ncores, keep=keep)
            recf, info = np.hsplit(recf, [recf.shape[1]-6])
            return recf, info

def S_osqp(y,
           q = np.empty(0),
           P = np.empty(0),
           S = np.empty(0),
           nn = False,
           settings = {},
           b_pos = np.empty(0), 
           bounds = np.empty(0),
           ncores = None, 
           keep = "recf"):
    q = q.dot(y.transpose())
    r = S.shape[0]
    c = b_pos.sum()
    A = sp.identity(c)
    l = np.repeat(-float("inf"), int(c))
    u = np.repeat(float("inf"), int(c))
    if nn:
        A = sp.identity(c)
        l = np.zeros(int(c))
        u = np.repeat(float("inf"), int(c))
    if bounds.size != 0:
        bounds_rows = np.asarray((np.sum(np.abs(bounds) == float("inf"), axis=1)<2).flatten())[0]
        A = sp.identity(int(c), format='csr')[bounds_rows, :]
    if len(settings)==0:
        settings = {
            "verbose": False,
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "polish_refine_iter": 100,
            "polish": True}
    P = sp.csc_matrix(P)
    A = sp.csc_matrix(A)
    qt=q.transpose()
    out=np.empty(qt.shape)
    if keep == "recf":
        if not qt.shape[0] == 1:
            if ncores == 1:
                out = map(partial(osqp_core, P=P, A=A, l=l, u=u, settings=settings), list(map(np.transpose, list(qt))))
            else:
                out = Pool(ncores).map(partial(osqp_core, P=P, A=A, l=l, u=u, settings=settings), list(map(np.transpose, list(qt))))
            out=np.vstack(list(out)) 
        else:
            out = osqp_core(qt.transpose(), P, A, l, u, settings=settings).reshape((1,qt.shape[1]))
        if nn:
            out[out<0] = 0
        return S.dot(out.transpose()).transpose() 
    else:
        if not qt.shape[0] == 1:
            if ncores == 1:
                out = map(partial(osqp_core_info, P=P, A=A, l=l, u=u, settings=settings), list(map(np.transpose, list(qt))))
            else:
                out = Pool(ncores).map(partial(osqp_core_info, P=P, A=A, l=l, u=u, settings=settings), list(map(np.transpose, list(qt))))
            out=np.vstack(list(out)) 
        else:
            out = osqp_core_info(qt.transpose(), P, A, l, u, settings=settings).reshape((1,qt.shape[1]+6))
        out, info = np.hsplit(out, [out.shape[1]-6])
        if nn:
            out[out<0] = 0
        out = S.dot(out.transpose()).transpose()
        out = np.hstack([out, info])
        return out


    """
    out=np.empty(q.shape)
    for i in range(q.shape[1]):
        prob = osqp.OSQP()
        prob.setup(P, q[:,i], A, l, u, **settings)
        results = prob.solve()
        if not results.info.status == "solved":
            print("OSQP flag " + str(results.info.status) + " OSQP pri_res " + str(results.info.pri_res))
        if nn:
            results.x[results.x<0] = 0
        out[:,i] = results.x
    if nn:
        out[out<0] = 0
    return S.dot(out).transpose()
    """


def M_osqp(y, 
           P = np.empty(0), 
           Ht = np.empty(0), 
           nn = False, 
           settings = {}, 
           b_pos = np.empty(0), 
           bounds = np.empty(0),
           ncores = None,
           keep = "recf"):
    c = Ht.shape[1]
    r = Ht.shape[0]
    l = np.zeros(r)
    u = np.zeros(r)
    A = copy.deepcopy(Ht)
    q = -P.transpose().dot(y.transpose())
    if nn:
        A = sp.vstack([A, sp.identity(c, format='csr')[b_pos==1, :]])
        l = np.concatenate([l, np.zeros(int(sum(b_pos)))])
        u = np.concatenate([u, np.repeat(float("inf"), int(sum(b_pos)))])
    if bounds.size != 0:
        bounds_rows = np.asarray((np.sum(np.abs(bounds) == float("inf"), axis=1)<2).flatten())[0]
        A = sp.vstack([A, sp.identity(c, format='csr')[bounds_rows, :]])
        #A = sp.vstack([A, sp.identity(c, format='csr')[np.asarray(bounds_rows.flatten())[0], :]])
        l = np.concatenate([l, np.asarray(bounds[bounds_rows, 0]).flatten()])
        u = np.concatenate([u, np.asarray(bounds[bounds_rows, 1]).flatten()])
    if len(settings)==0:
        settings = {
            "verbose": False,
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "polish_refine_iter": 100,
            "polish": True}
    P = sp.csc_matrix(P)
    A = sp.csc_matrix(A)
    qt=q.transpose()
    if keep == "recf":
        out=np.empty(qt.shape)
        if not qt.shape[0] == 1:
            if ncores == 1:
                out = map(partial(osqp_core, P=P, A=A, l=l, u=u, settings=settings), list(qt))
            else:
                pool = Pool(ncores)
                out = pool.map(partial(osqp_core, P=P, A=A, l=l, u=u, settings=settings), list(qt))
                pool.close()
                pool.join()
            out=np.vstack(list(out))  
        else: 
            out=osqp_core(qt.transpose(), P, A, l, u, settings=settings).reshape((1,qt.shape[1]))
        if nn:
            out[out<0] = 0 
    else:
        out = np.empty([qt.shape[0], qt.shape[1]+6])
        if not qt.shape[0] == 1:
            if ncores == 1:
                out = map(partial(osqp_core_info, P=P, A=A, l=l, u=u, settings=settings), list(qt))
            else:
                pool = Pool(ncores)
                out = pool.map(partial(osqp_core_info, P=P, A=A, l=l, u=u, settings=settings), list(qt))
                pool.close()
                pool.join()
            out=np.vstack(list(out))  
        else: 
            out=osqp_core_info(qt.transpose(), P, A, l, u, settings=settings).reshape((1,out.shape[1]))
        if nn:
            out[:,:out.shape[1]-6][out[:,:out.shape[1]-6]<0] = 0 
    return out


def osqp_core(q, P, A, l, u, settings={}):
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, **settings)
    results = prob.solve()
    # results = osqp.solve(P, q, A, l, u, **settings)
    if not results.info.status == "solved":
            print("OSQP flag " + str(results.info.status) + " OSQP pri_res " + str(results.info.pri_res))
    return results.x

def osqp_core_info(q, P, A, l, u, settings={}):
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, **settings)
    results = prob.solve()
    if not results.info.status == "solved":
            print("OSQP flag " + str(results.info.status) + " OSQP pri_res " + str(results.info.pri_res))
    return np.append(results.x, [results.info.obj_val, results.info.run_time, results.info.iter, results.info.pri_res, 
           results.info.status_val, results.info.status_polish])


def gmres_core(lm_dx, lm_sx):
    lm_sx1, andata = gmres(lm_sx, lm_dx, tol=1e-4)
    if not andata == 0:
        print("Warning! a linear system may be not solved\n")
    return lm_sx1.reshape((-1,1))

def cg_core(lm_dx, lm_sx):
    lm_sx1, andata = cg(lm_sx, lm_dx, tol=1e-4)
    if not andata == 0:
        print("Warning! a linear system may be not solved\n")
    return lm_sx1.reshape((-1,1))
