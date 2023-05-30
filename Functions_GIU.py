import math
import matplotlib 
import sys
sys.path.append('/home/ballerini/Desktop/NEWstudy/Programs/')
import numpy as np
import numpy.linalg as nplin
import numpy.polynomial.polynomial as nppol
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.colors import LogNorm 
import os
from RMs_module_collection_GB import *
from M_bib_GB import * 
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from mpl_toolkits.axes_grid1 import *
from mpl_toolkits.axisartist import *
import pickle



def get_data(day, hour, wheredataare):
    filenames = ['mms1_GSE_%s_%s.txt' %(day, hour), 'mms2_GSE_%s_%s.txt' %(day, hour),
                 'mms3_GSE_%s_%s.txt' %(day, hour), 'mms4_GSE_%s_%s.txt' %(day, hour)]
    list_names = ['x', 'y', 'z', 'bx', 'by', 'bz', 'ni', 'vix', 'viy', 'viz', 'ne', 'vex', 'vey', 'vez', 'Pixx', 'Piyy', 'Pizz', 'Pixy', 'Pixz', 'Piyz', 'Texx', 'Teyy', 'Tezz', 'Texy', 'Texz',  'Teyz', 'Ex', 'Ey', 'Ez', 'Jx',  'Jy',  'Jz']
    settings = [0]
    # Paramètre de lissage pour D1, gradB (et B, Bn sur les figures uniquement)  (1Hz ==  points?)
    paramlissage = 1 # 0 pour du filtrage, 1 pour du lissage (en second traitement)
    lissg = 20
    fltrg,Napods = 1,1 #filtrer à 50Hz pour imiter pas de filtrage
    ## building for the xlabel used in plots and in the "ShiMachinery" module
    x_label = 'time [s] since ' + ' '.join(filenames[0].split('_')[2:4])
    time, MMS_Data_or, MMS_Data_fr = lectureM(wheredataare, filenames, list_names, settings)
    ## the following lines order the MMS_Data_or, MMS_Data_fr arrays in global variables as explained before
    for i,superobj in enumerate(MMS_Data_or):
            for obj,name in zip(superobj,list_names):
                globals()[''.join([name,str(i+1)])] = obj
    for i,superobj in enumerate(MMS_Data_fr):
            for obj,name in zip(superobj,list_names):
                globals()[''.join([name,str(i+1),'fr'])] = obj
    ########################################################################################
    #Durée totale en nb de points
    N=len(time)
    n=np.arange(N)
    #pas de temps
    dt=1/128
    #Durée totale en secondes
    Dt=N*dt
    Vi = [vix1,viy1,viz1,vix2,viy2,viz2,vix3,viy3,viz3,vix4,viy4,viz4]
    Ve = [vex1,vey1,vez1,vex2,vey2,vez2,vex3,vey3,vez3,vex4,vey4,vez4]
    Vir=np.reshape(Vi,(4,3,N))     # s, i, t
    Ver=np.reshape(Ve,(4,3,N))          
    R_m_by_km = 1e3
    [X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4] = [R_m_by_km * obj for obj in [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]]
    coord = [X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4]
    kvec = np.array(ReciprocalVectors(coord))
     #### Champ magnétique ####
    B_T_by_nT = 1e-9
    [Bx_1,By_1,Bz_1,Bx_2,By_2,Bz_2,Bx_3,By_3,Bz_3,Bx_4,By_4,Bz_4] = [B_T_by_nT * obj for obj in [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]]
    # B et kvec ont 12 composantes
    B = np.array([Bx_1,By_1,Bz_1,Bx_2,By_2,Bz_2,Bx_3,By_3,Bz_3,Bx_4,By_4,Bz_4])
    # reshape (en separant les composantes et les satellites)
    Br=np.reshape(B,(4,3,N))
    kvecr=np.reshape(kvec,(4,3,N))
    nb_liss=8
    Ni=[ni1,ni2,ni3,ni4]
    Ni=np.reshape(Ni,(4,N))*1e6
    Ne=[ne1,ne2,ne3,ne4]
    Ne=np.reshape(Ne,(4,N))*1e6
    rhoVir=np.einsum('st,sjt->sjt',Ni, Vir)
    rhoVer=np.einsum('st,sjt->sjt',Ne, Ver)
    E = np.array([Ex1,Ey1,Ez1,Ex2,Ey2,Ez2,Ex3,Ey3,Ez3,Ex4,Ey4,Ez4])
    Er=np.reshape(E,(4,3,N))
    #For the conversation of the LINEAR MOMENTUM I need the pressure tensor
    conv_ev2K=1.160451812*1e4
    mu0=4*np.pi*1e-7
    m_i=1.67*1e-27
    m_e=9.1093*1e-31
    k_B=1.3*1e-23
    # reshape (en separant les composantes et les satellites)
    P_xj=np.array([Pixx1, Pixy1, Pixz1, Pixx2, Pixy2, Pixz2, Pixx3, Pixy3, Pixz3, Pixx4, Pixy4, Pixz4])
    P_xj=np.reshape(P_xj,(4,3,N))
    P_yj=np.array([Pixy1, Piyy1, Piyz1, Pixy2, Piyy2, Piyz2, Pixy3, Piyy3, Piyz3, Pixy4, Piyy4, Piyz4])
    P_yj=np.reshape(P_yj,(4,3,N))
    P_zj=np.array([Pixz1, Piyz1, Pizz1, Pixz2, Piyz2, Pizz2, Pixz3, Piyz3, Pizz3, Pixz4, Piyz4, Pizz4])
    P_zj=np.reshape(P_zj,(4,3,N))
    #electrons
    Te_xj=np.array([Texx1, Texy1, Texz1, Texx2, Texy2, Texz2, Texx3, Texy3, Texz3, Texx4, Texy4, Texz4])
    Te_xj=np.reshape(Te_xj,(4,3,N))
    Pe_xj=conv_ev2K*k_B*np.einsum('st,sjt->sjt',Ne,Te_xj)
    Te_yj=np.array([Texy1, Teyy1, Teyz1, Texy2, Teyy2, Teyz2, Texy3, Teyy3, Teyz3, Texy4, Teyy4, Teyz4])
    Te_yj=np.reshape(Te_yj,(4,3,N))
    Pe_yj=conv_ev2K*k_B*np.einsum('st,sjt->sjt',Ne,Te_yj) 
    Te_zj=np.array([Texz1, Teyz1, Tezz1, Texz2, Teyz2, Tezz2, Texz3, Teyz3, Tezz3, Texz4, Teyz4, Tezz4])
    Te_zj=np.reshape(Te_zj,(4,3,N))
    Pe_zj=conv_ev2K*k_B*np.einsum('st,sjt->sjt',Ne,Te_zj)
    Pe_tensor=np.array([Pe_xj,Pe_yj,Pe_zj])
    Pe_tensor=np.transpose(Pe_tensor,(1,0,2,3))
    Pe_tensor.shape
    Pi_tensor=np.array([P_xj,P_yj,P_zj])
    Pi_tensor=np.transpose(Pi_tensor,(1,0,2,3))
    Pi_tensor.shape
    print('            LES DONNEES SONT CHARGEES')    
    return N, time, Br, Vir, Ni, Ver, Ne, rhoVir, rhoVer, Er, Pi_tensor, Pe_tensor, kvec, kvecr, dt, coord

def equations(bcd):
    global i, j
    b=bcd[0]
    c=bcd[1]
    d=bcd[2]
    h=d*gt-c*I
    hT=np.transpose(h)
    
    Pperp = I-np.einsum("i,j -> ij", vt, vt)
    m = I + Pperp*alpha2/b
    m_inv = nplin.inv(m)
    hm=np.dot(m_inv, hT)
    hhm=np.dot(h, hm)
    hmh=np.dot(hm, h)
    
     # valeurs et vecteurs propres de hhm et hmh
    valn, vecn = nplin.eig(hhm)                     # solution, composante/solution 
    ivaln=np.argsort(valn)
    i_n=ivaln[2]
    nn = vecn[:,i_n]
    valu, vecu = nplin.eig(hmh)                  # solution, composante/solution 
    ivalu=np.argsort(valu)
    i_u=ivalu[2]
    uu = vecu[:,i_u]
    
    if np.dot(nn, n_init)<0: nn=-nn
    if np.dot(uu, u_init)<0: uu=-uu
    
    hmn=np.dot(hm, nn)
    uhmn=np.dot(uu,hmn)
    ng=np.dot(nn,gt)
    ngu=np.dot(ng, uu)
    nh=np.dot(nn,h)
    nhu=np.dot(nh, uu)
    eq1=(b-uhmn)/uhmn
    eq2=(d*d-nhu)/nhu
    eq3 = np.dot(nn, uu)
    return eq1, eq2, eq3

def B_normal(Br, Nw, N, dt, nrangez, time, n_inf, n_sup, coord, kvecr, nrangez4):
    
    def equations(bcd):
        global i, j
        b=bcd[0]
        c=bcd[1]
        d=bcd[2]
        h=d*gt-c*I
        hT=np.transpose(h)

        Pperp = I-np.einsum("i,j -> ij", vt, vt)
        m = I + Pperp*alpha2/b
        m_inv = nplin.inv(m)
        hm=np.dot(m_inv, hT)
        hhm=np.dot(h, hm)
        hmh=np.dot(hm, h)

         # valeurs et vecteurs propres de hhm et hmh
        valn, vecn = nplin.eig(hhm)                     # solution, composante/solution 
        ivaln=np.argsort(valn)
        i_n=ivaln[2]
        nn = vecn[:,i_n]
        valu, vecu = nplin.eig(hmh)                  # solution, composante/solution 
        ivalu=np.argsort(valu)
        i_u=ivalu[2]
        uu = vecu[:,i_u]

        if np.dot(nn, n_init)<0: nn=-nn
        if np.dot(uu, u_init)<0: uu=-uu

        hmn=np.dot(hm, nn)
        uhmn=np.dot(uu,hmn)
        ng=np.dot(nn,gt)
        ngu=np.dot(ng, uu)
        nh=np.dot(nn,h)
        nhu=np.dot(nh, uu)
        eq1=(b-uhmn)/uhmn
        eq2=(d*d-nhu)/nhu
        eq3 = np.dot(nn, uu)
        return eq1, eq2, eq3
    
    dtB_smooth=np.zeros((3,N))
    v_smooth=np.zeros((3,N))
    v_smooth_mod=np.zeros(N)
    Br4=np.mean(Br, axis=0)     # i, t
    Br4_tr=np.transpose(Br4)    # t, i
    dtB_raw=np.gradient(Br4, dt,  axis=1)
    nglob=np.arange(N)
    for t in nrangez:
        minw = max(t-Nw, 0)
        maxw = min(t+Nw+1, N)
        wrange=nglob[minw:maxw]
        trange=time[wrange]
        tc = wrange - t                              # indice t dans la fenêtre, centré au milieu de la fenêtre
        timec=time[tc]                               # temps centré
        ymean=np.mean(dtB_raw[:,wrange], axis=1)
        ytmean=np.mean(dtB_raw[:,wrange]*tc, axis=1)

        T2=np.mean(tc**2)

        vt=ymean/nplin.norm(ymean)#nplin.norm(ymean)*ymean + nplin.norm(ytmean)*ytmean/T2

        #vt=vt/nplin.norm(vt)
        # vt=vt/(nplin.norm(ymean)**2 + nplin.norm(ytmean)**2/T2)

        v_smooth[:,t] = vt
        v_smooth_mod[t]=nplin.norm(v_smooth[:,t])

        dtBt=vt*nplin.norm(ymean)
        dtB_smooth[:,t]=dtBt
    
    
            
    alpha2 = 0.5
    bad1=np.zeros(N)
    bad2=np.zeros(N)
    bad3=np.zeros(N)
    bad4=np.zeros(N)

    bad1[n_inf]=1
    bad2[n_inf]=1
    bad3[n_inf]=1
    bad4[n_inf]=1

    bad1[n_sup]=1
    bad2[n_sup]=1
    bad3[n_sup]=1
    bad4[n_sup]=1
    scal_tab=np.zeros(N)
    R=np.reshape(coord,(4,3,N))              # position
    Rm=np.mean(R, axis=0)                    # moyenne sur les 4 s/c
    eps0_tab=np.zeros(N)
    keep=list(range(N))
    I=np.identity(3)
    Vn=np.zeros(N)
    Vn_vect=np.zeros((3,N))
    Vn_init=np.zeros(N)
    Vn_init_vect=np.zeros((3,N))
    bcd=np.zeros(3)
    n=np.zeros((3,N))
    u=np.zeros((3,N))
    d=np.zeros(N)
    Bm=np.zeros((4,3,N))
    Gm=np.zeros((3,3,N))
    GGTm=np.zeros((3,3, N))
    GTGm=np.zeros((3,3, N))
    X=np.zeros(N)
    g=np.zeros((3,3,N))
    h=np.zeros((3,3,N))
    eq1tab=np.zeros(N)
    eq2tab=np.zeros(N)
    eq3tab=np.zeros(N)

    n_init0_tab=np.zeros((3,N))
    u_init0_tab=np.zeros((3,N))

    n_init_tab=np.zeros((3,N))
    u_init_tab=np.zeros((3,N))
    d_init_tab=np.zeros(N)

    G=np.einsum('sit,sjt->ijt',kvecr, Br)
    GGT=np.einsum('ijt,kjt->ikt',G,G)
    GTG=np.einsum('jit,jkt->ikt',G,G)

    nb_remove =0
    nb_remove1=0
    nb_remove2=0
    nb_remove3=0
    nb_remove4=0

    ##############################################
    # Fenêtre glissante de 2*NW+1 points en temps

    for t in nrangez:

        vt=v_smooth[:,t]
        dtBt=dtB_smooth[:,t]
        minw = max(t-Nw, 0)
        maxw = min(t+Nw+1, N)
        wrange=np.arange(minw,maxw)
        # Valeurs moyennes sur la fenêtre glissante

        Bmt=np.mean(Br[:,:,wrange], axis=2)    # s,i
        Bm[:,:,t]=Bmt    # s,i,t
        Gmt=np.mean(G[:,:,wrange], axis=2)     # i,j
        Gm[:,:,t]=Gmt     # i,j,t
        GGTmt=np.mean(GGT[:,:,wrange], axis=2)          # i,j
        GGTm[:,:,t]=GGTmt                             # i,j,t
        GTGmt=np.mean(GTG[:,:,wrange], axis=2)          # i,j
        GTGm[:,:,t]=GTGmt                              # i,j,t
        Xt=np.sqrt(np.trace(GGTmt))
        X[t]=Xt                                        # t
        gt=Gmt/Xt                                     # i,j
        gtT=np.transpose(gt)                          # j,i
        g[:,:,t]=gt    
        ####################
        # 1. Initialisation

        Tn = np.dot(gt, gtT)
        Tu = np.dot(gtT, gt)

        # valeurs et vecteurs propres de Tn
        valn, vecn = nplin.eig(Tn)                     # solution, composante/solution 
        ivaln=np.argsort(valn)
        in0=ivaln[2]

        n_init0 = vecn[:,in0]

         # valeurs et vecteurs propres de Tu
        valu, vecu = nplin.eig(Tu)                     # solution, composante/solution 
        ivalu=np.argsort(valu)
        iu0=ivalu[2]

        u_init0 = vecu[:,iu0]

         ############################################################
        # choix initial des signes par continuités temporelles

        scal=np.dot(n_init0, n_init0_tab[:, t-1])
        if scal<0:
            n_init0=-n_init0

        scal=np.dot(u_init0, u_init0_tab[:, t-1])
        if scal<0:
            u_init0=-u_init0

        n_init0_tab[:,t] = n_init0
        u_init0_tab[:,t] = u_init0

       #####################
        ng=np.dot(n_init0, gt)
        ngu=np.dot(ng, u_init0)
        ngn=np.dot(ng, n_init0)

        ug=np.dot(u_init0, gt)
        ugu=np.dot(ug, u_init0)


        ########################
        ########################
        d_init0=ngu
        c_init0=d_init0*ugu
        b_init0=d_init0*d_init0
        #########################
        #########################


        # On impose d>0 en changeant u0 s'il le faut

        if d_init0<0:
            u_init0=-u_init0
            d_init0=-d_init0
            c_init0=-c_init0


        # j0[:,t]=np.cross(n_init0, u_init0)

        #############################################

        #############################  correction pour assurer divB=0 dans l'initialisation  ##################################
        eps0=np.dot(n_init0, u_init0)

        un=np.einsum('i,j->ij', u_init0, n_init0)
        nu=np.einsum('i,j->ij', n_init0, u_init0)
        uu=np.einsum('i,j->ij', u_init0, u_init0)
        nn=np.einsum('i,j->ij', n_init0, n_init0)

        Mn = (1-eps0**2)*I + uu/2. - (un+nu)*eps0/2. + eps0**2*nn/2.
        Mu = (1-eps0**2)*I + nn/2. - (un+nu)*eps0/2. + eps0**2*uu/2.

        Mn_inv = nplin.inv(Mn)
        Mu_inv = nplin.inv(Mu)

        Sn2 = -(u_init0-eps0*n_init0)*eps0/2.
        Su2 = -(n_init0-eps0*u_init0)*eps0/2.

        delta_n = np.dot(Mn_inv, Sn2)
        delta_u = np.dot(Mu_inv, Su2)

        n_init = n_init0 + delta_n
        n_init = n_init/nplin.norm(n_init)

        u_init = u_init0 + delta_u
        u_init = u_init/nplin.norm(u_init)


         ################################################

        ng=np.dot(n_init, gt)
        ngu=np.dot(ng, u_init)
        ngn=np.dot(ng, n_init)

        ug=np.dot(u_init, gt)
        ugu=np.dot(ug, u_init)

        d_init=ngu
        c_init=d_init*ugu
        b_init=d_init*d_init

        ##############################################

        n_init_tab[:,t]= n_init
        u_init_tab[:,t]= u_init 
        d_init_tab[t]  = d_init

        ####################
        # calcul de Vn_init
        dtBt_norm=nplin.norm(dtBt)

        Vn_init[t] = d_init*dtBt_norm/X[t]                # abs(Vn)
        scal=np.dot(vt,u_init)
        if scal<0: Vn_init[t]=-Vn_init[t]               # Vn avec son signe

        if abs(scal)<0.5: bad1[t]=1                      # pas confiance si v et u sont trop mal alignes

        # Vn_init_vect
        Vn_init_vect[:,t]=Vn_init[t]* n_init            # Vn_vect


        ##################
        bcd_init=np.array([b_init, c_init, d_init])
        bcd=bcd_init
        ng=np.dot(n_init, gt)
        ngu=np.dot(ng, u_init)
        ngn=np.dot(ng, n_init)
        ug=np.dot(u_init, gt)
        ugu=np.dot(ug, u_init)
        ####################
        ####################
        d_init=ngu
        c_init=d_init*ugu
        b_init=d_init*d_init
        #####################
        #####################
        n_init_tab[:,t]=n_init
        u_init_tab[:,t]=u_init 
        d_init_tab[t]=d_init
        bcd_init=np.array([b_init, c_init, d_init])
        bcd=bcd_init

        # 2. Résolution complète (pour chaque t)
        ########################
        ier=1
        i=0
        j=0

        '******************************************************************************'
        bcd, info, ier, mesg =  fsolve(equations, bcd_init, full_output=1, xtol=1.e-12)
        '******************************************************************************'
        b_t,c_t,d_t = bcd    
        ht = d_t*gt-c_t*I
        htT = np.transpose(ht)
        Pperp = I - np.einsum("i,j -> ij", vt, vt)
        m = I + Pperp*alpha2/b_t
        m_inv = nplin.inv(m)
        hm=np.dot(m_inv, htT)
        hhm=np.dot(ht, hm)
        hmh=np.dot(hm, ht)


        # valeurs et vecteurs propres
        valn, vecn = nplin.eig(hhm)                     # solution, composante/solution 
        ivaln=np.argsort(valn)
        in0=ivaln[2]

        nt = vecn[:,in0]
        if np.dot(nt, n_init)<0: nt=-nt


         # valeurs et vecteurs propres
        valu, vecu = nplin.eig(hmh)                     # solution, composante/solution 
        ivalu=np.argsort(valu)
        iu0=ivalu[2]
        ut = vecu[:,iu0]
        if np.dot(nt, n_init)<0: nt=-nt               # sens de n et u par proximité avec l'initialisation
        if np.dot(ut, u_init)<0: ut=-ut
        n[:,t]= nt
        u[:,t]= ut
        d[t]=d_t

        # Limitons les rotations entre 2 points (pour n et pour u):
        dtheta=nplin.norm(np.cross(nt,n[:, t-1]))
        limit=3./(2.*Nw+1.)
        if dtheta > limit: bad2[t]=1

        dtheta=nplin.norm(np.cross(ut,u[:, t-1]))
        limit=3./(2.*Nw+1.)
        if dtheta > limit: bad2[t]=1

        hmn=np.dot(hm, nt)
        uhmn=np.dot(ut,hmn)
        nh=np.dot(nt,ht)
        nhu=np.dot(nh, ut)
        eq1=(b_t-uhmn)/uhmn
        eq2=(d_t*d_t-nhu)/nhu
        eq3 = np.dot(nt, ut)

        # calcul de Vn
        Vn[t] = d_t*dtBt_norm/X[t]                                # abs(Vn)
        vt=v_smooth[:,t]
        scal=np.dot(vt,ut)
        if scal<0: Vn[t]=-Vn[t]                                  # Vn avec son signe
        if abs(scal)<0.5: bad3[t]=1                      # pas confiance si v et u sont trop mal alignes

        # Vn_vect
        Vn_vect[:,t]=np.dot(Vn[t], nt)                           # Vn_vect
        error=eq1*eq1+eq2*eq2+eq3*eq3    
        if (error > 1.e-20): bad4[t]=1
        if (ier != 1) :    bad4[t] = 1


    for t in range(N):
        if bad1[t]==1 :
            if t in nrangez: nb_remove1 = nb_remove1+1
        if bad2[t]==1 :
            if t in nrangez: nb_remove2 = nb_remove2+1
        if bad3[t]==1 :
            if t in nrangez: nb_remove3 = nb_remove3+1
        if bad4[t]==1 :
            if t in nrangez: nb_remove4 = nb_remove4+1
        if bad1[t]==1 or bad2[t]==1 or bad3[t]==1 or bad4[t]==1:
            keep.remove(t)
            if t in nrangez: nb_remove = nb_remove+1

    tkeep=time[keep]
    ##############################################################################################################
    scal=np.einsum('it, it-> t', Rm, n_init0_tab)
    scal4=np.mean(scal[nrangez4])
    if scal4<0:
        n_init0_tab=-n_init0_tab
        u_nit0_tab=-u_init0_tab
    scal=np.einsum('it, it-> t', Rm, n_init_tab)
    scal4=np.mean(scal[nrangez4])
    if scal4<0:
        n_init_tab=-n_init_tab
        u_init_tab=-u_init_tab
    scal=np.einsum('it, it-> t', Rm, n)
    scal4=np.mean(scal[nrangez4])
    if scal4<0:
        n=-n
        u=-u
    print('nb_remove=', nb_remove, ' nb_remove1=', nb_remove1,' nb_remove2=', nb_remove2,' nb_remove3=', nb_remove3,' nb_remove4=', nb_remove4 )

    return v_smooth, tkeep, keep, n, u, Vn_vect, X, d


def rhov_normal(rhoVir,Ni, Nw, N, dt, nrangez, time, n_inf, n_sup, coord, kvecr, nrangez4):
    nglob=np.arange(N)
    def equations(bcd):
        global i, j

        b=bcd[0]
        c=bcd[1]
        d=bcd[2]
        h=d*gt-c*I
        hT=np.transpose(h)
        Pperp = I-np.einsum("i,j -> ij", vt, vt)
        m = I + Pperp*alpha2/b
        m_inv = nplin.inv(m)

        hm=np.dot(m_inv, hT)
        hhm=np.dot(h, hm)
        hmh=np.dot(hm, h)

         # valeurs et vecteurs propres de hhm et hmh
        valn, vecn = nplin.eig(hhm)                     # solution, composante/solution 
        ivaln=np.argsort(valn)
        i_n=ivaln[2]

        nn = vecn[:,i_n]
        valu, vecu = nplin.eig(hmh)                  # solution, composante/solution 
        ivalu=np.argsort(valu)
        i_u=ivalu[2]
        uu = vecu[:,i_u]

        if np.dot(nn, n_init)<0: nn=-nn
        if np.dot(uu, u_init)<0: uu=-uu

        hmn=np.dot(hm, nn)
        uhmn=np.dot(uu,hmn)
        ng=np.dot(nn,gt)
        ngu=np.dot(ng, uu)
        nh=np.dot(nn,h)
        nhu=np.dot(nh, uu)

        eq1=(b-uhmn)/uhmn
        eq2=(d*d-nhu+2*c*Z)/nhu
        eq3 = np.dot(nn, uu) + Z*d   
        return eq1, eq2, eq3


    keep=list(range(N))
    dtrhov_smooth=np.zeros((3,N))
    v_smooth=np.zeros((3,N))
    rhovir4=np.mean(rhoVir, axis=0)     # i, t
    rhovir4_tr=np.transpose(rhovir4)    # t, i
    dtrhov_raw=np.gradient(rhovir4, dt,  axis=1)
    for t in nrangez:
        minw = max(t-Nw, 0)
        maxw = min(t+Nw+1, N)
        wrange=nglob[minw:maxw]
        trange=time[wrange]
        tc = wrange - t                              # indice t dans la fenêtre, centré au milieu de la fenêtre
        timec=time[tc]                               # temps centrés
        ymean=np.mean(dtrhov_raw[:,wrange], axis=1)
        ytmean=np.mean(dtrhov_raw[:,wrange]*tc, axis=1)
        T2=np.mean(tc**2)
        vt=ymean/nplin.norm(ymean)
        v_smooth[:,t] = vt
        dtrhovt=vt*nplin.norm(ymean)
        dtrhov_smooth[:,t]=dtrhovt
        
        
    Ni_smooth=np.zeros(N)
    for t in nrangez:        
        minw = max(t-Nw, 0)
        maxw = min(t+Nw+1, N)
        wrange=np.arange(minw,maxw)
        Ni_smooth[t]=np.mean(np.sum(Ni[:,wrange],axis=0)/4)
    dNidt=np.gradient(Ni_smooth,dt)
    dNidt[nrangez[0]]=0.
    dNidt[nrangez[-1]]=0.
    alpha2 = 0.5

    bad1=np.zeros(N)
    bad2=np.zeros(N)
    bad3=np.zeros(N)
    bad4=np.zeros(N)
    bad1[n_inf]=1
    bad2[n_inf]=1
    bad3[n_inf]=1
    bad4[n_inf]=1
    bad1[n_sup]=1
    bad2[n_sup]=1
    bad3[n_sup]=1
    bad4[n_sup]=1


    scal_tab=np.zeros(N)
    R=np.reshape(coord,(4,3,N))              # position
    Rm=np.mean(R, axis=0)                    # moyenne sur les 4 s/c
    eps0_tab=np.zeros(N)
    keep=list(range(N))
    I=np.identity(3)
    Vn=np.zeros(N)
    Vn_vect=np.zeros((3,N))
    Vn_init=np.zeros(N)
    Vn_init_vect=np.zeros((3,N))   
    bcd=np.zeros(3)
    n=np.zeros((3,N))
    u=np.zeros((3,N))
    d=np.zeros(N)
    Bm=np.zeros((4,3,N))
    Gm=np.zeros((3,3,N))
    GGTm=np.zeros((3,3, N))
    GTGm=np.zeros((3,3, N))
    X=np.zeros(N)
    g=np.zeros((3,3,N))
    h=np.zeros((3,3,N))
    eq1tab=np.zeros(N)
    eq2tab=np.zeros(N)
    eq3tab=np.zeros(N)
    n_init0_tab=np.zeros((3,N))
    u_init0_tab=np.zeros((3,N))
    n_init_tab=np.zeros((3,N))
    u_init_tab=np.zeros((3,N))
    d_init_tab=np.zeros(N)



    Z_t=np.zeros(N)
    G=np.einsum('sit,sjt->ijt',kvecr, rhoVir)
    GGT=np.einsum('ijt,kjt->ikt',G,G)
    GTG=np.einsum('jit,jkt->ikt',G,G)

    nb_remove =0
    nb_remove1=0
    nb_remove2=0
    nb_remove3=0
    nb_remove4=0

    for t in nrangez:

        vt=v_smooth[:,t]
        dtrhovt=dtrhov_smooth[:,t]

        minw = max(t-Nw, 0)
        maxw = min(t+Nw+1, N)
        wrange=np.arange(minw,maxw)

        # Valeurs moyennes sur la fenêtre glissante

        Bmt=np.mean(rhoVir[:,:,wrange], axis=2)    # s,i
        Bm[:,:,t]=Bmt    # s,i,t

        Gmt=np.mean(G[:,:,wrange], axis=2)     # i,j
        Gm[:,:,t]=Gmt     # i,j,t

        GGTmt=np.mean(GGT[:,:,wrange], axis=2)          # i,j
        GGTm[:,:,t]=GGTmt                             # i,j,t

        GTGmt=np.mean(GTG[:,:,wrange], axis=2)          # i,j
        GTGm[:,:,t]=GTGmt                              # i,j,t

        Xt=np.sqrt(np.trace(GGTmt))
        X[t]=Xt                                        # t

        gt=Gmt/Xt                                     # i,j
        gtT=np.transpose(gt)                          # j,i

        g[:,:,t]=gt

        Z_t[t]=dNidt[t]/Xt                                                                       ###AGGIUNTO IO
        Z=Z_t[t]    
        #############################################################
        # Résolution du système

        ####################
        # 1. Initialisation

        Tn = np.dot(gt, gtT)
        Tu = np.dot(gtT, gt)

        # valeurs et vecteurs propres de Tn
        valn, vecn = nplin.eig(Tn)                     # solution, composante/solution 
        ivaln=np.argsort(valn)
        in0=ivaln[2]

        n_init0 = vecn[:,in0]

         # valeurs et vecteurs propres de Tu
        valu, vecu = nplin.eig(Tu)                     # solution, composante/solution 
        ivalu=np.argsort(valu)
        iu0=ivalu[2]

        u_init0 = vecu[:,iu0]

         ############################################################
        # choix initial des signes par continuités temporelles

        scal=np.dot(n_init0, n_init0_tab[:, t-1])
        if scal<0:
            n_init0=-n_init0

        scal=np.dot(u_init0, u_init0_tab[:, t-1])
        if scal<0:
            u_init0=-u_init0

        n_init0_tab[:,t] = n_init0
        u_init0_tab[:,t] = u_init0

       #####################
        ng=np.dot(n_init0, gt)
        ngu=np.dot(ng, u_init0)
        ngn=np.dot(ng, n_init0)

        ug=np.dot(u_init0, gt)
        ugu=np.dot(ug, u_init0)


        ########################
        ########################
        d_init0=ngu/(1+2*Z*ugu)  
        c_init0=d_init0*ugu
        b_init0=d_init0*d_init0
        #########################
        #########################


        # On impose d>0 en changeant u0 s'il le faut

        if d_init0<0:
            u_init0=-u_init0
            d_init0=-d_init0
            c_init0=-c_init0


        # j0[:,t]=np.cross(n_init0, u_init0)

        #############################################

        #############################  correction pour assurer divB=0 dans l'initialisation  ##################################
        eps0=np.dot(n_init0, u_init0)+Z*d_init0

        un=np.einsum('i,j->ij', u_init0, n_init0)
        nu=np.einsum('i,j->ij', n_init0, u_init0)
        uu=np.einsum('i,j->ij', u_init0, u_init0)
        nn=np.einsum('i,j->ij', n_init0, n_init0)

        Mn = (1-eps0**2)*I + uu/2. - (un+nu)*eps0/2. + eps0**2*nn/2.
        Mu = (1-eps0**2)*I + nn/2. - (un+nu)*eps0/2. + eps0**2*uu/2.

        Mn_inv = nplin.inv(Mn)
        Mu_inv = nplin.inv(Mu)

        Sn2 = -(u_init0-eps0*n_init0)*eps0/2.
        Su2 = -(n_init0-eps0*u_init0)*eps0/2.

        delta_n = np.dot(Mn_inv, Sn2)
        delta_u = np.dot(Mu_inv, Su2)

        n_init = n_init0 + delta_n
        n_init = n_init/nplin.norm(n_init)

        u_init = u_init0 + delta_u
        u_init = u_init/nplin.norm(u_init)


         ################################################

        ng=np.dot(n_init, gt)
        ngu=np.dot(ng, u_init)
        ngn=np.dot(ng, n_init)

        ug=np.dot(u_init, gt)
        ugu=np.dot(ug, u_init)

        d_init=ngu/(1+2*Z*ugu)
        c_init=d_init*ugu
        b_init=d_init*d_init

        ##############################################

        n_init_tab[:,t]= n_init
        u_init_tab[:,t]= u_init 
        d_init_tab[t]  = d_init

        ####################
        # calcul de Vn_init
        dtrhovt_norm=nplin.norm(dtrhovt)

        Vn_init[t] = d_init*dtrhovt_norm/X[t] # abs(Vn)
        scal=np.dot(vt,u_init)
        if scal<0: Vn_init[t]=-Vn_init[t]               # Vn avec son signe

        if abs(scal)<0.5: bad1[t]=1                      # pas confiance si v et u sont trop mal alignes

        # Vn_init_vect
        Vn_init_vect[:,t]=Vn_init[t]* n_init            # Vn_vect


        ##################

        bcd_init=np.array([b_init, c_init, d_init])
        bcd=bcd_init

        ng=np.dot(n_init, gt)
        ngu=np.dot(ng, u_init)
        ngn=np.dot(ng, n_init)

        ug=np.dot(u_init, gt)
        ugu=np.dot(ug, u_init)

        ####################
        ####################
        d_init=ngu/(1+2*Z*ugu)
        c_init=d_init*ugu
        b_init=d_init*d_init
        #####################
        #####################


        n_init_tab[:,t]=n_init
        u_init_tab[:,t]=u_init 
        d_init_tab[t]=d_init

        bcd_init=np.array([b_init, c_init, d_init])
        bcd=bcd_init
        #print(bcd)

        # 2. Résolution complète (pour chaque t)
        ########################

        ier=1

        i=0
        j=0

        '******************************************************************************'
        bcd, info, ier, mesg =  fsolve(equations, bcd_init, full_output=1, xtol=1.e-12)
        '******************************************************************************'

        b_t,c_t,d_t = bcd

        ht = d_t*gt-c_t*I
        htT = np.transpose(ht)

        Pperp = I - np.einsum("i,j -> ij", vt, vt)

        m = I + Pperp*alpha2/b_t
        m_inv = nplin.inv(m)

        hm=np.dot(m_inv, htT)

        hhm=np.dot(ht, hm)
        hmh=np.dot(hm, ht)


        # valeurs et vecteurs propres
        valn, vecn = nplin.eig(hhm)                     # solution, composante/solution 
        ivaln=np.argsort(valn)
        in0=ivaln[2]

        nt = vecn[:,in0]
        if np.dot(nt, n_init)<0: nt=-nt


         # valeurs et vecteurs propres
        valu, vecu = nplin.eig(hmh)                     # solution, composante/solution 
        ivalu=np.argsort(valu)
        iu0=ivalu[2]

        ut = vecu[:,iu0]
        if np.dot(nt, n_init)<0: nt=-nt               # sens de n et u par proximité avec l'initialisation
        if np.dot(ut, u_init)<0: ut=-ut

        n[:,t]= nt
        u[:,t]= ut
        d[t]=d_t

        # Limitons les rotations entre 2 points (pour n et pour u):

        dtheta=nplin.norm(np.cross(nt,n[:, t-1]))
        limit=3./(2.*Nw+1.)
        if dtheta > limit: bad2[t]=1


        dtheta=nplin.norm(np.cross(ut,u[:, t-1]))
        limit=3./(2.*Nw+1.)
        if dtheta > limit: bad2[t]=1


        hmn=np.dot(hm, nt)
        uhmn=np.dot(ut,hmn)

        nh=np.dot(nt,ht)
        nhu=np.dot(nh, ut)

        eq1=(b_t-uhmn)/uhmn
        eq2=(d_t*d_t-nhu+2*c_t*Z)/nhu
        eq3 = np.dot(nt, ut)+d_t*Z

        # calcul de Vn

        Vn[t] = d_t*dtrhovt_norm/X[t]                                # abs(Vn)

        vt=v_smooth[:,t]

        scal=np.dot(vt,ut)
        if scal<0: Vn[t]=-Vn[t]                                  # Vn avec son signe

        if abs(scal)<0.5: bad3[t]=1                      # pas confiance si v et u sont trop mal alignes

        # Vn_vect
        Vn_vect[:,t]=np.dot(Vn[t], nt)                           # Vn_vect

        error=eq1*eq1+eq2*eq2+eq3*eq3


        # Points à sauter pour les plots
        ################################

        if (error > 1.e-20): bad4[t]=1
        if (ier != 1) :    bad4[t] = 1


    for t in range(N):

        if bad1[t]==1 :
            if t in nrangez: nb_remove1 = nb_remove1+1

        if bad2[t]==1 :
            if t in nrangez: nb_remove2 = nb_remove2+1

        if bad3[t]==1 :
            if t in nrangez: nb_remove3 = nb_remove3+1

        if bad4[t]==1 :
            if t in nrangez: nb_remove4 = nb_remove4+1

        if bad1[t]==1 or bad2[t]==1 or bad3[t]==1 or bad4[t]==1:
            keep.remove(t)
            if t in nrangez: nb_remove = nb_remove+1

    tkeep=time[keep]

    ##############################################################################################################

    # Choix d'une normale sortante dans la zone centrale

    scal=np.einsum('it, it-> t', Rm, n_init0_tab)
    scal4=np.mean(scal[nrangez4])
    if scal4<0:
        n_init0_tab=-n_init0_tab
        u_nit0_tab=-u_init0_tab


    scal=np.einsum('it, it-> t', Rm, n_init_tab)
    scal4=np.mean(scal[nrangez4])
    if scal4<0:
        n_init_tab=-n_init_tab
        u_init_tab=-u_init_tab


    scal=np.einsum('it, it-> t', Rm, n)
    scal4=np.mean(scal[nrangez4])
    if scal4<0:
        n=-n
        u=-u

    print('nb_remove=', nb_remove, ' nb_remove1=', nb_remove1,' nb_remove2=', nb_remove2,' nb_remove3=', nb_remove3,' nb_remove4=', nb_remove4 )
    return v_smooth, tkeep, keep, n, u, Vn_vect, X, d