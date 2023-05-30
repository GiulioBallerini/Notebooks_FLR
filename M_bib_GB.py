###################################################################### import section ######################################################################

# Cette version _GB est identique au M_bib original, sauf qu'il appelle RMs_module_collection_GB au lieu de l'original
# Les importations de toolkits sont aussi un peu differentes.

#__main modules                                                                      
import numpy as np                                                                            
import glob                                                                                     
import os                                                                                     
import datetime                                                                                 
import urllib                                                                                   
import matplotlib                                                                               
import math                                                                                     
import random as rnd                                                                           

#__sub-modules
from sympy import Function, simplify, symbols                                                  
from sympy import cos, sin                                                                      
from os import system                                                                          
from spacepy import pycdf                                                                      

#__sub-sub-modules
from scipy.fftpack import fft, ifft                                                             
from numpy.linalg import det, inv                                                              
from scipy.ndimage import gaussian_filter                                                       
import matplotlib.pyplot as plt                                                                
import matplotlib.pyplot as plt2                                                                
from scipy.optimize import curve_fit, minimize
import scipy.fftpack as fftpack                                                                 
from scipy.special import factorial                                                             
from scipy.interpolate import interp1d                                                          
from matplotlib.colors import Normalize                                                         
from mpl_toolkits.mplot3d import Axes3D                                                        
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from mpl_toolkits.axes_grid1 import *
from mpl_toolkits.axisartist import *


from RMs_module_collection_GB import *


#Fonction calcul du produit vectoriel
def vector_product(Ax,Ay,Az,Bx,By,Bz):
    Cx = Ay*Bz - Az*By
    Cy = Az*Bx - Ax*Bz
    Cz = Ax*By - Ay*Bx
    return([Cx,Cy,Cz])


#Fonction calcul du rotationnel ???????????
def curl(coord, Q):
    rv = ReciprocalVectors(coord) #fonction de RMs_module_collection
    EspGradQ(rv,Q) #fonction de RMs_module_collection
    #returns [d(Qx)/dx, d(Qy)/dx, d(Qz)/dx, d(Qx)/dy, d(Qy)/dy, d(Qz)/dy, d(Qx)/dz, d(Qy)/dz, d(Qz)/dz]
    curl_x = EspGradQ(rv,Q)[5] -  EspGradQ(rv,Q)[7] # d(Qz)/dy - d(Qy)/dz
    curl_y = EspGradQ(rv,Q)[6] -  EspGradQ(rv,Q)[2] # d(Qx)/dz - d(Qz)/dx
    curl_z = EspGradQ(rv,Q)[1] -  EspGradQ(rv,Q)[3] # d(Qy)/dx - d(Qx)/dy
    return(np.array([curl_x, curl_y, curl_z])) 


#Fonctions pour le filtrage par la méthode SVD

#--Fonction de troncature
def svd_tronc(s,i):
    stronque=np.zeros(len(s))
    for k in range(1,i+1):
        stronque[k-1]=s[k-1]
    erreur=1.-np.sum(stronque**2)/np.sum(s**2)
    return k, erreur, stronque

#--Fonction de recontruction
def reconstruction_svd(a,tronc):
# a matrice d'entrée, seuil = valeur de l'erreur pour calculer sur combien de composantes on reconstruit
# analyse de a
    nlign,ncol = a.shape
# calcul de SVD
    u, s, vh = np.linalg.svd(a, full_matrices=True)
# s = valeurs singulières rangées par ordre décroissant
    print("s=", s)
    L = int(len(s))
    print("len(s)=", L)
    plt.figure(1, figsize=[10,4])
    plt.plot(s)
    plt.yscale('log')
    plt.title("Singular values")
    k, erreur_def, stronque = svd_tronc(s,tronc)
    smat2 = np.zeros((nlign, ncol))
    smat2[:ncol, :ncol] = np.diag(stronque)
    asvd2=np.dot(u, np.dot(smat2, vh))
    print('troncature à ',k,' composante(s)')
    print('erreur =',erreur_def)
    return asvd2

#--Fonction de construction de la matrice de départ
def mat_init(nbcompo,nbsat):
    if nbsat == 1:
        composantes = [j for j in range(0,nbcompo)]
    else:
        composantes = [0]*nbcompo*nbsat #liste de zéros
        for l in range(0,nbsat):
            for m in range(0,nbcompo):
                compos = (l*29)+m
                composantes[l*nbcompo+m] = compos  
    print(composantes)
    datanew = datatotfr[:, composantes] #SVD appliqué aux données filtrées
    mat = np.zeros((len(time),nbcompo*nbsat))
    for i in range (1,(nbcompo*nbsat)+1):
        mati = datanew[:,i-1]
        mat[:,i-1] = mati
    print("matrice de données", np.shape(mat), "= \n", mat)
    return mat

#--Fonction de normalisation de la matrice de départ
def norm_mat(mat):
    dim = np.shape(mat)
    norm = np.zeros(dim[1])
    for i in range(0,dim[1]):
        maxabs = np.max(np.abs(mat[:,i]))
        norm[i] = maxabs
    nmat = mat/norm
    return nmat


#Fonction de lecture des données MMS          
def lectureM(path, filenames, lista, settings):
    
    '''
    Explanation:
        Multiple application of single_lecture(...)
    Inputs:
        path:       address of the folder which contains all the txt_files where MMS data are stored in columns
        filenames:  array which contains the names (strings) of txt_files (for MMS1,2,3,4)
        lista:      array which contains the names (strings) of the variables that are needed to be extracted
        settings:   array which contains the frequency cut (double) above which data are filtered by means of the
                    Filter_S(....) module and the Napod for Belmont's apodisation window Belmont_window(..)
    Outputs:
        time:       array which contains the time (the first column of the first file indicated in filenames)
        Data_or:    array of arrays which contains data extracted
        Data_fr:    same as Data_or but filtered
    '''
    
    time = single_lecture(path, filenames[0], 'time')

    Data_or, Data_fr = [], []
    Data_frr = []
    
    for filename in filenames:
        Data_or.append([single_lecture(path, filename, obj) for obj in lista])
        
    if len(settings) == 2:
        #_filtrage simple
        frq_cut, Napod = settings[0], settings[1]
        for data_or in Data_or:
            Data_fr.append([Filter_S(time, obj, frq_cut, Napod) for obj in data_or])
   
    elif len(settings) == 3:
        #_filtrage + lissage
        frq_cut, Napod, fl = settings[0], settings[1], settings[2]
        for data_or in Data_or:
            Data_frr.append([Filter_S(time, obj, frq_cut, Napod) for obj in data_or]) 
        for data_frr in Data_frr:
            Data_fr.append([gaussian_filter(obj, fl)  for obj in data_frr]) 
            
#    elif len(settings) == 3:
#        #_filtrage2 (avec Filter_S)
#        frq_cut, Napod = settings[0], settings[1]
#        def derivfrint(F,t):
#            dx0=t[1]-t[0] #échantillonnage
#            dF = np.gradient(F,dx0)
#            dFfr = Filter_S(t, dF, frq_cut, Napod)
#            return primF
#            primF = np.zeros(len(F))
#            for k in range(0,len(t)):
#                primF[k] = F[0] + np.sum(dFfr[0:k])*dx0
#            return primF
#        for data_or in Data_or:
#            Data_fr.append([derivfrint(obj,time) for obj in data_or])
            
#    elif len(settings) == 4:
#        #_filtrage2 (avec gaussien_filter)
#        fl = settings[0]
#        def derivfrint(F,t):
#            dx0=t[1]-t[0] #échantillonnage
#            dF = np.gradient(F,dx0)
#            dFfr = gaussian_filter(dF,fl)
#            primF = np.zeros(len(F))
#            for k in range(0,len(t)):
#                primF[k] = F[0] + np.sum(dFfr[0:k])*dx0
#            return primF
#        for data_or in Data_or:
#            Data_fr.append([derivfrint(obj,time) for obj in data_or])    
     
    elif len(settings) == 1:
        #_lissage simple
        fl = settings[0]
        for data_or in Data_or:
            Data_fr.append([gaussian_filter(obj,fl) for obj in data_or])
            
    else:
        Data_fr = Data_or      
    
    return(time, Data_or, Data_fr)


# Fonction de calcul du gradient d'un scalaire
def Gradscal(rv,S):
    
    S1 = S[0]
    S2 = S[1]
    S3 = S[2]
    S4 = S[3]
    
    gradxS = rv[0]*S1 + rv[3]*S2 + rv[6]*S3 + rv[9]*S4      # comp 0 : dS/dx
    gradyS = rv[1]*S1 + rv[4]*S2 + rv[7]*S3 + rv[10]*S4     # comp 1 : dS/dy
    gradzS = rv[2]*S1 + rv[5]*S2 + rv[8]*S3 + rv[11]*S4     # comp 2 : dS/dz
    
    return(np.array([gradxS,gradyS,gradzS]))


# Fonction couleur = f(valeur D1)
def donner_couleur(x,y):
    couleurs=[]
    for i in range(len(x)):
        if 0.98 < y[i] < 1 : couleurs.append('#1f77b4') #bleu
        elif 0.95 < y[i] < 0.98 : couleurs.append('#8c564b') #marron
        elif 0.90 < y[i] < 0.95 : couleurs.append('#7f7f7f') #gris
        elif 0.85 < y[i] < 0.95 : couleurs.append('#9467bd') #mauve #
        elif 0.80 < y[i] < 0.85 : couleurs.append('#e377c2') #rose #
        elif 0.70 < y[i] < 0.80 : couleurs.append('#d62728') #rouge #
        elif 0.60 < y[i] < 0.70 : couleurs.append('#ff7f0e') #orange #
        else : couleurs.append('#2ca02c') #vert #
        #else : couleurs.append('w') #vert
    return couleurs




#D = fltrg(obj, cut, napods) #cut = Freqcut en Hz si on veut un filtrage, cut = Pntscut en nombre de points si on veut un lissage
#def fltrg(obj, cut, nbapodisation):
#    if napods==[]:
#        paramfr = gaussian_filter(obj,cut)
#    else:
#        paramfr = Filter_S(time, obj, cut)
#param = paramfr