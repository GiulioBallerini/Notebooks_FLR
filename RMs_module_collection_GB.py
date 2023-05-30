#   matches (without capitol letter) of functions or modules inside this file are indicated for future optimization.
#   Matches (with capitol letter) inside the whole amount of programs writtend by RM, can be easily visualized from terminal using the
#   following command:  grep -RIci "KEYWORD" . | awk -v FS=":" -v OFS="\t" '$2>0 { print $2, $1 }' | sort -hr
#   where KEYWORD is substituted with some word, for instance "KEYWORD" => "np"
#   Both matches and Matches are computed at the end of RM's PhD (exactly on 21/08/2019) for who will come after.
#   When functions are not written by RM, source links are provided.

###################################################################### import section ######################################################################

#__main modules                                                                      MATCHES INSIDE THE RMs MODULEs COLLECTION
import numpy as np                                                                              #....653 matches
import glob                                                                                     #....190 matches
import os                                                                                       #....135 matches
import datetime                                                                                 #....29  matches
import urllib                                                                                   #....1   matches
import matplotlib                                                                               #....6   matches
import math                                                                                     #....4   matches
import random as rnd                                                                            #....1   matches

#__sub-modules
from sympy import Function, simplify, symbols                                                   #....3,1,2  matches
from sympy import cos, sin                                                                      #....16,27  matches
from os import system                                                                           #....1      matches
from spacepy import pycdf                                                                       #....1      matches

#__sub-sub-modules
from scipy.fftpack import fft, ifft                                                             #....21,4   matches
from numpy.linalg import det, inv                                                               #....82,21  matches
from scipy.ndimage import gaussian_filter                                                       #....2      matches
import matplotlib.pyplot as plt                                                                 #....75     matches
import matplotlib.pyplot as plt2                                                                #....35     matches
from scipy.optimize import curve_fit                                                            #....10     matches
import scipy.fftpack as fftpack                                                                 #....9      matches
from scipy.special import factorial                                                             #....4      matches
from scipy.interpolate import interp1d                                                          #....30     matches
from matplotlib.colors import Normalize                                                         #....19     matches
from mpl_toolkits.mplot3d import Axes3D                                                         #....1      matches
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)        #....1,3,1      matches
from mpl_toolkits.axes_grid1 import *
from mpl_toolkits.axisartist import *
#
###################################################################### data analysis function section ######################################################################

def calcolo_distanza(array_riferimento, array_soggetto):    # 1 matches; 219 Matches
    '''
        Inputs & Outputs:
        Inputs:
            -) array_riferimento    have to contain elements like [time, space] and have to be numpy arrays.
            -) array_soggetto       have to contain elements like [time, space] and have to be numpy arrays.
        Outputs:
            -) std of the y distance between array_riferimento & array_soggetto
        Explanation:
            This function computes the standard deviation of the y distance between two arrays sharing roughly the same interval.
            The time array of reference is that of "array_riferimento", therefore for every point the function look at the closest point (in time)
            of "array_soggetto" to each point of "array_riferimento" and compute the distance between the y values of the two arrays.
    '''
    return(np.std([array_soggetto[np.argmin(abs(array_soggetto[:,0]-obj[0])),1]-obj[1] for obj in array_riferimento if obj[1]]))
############################################################

def set_axes_equal(ax):    # 1 matches; 1 Matches
    '''
    NOTE FROM RManuzzo:
        I did not write this module.
        Source: https://stackoverflow.com/a/31364297
    
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
####################################################################

class MidpointNormalize(Normalize):    # 1 matches; 29 Matches
    '''
        NOTE FROM RManuzzo:
        I did not write this module.
        Source: https://matplotlib.org/3.1.1/gallery/userdemo/colormap_normalizations_custom.html#sphx-glr-gallery-userdemo-colormap-normalizations-custom-py
        
        This module is needed to normalize the colorbar used in python contour plots (and others) in order to have the white colour of divergent color maps (cmaps) at zero.
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
####################################################################
    
def seleziona(array,indexes):    # 1 matches; 1 Matches
    '''
        An old but not deprecated way to select all the ith components of elements of a array indipendently by it is a numpy array or not.
        Possible alternative:
            -) numpy array:         array[:,i]
            -) not numpy array:     numpy.array(array)[:,i]
        '''
    return np.array([array[i] for i in indexes])


##########################################################################
def rotation_matrix(axis, theta):   # 1 matches; 30 Matches
    """
        NOTE FROM RManuzzo:
        I did not write this module.
        Source: https://stackoverflow.com/a/6802723
        
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
####################################################################

def get_key_0(s):   # 1 matches; 7 Matches
    '''
        This function is provided as argument to certain nnumpy plot functions (such as contour) in order to have code colors which depend by the 0th component of an array
        '''
    return(s[0])
####################################################################


def get_key_1(s):   # 1 matches; 17 Matches
    '''
        This function is provided as argument to certain nnumpy plot functions (such as contour) in order to have code colors which depend by the 1st component of an array
        '''
    return(s[1])
####################################################################


def get_key_2(s):   # 1 matches; 11 Matches
    '''
        This function is provided as argument to certain nnumpy plot functions (such as contour) in order to have code colors which depend by the 2nd component of an array
        '''
    return(s[2])
####################################################################

def exp(t,sig,obj): # 30 matches; a lot Matches
    '''
        Function especially used in the suppression of the singularities in the STD+ method.
        '''
    if sig > 0 or sig < 0:
        return(obj[1]*np.exp(-((t-obj[0])**2)/(sig**2)))
    else:
        return(obj[1]*np.exp(-((t-obj[0])**2))) # hp forte
####################################################################

def project(vec,direction):# 29 matches; a lot of Matches
    '''
        Function especially used in the suppression of the singularities in the STD+ method.
        '''
    new_vec = np.dot(vec,normalize(direction))
    return(new_vec)
####################################################################

def filtre(array, indexes):# 3 matches; 13 Matches
    '''
        Same as array[:,indexes] but for non-numpy arrays
        It may be deprecated but with caution.
        '''
    array_filtered = ([array[i] for i in indexes])
    return(array_filtered)
####################################################################
   
def normalize(v):
    '''
        Compute the normalization of v=[i,j,k,...] as norm_v = v/|v| where |v|=sqrt(i**2+j**2+k**2+...)
        '''
    norm = np.linalg.norm(v)
    if norm!=0: 
       return([obj/norm for obj in v])
    else:
        print('1/0')
####################################################################
 
def feedback(message, flag):
    '''
        Print message if flag is true.
        '''
    if flag:
        print(message)
####################################################################

def globator(name,data):
    '''
        This function transforms a string name in a variable which contains data.
        It is useful when multiple data contained in a general array have to be saved
        with different names. This function is used everytime MMS data are called in
        jupyter notebooks from txt tables.
        '''
    globals()[name]=data
####################################################################

def single_lecture(directory, txt_file, quantity):
    '''
        Explanation:
            Extraction of quantity from the txt_file contained in directory.
            The txt file needs to be organized as follows:
            
            time(s)  x(km)  y(km)  z(km)  bx(nT)  ...
            0.004936        53035.8573206        54238.6673723        -4495.46396245        14.9908981323       ...
            0.012748        53035.8506072        54238.6712833        -4495.46460595        15.001955986        ...
            0.020561        53035.8438929        54238.6751949        -4495.46524953        15.0196142197       ...
            ...             ...                  ...                  ...                   ...                 ...
            
            whereas the quantity need to be a string denoting the name of the variable (e.g. "time")
        
        Inputs:
            directory:
            txt_file:
            quantity:
        Output:
            a numpy array containig the numerical data of the column of txt_file under quantity.
        '''
    cwd = os.getcwd()
    os.chdir(directory)

    f = open(txt_file, mode='r')
    content = f.readlines()

    f.close()

    variables_names = [obj.split('(')[0] for obj in content[0].split(' ')[:-1] if obj]

    content_numed = [float(final_obj) for final_obj in np.array([[subobj for subobj in obj.split(' ') if subobj] for obj in content[1:]])[:,variables_names.index(quantity)]]

    return(np.array(content_numed))
    os.chdir(cwd)
####################################################################

def lecture(path, filenames, lista, settings):# 20 matches; ? Matches
    '''
        Explanation:
            Multiple application of single_lecture(...)
        Inputs:
            path:       address of the folder which contains all the txt_files where MMS data are stored in columns
            filenames:  array which contains the names (strings) of txt_files (for MMS1,2,3,4)
            lista:      array which contains the names (strings) of the variables that are needed to be extracted
            settings:   array which contains the frequency cut (double) above which data are filtered by means of the Filter_S(....) module and the Napod for Belmont's apodisation window Belmont_window(..)
        Outputs:
            time:       array which contains the time (the first column of the first file indicated in filenames)
            Data_or:    array of arrays which contains data extracted
            Data_fr:    same as Data_or but filtered
        '''
    time = single_lecture(path, filenames[0], 'time')

    Data_or, Data_fr = [], []
    for filename in filenames:
        Data_or.append([single_lecture(path, filename, obj) for obj in lista])
        
    if len(settings) == 2:
        #_filtrage
        frq_cut, Napod = settings[0], settings[1]
        for data_or in Data_or:
            Data_fr.append([Filter_S(time, obj, frq_cut, Napod) for obj in data_or])
    else:
        #_lissage
        fl = settings[0]
        for data_or in Data_or:
            Data_fr.append([gaussian_filter(obj,fl) for obj in data_or])
    
    return(time, Data_or, Data_fr)
####################################################################

def SpectrumLecturer(path,filename):    # 1 matches; ? Matches
    '''
        Explanation:
            Specific function to read txt_files which contains the spectrograms.
            Names of such files are similar to: mms1_SPCTR_Ions_20151016_130530_60.txt
            (with respect to txt files which contains standard temporal data such as mms1_GSE_20151016_130530_60.txt)
        Inputs:
            path:       address of the txt file
            filename:   name of the txt file
        Outputs:
            aa:         rectangular array which contains spectrogram
        '''
    cwd = os.getcwd()
    os.chdir(path)

    f = open(filename, mode='r')
    content = f.readlines()
    f.close()

    variables_names = [obj.split('(')[0] for obj in content[0].split(' ')[:-1] if obj]
    content_numed = np.array([[float(final_obj) for final_obj in np.array([[subobj for subobj in obj.split(' ') if subobj] for obj in content[1:]])[:,variables_names.index(quantity)]] for quantity in variables_names])

    appo = []
    for Bin in ['bin'+str(i) for i in np.arange(32)]:
        appo.append(single_lecture(path, filename, Bin))

    aa = np.zeros([32,len(appo[0])])
    for i in range(np.shape(aa)[0]):
        for j in range(np.shape(aa)[1]):
            aa[i, j] = appo[i][j]
        
    return(aa)
    os.chdir(cwd) 
####################################################################

def DateToSec(x_data_date,year,month,day):    # 1 matches; 273 Matches
    '''
        Explanation:
            It changes the origin of x_data_date from epoch to the midnight of the same day when data are recorded.
            Deprecated in MMSData_to_txt.ipynb in favour of a new definition.
        Inputs:
            x_data_date:    double which indicates the time interval from EPOCH to the instant when data have been recorded
            year:           string which indicates the year when data have been measured
            month:          string which indicates the month when data have been measured
            day:            string which indicates the day when data have been measured
        Outputs:
            x_data_sec:     time interval between the midnight of day and the instant when data have been recorded
        '''
    x_data_sec = []
    mid_night = datetime.datetime(year, month, day, 0, 0, 0, 0)
    for i in range(len(x_data_date)):
        dt = x_data_date[i] - mid_night
        x_data_sec.append(dt.microseconds * 1e-6 + dt.seconds + dt.days * 86400)
    return(x_data_sec)
####################################################################

def InterPolator(time,main_obj,comp,way,convfct):    # 1 matches; >100 Matches
    '''
        Explanation:
        Interpolation of the vector "main_obj[:,1]" (main_obj[:,0] being the time coordinate of main_obj[:,1]) to the "time" array. If main_obj[:,1] are arrays, this routine interpolates the component "comp" of "main_obj[:,1]". Interpolation kind is "way". Data are eventually modulated by the factor "convfct".
        '''
    tt  = [obj[0] for obj in main_obj]
    if np.shape(main_obj[0][1]) is ():
        data = [convfct*obj[1] for obj in main_obj]
    else:
        if np.shape(comp) is ():
            data = [convfct*obj[1][comp] for obj in main_obj]
        else:
            data = [convfct*obj[1][comp[0]][comp[1]] for obj in main_obj]
    data_interpol = interp1d(tt, data, kind=way) #_interpolation
    result = data_interpol(time)
    return(result)
####################################################################
    
def PutSpectrInMatrix(main_obj):    # 1 matches; 3 Matches
    '''
        Explanation:
            Other function that likewise SpectrumLecturer put an already read spectrogram in a matrix.
            To be deprecated with caution.
        '''
    tt = [main_obj[i][0] for i in range(len(main_obj))]
    appo = []
    for j in np.arange(32):
        oo = [main_obj[i][1][j] for i in range(len(main_obj))]
        data_interpol = interp1d(tt, oo, kind='linear') #_interpolation
        new_data = data_interpol(time)
        appo.append(new_data)

    aa = np.zeros([32,len(new_data)])

    for i in range(np.shape(aa)[0]):
        for j in range(np.shape(aa)[1]):
            aa[i, j] = appo[i][j]
    
    return(aa)
####################################################################

def Belmont_window(time,Napod):    # 4 matches; ? Matches
    '''
        Explanation:
            Create a Belmont apodisation window having same array length of time. It is used in Filter_S.
        Inputs:
            time:   double array which contains the temporal coordinate of data to be filtered
            Napod:  integer which sets of Belmont's window.
        Outputs:
            apodgb: double array which contains the apodisation profile
        '''
    apodgb=np.zeros_like(time)
    p=2*Napod+1
    binome=np.zeros(Napod+1)
    puis=np.zeros(Napod+1)
    coeffgb=np.zeros(Napod+1)
    valeur1=0.0
    valeur0=0.0
    for i in range(0,Napod+1):
        binome[i]= factorial(Napod)/(factorial(i)*factorial(Napod-i))
        puis[i]=2.*i+1.0
        coeffgb[i]=((-1)**i)/(2.0*i+1.0)
        valeur1=valeur1+binome[i]*coeffgb[i]*(-1.0)**puis[i]
        valeur0=valeur0+binome[i]*coeffgb[i]
    k=1.0/(valeur1-valeur0)
    c=valeur1/(valeur1-valeur0)  
    apodgb=apodgb+c
    t0=(time.min()+time.max()+time[1]-time[0])/2.0
    for i in range(0,Napod+1):
        apodgb=apodgb-k*(binome[i]*coeffgb[i]* (np.cos(2*np.pi*(time-t0)/(time.max()-time.min()+time[1]-time[0])))**(puis[i]))
    return apodgb
####################################################################

def Blackman(i,N,alpha):    # 11 matches; ? Matches
    '''
        Explanation:
            Blackman function that defines the Blackman apodisation window.
        Inputs:
        N:      total number of data sample of the signal to be filtered (it is an apodisation function).
        i:      integer defined in the 0<i<N interval which denotes the position where to evalueate the Blackman apodisation function.
        alpha:  double internal parameter of the Blackman apodisation function
        Outputs:
        ...:    the value of the Blackman apodisation function at position i.
        '''
    a0,a1,a2 = (1-alpha)/2, 1/2, alpha/2
    return(a0 - a1*np.cos((2*np.pi*i)/(N-1)) + a2*np.cos((4*np.pi*i)/(N-1)))
####################################################################

def Blackman_window(time,a0):    # 1 matches; ? Matches
    '''
        Explanation:
            It creates a Blackman apodisation function profile stored in an array which has the same length of time.
        Inputs:
            time:   double array which contains the temporal coordinate of data to be filtered
            a0:     double internal parameter of the Blackman apodisation function
        Outputs:
            window: double array which contains the Blackman apodisation function profile
        '''
    N = len(time)
    window = np.array([Blackman(obj,N,a0) for obj in np.arange(N)])
    return(window)
####################################################################

def Hamming(i,N,a0):    # 11 matches; ? Matches
    '''
        Explanation:
        Hamming function that defines the Hamming apodisation window.
        Inputs:
        N:      total number of data sample of the signal to be filtered (it is an apodisation function).
        i:      integer defined in the 0<i<N interval which denotes the position where to evalueate the Hamming apodisation function.
        a0:     double internal parameter of the Hamming apodisation function
        Outputs:
        ...:    the value of the Hamming apodisation function at position i.
        '''
    a1 = 1-a0
    return(a0 - a1*np.cos((2*np.pi*i)/(N-1)))
####################################################################

def Hamming_window(time,a0):    # 1 matches; ? Matches
    '''
        Explanation:
        It creates a Hamming apodisation function profile stored in an array which has the same length of time.
        Inputs:
        time:   double array which contains the temporal coordinate of data to be filtered
        a0:     double internal parameter of the Hamming apodisation function
        Outputs:
        window: double array which contains the Hamming apodisation function profile
        '''
    N = len(time)
    window = np.array([Hamming(obj,N,a0) for obj in np.arange(N)])
    return(window)
####################################################################

def apo_R(array,window):    # 6 matches; ? Matches
    '''
        Explanation:
            Modulation of array by means of window
        Inputs:
            array:      array to be modulated
            window:     apodisation window
        Outputs:
            array*window        ... Well, I wonder why I did it...
        '''
    return(array*window)
####################################################################

def deapo_R(array,window):    # 3 matches; ? Matches
    '''
        Explanation:
            Modulation of array by means of the point-by-point inverse of window
        Inputs:
            array:      array to be de-modulated
            window:     apodisation window
        '''
    dewindow = np.array([1/obj  for obj in window])
    return(array*dewindow)
####################################################################

def shueator(bz,dp,pos):    # 6 matches; ? Matches
    '''
            Computation of
                1) the spherical coordinates r,T,P of a point on the Shue paraboloid and
                2) the "normal" to the Shue surface in that point
            as a function of
                1) the Solar Wind magnetic field GSE z component "bz" (float, scalar),
                2) the dynamical pressure "dp" (float, scalar) and
                3) the GSE position of an observer (usually the MMS spacecraft) which belongs
                by definition to the line joining the point r,T,P to the center of the Earth.
            T,P are provided in degrees.
        '''
    #_symbolic computations for tangent and normal vectors
    t, o, p, a = symbols('t, o, p, a')
    ang_fct = ((2/(1+cos(t)))**a)
    ang_prt = [ang_fct*cos(t),ang_fct*sin(t),o]
    r_t = [obj.diff(t) for obj in ang_prt]
    normal_vec = np.cross(np.cross(ang_prt,r_t),r_t)
    
    alpha = (0.58-0.01*bz)*(1+0.01*dp)
    T = np.arccos(pos[0]/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2))
    normal = np.array([-obj.subs(o,0).subs(a,alpha).subs(t,T) for obj in normal_vec])
    
    #_rotation from the Shue to the GSE frame
    rot_mat = np.matrix([[1,0,0],[0,cos(p),-sin(p)],[0,sin(p),cos(p)]])
    #_last substitutions
    P = np.arcsin(pos[2]/np.sqrt(pos[1]**2+pos[2]**2))
    normal = [obj.subs(p,P) for obj in np.array(np.dot(rot_mat,normal))[0]]
    normal = normalize([float(obj) for obj in normal])
    #_distance computations from [0,0,0]_GSE to the surface at the Shue angle
    if bz >= 0:
        b = 0.013
    else:
        b = 0.14
    r0 = (11.4+b*bz)*(dp**(-1/6.6))
    r = r0 * ang_fct.subs(a,alpha).subs(t,T)
    
    return([r,180/np.pi * T,180/np.pi * P,normal])
####################################################################

# procedure for the MVA(B) computations
def MVA(vecx,vecy,vecz):    # 1 matches; ? Matches
    '''
        Explanation:
            Computation of the MVA eigenvalues "EVals" (array, lenght = 3, float) and eigenvectors "EVecs" (array of arrays, lenght = 3, sub-lenght = 3, floats)
            of MVA method as a function of the three arrays "vecx","vecy","vecz" (arrays, lenght = N, floats) containing respectively the three GSE components
            of a general vectorial quantity probed during a magnetopause crossing.
        '''
    bxm,bym,bzm = np.mean(vecx),np.mean(vecy),np.mean(vecz)
    bxxm = np.mean(np.array([a*b for a,b in zip(vecx,vecx)]))
    byym = np.mean(np.array([a*b for a,b in zip(vecy,vecy)]))
    bzzm = np.mean(np.array([a*b for a,b in zip(vecz,vecz)]))
    bxym = np.mean(np.array([a*b for a,b in zip(vecx,vecy)]))
    bxzm = np.mean(np.array([a*b for a,b in zip(vecx,vecz)]))
    byzm = np.mean(np.array([a*b for a,b in zip(vecy,vecz)]))

    mxx = bxxm - bxm*bxm
    mxy = bxym - bxm*bym
    mxz = bxzm - bxm*bzm
    myx = bxym - bym*bxm
    myy = byym - bym*bym
    myz = byzm - bym*bzm
    mzx = bxzm - bzm*bxm
    mzy = byzm - bzm*bym
    mzz = bzzm - bzm*bzm

    matrix = np.array([[mxx,mxy,mxz],[myx,myy,myz],[mzx,mzy,mzz]], np.float64)
    obj = np.linalg.eigh(matrix) #_Eigenvalues and eigenvectors of a Hermitian matrix

    EVals = obj[0] #_eigenvalues sorted as -> small medium large
    EVecs = [[float(obj) for obj in obj[1][:,0]],[float(obj) for obj in obj[1][:,1]],[float(obj) for obj in obj[1][:,2]]]
    
    return EVals, EVecs
####################################################################

def ReciprocalVectors(coord):
    '''
        Explanation:
            Computation of the reciprocal vectors
            
                kx1,ky1,kz1,kx2,ky2,kz2,kx3,ky3,kz3,kx4,ky4,kz4
                
            associated to the any multi-spacecraft missions and needed for MDD and STD methods
            as a function of the spacecraft positions
            
                x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
                
            contained in the array of arrays "coord", being any of x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4 an array of floats containing
            the respective spacecraft coordinates.
            
        '''
    x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4 = coord[0],coord[1],coord[2],coord[3],coord[4],coord[5],coord[6],coord[7],coord[8],coord[9],coord[10],coord[11]
    
    ax4 = (y2-y1)*(z3-z1) -(z2-z1)*(y3-y1)
    ay4 = (z2-z1)*(x3-x1) -(x2-x1)*(z3-z1)
    az4 = (x2-x1)*(y3-y1) -(y2-y1)*(x3-x1)

    v4 = ax4*(x4-x1) +ay4*(y4-y1) +az4*(z4-z1)
    kx4 = ax4/v4
    ky4 = ay4/v4
    kz4 = az4/v4

    ax3 = (y1-y4)*(z2-z4) -(z1-z4)*(y2-y4)
    ay3 = (z1-z4)*(x2-x4) -(x1-x4)*(z2-z4)
    az3 = (x1-x4)*(y2-y4) -(y1-y4)*(x2-x4)

    v3 = ax3*(x3-x4) +ay3*(y3-y4)+az3*(z3-z4)
    kx3 = ax3/v3
    ky3 = ay3/v3
    kz3 = az3/v3

    ax2 = (y4-y3)*(z1-z3) -(z4-z3)*(y1-y3)
    ay2 = (z4-z3)*(x1-x3) -(x4-x3)*(z1-z3)
    az2 = (x4-x3)*(y1-y3) -(y4-y3)*(x1-x3)

    v2 = ax2*(x2-x3) +ay2*(y2-y3)+az2*(z2-z3)
    kx2 = ax2/v2
    ky2 = ay2/v2
    kz2 = az2/v2

    ax1 = (y3-y2)*(z4-z2) -(z3-z2)*(y4-y2)
    ay1 = (z3-z2)*(x4-x2) -(x3-x2)*(z4-z2)
    az1 = (x3-x2)*(y4-y2) -(y3-y2)*(x4-x2)

    v1 = ax1*(x1-x2) +ay1*(y1-y2)+az1*(z1-z2)
    kx1 = ax1/v1
    ky1 = ay1/v1
    kz1 = az1/v1

    return([kx1,ky1,kz1,kx2,ky2,kz2,kx3,ky3,kz3,kx4,ky4,kz4])
####################################################################

def EspGradQ(rv,Q):
    '''
        Explanation:
            Computation of the gradients of any vector "Q" using the reciprocal vectors "rv"
            computed by the routine "ReciprocalVectors".
            Computation can be performed on a single acquisition or on a series of acquisitions.
        '''
    gQxx = rv[0]*Q[0] + rv[3]*Q[3] + rv[6]*Q[6] + rv[9]*Q[9]     # comp 0 : d(Qx)/dx
    gQxy = rv[0]*Q[1] + rv[3]*Q[4] + rv[6]*Q[7] + rv[9]*Q[10]    # comp 1 : d(Qy)/dx
    gQxz = rv[0]*Q[2] + rv[3]*Q[5] + rv[6]*Q[8] + rv[9]*Q[11]    # comp 2 : d(Qz)/dx
    gQyx = rv[1]*Q[0] + rv[4]*Q[3] + rv[7]*Q[6] + rv[10]*Q[9]    # comp 3 : d(Qx)/dy
    gQyy = rv[1]*Q[1] + rv[4]*Q[4] + rv[7]*Q[7] + rv[10]*Q[10]   # comp 4 : d(Qy)/dy
    gQyz = rv[1]*Q[2] + rv[4]*Q[5] + rv[7]*Q[8] + rv[10]*Q[11]   # comp 5 : d(Qz)/dy
    gQzx = rv[2]*Q[0] + rv[5]*Q[3] + rv[8]*Q[6] + rv[11]*Q[9]    # comp 6 : d(Qx)/dz
    gQzy = rv[2]*Q[1] + rv[5]*Q[4] + rv[8]*Q[7] + rv[11]*Q[10]   # comp 7 : d(Qy)/dz
    gQzz = rv[2]*Q[2] + rv[5]*Q[5] + rv[8]*Q[8] + rv[11]*Q[11]   # comp 8 : d(Qz)/dz
    
    return(np.array([gQxx,gQxy,gQxz,gQyx,gQyy,gQyz,gQzx,gQzy,gQzz]))
####################################################################

def ShiEarth(gb_obj):
    '''
        Explanation:
            heart of routine "ShiMachinery".
            It computes the eigenvalues "EVals" and the eigenvectors "EVecs" of matrix L = gb_obj . gb_obj^T (method MDD of Shi_2005)
            where "gb_obj" is the gradient of any vector given in input to the routine "EspGradQ".
        '''
    EVals,EVecs = [], []
    for i in np.arange(len(gb_obj[0])):
        mat = np.matrix(np.array([[gb_obj[0][i],gb_obj[1][i],gb_obj[2][i]],[gb_obj[3][i],gb_obj[4][i],gb_obj[5][i]],[gb_obj[6][i],gb_obj[7][i],gb_obj[8][i]]]))
        tenseurL = np.dot(mat,mat.T)
        obj = np.linalg.eigh(tenseurL) #_Eigenvalues and eigenvectors of a Hermitian matrix
        EVals.append(obj[0]) #_eigenvalues sorted as -> small medium large
        EVecs.append([[float(obj) for obj in obj[1][:,0]],[float(obj) for obj in obj[1][:,1]],[float(obj) for obj in obj[1][:,2]]])
    return(EVals,EVecs)
####################################################################

def ShiGenEarth(gb_obj,ge_obj):
    '''
        Explanation:
        Generalization of routine "ShiEarth".
        It computes the eigenvalues "EVals" and the eigenvectors "EVecs" of matrix L = (gb_obj,ge_obj) . (gb_obj,ge_obj)^T (method MDD of Shi_2005 +
        generalization discussed in Rezeau_2018) where "gb_obj" and "ge_obj" are the gradients of any two vectors given in input to the routine "EspGradQ".
        '''
    EVals,EVecs = [], []
    for i in np.arange(len(gb_obj[0])):
        mat = np.matrix(np.array([[gb_obj[0][i],gb_obj[1][i],gb_obj[2][i],ge_obj[0][i],ge_obj[1][i],ge_obj[2][i]],[gb_obj[3][i],gb_obj[4][i],gb_obj[5][i],ge_obj[3][i],ge_obj[4][i],ge_obj[5][i]],[gb_obj[6][i],gb_obj[7][i],gb_obj[8][i],ge_obj[3][i],ge_obj[4][i],ge_obj[5][i]]]))
        tenseurL = np.dot(mat,mat.T)
        obj = np.linalg.eigh(tenseurL) #_Eigenvalues and eigenvectors of a Hermitian matrix
        EVals.append(obj[0]) #_eigenvalues sorted as -> small medium large
        EVecs.append([[float(obj) for obj in obj[1][:,0]],[float(obj) for obj in obj[1][:,1]],[float(obj) for obj in obj[1][:,2]]])
    return(EVals,EVecs)
####################################################################

def JumpAdjust(vec):
    '''
        Explanation:
            Correction for jumps observed in the components of the MDD normal ("vec").
        '''
    i_array = np.arange(len(vec[:,0]))
    check = [1 for i in i_array]
    v = 1
    for i in i_array[1:]:
        if np.dot([vec[:,0][i-1],vec[:,1][i-1],vec[:,2][i-1]],[vec[:,0][i],vec[:,1][i],vec[:,2][i]]) < 0:
            v = -1*v
        check[i] *= v*check[i]
    for i in i_array:
        vec[i,0] = check[i]*vec[i,0]
        vec[i,1] = check[i]*vec[i,1]
        vec[i,2] = check[i]*vec[i,2]
    return(vec)
####################################################################

def ShueAdjust(Shue_normal,vec):
    '''
        Explanation:
            Simple inversion of the normal ("Shue_normal") to the Shue surface
            if it is antiparallel to a reference direction "vec".
        '''
    #Shue_normal = shueator(Bsw,DpSw,posSc)[3]
    check = [np.dot(obj,Shue_normal) for obj in vec]
    for i in range(len(check)):
        if check[i] < 0:
            vec[i,0] = - vec[i,0]
            vec[i,1] = - vec[i,1]
            vec[i,2] = - vec[i,2]
    return(vec)
####################################################################

def ZgseAdjust(vec):
    '''
        Explanation:
            Simple inversion of a vector ("vec")
            if it is antiparallel to the Z direction in GSE (0,0,1).
        '''
    check = [np.dot(obj,[0,0,1]) for obj in vec]
    for i in range(len(check)):
        if check[i] < 0:
            vec[i,0] = - vec[i,0]
            vec[i,1] = - vec[i,1]
            vec[i,2] = - vec[i,2]
    return(vec)
####################################################################

def DirAdjust(direction, vec):
    '''
        Explanation:
            Simple inversion of a vector "vec"
            if it is antiparallel to a reference "direction".
        '''
    dir_normed = normalize(direction)
    check = [np.dot(obj,dir_normed) for obj in vec]
    for i in range(len(check)):
        if check[i] < 0:
            vec[i,0] = - vec[i,0]
            vec[i,1] = - vec[i,1]
            vec[i,2] = - vec[i,2]
    return(vec)
####################################################################

def ShiMachinery(time,pos_field,mag_field,el_field,Denton,Jumps,Shue_normal,ZGSE,time_flag = ''): #....5 matches
    '''
        Explanation:
            Computation of the MDD (Shi_2005) eigenvalues ("l1","l2","l3") and the eigenvectors ("N","M","L")
            as a function of
                time:       double array which contains the temporal coordinate of data
                pos_field:  array of double arrays which contains the spatial coordinates of data.
                            Organization of the array: likewise the ReciprocalVectors routine input.
                mag_field:  array of double arrays which contains ANY vectorial data.
                            Organization of the array: likewise the ReciprocalVectors routine input.
            and where the following (optional) possibilities can be applied:
                el_field:   supplementary array of double arrays which contains ANY other vectorial data
                            in order to generalize (Rezeau_2018) the MDD method. el_field is not empty, ShiGenEarth is used. Otherwhise ShiGen is used.
                            If no generalization is wanted, el_field must be [].
                            Organization of the array: likewise mag_field.
                Denton:     boolean variable indicating wether to substract to grad(B) its mean.
                Jumps:      boolean variable indicating wether to fix the jumps observed in the MDD normal
                Shue_normal:boolean variable indicating wether to orient the MDD normal toward the same half-space indicated by the Shue normal.
                ZGSE:       boolean variable indicating wether to orient the MDD normal toward the same half-space indicated by [0,0,1] direction.
            the following parameters are taken into accout too:
                time_flag   string variable indicating what to join to the name of the output txt file which saves the output.
            The output are the MDD (Shi_2005) eigenvalues ("l1","l2","l3") and the eigenvectors ("N","M","L") which are saved also in a txt file adopting the same format for the MMS data txt files. The columns are time l1 l2 l3 Nx Ny Nz Mx My Mz Lx Ly Lz.
        '''
    i_array = np.arange(len(pos_field[0]))
    
    RecVecs = ReciprocalVectors(pos_field)
    
    #___B field
    gb_obj = EspGradQ(RecVecs,mag_field)
    
    if Denton:
        for i in range(len(gb_obj)):
            gb_obj[i] = gb_obj[i] - np.mean(gb_obj[i])    
    
    frob_B = max([np.linalg.norm([obj[i] for obj in gb_obj]) for i in i_array])
    for i in range(len(gb_obj)):
        gb_obj[i] = gb_obj[i]/frob_B #  Frobenius' Norm
   
    #___other field
    if el_field:
        ge_obj = EspGradQ(RecVecs,el_field)

        if Denton:
            for i in range(len(ge_obj)):
                ge_obj[i] = ge_obj[i] - np.mean(ge_obj[i]) 

        frob_E = max([np.linalg.norm([obj[i] for obj in ge_obj]) for i in i_array])
        for i in range(len(ge_obj)):
            ge_obj[i] = ge_obj[i]/frob_E
           
        
        w,v = ShiGenEarth(gb_obj,ge_obj)
        
    else:
        w,v = ShiEarth(gb_obj)
        
    l1 = np.array([obj[2] for obj in w]) #_array for the largest eigenvalues
    l2 = np.array([obj[1] for obj in w]) #_array for the medium eigenvalues
    l3 = np.array([obj[0] for obj in w]) #_array for the smallest eigenvalues
        
    N = np.array([[-obj[2][0],-obj[2][1],-obj[2][2]] for obj in v]) #_eigenvector associated to the largest  eigenvalue
    M = np.array([[-obj[1][0],-obj[1][1],-obj[1][2]] for obj in v]) #_eigenvector associated to the medium   eigenvalue
    L = np.array([[-obj[0][0],-obj[0][1],-obj[0][2]] for obj in v]) #_eigenvector associated to the smallest eigenvalue
    
    if Shue_normal:
        N = DirAdjust(Shue_normal,N)
        
    if ZGSE:
        L = DirAdjust([0,0,1],L)
        
    if Jumps:
        N = JumpAdjust(N) # ===> Per la N1
        M = JumpAdjust(M) # ===> Per la N2
        L = JumpAdjust(L) # ===> Per la N3
    
    if time_flag:
        filename = 'mms_dim_' + '_'.join(time_flag.split(' ')[-2:]) + '.txt'
        fw = open(filename, "w")
        fw.write('time λ1 λ2 λ3 n1x n1y n1z n2x n2y n2z n3x n3y n3z '+ '\n')
        for i,t in enumerate(time):
            array = [str(t)] + [str(obj) for obj in [l1[i],l2[i],l3[i],N[i,0],N[i,1],N[i,2],M[i,0],M[i,1],M[i,2],L[i,0],L[i,1],L[i,2]]] + ['\n']
            stringa = '  '.join(array)
            fw.write(stringa)  
        fw.close()
 
    return(l1,l2,l3,N,M,L)
####################################################################

def divPe_Computator(pos_field,Te):
    
    '''
        Explanation
            This function returns the div(Te) from multipoint (4) MMS measurements.
            Te can be substituted to any other tensor (it is only a name).
            Inputs are:
            -) the position of measurements ordered as:
            
                [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]

            where x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4 are double arrays containing the components of the positions of spacecraft.
            -) the Te tensors is organised as:
            
                Te = [
                    [Texx1fr,Teyy1fr,Tezz1fr,Texy1fr,Texz1fr,Teyz1fr],
                    [Texx2fr,Teyy2fr,Tezz2fr,Texy2fr,Texz2fr,Teyz2fr],
                    [Texx3fr,Teyy3fr,Tezz3fr,Texy3fr,Texz3fr,Teyz3fr],
                    [Texx4fr,Teyy4fr,Tezz4fr,Texy4fr,Texz4fr,Teyz4fr]
                     ]
    '''
    RVs = ReciprocalVectors(pos_field)

    dPexxdx = RVs[0]*Te[0][0] + RVs[3]*Te[1][0] + RVs[6]*Te[2][0] + RVs[9]*Te[3][0]
    dPeyxdy = RVs[1]*Te[0][3] + RVs[4]*Te[1][3] + RVs[7]*Te[2][3] + RVs[10]*Te[3][3]
    dPezxdz = RVs[2]*Te[0][4] + RVs[5]*Te[1][4] + RVs[8]*Te[2][4] + RVs[11]*Te[3][4]
    divPex = dPexxdx + dPeyxdy + dPezxdz

    dPexydx = RVs[0]*Te[0][3] + RVs[3]*Te[1][3] + RVs[6]*Te[2][3] + RVs[9]*Te[3][3]
    dPeyydy = RVs[1]*Te[0][1] + RVs[4]*Te[1][1] + RVs[7]*Te[2][1] + RVs[10]*Te[3][1]
    dPezydz = RVs[2]*Te[0][5] + RVs[5]*Te[1][5] + RVs[8]*Te[2][5] + RVs[11]*Te[3][5]
    divPey = dPexydx + dPeyydy + dPezydz

    dPexzdx = RVs[0]*Te[0][4] + RVs[3]*Te[1][4] + RVs[6]*Te[2][4] + RVs[9]*Te[3][4]
    dPeyzdy = RVs[1]*Te[0][5] + RVs[4]*Te[1][5] + RVs[7]*Te[2][5] + RVs[10]*Te[3][5]
    dPezzdz = RVs[2]*Te[0][2] + RVs[5]*Te[1][2] + RVs[8]*Te[2][2] + RVs[11]*Te[3][2]
    divPez = dPexzdx + dPeyzdy + dPezzdz

    divPe = np.array([[i,j,k] for i,j,k in zip(divPex,divPey,divPez)])
    return(divPe)
####################################################################

#fit cubique sur une longueur donnee ncub
def fit_cubic(x,y,ncub):
    '''
        Explanation:
            Filtering of data "y" (having x coordinate "x") by a cubic filter of order "ncub".
            Translation of an IDL implementation by Rezeau.
        '''
    yfit=[]
    dyfit=[]
    p=int(ncub/2)
    remplissage=np.zeros(p)
    for index in range(p, len(x)-p):
        xl = x[index-p : index+p+1]
        yl = y[index-p : index+p+1]
        res=np.polyfit(xl,yl,3)
        y0=res[0]*x[index]**3+res[1]*x[index]**2+res[2]*x[index]+res[3]
        y1=3*res[0]*x[index]**2+2*res[1]*x[index]+res[2]
        yfit.append(y0)
        dyfit.append(y1)
    yfit=np.concatenate((remplissage,yfit,remplissage))
    dyfit=np.concatenate((remplissage,dyfit,remplissage))
    return yfit, dyfit
####################################################################

def Filter_S_notrend(time, signal, frq_cut, Napod):
    '''
        Explanation:
            Filtering of "signal" (having x coordinate "time") by a Fourier cut of frequencies above "frq_cut".
            The "signal" is modulated by the Gerard's apodisation profile of order "Napod".
            "signal" is not de-trended before filtering (differently from what have been done in Filter_S).
        '''
    te, tlen = (time[-1]-time[0])/(len(time)), len(time)
    mirrored_signal = np.concatenate((signal[::-1], signal, signal[::-1],signal), axis=0)
    new_time = np.linspace(time[0],time[-1],4*tlen)
    window = Belmont_window(new_time,Napod)
    mirrored_signal_apoded = apo_R(mirrored_signal,window)
    spectrum, freq = fftpack.fft(mirrored_signal_apoded), fftpack.fftfreq(4*tlen,te)
    ind_cut = np.argmin(abs(freq-frq_cut))
    spectrum[ind_cut:4*tlen-ind_cut] = 0
    new_signal = deapo_R(ifft(spectrum).real,window)[tlen:2*tlen]
    return(new_signal)
####################################################################

def Filter_S(time, S, frq_cut, Napod):
    '''
        Explanation:
            Filtering of "signal" (having x coordinate "time") by a Fourier cut of frequencies above "frq_cut".
            The "signal" is modulated by the Gerard's apodisation profile of order "Napod".
            "signal" is de-trended before filtering (differently from what have been done in Filter_S_notrend).
        '''
    def func(x, a, b, c):
        return a + b * np.cos(omega/2 * x) + c * np.sin(omega/2 * x)
    #_subtraction of the trend
    omega = 2 * np.pi / (time[-1]-time[0])
    semi_amp, cst = abs(max(S)-min(S))/2, abs(np.mean(S))
    bound1 = [cst - semi_amp, -semi_amp, -semi_amp]
    bound2 = [cst + semi_amp, semi_amp, semi_amp]
    if bound1 == bound2:
        popt, pcov = curve_fit(func, time, S)
    else:
        popt, pcov = curve_fit(func, time, S, bounds=(bound1, bound2))
    S = S - func(time, *popt)
    #_mirroring and windowing
    te, tlen = (time[-1]-time[0])/(len(time)), len(time)
    S = np.concatenate((S[::-1], S, S[::-1],S), axis=0)
    new_time = np.linspace(time[0],time[-1],4*tlen)
    window = Belmont_window(new_time,Napod)
    S = apo_R(S,window)
    #_FFT and filtering
    spectrum, freq = fftpack.fft(S), fftpack.fftfreq(4*tlen,te)
    ind_cut = np.argmin(abs(freq-frq_cut))
    spectrum[ind_cut:4*tlen-ind_cut] = 0
    #_demirroring and adding the trend
    S = deapo_R(ifft(spectrum).real,window)[tlen:2*tlen]
    S = S + func(time, *popt)
    return(S)
####################################################################

def Filter_dSdt(time, S, frq_cut, Napod):
    '''
        Explanation:
            Filtering of the time derivative of "signal" (having x coordinate "time") by a Fourier cut of frequencies above "frq_cut".
            The "signal" is modulated by the Gerard's apodisation profile of order "Napod".
            "signal" is not de-trended before filtering.
        '''
    te, tlen = (time[-1]-time[0])/(len(time)), len(time)
    dSdt = np.gradient(S,te)
    dSdt_fr = Filter_S(time, dSdt, frq_cut, Napod)
    S_fr_bydt = np.cumsum(dSdt_fr)
    fct = abs(max(S)-min(S))/abs(max(S_fr_bydt)-min(S_fr_bydt))
    S_fr_bydt = fct * S_fr_bydt
    cst = np.mean(S)-np.mean(S_fr_bydt)
    S_fr_bydt = S_fr_bydt + cst
    return(S_fr_bydt)
####################################################################

def LNA(time, currents, mag_field, sel_fct, D1, D1_lim, direction, time_flag):
    '''
        Explanation:
            Implementation of the LNA technique (Rezeau_2018).
            Inputs:
                time:       float array which contains the temporal coordinate of data
                currents:   float array which contains the currents J = niVi-neVe
                mag_field:  float array which contains the magnetic field
                sel_fct:    float scalar indicating the threshold in temporal variations of the magnitude of B to select data
                D1:         float array which contains the D1 values ((l1-l2)/l1, with l1,l2 eigenvalues of MDD (Shi_2005), see Rezeau_2018)
                D1_lim:     float scalar indicating the threshold in D1
                direction:  float array which contains the direction indicating the half-space where to orientate the LNA normal (it uses DirAdjust)
                time_flag:  string variable indicating what to join to the name of the output txt file which saves the output.
            Outputs:
                LNA_time:   float array which contains the temporal coordinate of the LNA normal
                LNA_norm:   float array of arrays which contain the LNA normal
                txt file:   it contains in column form titled in the following way: time nx ny nz
        '''
    J = np.array([[i,j,k] for i,j,k in zip(currents[0],currents[1],currents[2])])
    
    step = (time[-1]-time[0])/len(time)
    dbxdt = np.gradient(mag_field[0],step)
    dbydt = np.gradient(mag_field[1],step)
    dbzdt = np.gradient(mag_field[2],step)
    dBdt = np.array([[i,j,k] for i,j,k in zip(dbxdt,dbydt,dbzdt)])
    
    #selection according to MDD dimensionality
    time = np.array([obj for obj,d1 in zip(time,D1) if d1 > D1_lim])
    dBdt = np.array([obj for obj,d1 in zip(dBdt,D1) if d1 > D1_lim])
    J = np.array([obj for obj,d1 in zip(J,D1) if d1 > D1_lim])
    
    #selection according to |db/dt|^2
    amplitude = np.array([np.linalg.norm(obj)**2 for obj in dBdt])
    amplitude_max = max(amplitude)
    LNA_norm = np.array([normalize(-np.cross(i,j)) for i,j,a in zip(J,dBdt,amplitude) if a > sel_fct * amplitude_max])
    LNA_time = np.array([t for t,a in zip(time,amplitude) if a > sel_fct * amplitude_max])
    
    if direction:
        LNA_norm = DirAdjust(direction,LNA_norm)
        
    filename = 'LNA_normal_' + '_'.join(time_flag.split(' ')[-2:]) + '.txt'
    fw = open(filename, "w")
    fw.write('time nx ny nz '+ '\n')
    for i,t in enumerate(LNA_time):
        array = [str(t)] + [str(obj) for obj in [LNA_norm[i,0],LNA_norm[i,1],LNA_norm[i,2]]] + ['\n']
        stringa = '  '.join(array)
        fw.write(stringa)  
    fw.close()
        
    return(LNA_time, LNA_norm)
####################################################################

def depth(time,pos_field,B_field,V_field,N,λ1,λ2,D1_lim,indexes_where_1D,global_norm,time_flag,control): #1 matches; 3 Matches.
    """
        Explanation:
            Old function to compute the spacecraft trajectory.
            To be deprecated with caution.
    """
    # advancing control
    print('Inizio',datetime.datetime.now().isoformat())
    #_computations for det(grad(B)) * dX/dt by the inversion procedure for dB = dX/dt . grad(B)
    RecVecs = ReciprocalVectors(pos_field)
    gb_obj = EspGradQ(RecVecs,B_field)
    gb_reorg = np.array([[[gb_obj[0][i],gb_obj[3][i],gb_obj[6][i]],[gb_obj[1][i],gb_obj[4][i],gb_obj[7][i]],[gb_obj[2][i],gb_obj[5][i],gb_obj[8][i]]] for i in np.arange(len(gb_obj[0]))])
    dets = np.array([det(matr) for matr in gb_reorg])
    indexes = [0] + indexes_where_1D + [len(time)-1]
    
    #_calcolo delle medie sui satelliti (mediati i valori e non i gradienti, di cui non sappiamo nulla)
    Fxfr = np.array([np.mean([i,j,k,r]) for i,j,k,r in zip(B_field[0],B_field[3],B_field[6],B_field[9])])
    Fyfr = np.array([np.mean([i,j,k,r]) for i,j,k,r in zip(B_field[1],B_field[4],B_field[7],B_field[10])])
    Fzfr = np.array([np.mean([i,j,k,r]) for i,j,k,r in zip(B_field[2],B_field[5],B_field[8],B_field[11])])
    
    mean_dt = np.mean(np.diff(time))
    dFx,dFy,dFz = np.gradient(Fxfr, mean_dt),np.gradient(Fyfr, mean_dt),np.gradient(Fzfr, mean_dt)
    
    #_inizio a eseguire i calcoli per det(grad(B)) * dX/dt
    ##_calcolo dappertutto l'inversione 
    loc_norm_steps = [[t, project(d*inv(matr).dot([x,y,z]),n)] for t,d,matr,n,x,y,z in zip(time,dets,gb_reorg,N,dFx,dFy,dFz)]
    ##_filtro solo i punti in cui ritengo si sia monodimensionali 
    loc_norm_steps = np.array(filtre(loc_norm_steps,indexes))
    ##_interpolo questi per sostituire le mancanze
    model = interp1d(loc_norm_steps[:,0], loc_norm_steps[:,1], kind='cubic')
    norm_steps = model(time)
    print(datetime.datetime.now().isoformat())
    #_computations of the exponential corrections needed for having corresponding zeros of dets and det(grad(B)) * dX
    corrections_steps = [loc_norm_steps[0]]
    corrections_indexes = [i for i,p,s in zip(np.arange(len(dets)-1),dets[:-1],dets[1:]) if p*s<0]
    print(datetime.datetime.now().isoformat())
    for i in corrections_indexes:
        to,m,q = time[i],(dets[i+1]-dets[i])/(time[i+1]-time[i]),dets[i]
        t = - q/m + to
        corrections_steps.append([t,model(t)])
    print(datetime.datetime.now().isoformat())
    corrections_steps.append(loc_norm_steps[-1])
    corrections_steps = np.array(corrections_steps)
    print(datetime.datetime.now().isoformat())
    finer_time = np.array(sorted(np.concatenate((corrections_steps[:,0], time), axis=0)))
    expwidths = np.diff(corrections_steps[:,0])/4
    widths = np.array([expwidths[0]]+[min(expwidths[i],expwidths[i+1]) for i in np.arange(len(expwidths))[:-1]]+[expwidths[-1]])
    print(datetime.datetime.now().isoformat())
    corrections = [np.sum([exp(t,w,obj) for obj,w in zip(corrections_steps,widths)]) for t in finer_time]
    norm_steps = model(finer_time)
    dets_improved = interp1d(time, dets, kind='linear')(finer_time)
    print(datetime.datetime.now().isoformat())
    #_subtraction of corrections on the det(grad(B)) * dX
    norm_steps_corrcted_old = norm_steps - corrections
    norm_steps_corrcted = np.array([i/j for i,j in zip(norm_steps_corrcted_old,dets_improved)])
    norm_steps_corrcted = interp1d(finer_time, norm_steps_corrcted, kind = 'cubic')(time)
    print(datetime.datetime.now().isoformat())
    
    #_ rimetto la costante dt
    norm_steps_corrcted = norm_steps_corrcted * mean_dt
    
    #_eventual projection toward a global normal
    if global_norm:
        steps = [step * np.dot(normalize(n),normalize(global_norm)) for step,n in zip(norm_steps_corrcted,N)]
        depth = np.cumsum(steps)
    else:
        depth = np.cumsum(norm_steps_corrcted)

    if time_flag:
        filename = 'depth_' + '_'.join(time_flag.split(' ')[-2:]) + '.txt'
        fw = open(filename, "w")
        fw.write('time(s) λ1(1) λ2(1) n1x(1) n1y(1) n1z(1) loc_norm_depth(km) depth(km) '+ '\n')
        for i,t in enumerate(time):
            array = [str(t)] + [str(obj) for obj in [λ1[i],λ2[i],N[i,0],N[i,1],N[i,2],norm_steps_corrcted[i],depth[i]]] + ['\n']
            stringa = '  '.join(array)
            fw.write(stringa)  
        fw.close()    

    if control:
        f, axarrs = plt.subplots(3,figsize=(10, 9),sharex = True)
        axarrs[0].plot(time, B_field[0], label='Other field X', linewidth = 1)
        axarrs[0].plot(time, B_field[1], label='Other field Y', linewidth = 1)
        axarrs[0].plot(time, B_field[2], label='Other field Z', linewidth = 1)
        axarrs[1].plot(time, dets, label='dets', linewidth = 1)
        axarrs[1].plot(loc_norm_steps[:,0], loc_norm_steps[:,1], ',', label='loc_norm_steps', linewidth = 1)
        axarrs[1].plot(finer_time, norm_steps, '--', label='loc_norm_steps_interpolated where 1D', linewidth = 1)
        axarrs[1].plot(finer_time, corrections, '--', label='corrections', linewidth = 1)
        axarrs[2].plot(time, np.cumsum(norm_steps_corrcted), label='norm_depth without projection', linewidth = 1)
        axarrs[2].plot(time, depth, label='norm_depth', linewidth = 1)
        for axarr in axarrs:
            axarr.legend(loc = 'upper left')
            axarr.grid()
        axarrs[0].set_ylabel('2^ field')
        axarrs[-1].set_ylabel('depths [km]')
        if time_flag:
            axarrs[-1].set_xlabel(time_flag)

        f.subplots_adjust(hspace=0.05)
        plt.show()
    print(datetime.datetime.now().isoformat())
    return(depth)
####################################################################

def zeros_corrector(time, treD_path_and_det, dets, kind, doyoureallywanttocorrrect):
    '''
        Explanation:
            Corrector for singularities during application of the STD method (Shi_2006).
        Inputs:
        time:                       float array which contains the temporal coordinate of data to be filtered
        treD_path_and_det:          3D GSE spacecraft path and coordinates and determinant of grad(B) (or any other quantity) (see pathfinder_STD)
        dets:                       Already contained in treD_path_and_det. To be ameliorated.
        kind:                       interpolation kind.
        doyoureallywanttocorrrect:  simple flag to avoid the correction.
        Outputs:
            3D GSE spacecraft corrected from any singularity.
        '''
    #_computations of the exponential corrections needed for having corresponding zeros of dets and det(grad(B)) * dX
    
    def comp_sel(array, ind):
        return np.array([obj[ind] for obj in array])
    
    comps_steps_corrcted = []
    all_corrections=[]
    for num_comp,comp in zip([0,1,2],['x','y','z']):
        #print('I am computing the ' + comp + ' component')
        corrections_steps = [[time[0],treD_path_and_det[0,num_comp]]]                                      # inputs: time&3Ddets
        corrections_indexes = [i for i,p,s in zip(np.arange(len(dets)-1),dets[:-1],dets[1:]) if p*s<0]     # input: dets
        model = interp1d(time, treD_path_and_det[:,num_comp], kind=kind) ## cambiato da cubic a linear
        #print('A',datetime.datetime.now().isoformat())
        for i in corrections_indexes:
            to,m,q = time[i],(dets[i+1]-dets[i])/(time[i+1]-time[i]),dets[i]
            t = - q/m + to
            corrections_steps.append([t,model(t)])
        #print('B',datetime.datetime.now().isoformat())
        corrections_steps.append([time[-1],treD_path_and_det[-1,num_comp]])
        #corrections_steps = np.array(corrections_steps)
        finer_time = np.array(sorted(np.concatenate((comp_sel(corrections_steps, 0), time), axis=0)))
        expwidths = np.diff(comp_sel(corrections_steps, 0))/4
        #print('C',datetime.datetime.now().isoformat())
        widths = np.array([expwidths[0]]+[min(expwidths[i],expwidths[i+1]) for i in np.arange(len(expwidths))[:-1]]+[expwidths[-1]])
        #print('D',datetime.datetime.now().isoformat())
        corrections = [np.sum([exp(t,w,obj) for obj,w in zip(corrections_steps,widths)]) for t in finer_time]
        comp_steps = model(finer_time)
        dets_improved = interp1d(time, dets, kind=kind)(finer_time)
        if comp is "x":
            f = open('/Users/manuzzo/Desktop' + '/zeros_corrector_x_output.txt', mode='a')
            for i,a in enumerate(finer_time):
                array = [str(a)] + [str(obj) for obj in [dets_improved[i],comp_steps[i],comp_steps[i] - corrections[i]]] + ['\n']
                stringa = '  '.join(array)
                f.write(stringa)
            f.close()
        ##_subtraction of corrections on the det(grad(B)) * dX
        if doyoureallywanttocorrrect:
            print('Sto correggendo ciao!')
            comp_steps_corrcted_old = comp_steps - corrections
        else:
            print('NON sto correggendo')
            comp_steps_corrcted_old = comp_steps
        comp_steps_corrcted = np.array([i/j for i,j in zip(comp_steps_corrcted_old,dets_improved)])
        all_corrections.append([[i,j] for i,j in zip(comp_steps, corrections)])
        comp_steps_corrcted = interp1d(finer_time, comp_steps_corrcted, kind = kind)(time)   ## cambiato da cubic a linear
        comps_steps_corrcted.append(comp_steps_corrcted)
    comps_steps_corrcted = np.array(comps_steps_corrcted)
    comps_steps_corrcted = np.array([[i,j,k] for i,j,k in zip(comps_steps_corrcted[0],comps_steps_corrcted[1],comps_steps_corrcted[2])])
    return(comps_steps_corrcted)
####################################################################

def pathfinder_STD(time,pos_field,B_field,V_field,D1_lim,LNA_fct, global_norm, timesxGlobNorm, kind, data_path, log_path, time0,time_flag, reallycorrection):#, specI = []):
    '''
        Explanation:
            Implementation of the STD+ method to compute the spacecraft trajectory across the magnetopause. See Manuzzo's thesis. By the way:
        Inputs:
            time:                   double array which contains the temporal coordinate of pos_field, B_field, V_field
            pos_field:              the position of measurements ordered as: [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]
            B_field:                the magnetic field measurements ordered as: [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]
            V_field:                *NOT USED* the magnetic field measurements ordered as: [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]
            D1_lim:                 float scalar indicating the threshold in D1
            LNA_fct:                float scalar indicating the threshold in temporal variations of the magnitude of B to select data
            global_norm:            float array which contains the direction indicating the half-space where to orientate the MDD normal
            timesxGlobNorm:         time interval within which to compute the MDD normal
            kind:                   interpolation kind
            data_path:              path where to find data
            log_path:               path where to keep a log file
            time0:                  time at which the MDD_glob_normal_depth (see output) is zero
            time_flag:              *NOT USED* anymore
            reallycorrection:       boolean variable to enable correction of singularities caused by the STD method
        Outputs:
            MDD_glob_normal_depth:  1D spacecraft trajectory along the normal to the magnetopause in arbitrary frame (format [t,x).
            MDD_GSE_trajectory:     3D spacecraft trajectory in GSE coordinate (format [t,[x,y,z])
            N:                      eigenvector of MDD method assosiated to the greatest eigenvalue of the MDD method
            M:                      eigenvector of MDD method assosiated to the medium eigenvalue of the MDD method
            L:                      eigenvector of MDD method assosiated to the smallest eigenvalue of the MDD method
            indexes_where_1D:
            λ1:                     greatest eigenvalue of the MDD method
            λ2:                     medium eigenvalue of the MDD method
            λ3:                     smallest eigenvalue of the MDD method
        '''
    message = 'STD+ Method begins' ; log_note(message,log_path)
    
    if not global_norm:
        BzSW = -1.2; PSW = 2
        message = ' '.join(['STD+ warning: I am computing the global norm with Shue. Are Bz_SW', str(BzSW) , ' and P_SW', str(PSW), ' the right features of the SW?'])  ; log_note(message,log_path)
        global_norm = shueator(BzSW,PSW,np.array(pos_field)[:3,0])[-1]

    Denton,Shue,Jumps,ZGSE = False,True,True,True
    λ1,λ2,λ3,N,M,L = ShiMachinery(time,pos_field,B_field,[],Denton,Jumps,[],ZGSE)
    N = DirAdjust(global_norm,N) # TO BE DONE: verify why I need it

    mean_dt = np.mean(np.diff(time))
    bxfr = (B_field[0]+B_field[3]+B_field[6]+B_field[9])/4
    byfr = (B_field[1]+B_field[4]+B_field[7]+B_field[10])/4
    bzfr = (B_field[2]+B_field[5]+B_field[8]+B_field[11])/4

    # selection of points according to the Rezeau(2017) (l1-l2)/l1 parameter
    indexes_where_1D = [i for i,l1,l2 in zip(np.arange(len(time)),λ1,λ2) if ((l1-l2)/l1>=D1_lim)]
    # selection of points according to |db/dt|^2 amplitude parameter
    dbdt2 = [np.linalg.norm([i,j,k]) for i,j,k in zip(np.gradient(bxfr,mean_dt),np.gradient(byfr,mean_dt),np.gradient(bzfr,mean_dt))]
    dbdt2_max = max(dbdt2)
    indexes_where_1D = [i for i in indexes_where_1D if dbdt2[i]>=LNA_fct * dbdt2_max]
    
    # qui calcolo una nuova global_norm alla luce della maggiore correttezza di MDD rispetto a Shue
    # seleziono un certo numero di Nloc per calcolare Nglob
    N_sel = np.array([N[ind] for ind in indexes_where_1D if time[ind]>=timesxGlobNorm[0] and time[ind]<=timesxGlobNorm[1]])
    if len(N_sel) > 0:
        #global_norm = [np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])]
        global_norm = normalize([np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])])
        #global_T1 = np.cross([0,0,1],global_norm)
        #global_T2 = np.cross(global_norm,global_T1)
        message = ' '.join(['Global norm (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(global_norm)]); log_note(message,log_path)
        #message = ' '.join(['Global T1 (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(global_T1)]); log_note(message,log_path)
        #message = ' '.join(['Global T2 (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(global_T2)]); log_note(message,log_path)
        
        # advancing control
        #_computations for det(grad(B)) * dX/dt by the inversion procedure for dB = dX/dt . grad(B)
        RecVecs = ReciprocalVectors(pos_field) #; pd_ind = 0; message = str(pd_ind); log_note(message,log_path)
        gb_obj = EspGradQ(RecVecs,B_field)#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        gb_reorg = np.array([[[gb_obj[0][i],gb_obj[3][i],gb_obj[6][i]],[gb_obj[1][i],gb_obj[4][i],gb_obj[7][i]],[gb_obj[2][i],gb_obj[5][i],gb_obj[8][i]]] for i in np.arange(len(gb_obj[0]))])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        dets = np.array([det(matr) for matr in gb_reorg])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        indexes = [0] + indexes_where_1D + [len(time)-1]#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        
        
        
        #_calcolo delle medie sui satelliti (mediati i valori e non i gradienti, di cui non sappiamo nulla)
        Fxfr = np.array([np.mean([i,j,k,r]) for i,j,k,r in zip(B_field[0],B_field[3],B_field[6],B_field[9])])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        Fyfr = np.array([np.mean([i,j,k,r]) for i,j,k,r in zip(B_field[1],B_field[4],B_field[7],B_field[10])])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        Fzfr = np.array([np.mean([i,j,k,r]) for i,j,k,r in zip(B_field[2],B_field[5],B_field[8],B_field[11])])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        
        mean_dt = np.mean(np.diff(time))#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        dFx,dFy,dFz = np.gradient(Fxfr, mean_dt),np.gradient(Fyfr, mean_dt),np.gradient(Fzfr, mean_dt)#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        
        # qui devo introdurre la correzione dB/dt - rot(UexB+...)
        #vedi relativo notebook officina
        
        # calcolo rotB = J per delimitare il periodo a cui si puo' applicare questa routine
        # rotB_x
        rotBx = (gb_obj[5] - gb_obj[7])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        rotBy = (gb_obj[6] - gb_obj[2])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        rotBz = (gb_obj[1] - gb_obj[3])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        rotB = np.array([np.linalg.norm([i,j,k]) for i,j,k in zip(rotBx,rotBy,rotBz)])/(4*np.pi*10**(-7))#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path) #SI
        
        #_inizio a eseguire i calcoli per dX/dt*det(grad(B))
        treD_path = np.array([inv(matr).dot([x,y,z]) for matr,x,y,z in zip(gb_reorg,dFx,dFy,dFz)])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        treD_path_and_det = np.array([d*tp3D for d,tp3D in zip(dets,treD_path)])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        #treD_path_and_det = np.array([d*inv(matr).dot([x,y,z]) for d,matr,x,y,z in zip(dets,gb_reorg,dFx,dFy,dFz)])
        
        #_computations of the exponential corrections needed for having corresponding zeros of dets and det(grad(B)) * dX
        comps_steps_corrcted = zeros_corrector(time, treD_path_and_det, dets, kind, reallycorrection)#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        
        MDD_GSE_vel = comps_steps_corrcted * mean_dt#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        time_where_1D = [time[i] for i in indexes_where_1D]#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        time_reduced = time[indexes_where_1D[0]:indexes_where_1D[-1]]
        MDD_loc_normal_vel = np.array([np.dot(MDD_GSE_vel[i],N[i])*N[i] for i in indexes_where_1D])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)

        MDD_loc_normal_vel_x = interp1d(time_where_1D,MDD_loc_normal_vel[:,0])(time_reduced)/1000#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path) ## so units of km/s!!
        MDD_loc_normal_vel_y = interp1d(time_where_1D,MDD_loc_normal_vel[:,1])(time_reduced)/1000#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path) ## so units of km/s!!
        MDD_loc_normal_vel_z = interp1d(time_where_1D,MDD_loc_normal_vel[:,2])(time_reduced)/1000#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path) ## so units of km/s!!
        MDD_loc_normal_vel = np.array([[i,j,k] for i,j,k in zip(MDD_loc_normal_vel_x,MDD_loc_normal_vel_y,MDD_loc_normal_vel_z)])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path) ## so units of km/s!!
        
        MDD_glob_normal_vel = np.array([np.dot(obj,global_norm) for obj in MDD_loc_normal_vel])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        cumsum = np.cumsum(MDD_glob_normal_vel)
        cumsum_at_time0 = cumsum[np.argmin(np.abs(time_reduced-time0))]
        MDD_glob_normal_depth = np.array([[i,j-cumsum_at_time0] for i,j in zip(time_reduced,cumsum)])#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        MDD_GSE_trajectory = [[t,[i,j,k]] for t,i,j,k in zip(time_reduced,np.cumsum(MDD_loc_normal_vel_x),np.cumsum(MDD_loc_normal_vel_y),np.cumsum(MDD_loc_normal_vel_z))]#; pd_ind = pd_ind + 1 ; message = str(pd_ind); log_note(message,log_path)
        
        return([MDD_glob_normal_depth, MDD_GSE_trajectory, N, M, L, indexes_where_1D, λ1,λ2,λ3])
    
    else:
        message = ' '.join(['STD+ warning: no MDD normals within the timesxGlobNorm period: ', str(timesxGlobNorm), ' s']) ; log_note(message,log_path)
        return(0)
####################################################################

def log_note(note,path):
    '''
        Explanation:
            Just a function to write a log
        '''
    if path:
        f = open(path + '/log_file_'+str(os.getpid())+'.txt', mode='a')
        f.write(note+"\n")
        f.close()
    else:
        print(note)
####################################################################

def pathfinder_MVF(time,pos_field,mag_field,V_field,D1_lim,LNA_fct, D_fct, num_point_fit, global_norm, timesxGlobNorm, kind, data_path, log_path, day, hour,  scelta_paletti_ignorante, evita_calcolo_fit_points, visualizzo_paletti, visualizzo_fit):
    '''
        Explanation:
            Implementation of the STD+ method to compute the spacecraft trajectory across the magnetopause. See Manuzzo's thesis. By the way:
            Inputs:
                time:               double array which contains the temporal coordinate of pos_field, B_field, V_field
                pos_field:          the position of measurements ordered as: [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]
                B_field:            the magnetic field measurements ordered as: [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]
                V_field:            *NOT USED* the magnetic field measurements ordered as: [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]
                D1_lim:             float scalar indicating the threshold in D1
                LNA_fct:            float scalar indicating the threshold in temporal variations of the magnitude of B to select data
                D_fct:              float scalar indicating the threshold in error D (see Manuzzo 2019)
                num_point_fit:      number of point to fit (see Manuzzo 2019, there called "p")
                global_norm:        float array which contains the direction indicating the half-space where to orientate the MDD normal
                timesxGlobNorm:     time interval within which to compute the MDD normal
                kind:               interpolation kind
                data_path:          path where to find data
                log_path:           path where to keep a log file
                day:                string to add at the txt file where to save the fits periods
                hour:               string to add at the txt file where to save the fits periods
                scelta_paletti_ignorante:   boolean value indicating wheter avoid the optimization of the length of the fits periods
                evita_calcolo_fit_points:   further boolean value indicating wheter avoid the optimization of the length of the fits periods
                visualizzo_paletti:         boolean value indicating wheter avoid the visualization of the intermediate procedures optimizing the choice of the fits intervals boundaries
                visualizzo_fit:             boolean value indicating wheter avoid the visualization of fits
            Outputs:
                depth:                          1D spacecraft path across the magnetopause
                indexes_where_1D:               array which contains the indexes of times where the magnetopause is 1D
                indexes_where_DmnrDp_MltVrt:    array which contains the indexes of times where the error D is less than the respective threshold
                D_array:                        array which contains the errors D
                N:                              array of arrays that contain the point-by-point normals
                rotB:                           array of arrays that contain the point-by-point computation of rot(B)
                dB_reco_array:                  array of arrays that contain the point-by-point computation of d(B)/dt_spacecraft
        '''

    log_note('MultiVariate Fit Method begins',log_path)
    
    def func(X,A,B):
        return A*X + B

    RecVecs = ReciprocalVectors(pos_field)
    gb_obj = EspGradQ(RecVecs,mag_field) #_[gQxx,gQxy,gQxz,gQyx,gQyy,gQyz,gQzx,gQzy,gQzz]
    gb_reorg = np.array([[[gb_obj[0][i],gb_obj[3][i],gb_obj[6][i]],[gb_obj[1][i],gb_obj[4][i],gb_obj[7][i]],[gb_obj[2][i],gb_obj[5][i],gb_obj[8][i]]] for i in np.arange(len(gb_obj[0]))])

    mean_dt = np.mean(np.diff(time))

    bxfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[0],mag_field[3],mag_field[6],mag_field[9])]
    byfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[1],mag_field[4],mag_field[7],mag_field[10])]
    bzfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[2],mag_field[5],mag_field[8],mag_field[11])]
    dbx,dby,dbz = np.gradient(bxfr_mean),np.gradient(byfr_mean),np.gradient(bzfr_mean)
    
    # reorganization of the reconstruction of dB observed
    db = np.array([[i,j,k] for i,j,k in zip(dbx,dby,dbz)])

    dbxdx,dbydx,dbzdx = gb_obj[0],gb_obj[1],gb_obj[2]
    dbxdy,dbydy,dbzdy = gb_obj[3],gb_obj[4],gb_obj[5]
    dbxdz,dbydz,dbzdz = gb_obj[6],gb_obj[7],gb_obj[8]

    # calcolo il rotore di B
    rotBx = dbzdy - dbydz ; rotBy = dbxdz - dbzdx ; rotBz = dbydx - dbxdy
    rotB = np.array([np.linalg.norm([i,j,k]) for i,j,k in zip(rotBx,rotBy,rotBz)])/(4*np.pi*10**(-7)) #SI


    # analisi MDD
    if not global_norm:
        BzSW = -1.2; PSW = 2
        message = ' '.join(['STD+ warning: I am computing the global norm with Shue. Are Bz_SW', str(BzSW) , ' and P_SW', str(PSW), ' the right features of the SW?'])
        log_note(message,log_path)
        global_norm = shueator(BzSW,PSW,np.array(pos_field)[:3,0])[-1]
    
    Denton,Shue,Jumps,ZGSE,time_flag = False,True,True,True,'_'
    λ1,λ2,λ3,N,M,L = ShiMachinery(time,pos_field,mag_field,[],Denton,Jumps,global_norm,ZGSE,time_flag)
    N = DirAdjust(global_norm,N)

    bxfr = (mag_field[0]+mag_field[3]+mag_field[6]+mag_field[9])/4
    byfr = (mag_field[1]+mag_field[4]+mag_field[7]+mag_field[10])/4
    bzfr = (mag_field[2]+mag_field[5]+mag_field[8]+mag_field[11])/4

    # selection of points according to the Rezeau(2017) (l1-l2)/l1 parameter

    indexes_where_1D = [i for i,l1,l2 in zip(np.arange(len(time)),λ1,λ2) if ((l1-l2)/l1>=D1_lim)]
    # selection of points according to |db/dt|^2 amplitude parameter
    dbdt2 = [np.linalg.norm([i,j,k]) for i,j,k in zip(np.gradient(bxfr,mean_dt),np.gradient(byfr,mean_dt),np.gradient(bzfr,mean_dt))]
    dbdt2_max = max(dbdt2)
    indexes_where_1D = [i for i in indexes_where_1D if dbdt2[i]>=LNA_fct * dbdt2_max]

    #Seleziono un certo numero di Nloc per calcolare Nglob
    N_sel = np.array([N[ind] for ind in indexes_where_1D if time[ind]>timesxGlobNorm[0] and time[ind]<timesxGlobNorm[1]])
    if len(N_sel) > 0:
        #N1_glob,N2_glob,N3_glob = np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])
        [N1_glob,N2_glob,N3_glob] = normalize([np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])])
        message = ' '.join(['Global norm (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str([N1_glob,N2_glob,N3_glob])]); log_note(message,log_path)
        #message = ' '.join(['Global T1 (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(N2_glob)]); log_note(message,log_path)
        #message = ' '.join(['Global T2 (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(N3_glob)]); log_note(message,log_path)
        

        def multivariate_heart(da,a,step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz):
            
            da = int(da) ; a = int(a)
            
            # computo i coefficienti del sistema di minimizzazione (vedi documento Lyx relativo)
            a11 = np.sum([i*j for i,j in zip(dbxdx[da:a],dbxdx[da:a])]) - (1/step) * np.sum(dbxdx[da:a])*np.sum(dbxdx[da:a])
            a12 = np.sum([i*j for i,j in zip(dbydx[da:a],dbydx[da:a])]) - (1/step) * np.sum(dbydx[da:a])*np.sum(dbydx[da:a])
            a13 = np.sum([i*j for i,j in zip(dbzdx[da:a],dbzdx[da:a])]) - (1/step) * np.sum(dbzdx[da:a])*np.sum(dbzdx[da:a])

            b11 = np.sum([i*j for i,j in zip(dbxdy[da:a],dbxdx[da:a])]) - (1/step) * np.sum(dbxdy[da:a])*np.sum(dbxdx[da:a])
            b12 = np.sum([i*j for i,j in zip(dbydy[da:a],dbydx[da:a])]) - (1/step) * np.sum(dbydy[da:a])*np.sum(dbydx[da:a])
            b13 = np.sum([i*j for i,j in zip(dbzdy[da:a],dbzdx[da:a])]) - (1/step) * np.sum(dbzdy[da:a])*np.sum(dbzdx[da:a])

            c11 = np.sum([i*j for i,j in zip(dbxdz[da:a],dbxdx[da:a])]) - (1/step) * np.sum(dbxdz[da:a])*np.sum(dbxdx[da:a])
            c12 = np.sum([i*j for i,j in zip(dbydz[da:a],dbydx[da:a])]) - (1/step) * np.sum(dbydz[da:a])*np.sum(dbydx[da:a])
            c13 = np.sum([i*j for i,j in zip(dbzdz[da:a],dbzdx[da:a])]) - (1/step) * np.sum(dbzdz[da:a])*np.sum(dbzdx[da:a])

            d11 = np.sum([i*j for i,j in zip(dbx[da:a],dbxdx[da:a])]) - (1/step) * np.sum(dbx[da:a])*np.sum(dbxdx[da:a])
            d12 = np.sum([i*j for i,j in zip(dby[da:a],dbydx[da:a])]) - (1/step) * np.sum(dby[da:a])*np.sum(dbydx[da:a])
            d13 = np.sum([i*j for i,j in zip(dbz[da:a],dbzdx[da:a])]) - (1/step) * np.sum(dbz[da:a])*np.sum(dbzdx[da:a])

            a1 = a11 + a12 + a13
            b1 = b11 + b12 + b13
            c1 = c11 + c12 + c13
            d1 = d11 + d12 + d13

            a21 = np.sum([i*j for i,j in zip(dbxdx[da:a],dbxdy[da:a])]) - (1/step) * np.sum(dbxdx[da:a])*np.sum(dbxdy[da:a])
            a22 = np.sum([i*j for i,j in zip(dbydx[da:a],dbydy[da:a])]) - (1/step) * np.sum(dbydx[da:a])*np.sum(dbydy[da:a])
            a23 = np.sum([i*j for i,j in zip(dbzdx[da:a],dbzdy[da:a])]) - (1/step) * np.sum(dbzdx[da:a])*np.sum(dbzdy[da:a])

            b21 = np.sum([i*j for i,j in zip(dbxdy[da:a],dbxdy[da:a])]) - (1/step) * np.sum(dbxdy[da:a])*np.sum(dbxdy[da:a])
            b22 = np.sum([i*j for i,j in zip(dbydy[da:a],dbydy[da:a])]) - (1/step) * np.sum(dbydy[da:a])*np.sum(dbydy[da:a])
            b23 = np.sum([i*j for i,j in zip(dbzdy[da:a],dbzdy[da:a])]) - (1/step) * np.sum(dbzdy[da:a])*np.sum(dbzdy[da:a])

            c21 = np.sum([i*j for i,j in zip(dbxdz[da:a],dbxdy[da:a])]) - (1/step) * np.sum(dbxdz[da:a])*np.sum(dbxdy[da:a])
            c22 = np.sum([i*j for i,j in zip(dbydz[da:a],dbydy[da:a])]) - (1/step) * np.sum(dbydz[da:a])*np.sum(dbydy[da:a])
            c23 = np.sum([i*j for i,j in zip(dbzdz[da:a],dbzdy[da:a])]) - (1/step) * np.sum(dbzdz[da:a])*np.sum(dbzdy[da:a])

            d21 = np.sum([i*j for i,j in zip(dbx[da:a],dbxdy[da:a])]) - (1/step) * np.sum(dbx[da:a])*np.sum(dbxdy[da:a])
            d22 = np.sum([i*j for i,j in zip(dby[da:a],dbydy[da:a])]) - (1/step) * np.sum(dby[da:a])*np.sum(dbydy[da:a])
            d23 = np.sum([i*j for i,j in zip(dbz[da:a],dbzdy[da:a])]) - (1/step) * np.sum(dbz[da:a])*np.sum(dbzdy[da:a])

            a2 = a21 + a22 + a23
            b2 = b21 + b22 + b23
            c2 = c21 + c22 + c23
            d2 = d21 + d22 + d23

            a31 = np.sum([i*j for i,j in zip(dbxdx[da:a],dbxdz[da:a])]) - (1/step) * np.sum(dbxdx[da:a])*np.sum(dbxdz[da:a])
            a32 = np.sum([i*j for i,j in zip(dbydx[da:a],dbydz[da:a])]) - (1/step) * np.sum(dbydx[da:a])*np.sum(dbydz[da:a])
            a33 = np.sum([i*j for i,j in zip(dbzdx[da:a],dbzdz[da:a])]) - (1/step) * np.sum(dbzdx[da:a])*np.sum(dbzdz[da:a])

            b31 = np.sum([i*j for i,j in zip(dbxdy[da:a],dbxdz[da:a])]) - (1/step) * np.sum(dbxdy[da:a])*np.sum(dbxdz[da:a])
            b32 = np.sum([i*j for i,j in zip(dbydy[da:a],dbydz[da:a])]) - (1/step) * np.sum(dbydy[da:a])*np.sum(dbydz[da:a])
            b33 = np.sum([i*j for i,j in zip(dbzdy[da:a],dbzdz[da:a])]) - (1/step) * np.sum(dbzdy[da:a])*np.sum(dbzdz[da:a])

            c31 = np.sum([i*j for i,j in zip(dbxdz[da:a],dbxdz[da:a])]) - (1/step) * np.sum(dbxdz[da:a])*np.sum(dbxdz[da:a])
            c32 = np.sum([i*j for i,j in zip(dbydz[da:a],dbydz[da:a])]) - (1/step) * np.sum(dbydz[da:a])*np.sum(dbydz[da:a])
            c33 = np.sum([i*j for i,j in zip(dbzdz[da:a],dbzdz[da:a])]) - (1/step) * np.sum(dbzdz[da:a])*np.sum(dbzdz[da:a])

            d31 = np.sum([i*j for i,j in zip(dbx[da:a],dbxdz[da:a])]) - (1/step) * np.sum(dbx[da:a])*np.sum(dbxdz[da:a])
            d32 = np.sum([i*j for i,j in zip(dby[da:a],dbydz[da:a])]) - (1/step) * np.sum(dby[da:a])*np.sum(dbydz[da:a])
            d33 = np.sum([i*j for i,j in zip(dbz[da:a],dbzdz[da:a])]) - (1/step) * np.sum(dbz[da:a])*np.sum(dbzdz[da:a])

            a3 = a31 + a32 + a33
            b3 = b31 + b32 + b33
            c3 = c31 + c32 + c33
            d3 = d31 + d32 + d33

            #risolvo C dX = T e calcolo finalmente i dX
            C = np.array([[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]])
            T = np.array([d1,d2,d3])
            dX = np.linalg.solve(C, T)

            # calcolo i dB0
            dB0x = np.array((np.sum(dbx[da:a]) - np.dot(dX, [np.sum(dbxdx[da:a]),np.sum(dbxdy[da:a]),np.sum(dbxdz[da:a])])))/step
            dB0y = np.array((np.sum(dby[da:a]) - np.dot(dX, [np.sum(dbydx[da:a]),np.sum(dbydy[da:a]),np.sum(dbydz[da:a])])))/step
            dB0z = np.array((np.sum(dbz[da:a]) - np.dot(dX, [np.sum(dbzdx[da:a]),np.sum(dbzdy[da:a]),np.sum(dbzdz[da:a])])))/step
            # ricostruisco dB
            dBx_reco = [i*dX[0]+j*dX[1]+k*dX[2]+dB0x  for i,j,k in zip(dbxdx[da:a], dbxdy[da:a], dbxdz[da:a])]
            dBy_reco = [i*dX[0]+j*dX[1]+k*dX[2]+dB0y  for i,j,k in zip(dbydx[da:a], dbydy[da:a], dbydz[da:a])]
            dBz_reco = [i*dX[0]+j*dX[1]+k*dX[2]+dB0z  for i,j,k in zip(dbzdx[da:a], dbzdy[da:a], dbzdz[da:a])]
            # calcolo un errore globale
            Dx = np.sum([(obj-z)**2 for obj,z in zip(dBx_reco,dbx[da:a])])
            Dy = np.sum([(obj-z)**2 for obj,z in zip(dBy_reco,dby[da:a])])
            Dz = np.sum([(obj-z)**2 for obj,z in zip(dBz_reco,dbz[da:a])])

            dB_mean = np.mean([np.linalg.norm([i,j,k]) for i,j,k in zip(dbx[da:a],dby[da:a],dbz[da:a])])
            DnotNorm = np.sqrt((Dx + Dy + Dz)/step)#/dB_mean
            DNorm = np.sqrt((Dx + Dy + Dz)/step)/dB_mean

            return(dX, dB0x, dB0y, dB0z, dBx_reco, dBy_reco, dBz_reco, DnotNorm, DNorm)

        # inizializzo le quantita' che serviranno per i fits
        dX_array,dX_array_Err_p,dB0_array,D_array,dB_reco_array,dB0_reco_array,all_array,indexes_where_DmnrDp_MltVrt,dB0ns_prop,std_errors_n_array,std_errors_m_array,std_errors_l_array = [],[],[],[],[],[],[],[],[],[],[],[]
        step = num_point_fit
        
        # decido i boundaries per i fits
        if scelta_paletti_ignorante:
            # qui setto i paletti ogni tot, indipendentemente dai dati
            da_array = np.arange(0,len(time),step,dtype=int)
            da_a_array = [[obj, obj + step] for obj in da_array[:-1]]
        else:
            path = data_path + '/fit_steps/multifit/paletti'
            os.chdir(path)
            noise_file = glob.glob('fit_periods_*')
            if noise_file and evita_calcolo_fit_points:
                message = 'I am reading the fit_periods from a file' ; log_note(message,log_path)
                da_a_array = np.load('fit_periods_' + day + '_' + hour + '.npy')
            else:
                message = 'I am creating ex novo the fit_periods' ; log_note(message,log_path)
                # qui invece scelgo la loro posizione in base ai dati
                #print('Entro nel while')
                min_p = num_point_fit
                # while cicle for the Multivariate fit
                MTAD_D_array, not_considered, new_da_a_array  = [], [], []
                da_a_array = [[0,len(time)//2],[len(time)//2,len(time)]]
                condition = True
                j = 0
                while condition:
                    not_considered, new_da_a_array, ind_min_MTAD_D_new_born = [], [], []
                    for i,obj in enumerate(da_a_array):
                        if obj[1]-obj[0] > 2 * min_p:
                            fit_indexes = [obj[0]+min_p,obj[1]-min_p]
                            #print('Fitto tra gli indici ', fit_indexes)
                            MTAD_D_array = []
                            fit_range = np.arange(fit_indexes[0],fit_indexes[1])
                            for a in fit_range:
                                DSum = 0
                                #print('fitto intervallo ', [obj[0],a])
                                dX, dB0x, dB0y, dB0z, dBx_reco, dBy_reco, dBz_reco, DnotNorm, D = multivariate_heart(obj[0],a,step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz)
                                DSum += DnotNorm
                                #print('fitto intervallo ', [a,obj[1]])
                                dX, dB0x, dB0y, dB0z, dBx_reco, dBy_reco, dBz_reco, DnotNorm, D = multivariate_heart(a,obj[1],step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz)
                                DSum += DnotNorm
                                MTAD_D_array.append(DSum)
                            minimo = np.argmin(MTAD_D_array)
                            ind_min_MTAD_D = obj[0] + min_p + minimo
                            ind_min_MTAD_D_new_born.append(ind_min_MTAD_D)
                            if visualizzo_paletti:
                                fig, ax1 = plt.subplots()
                                ax1.plot(np.arange(len(time)),time)
                                ax1.plot(np.arange(obj[0],obj[1]),[time[k] for k in np.arange(obj[0],obj[1])])
                                ax1.plot([obj for obj in np.array(da_a_array)[:,0]],[time[obj] for obj in np.array(da_a_array)[:,0]], '+', color = 'red')
                                ax1.plot(ind_min_MTAD_D_new_born,[time[k] for k in ind_min_MTAD_D_new_born], '+', color = 'yellow')
                                ax1.plot(ind_min_MTAD_D,time[ind_min_MTAD_D], '+', color = 'green')
                                ax1.set_title('Analysed periods visualisation')
                                ax1.set_xlabel('indexes'); ax1.set_ylabel('time'); ax1.grid()
                                ax2 = plt.axes([0,0,1,1])
                                ip = InsetPosition(ax1, [0.6,0.1,0.5,0.3])
                                ax2.set_axes_locator(ip)
                                ax2.plot(fit_range,MTAD_D_array)
                                ax2.set_title('DnotNorm Minimum @ ' + str(obj[0] + min_p +minimo))
                                ax2.set_ylabel('D')
                                ax2.grid()
                                ax2.ticklabel_format(axis='y', style='sci', scilimits=(-0,1))
                                ax2.set_xlim(obj)
                                j += 1 ; fig.savefig(data_path + '/fit_steps/multifit/considered_steps/'+ str(j) + ' time_considered' +str(obj[0]) + '_'+str(obj[1])+'.pdf')
                                plt.clf(); plt.cla(); plt.close()

                            new_da_a_array.append([obj[0],ind_min_MTAD_D])
                            new_da_a_array.append([ind_min_MTAD_D,obj[1]])
                            new_da_a_array.sort()
                        else:
                            new_da_a_array.append(obj)
                            not_considered.append(obj)
                    da_a_array = new_da_a_array
                    if len(not_considered) >= len(da_a_array):
                        #print('Esco dal while')
                        condition = False
                #print('Uscito dal while')
                filename = 'fit_periods_' + day + '_' + hour + '.npy'
                np.save(filename,da_a_array)

        # fitto
        for obj in da_a_array:
            da = int(obj[0])
            a = int(obj[1])#da + step

            # PRIMO METODO: multivariate fit in GSE
            dX, dB0x, dB0y, dB0z, dBx_reco, dBy_reco, dBz_reco, DnotNorm, D = multivariate_heart(da,a,step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz)
            
            taken = 'THREW due D > D_fct'
            # salvo i dati in array a parte per post analisi e seleziono di gia' a seconda che D < D_fct
            if da + step//2 < len(time) and D < D_fct:
                dX_array.append([time[int(da + step//2)],dX])
                dB0_array.append([time[int(da + step//2)],[dB0x,dB0y,dB0z]])
                D_array.append([time[int(da + step//2)],D])
                indexes_where_DmnrDp_MltVrt.append(da + step//2)
                taken = 'TAKEN due D < D_fct'
                dB_obs = [[i,j,k] for i,j,k in zip(dbx[da:a],dby[da:a],dbz[da:a])]
                dB_reco = [[i,j,k] for i,j,k in zip(dBx_reco,dBy_reco,dBz_reco)]
                #for index in np.arange(da,a):
                #if da + step <= len(time) and D < D_fct:
                dB_reco_array.append([time[int(da + step//2)],dB_obs,dB_reco,dX,[dB0x,dB0y,dB0z]])
            
            # visualizzo i fits nel frame "prop"
            # porto tutto nel sistema degli autovettori di gradB e li faccio il fit lineare per ogni componente
            ## calcolo gli autovettori e gli autovalori di gradB per i p punti di questo loop:
            matr = [[[dbxdx[ind], dbxdy[ind], dbxdz[ind]],[dbydx[ind], dbydy[ind], dbydz[ind]],[dbzdx[ind], dbzdy[ind], dbzdz[ind]]] for ind in np.arange(da,a)]
            eigens = [np.linalg.eigh(obj) for obj in matr]
            autovalori = [obj[0] for obj in eigens]
            matrici_di_rotazione = [obj[1] for obj in eigens]
            ## ruoto i punti sperimentali dbx,dby,dbz nel sistema di riferimento di gradB:
            db_prop = [np.dot([obj1x,obj1y,obj1z],obj2) for obj1x,obj1y,obj1z,obj2 in zip(dbx[da:a],dby[da:a],dbz[da:a], matrici_di_rotazione)]
            ## eseguo il fit per ciascuna componente:
            popt_n, cov_n = curve_fit(func, [obj[0] for obj in autovalori], [obj[0] for obj in db_prop])
            popt_m, cov_m = curve_fit(func, [obj[1] for obj in autovalori], [obj[1] for obj in db_prop])
            popt_l, cov_l = curve_fit(func, [obj[2] for obj in autovalori], [obj[2] for obj in db_prop])
            
            std_errors_n = np.sqrt(np.diag(cov_n)); std_errors_n_array.append([time[int(da + step//2)],std_errors_n])
            std_errors_m = np.sqrt(np.diag(cov_m)); std_errors_m_array.append([time[int(da + step//2)],std_errors_m])
            std_errors_l = np.sqrt(np.diag(cov_l)); std_errors_l_array.append([time[int(da + step//2)],std_errors_l])
            

            plt.clf(); plt.cla(); plt.close()
            if visualizzo_fit:
                for k,popt,std_errors,dir_lab in zip([0,1,2],[popt_n,popt_m,popt_l],[std_errors_n,std_errors_m,std_errors_l],['n','m','l']):
                    plt2.clf(); plt2.cla(); plt2.close()
                    plt2.plot([obj[k] for obj in autovalori],[obj[k] for obj in db_prop], '+', label = 'data points')
                    plt2.plot([obj[k] for obj in autovalori],popt[0] * np.array([obj[k] for obj in autovalori]) + popt[1], label = 'simple fit in Prop: '+str([[obj1,obj2] for obj1,obj2 in zip(["{:.1E}".format(obj) for obj in popt],["{:.1E}".format(obj) for obj in std_errors])]))
                    plt2.grid(); plt2.legend()
                    plt2.xlabel(r'$\partial B_{'+dir_lab+'}/\partial X_{'+dir_lab+'}$'); plt2.ylabel(r'$d B_{'+dir_lab+'}/d t$')
                    plt2.title(taken + r'$ \thickspace d B_{'+dir_lab+'}/dt \thickspace vs \thickspace \partial B_{'+dir_lab+'}/\partial X_{'+dir_lab+'} \thickspace @ \thickspace time: \thickspace $' + str(time[int(da + step//2)]) + ' s')
                    path = data_path + '/fit_steps/multifit/'+dir_lab+''
                    plt2.savefig(path + '/' + 'fit_dB'+dir_lab+'_'+str(time[int(da + step//2)])+'.pdf')
                    plt2.clf(); plt2.cla(); plt2.close()


        if len(indexes_where_DmnrDp_MltVrt) > 0 :
            indexes_where_DmnrDp_MltVrt = [int(obj) for obj in indexes_where_DmnrDp_MltVrt]
            
            if len(dX_array) > 2:
                dX_array_x_inter = np.concatenate((np.zeros(indexes_where_DmnrDp_MltVrt[0]), interp1d([obj[0] for obj in dX_array],[obj[1][0] for obj in dX_array], kind = kind)(time[indexes_where_DmnrDp_MltVrt[0]:indexes_where_DmnrDp_MltVrt[-1]]), np.zeros(len(time)-indexes_where_DmnrDp_MltVrt[1])), axis=0)
                dX_array_y_inter = np.concatenate((np.zeros(indexes_where_DmnrDp_MltVrt[0]), interp1d([obj[0] for obj in dX_array],[obj[1][1] for obj in dX_array], kind = kind)(time[indexes_where_DmnrDp_MltVrt[0]:indexes_where_DmnrDp_MltVrt[-1]]), np.zeros(len(time)-indexes_where_DmnrDp_MltVrt[1])), axis=0)
                dX_array_z_inter = np.concatenate((np.zeros(indexes_where_DmnrDp_MltVrt[0]), interp1d([obj[0] for obj in dX_array],[obj[1][2] for obj in dX_array], kind = kind)(time[indexes_where_DmnrDp_MltVrt[0]:indexes_where_DmnrDp_MltVrt[-1]]), np.zeros(len(time)-indexes_where_DmnrDp_MltVrt[1])), axis=0)
                dX_array_inter = [[i,j,k] for i,j,k in zip(dX_array_x_inter,dX_array_y_inter,dX_array_z_inter)]
                dX_array_globally_projected = [[time[ind],np.dot(N[ind],dX_array_inter[ind])*np.dot(N[ind],[N1_glob,N2_glob,N3_glob])] for ind in indexes_where_1D if ind in indexes_where_DmnrDp_MltVrt] # steps normali [km]
                if len(dX_array_globally_projected) > 2:
                    indice_di_restrizione_1 = np.argmin(abs(time-dX_array_globally_projected[0][0]))
                    indice_di_restrizione_2 = np.argmin(abs(time-dX_array_globally_projected[-1][0]))
                    if indice_di_restrizione_1 < indice_di_restrizione_2 and indice_di_restrizione_2 < len(time)-1:
                        time_restr = time[indice_di_restrizione_1:indice_di_restrizione_2]
                        
                        if len(dX_array_globally_projected) > 2:
                            dX_array_globally_projected_inter = np.array([[obj1,obj2] for obj1,obj2 in zip(time_restr,interp1d([obj[0] for obj in dX_array_globally_projected],[obj[1] for obj in dX_array_globally_projected], kind = kind)(time_restr))])
                            depth = np.array([[obj1,obj2/1000] for obj1,obj2 in zip(dX_array_globally_projected_inter[:,0],np.cumsum(dX_array_globally_projected_inter[:,1]))]) # unita' finale in km
                        
                            return([depth, indexes_where_1D, indexes_where_DmnrDp_MltVrt, D_array, N, rotB, dB_reco_array])
                        else:
                            message = 'MultiVariateFit Method warning: len(dX_array_globally_projected) <=2 ' ; log_note(message,log_path)
                            return(0)
                    else:
                        message = 'MultiVariateFit Method warning: ndice_di_restrizione_1 > indice_di_restrizione_2 or indice_di_restrizione_2 > len(time)-1 ' ; log_note(message,log_path)
                        return(0)
                else:
                    message = 'MultiVariateFit Method warning: len(dX_array_globally_projected) <= 2 ' ; log_note(message,log_path)
                    return(0)
    
            else:
                message = 'MultiVariateFit Method warning: len(dX_array) <=2 ' ; log_note(message,log_path)
                return(0)
        else:
            message = 'MultiVariateFit Method warning: no indexes_where_DmnrDp_MltVrt' ; log_note(message,log_path)
            return(0)
    else:
        message = ' '.join(['MultiVariateFit Method warning: no MDD normals within the timesxGlobNorm period: ', str(timesxGlobNorm), ' s']) ; log_note(message,log_path)
        return(0)
####################################################################

def pathfinder_SVF(time, pos_field, mag_field, D1_lim, LNA_fct, num_point_fit, DSVFlim, global_norm, timesxGlobNorm, time0, log_path):
    '''
        Explanation:
            See Manuzzo's thesis
            Inputs:
                time:               double array which contains the temporal coordinate of pos_field, B_field, V_field
                pos_field:          the position of measurements ordered as: [x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4]
                B_field:            the magnetic field measurements ordered as: [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]
                V_field:            *NOT USED* the magnetic field measurements ordered as: [bx1,by1,bz1,bx2,by2,bz2,bx3,by3,bz3,bx4,by4,bz4]
                D1_lim:             float scalar indicating the threshold in D1
                LNA_fct:            float scalar indicating the threshold in temporal variations of the magnitude of B to select data
                D_fct:              *NOT USED*
                num_point_fit:      number of point to fit (see Manuzzo 2019, there called "p")
                DSVFlim:            float scalar indicating the threshold in error D (see Manuzzo 2019)
                global_norm:        float array which contains the direction indicating the half-space where to orientate the MDD normal
                timesxGlobNorm:     time interval within which to compute the MDD normal
                time0:              time at which the MDD_glob_normal_depth (see output) is zero
                log_path:           path where to keep a log file
            Outputs:
                pathfinder_SVF_result:          1D spacecraft path across the magnetopause
                [obj[0] for obj in X_buoni]:    spatial steps for which D < DSVFlim
        '''
    def func(X,A,B):
        return A*X + B
    
    message = 'SingleVariate Fit Method begins' ; log_note(message,log_path)
    
    RecVecs = ReciprocalVectors(pos_field)
    gb_obj = EspGradQ(RecVecs,mag_field) #_[gQxx,gQxy,gQxz,gQyx,gQyy,gQyz,gQzx,gQzy,gQzz]
    
    mean_dt = np.mean(np.diff(time))
    bxfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[0],mag_field[3],mag_field[6],mag_field[9])]
    byfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[1],mag_field[4],mag_field[7],mag_field[10])]
    bzfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[2],mag_field[5],mag_field[8],mag_field[11])]
    dbx,dby,dbz = np.gradient(bxfr_mean),np.gradient(byfr_mean),np.gradient(bzfr_mean)
    # ricorda che gb_obj = gQxx (=dQx/dx),gQxy (=dQy/dx),gQxz (=dQz/dx),gQyx,gQyy,gQyz,gQzx,gQzy,gQzz
    dbxdx,dbydx,dbzdx = gb_obj[0],gb_obj[1],gb_obj[2]
    dbxdy,dbydy,dbzdy = gb_obj[3],gb_obj[4],gb_obj[5]
    dbxdz,dbydz,dbzdz = gb_obj[6],gb_obj[7],gb_obj[8]
    
    Denton,Shue,Jumps,ZGSE,time_flag = False,True,True,True,'_'
    λ1,λ2,λ3,N,M,L = ShiMachinery(time,pos_field,mag_field,[],Denton,Jumps,global_norm,ZGSE,time_flag)
    N = DirAdjust(global_norm,N)
    
    # selection of points according to the Rezeau(2017) (l1-l2)/l1 parameter
    indexes_where_1D = [i for i,l1,l2 in zip(np.arange(len(time)),λ1,λ2) if ((l1-l2)/l1>=D1_lim)]
    # selection of points according to |db/dt|^2 amplitude parameter
    dbdt2 = [np.linalg.norm([i,j,k]) for i,j,k in zip(np.gradient(bxfr_mean,mean_dt),np.gradient(byfr_mean,mean_dt),np.gradient(bzfr_mean,mean_dt))]
    dbdt2_max = max(dbdt2)
    indexes_where_1D = [i for i in indexes_where_1D if dbdt2[i]>=LNA_fct * dbdt2_max]
    
    #Seleziono un certo numero di Nloc per calcolare Nglob
    N_sel = np.array([N[ind] for ind in indexes_where_1D if time[ind]>timesxGlobNorm[0] and time[ind]<timesxGlobNorm[1]])
    if len(N_sel) > 0:
        [N1_glob,N2_glob,N3_glob] = normalize([np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])])
        message = ' '.join(['Global norm (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str([N1_glob,N2_glob,N3_glob])]); log_note(message,log_path)
        
        Ps = [] # dove staranno gli autovettori di gradB di ogni punto
        X_buoni = [] # dove stanno i risultati dei fits che hanno passato il test del fit
        
        for da in np.arange(0,len(time)-num_point_fit,num_point_fit):
            a = da + num_point_fit
            
            # in GSE:
            # Y = A * X + B dove
            # comp1: dBx/dt = {∂Bx/∂Xn,∂Bx/∂Xm,∂Bx/∂Xl} * {∂Xn/∂t,∂Xm/∂t,∂Xl/∂t}  + ∂Bx/∂t
            # comp2: ...
            # comp3: ...
            # e quindi:
            # Y = {dBx/dt,dBy/dt,dBz/dt},
            # X = {∂Xn/∂t,∂Xm/∂t,∂Xl/∂t}
            # B = {∂Bx/∂t,∂By/∂t,∂Bz/∂t}
            # A = {{∂Bx/∂Xn,∂Bx/∂Xm,∂Bx/∂Xl},
            #      {∂By/∂Xn,∂By/∂Xm,∂By/∂Xl},
            #      {∂Bz/∂Xn,∂Bz/∂Xm,∂Bz/∂Xl}}
            
            # preparo quindi la matrice gradB:
            matr = np.array([[[dbxdx[ind], dbxdy[ind], dbxdz[ind]],[dbydx[ind], dbydy[ind], dbydz[ind]],[dbzdx[ind], dbzdy[ind], dbzdz[ind]]] for ind in np.arange(da,a)])
            # voglio ora portare la relazione Y = A * X + B nel sistema di riferimento proprio di gradB.
            
            # inverto la relazione Y = A * X + B grazie all'inversa di A:
            # Y = A * X + B
            # Y = PDP^-1 * X + B
            # P^-1Y = P^-1PDP^-1 * X + P^-1B
            # Y' = DX' + B'
            # dove:
            # Y' = P^-1 Y => Y = P Y'
            # X' = P^-1 X => X = P X'
            # B' = P^-1 B => B = P B'
            # e D e' la matrice diagonale formata dagli autovalori Av1, Av2 e Av3 provenienti dalla diagonalizzazione di A
            # mentre P = ha le colonne come autovettori, P^-1 ne e' l'inversa!
            
            # quindi:
            # calcolo autovalori e autovettori di gradB:
            eigens = [np.linalg.eigh(obj) for obj in matr]
            # mi salvo a parte gli autovalori per ogni punto
            autovalori = np.array([obj[0] for obj in eigens])
            # mi salvo P per ogni punto:
            P = np.array([obj[1] for obj in eigens])
            for obj in P:
                Ps.append(obj)
            # e la sua inversa:
            Pinv = np.array([np.linalg.inv(obj) for obj in P])
            # costruisco quindi Y' e B':
            Yp = np.array([np.dot(m,[i,j,k]) for m,i,j,k in zip(Pinv,dbx[da:a],dby[da:a],dbz[da:a])])

            # eseguo quindi i 3 fits:
            popt0, cov0  = curve_fit(func,autovalori[:,0],Yp[:,0])
            popt1, cov1  = curve_fit(func,autovalori[:,1],Yp[:,1])
            popt2, cov2  = curve_fit(func,autovalori[:,2],Yp[:,2])

            # riordino i X' e i B' in due arrays:
            Xp = np.array([popt0[0],popt1[0],popt2[0]])
            Bp = np.array([popt0[1],popt1[1],popt2[1]])
            
            # e ora riporto Xp e Bp a X e B.
            X = np.array([np.dot(m,Xp) for m in P])
            B = np.array([np.dot(m,Bp) for m in P])
            
            # calcolo la std per il filtering dei risultati in base alla bonta' dei fits:
            StDevS = [(np.std(Yp[:,comp] - (autovalori[:,comp]*Xp[comp]+Bp[comp])))/np.mean(np.abs(Yp[:,comp])) for comp in [0,1,2]]
            meanStDev = np.mean(StDevS)
            
            if meanStDev > DSVFlim:
                mark = 'rejected'
            else:
                for ind,obj in enumerate(X):
                    X_buoni.append([da+ind,obj])
                mark = 'retained'
            
            if False:
                # check dei fits:
                fig = plt.figure(figsize=(10,10))
                grid = plt.GridSpec(3, 1, wspace=0.1, hspace=0.1)
                plt0 = fig.add_subplot(grid[0])
                plt1 = fig.add_subplot(grid[1], sharex=plt0)
                plt2 = fig.add_subplot(grid[2], sharex=plt0)
                comp = 0
                plt0.set_title('Comps in the gradB frame, mark: ' + mark + ' <D> < ' + str(DSVFlim))
                plt0.plot(time[da:a],Yp[:,comp], label = 'Yp comp' + str(comp))
                plt0.plot(time[da:a],autovalori[:,comp]*Xp[comp]+Bp[comp], label = 'DXp+Bp comp' + str(comp))
                plt0.legend(); plt0.grid(); plt0.tick_params(axis='x',top='on',bottom='on',labeltop='off',labelbottom='off')
                comp = 1
                plt1.plot(time[da:a],Yp[:,comp], label = 'Yp comp' + str(comp))
                plt1.plot(time[da:a],autovalori[:,comp]*Xp[comp]+Bp[comp], label = 'DXp+Bp comp' + str(comp))
                plt1.legend(); plt1.grid(); plt1.tick_params(axis='x',top='on',bottom='on',labeltop='off',labelbottom='off')
                comp = 2
                plt2.plot(time[da:a],Yp[:,comp], label = 'Yp comp' + str(comp))
                plt2.plot(time[da:a],autovalori[:,comp]*Xp[comp]+Bp[comp], label = 'DXp+Bp comp' + str(comp))
                plt2.set_xlabel(xflag)
                plt2.legend(); plt2.grid()
                path = data_path + '/fit_steps/singlefit/fits_in_prop'
                fig.savefig(path + '/' + 'fits_'+str(time[da])+'-'+str(time[a])+'.pdf')

    
        # proietto i {dX,dY,dZ} lungo la normale locale e globale solo dove esiste (selezionato secondo l'errore unitario D e dove la MP e' 1D)
        proiezione_appo = [[obj[0],time[obj[0]],np.dot(obj[1],N[obj[0]])*np.dot(N[obj[0]],[N1_glob,N2_glob,N3_glob])] for obj in X_buoni if obj[0] in indexes_where_1D]
        if len(proiezione_appo) > 2:
            # interpolo i risultati per riempire i buchi creatosi a causa della selezione
            time_reduced = time[proiezione_appo[0][0]:proiezione_appo[-1][0]]
            proiezione = interp1d([obj[1] for obj in proiezione_appo],[obj[2] for obj in proiezione_appo])(time_reduced)
            # integro e porto in unita' di km
            cumulative_sum = np.cumsum(proiezione)/1000
            # riordino e shifto in y a seconda di time0
            pathfinder_SVF_result = np.array([[i,j] for i,j in zip(time_reduced,cumulative_sum-cumulative_sum[np.argmin(abs(time_reduced-time0))])])
            return([pathfinder_SVF_result,[obj[0] for obj in X_buoni]])
        else:
            message = ' '.join(['SVF warning: the length of proiezione_appo is null']); log_note(message,log_path)
            return(0)
    else:
        message = ' '.join(['SingleVariateFit Method warning: no MDD normals within the timesxGlobNorm period: ', str(timesxGlobNorm), ' s']) ; log_note(message,log_path)
        return(0)
####################################################################

# sbagliata e riscritta nella routine pathfinder_SVF (vedi sopra):
def pathfinder_SinglesFit(time,pos_field,mag_field,V_field,D1_lim, LNA_fct, D_prop_fct, num_point_fit, global_norm, timesxGlobNorm, kind, data_path, log_path, day, hour,  scelta_paletti_ignorante, evita_calcolo_fit_points, visualizzo_paletti, visualizzo_fit):
    '''
        Explanation:
        SVF version to be deprecated BUT WITH CAUTION: it has been used several times. The new version is pathfinder_SVF.
        '''
    message = 'SingleVariate Fit Method begins' ; log_note(message,log_path)

    RecVecs = ReciprocalVectors(pos_field)
    gb_obj = EspGradQ(RecVecs,mag_field) #_[gQxx,gQxy,gQxz,gQyx,gQyy,gQyz,gQzx,gQzy,gQzz]
    gb_reorg = np.array([[[gb_obj[0][i],gb_obj[3][i],gb_obj[6][i]],[gb_obj[1][i],gb_obj[4][i],gb_obj[7][i]],[gb_obj[2][i],gb_obj[5][i],gb_obj[8][i]]] for i in np.arange(len(gb_obj[0]))])

    mean_dt = np.mean(np.diff(time))

    bxfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[0],mag_field[3],mag_field[6],mag_field[9])]
    byfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[1],mag_field[4],mag_field[7],mag_field[10])]
    bzfr_mean = [np.mean([i,j,k,s]) for i,j,k,s in zip(mag_field[2],mag_field[5],mag_field[8],mag_field[11])]
    dbx,dby,dbz = np.gradient(bxfr_mean),np.gradient(byfr_mean),np.gradient(bzfr_mean)
    
    # reorganization of the reconstruction of dB observed
    db = np.array([[i,j,k] for i,j,k in zip(dbx,dby,dbz)])

    dbxdx,dbydx,dbzdx = gb_obj[0],gb_obj[1],gb_obj[2]
    dbxdy,dbydy,dbzdy = gb_obj[3],gb_obj[4],gb_obj[5]
    dbxdz,dbydz,dbzdz = gb_obj[6],gb_obj[7],gb_obj[8]

    # calcolo il rotore di B
    rotBx = dbzdy - dbydz ; rotBy = dbxdz - dbzdx ; rotBz = dbydx - dbxdy
    rotB = np.array([np.linalg.norm([i,j,k]) for i,j,k in zip(rotBx,rotBy,rotBz)])/(4*np.pi*10**(-7)) #SI


    # analisi MDD    
    if not global_norm:
        BzSW = -1.2; PSW = 2
        message = ' '.join(['STD+ warning: I am computing the global norm with Shue. Are Bz_SW', str(BzSW) , ' and P_SW', str(PSW), ' the right features of the SW?'])
        log_note(message,log_path)
        global_norm = shueator(BzSW,PSW,np.array(pos_field)[:3,0])[-1]

    Denton,Shue,Jumps,ZGSE,time_flag = False,True,True,True,'_'
    λ1,λ2,λ3,N,M,L = ShiMachinery(time,pos_field,mag_field,[],Denton,Jumps,global_norm,ZGSE,time_flag)
    N = DirAdjust(global_norm,N)

    bxfr = (mag_field[0]+mag_field[3]+mag_field[6]+mag_field[9])/4
    byfr = (mag_field[1]+mag_field[4]+mag_field[7]+mag_field[10])/4
    bzfr = (mag_field[2]+mag_field[5]+mag_field[8]+mag_field[11])/4

    # selection of points according to the Rezeau(2017) (l1-l2)/l1 parameter

    indexes_where_1D = [i for i,l1,l2 in zip(np.arange(len(time)),λ1,λ2) if ((l1-l2)/l1>=D1_lim)]
    # selection of points according to |db/dt|^2 amplitude parameter
    dbdt2 = [np.linalg.norm([i,j,k]) for i,j,k in zip(np.gradient(bxfr,mean_dt),np.gradient(byfr,mean_dt),np.gradient(bzfr,mean_dt))]
    dbdt2_max = max(dbdt2)
    indexes_where_1D = [i for i in indexes_where_1D if dbdt2[i]>=LNA_fct * dbdt2_max]

    #Seleziono un certo numero di Nloc per calcolare Nglob
    N_sel = np.array([N[ind] for ind in indexes_where_1D if time[ind]>timesxGlobNorm[0] and time[ind]<timesxGlobNorm[1]])

    if len(N_sel) > 0:
        #N1_glob,N2_glob,N3_glob = np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])
        [N1_glob,N2_glob,N3_glob] = normalize([np.mean(N_sel[:,0]), np.mean(N_sel[:,1]), np.mean(N_sel[:,2])])
        message = ' '.join(['Global norm (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str([N1_glob,N2_glob,N3_glob])]); log_note(message,log_path)
        #message = ' '.join(['Global T1 (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(N2_glob)]); log_note(message,log_path)
        #message = ' '.join(['Global T2 (<MDD>), between t: ', str(timesxGlobNorm), ' is: ', str(N3_glob)]); log_note(message,log_path)

        def singlevariate_heart(dir_flag,da,a,step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz):
            def func(X,A,B):
                return A*X + B
            message = ' '.join(['comp: ',dir_flag])
            log_note(message,log_path)
            da,a,step = int(da),int(a),int(step)
            message = ' '.join(['da,a,step: ',str([da,a,step])])
            log_note(message,log_path)
            matr = [[[dbxdx[ind], dbxdy[ind], dbxdz[ind]],[dbydx[ind], dbydy[ind], dbydz[ind]],[dbzdx[ind], dbzdy[ind], dbzdz[ind]]] for ind in np.arange(da,a)]
            message = ' '.join(['gradB matrix [0]: ',str(matr[0])])
            #log_note(message,log_path)
            #eigens = [np.linalg.eigh(obj) for obj in matr]
            eigens = [np.linalg.eig(obj) for obj in matr]
            message = ' '.join(['gradB eigens [0]: ',str(eigens[0])])
            #log_note(message,log_path)
            autovalori = np.array([obj[0] for obj in eigens]).real
            message = ' '.join(['gradB eigenvalues [0]: ',str(autovalori[0])])
            #log_note(message,log_path)
            matrici_di_rotazione = [obj[1] for obj in eigens]
            autovec1 = np.array([obj[1][:,0] for obj in eigens]).real
            autovec2 = np.array([obj[1][:,1] for obj in eigens]).real
            autovec3 = np.array([obj[1][:,2] for obj in eigens]).real
            message = ' '.join(['rot matrix [0]: ',str(matrici_di_rotazione[0])])
            #log_note(message,log_path)
            message = ' '.join(['autovettore 1: ',str(autovec1)])
            #log_note(message,log_path)
            message = ' '.join(['autovettore 2: ',str(autovec2)])
            #log_note(message,log_path)
            message = ' '.join(['autovettore 3: ',str(autovec3)])
            #log_note(message,log_path)
            autovettori = [autovec1,autovec2,autovec3]
            comp_ind = ['n','m','l'].index(dir_flag)
            ## proietto i punti sperimentali dbx,dby,dbz lungo la direzione voluta:
            db_prop = np.array([np.dot([obj1x,obj1y,obj1z],obj2) for obj1x,obj1y,obj1z,obj2 in zip(dbx[da:a],dby[da:a],dbz[da:a], autovettori[comp_ind])])
            message = ' '.join(['db_prop: ',str(db_prop)])
            #log_note(message,log_path)
            ## eseguo il fit per ciascuna componente:
            
            xdata = np.array([obj[comp_ind] for obj in autovalori])
            message = ' '.join(['xdata: ',str(xdata)])
            #log_note(message,log_path)
            #ydata = np.array([obj for obj in db_prop])
            popt, cov = curve_fit(func, xdata, db_prop)
            message = ' '.join(['results from fit: ',str(popt)])
            #log_note(message,log_path)
            selautoval = [obj for obj in xdata]
            #seldb_prop = np.array([obj for obj in db_prop])
            D = np.std((db_prop-(popt[0]*xdata+popt[1]))) / np.abs(np.mean(db_prop))
            message = ' '.join(['D: ',str(D)])
            log_note(message,log_path)
            std_errors = np.sqrt(np.diag(cov))
            message = ' '.join(['std_errors: ',str(std_errors)])
            log_note(message,log_path)
            return(popt,std_errors,selautoval,db_prop,D)
        
        # inizializzo le quantita' che serviranno per i fits
        dX_array,dX_array_Err_p,dB0_array,D_array,dB_reco_array,dB0_reco_array,all_array,dXnsteps,dXmsteps,dXlsteps,dB0ns_prop,std_errors_n_array,std_errors_m_array,std_errors_l_array = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        indexes_where_DmnrDp_SnglVrt = [[],[],[]]
        step = num_point_fit
        da_a_array_superarray = []
        # decido i boundaries per i fits
        if scelta_paletti_ignorante:
            # qui setto i paletti ogni tot, indipendentemente dai dati
            da_array = np.arange(0,len(time),step)
            da_a_array = [[obj, obj + step] for obj in da_array[:-1]]
            da_a_array_superarray = [da_a_array,da_a_array,da_a_array]
        else:
            for direction_ind in ['n','m','l']:
                #print('Sto analizzando la direzione ',direction_ind)
                path = data_path + '/fit_steps/singlefit/'+direction_ind+'/paletti'
                os.chdir(path)
                noise_file = glob.glob('fit_periods_'+direction_ind+'_*')
                if noise_file and evita_calcolo_fit_points:
                    message = 'I am reading the fit_periods from a file' ; log_note(message,log_path)
                    da_a_array = np.load('fit_periods_'+direction_ind+ '_' + day + '_' + hour + '.npy')
                else:
                    message = 'I am creating ex novo the fit_periods for direction ' + direction_ind ; log_note(message,log_path)
                    # qui invece scelgo la loro posizione in base ai dati
                    #print('Entro nel while')
                    min_p = num_point_fit
                    # while cicle for the Singlevariate fit
                    MTAD_D_array, not_considered, new_da_a_array  = [], [], []
                    da_a_array = [[0,len(time)//2],[len(time)//2,len(time)]]
                    condition = True
                    j = 0
                    while condition:
                        not_considered, new_da_a_array, ind_min_MTAD_D_new_born = [], [], []
                        for i,obj in enumerate(da_a_array):
                            if obj[1]-obj[0] > 2 * min_p:
                                fit_indexes = [obj[0]+min_p,obj[1]-min_p]
                                #print('Fitto tra gli indici ', fit_indexes)
                                MTAD_D_array = []
                                fit_range = np.arange(fit_indexes[0],fit_indexes[1])
                                for a in fit_range:
                                    DSum = 0
                                    #print('fitto intervallo ', [obj[0],a])
                                    popt_n,std_errors,av,dbp,D = singlevariate_heart(direction_ind,obj[0],a,step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz)
                                    #DSum += std_errors[0]
                                    DSum += D
                                    #print('fitto intervallo ', [a,obj[1]])
                                    popt_n,std_errors,av,dbp,D = singlevariate_heart(direction_ind,a,obj[1],step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz)
                                    #DSum += std_errors[0]
                                    DSum += D
                                    MTAD_D_array.append(DSum)
                                minimo = np.argmin(MTAD_D_array)
                                ind_min_MTAD_D = obj[0] + min_p + minimo
                                #print(len(MTAD_D_array),obj[0],min_p,minimo,ind_min_MTAD_D)
                                ind_min_MTAD_D_new_born.append(ind_min_MTAD_D)
                                if visualizzo_paletti:
                                    fig, ax1 = plt.subplots()
                                    ax1.plot(np.arange(len(time)),time)
                                    ax1.plot(np.arange(obj[0],obj[1]),[time[k] for k in np.arange(obj[0],obj[1])])
                                    ax1.plot([obj for obj in np.array(da_a_array)[:,0]],[time[obj] for obj in np.array(da_a_array)[:,0]], '+', color = 'red')
                                    ax1.plot(ind_min_MTAD_D_new_born,[time[k] for k in ind_min_MTAD_D_new_born], '+', color = 'yellow')
                                    ax1.plot(ind_min_MTAD_D,time[ind_min_MTAD_D], '+', color = 'green')
                                    ax1.set_title('Analysed periods visualisation')
                                    ax1.set_xlabel('indexes'); ax1.set_ylabel('time'); ax1.grid()
                                    ax2 = plt.axes([0,0,1,1])
                                    ip = InsetPosition(ax1, [0.6,0.1,0.5,0.3])
                                    ax2.set_axes_locator(ip)
                                    ax2.plot(fit_range,MTAD_D_array)
                                    ax2.set_title('DnotNorm Minimum @ ' + str(obj[0] + min_p +minimo))
                                    ax2.set_ylabel('D')
                                    ax2.grid()
                                    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-0,1))
                                    ax2.set_xlim(obj)
                                    j += 1 ; fig.savefig(data_path + '/fit_steps/singlefit/'+direction_ind+'/considered_steps/'+ str(j) + ' time_considered' +str(obj[0]) + '_'+str(obj[1])+'.pdf')
                                    plt.clf(); plt.cla(); plt.close()

                                new_da_a_array.append([obj[0],ind_min_MTAD_D])
                                new_da_a_array.append([ind_min_MTAD_D,obj[1]])
                                new_da_a_array.sort()
                            else:
                                new_da_a_array.append(obj)
                                not_considered.append(obj)
                        da_a_array = new_da_a_array
                        if len(not_considered) >= len(da_a_array):
                            #print('Esco dal while')
                            condition = False
                    #print('Uscito dal while')
                    filename = 'fit_periods_'+ direction_ind +'_'+ day + '_' + hour + '.npy'
                    np.save(filename,da_a_array)
                da_a_array_superarray.append(da_a_array)

        # FITTO
        autovalori_n_array, autovalori_m_array, autovalori_l_array, db_prop_n_array, db_prop_m_array, db_prop_l_array = [], [], [], [], [], []
        Ds = [[],[],[]]

        for superarray_index,da_a_array,dir_lab in zip([0,1,2],da_a_array_superarray,['n','m','l']):
            #indexes_where_DmnrDp.append([int(np.mean(obj)) for obj in da_a_array if int(np.mean(obj)) < len(time)])
            for obj in da_a_array:
                da, a = obj[0], obj[1]

                popt,std_errors, autovalori, db_prop, D = singlevariate_heart(dir_lab,da,a,step,dbx,dby,dbz,dbxdx,dbxdy,dbxdz,dbydx,dbydy,dbydz,dbzdx,dbzdy,dbzdz)
                
                taken = 'THREW'
                if (da + step//2 < len(time)) and (D <= D_prop_fct): #(std_errors[0] <= D_prop_fct * abs(popt[0])):
                    taken = 'TAKEN'
                    message = taken
                    log_note(message,log_path)
                    log_note(str(popt),log_path)
                    indexes_where_DmnrDp_SnglVrt[superarray_index].append(da + step//2)
                    if superarray_index == 0:
                        dXnsteps.append([da + step//2,time[int(da + step//2)],popt[0]])
                    elif superarray_index == 1:
                        dXmsteps.append([da + step//2,time[int(da + step//2)],popt[0]])
                    elif superarray_index == 2:
                        dXlsteps.append([da + step//2,time[int(da + step//2)],popt[0]])
                
                if visualizzo_fit:
                    plt2.clf(); plt2.cla(); plt2.close()
                    plt2.plot(autovalori,db_prop, '+', label = 'data points')
                    plt2.plot(autovalori,popt[0] * np.array(autovalori) + popt[1], label = 'simple fit in Prop: '+str([[obj1,obj2] for obj1,obj2 in zip(["{:.1E}".format(obj) for obj in popt],["{:.1E}".format(obj) for obj in std_errors])]))
                    plt2.grid(); plt2.legend()
                    plt2.xlabel(r'$\partial B_{'+dir_lab+'}/\partial X_{'+dir_lab+'}$'); plt2.ylabel(r'$d B_{'+dir_lab+'}/d t$')
                    plt2.title(taken + r'$ d B_{'+dir_lab+'}/dt vs \partial B_{'+dir_lab+'}/\partial X_{'+dir_lab+'} @ time: $' + str(time[int(da + step//2)]) + ' s')
                    path = data_path + '/fit_steps/singlefit/'+dir_lab+dir_lab
                    plt2.savefig(path + '/' + 'fit_dB'+dir_lab+'_'+str(time[int(da + step//2)])+'.pdf')
                    plt2.clf(); plt2.cla(); plt2.close()

        #dXn_array = [[time[0],dXnsteps[0][1]]] + [obj for obj in dXnsteps] + [[time[-1],dXnsteps[-1][1]]]
        #dXm_array = [[time[0],dXmsteps[0][1]]] + [obj for obj in dXmsteps] + [[time[-1],dXmsteps[-1][1]]]
        #dXl_array = [[time[0],dXlsteps[0][1]]] + [obj for obj in dXlsteps] + [[time[-1],dXlsteps[-1][1]]]

        plt.plot([obj[1] for obj in dXnsteps],[obj[2] for obj in dXnsteps], label = 'dXnsteps')
        plt.plot([obj[1] for obj in dXmsteps],[obj[2] for obj in dXmsteps], label = 'dXmsteps')
        plt.plot([obj[1] for obj in dXlsteps],[obj[2] for obj in dXlsteps], label = 'dXlsteps')
        plt.show()
        if len(dXnsteps)>2 and len(dXmsteps)>2 and len(dXlsteps)>2:
            #ind_cuts = [dXnsteps[0][0],dXnsteps[-1][0],dXmsteps[0][0],dXmsteps[-1][0],dXlsteps[0][0],dXlsteps[-1][0]]
            #print(ind_cuts)
            #ind_cuts = np.sort(ind_cuts)
            #ind_cuts = ind_cuts[2:4]
            ind_cuts = [max([dXnsteps[0][0],dXmsteps[0][0],dXlsteps[0][0]]),min([dXnsteps[-1][0],dXmsteps[-1][0],dXlsteps[-1][0]])]
            #interpolo
            ind_cuts = [int(obj) for obj in ind_cuts]
            dXn_array_inter = np.concatenate((np.zeros(ind_cuts[0]), interp1d([obj[1] for obj in dXnsteps],[obj[2] for obj in dXnsteps], kind = kind)(time[ind_cuts[0]:ind_cuts[1]]), np.zeros(len(time)-ind_cuts[1])), axis=0)
            dXm_array_inter = np.concatenate((np.zeros(ind_cuts[0]), interp1d([obj[1] for obj in dXmsteps],[obj[2] for obj in dXmsteps], kind = kind)(time[ind_cuts[0]:ind_cuts[1]]), np.zeros(len(time)-ind_cuts[1])), axis=0)
            dXl_array_inter = np.concatenate((np.zeros(ind_cuts[0]), interp1d([obj[1] for obj in dXlsteps],[obj[2] for obj in dXlsteps], kind = kind)(time[ind_cuts[0]:ind_cuts[1]]), np.zeros(len(time)-ind_cuts[1])), axis=0)

            dX_array_inter = [[i,j,k] for i,j,k in zip(dXn_array_inter,dXm_array_inter,dXl_array_inter)]

            #-----#s
            #Proietto i dati lungo la MDD locale e globale
            dX_array_globally_projected = [[time[ind],np.dot(N[ind],dX_array_inter[ind])*np.dot(N[ind],[N1_glob,N2_glob,N3_glob])] for ind in indexes_where_1D if ind > ind_cuts[0] and ind < ind_cuts[1]]#> ind_cuts[0] and ind < ind_cuts[1]] # steps normali [km]
            # costruisco il time array ristretto all'intersezione di indexes_where_1D e ind_cuts
            time_restr = np.array([time[ind] for ind in indexes_where_1D if ind > ind_cuts[0] and ind < ind_cuts[1]])
            #time_restr = time[np.argmin(abs(time-dX_array_globally_projected[0][0])):np.argmin(abs(time-dX_array_globally_projected[-1][0]))]
            
            if len(dX_array_globally_projected) > 2:
                dX_array_globally_projected_inter = np.array([[obj1,obj2] for obj1,obj2 in zip(time_restr,interp1d([obj[0] for obj in dX_array_globally_projected],[obj[1] for obj in dX_array_globally_projected])(time_restr))])
                # interpolo prima di eseguire l'integrale temporale
                depth = np.array([[obj1,obj2/1000] for obj1,obj2 in zip([obj[0] for obj in dX_array_globally_projected_inter],np.cumsum([obj[1] for obj in dX_array_globally_projected_inter]))])

                return([depth, indexes_where_1D, indexes_where_DmnrDp_SnglVrt])
            else:
                message = 'SingleVariateFit Method warning: len(dX_array_globally_projected) <=2 ' ; log_note(message,log_path)
                return(0)
        else:
            message = 'SingleVariateFit Method warning: some between dXnsteps, dXmsteps or dXlsteps has length < 2' ; log_note(message,log_path)
            return(0)
    else:
        message = ' '.join(['SingleVariateFit Method warning: no MDD normals within the timesxGlobNorm period: ', str(timesxGlobNorm), ' s']) ; log_note(message,log_path)
        return(0)
####################################################################
#simulation function section 

def myload(Quantity,simu_paths):
    """
        Working example:
            Bx = myload('Bx',paths)
            where paths = [path1,path2,...]
            are paths of files *.npy
            containing data belonging to ordered and successives simulations.
            that will be merged.
    """
    print('Sto leggendo ' + Quantity)
    array = []
    for simu_path in simu_paths:
        print('    nella cartella ' + simu_path)
        os.chdir(simu_path)
        appo = np.load(Quantity + '.npy')
        array.append(appo)
    
    new_appo = []
    for ind,obj in enumerate(array):
        shape = np.shape(obj)
        for t in np.arange(shape[-1]):
            new_appo.append(obj[:,:,:,t])
    new_appo = np.array(new_appo)
    return new_appo
