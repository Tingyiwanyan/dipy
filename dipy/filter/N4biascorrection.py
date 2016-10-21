import numpy as np
import math
import scipy
import scipy.stats
from scipy.fftpack import fft,ifft
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
import scipy.ndimage as ndimage
from scipy import interpolate


#def Smoothing_field(f):


def N3biasfieldcorrection(t1_slic,log_t1,estimate_u,sample_f,numerator_u,h,expect,sample_rate,zero_pad_u):
    """
    compute fft and ifft
    """
    f_fft = fft(sample_f)
#    v_fft = fft(sample_v)
    u_fft = fft(estimate_u)

    """
    create winer filer to calculate estimate u
    """
    """
    for i in range(int(length) + int(zero_pad) + 100 - 1):
        estimate_u_fft[i] = (f_fft_conjugate[i]/(f_fft[i].real * f_fft[i].real + f_fft[i].imag * f_fft[i].imag + z * z)) * v_fft[i]

    estimate_u = ifft(estimate_u_fft)
    """
    """
    compute E[u|v]
    """

    numerator_ufft = fft(numerator_u)

    product_fft = f_fft * u_fft

    product_uf = numerator_ufft * f_fft

    numerator_e = ifft(product_uf)
    denominator_e = ifft(product_fft)

    for i in range(len(estimate_u)):
        if denominator_e[i].real == 0:
            expect[i] = 0
        else:
            expect[i] = numerator_e[i].real / denominator_e[i].real

    lengh_expect = len(expect)

    """
    compute bias field(2D)
    """
    f_new = np.zeros(shape = (t1_slic.shape[0],t1_slic.shape[1]),dtype = float)
    u_new = np.zeros(shape = (t1_slic.shape[0],t1_slic.shape[1]),dtype = float)
    f_real = np.zeros(shape = (t1_slic.shape[0],t1_slic.shape[1]),dtype = float)
    u_real = np.zeros(shape = (t1_slic.shape[0],t1_slic.shape[1]),dtype = float)
    for i in range(t1_slic.shape[0]):
        for j in range(t1_slic.shape[1]):
            index = math.floor((log_t1[i,j] - h[0]) / sample_rate)
            iindex = int(zero_pad_u) + index - 1
            if iindex > lengh_expect or iindex == lengh_expect:
                iindex = lengh_expect - 1
            e_single = expect[iindex]
            f_new[i,j] = log_t1[i,j] - e_single
#            u_new[i,j] = log_t1[i,j] - f_new[i,j]
#            f_real[i,j] = np.power(10,f_new[i,j])
#            u_real[i,j] = np.power(10,u_new[i,j])

#    for i in range(4):
    f_new = ndimage.gaussian_filter(f_new, sigma= 1)
#    f_new = ndimage.gaussian_filter(f_new, sigma= 1)
    """
    x_old = np.mgrid[0:160]
    y_old = np.mgrid[0:239]
    print(f_new.shape)
    print(x_old)
    print(y_old.shape)
    x_new , y_new = np.mgrid[0:160:320j,0:239:478j]
    tck = interpolate.bisplrep(x_old,y_old,f_new,s=0)
    f_new = interpolate.bisplev(x_new[:,0],y_new[0,:],tck)
    """


    for i in range(t1_slic.shape[0]):
        for j in range(t1_slic.shape[1]):
            u_new[i,j] = log_t1[i,j] - f_new[i,j]
            f_real[i,j] = np.power(10,f_new[i,j])
            u_real[i,j] = np.power(10,u_new[i,j])

    return u_real,f_real,u_new,f_new

def Initial_estimate(t1_slic):

    data = np.squeeze(t1_slic)

    b0_mask,mask = median_otsu(data, 2, 1)

    log_t1 = np.log10(t1_slic)

    data = t1_slic.reshape((t1_slic.shape[0] * t1_slic.shape[1],1))

    log_data = np.log10(data)

    g,h = np.histogram(log_data,bins = 100)

    g = g /(t1_slic.shape[0] * t1_slic.shape[1])
    #Dist_v = np.zeros((h[h.size-1],1))

    """
    Gaussian distribution function
    """
    pi = 3.1415926
    u = 0
    phi = (h[100] - h[0])/4

    sample = 100
    sample_rate = (4 * phi) / sample


    def Gaussain_distribution(x,u,phi):
        """
        return Gaussian distribution values
        """
        numerator = math.exp(-(x-u)*(x-u)/(2*phi*phi))
        denominator = phi*math.sqrt(2*pi)
        return numerator/denominator

    zero_pad = round((h[0] + 2 * phi) / sample_rate)
    zero_pad = int(zero_pad)

    length = (h[100]-h[0])/sample_rate

    sample_f = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    sample_v = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    estimate_u = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    f_fft_comp = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)

    numerator_e = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    denominator_e = np.zeros((int(length) + int(zero_pad) + 100 - 1,), dtype = float)
    numerator_u = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    product_ifft = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    expect = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)

    """
    Assign values to zero paded v signal
    """
    for i in range(100):
        sample_v[i + zero_pad] = g[i]

    """
    Create Gausian distribution for bias field
    """
    for i in range(100):
        estimate_u[i + zero_pad] = g[i]

    for i in range(100):
        sample_f[i] = Gaussain_distribution(-2 * phi + i*sample_rate, u, phi)

    for i in range(100):
        numerator_u[i + zero_pad] = estimate_u[i + zero_pad] * (h[0] + i * sample_rate)

    u_real,f_real,u_log,f_log = N3biasfieldcorrection(t1_slic,log_t1,sample_v,sample_f,numerator_u,h,expect,sample_rate,zero_pad)

    return u_real,f_real,u_log,f_log

def iterative_step(u_real,f_real,u_log,f_log,t1_slic):
    """
    iteratively implement the loop
    """
    data = np.squeeze(t1_slic)

    b0_mask,mask = median_otsu(data, 2, 1)

    log_t1 = np.log10(t1_slic)

    data = t1_slic.reshape((t1_slic.shape[0] * t1_slic.shape[1],1))

    log_data = np.log10(data)

    g_u,h_u = np.histogram(u_log,bins = 100)

    g_u = g_u /(t1_slic.shape[0] * t1_slic.shape[1])

    g_f,h_f = np.histogram(f_log,bins = 100)

    g_f = g_f /(t1_slic.shape[0] * t1_slic.shape[1])

    sample_rate = h_u[1] - h_u[0]
    sample_rate_f = h_f[1] - h_f[0]

    length = math.floor((f_log.max() - f_log.min())/sample_rate)

    if f_log.min() < u_log.min():
        zero_pad_u = round((u_log.min() - f_log.min())/sample_rate)
        zero_pad_u = int(zero_pad_u)
        zero_pad_f = 0
    if f_log.min() > u_log.min():
        zero_pad_f = round((f_log.min()-u_log.min())/sample_rate)
        zero_pad_f = int(zero_pad_f)
        zero_pad_u = 0

    zero_pad = max(zero_pad_u,zero_pad_f)
    sample_f = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    sample_v = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    estimate_u = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    f_fft_comp = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)

    numerator_e = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    denominator_e = np.zeros((int(length) + int(zero_pad) + 100 - 1,), dtype = float)
    numerator_u = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    product_ifft = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)
    expect = np.zeros((int(length) + int(zero_pad) + 100 - 1,),dtype = float)

    """
    Assign values to zero paded estimate u signal
    """
    for i in range(100):
        estimate_u[i + zero_pad_u] = g_u[i]


    """
    Assign values to estimate f
    """
    for i in range(length):
        index = math.floor(i * (sample_rate/sample_rate_f))
        sample_f[i + zero_pad_f] = g_f[index]

    for i in range(100):
        numerator_u[i + zero_pad_u] = estimate_u[i + zero_pad_u] * (h_u[0] + i * sample_rate)

    u_real,f_real,u_log,f_log = N3biasfieldcorrection(t1_slic,log_t1,estimate_u,sample_f,numerator_u,h_u,expect,sample_rate,zero_pad_u)

    return u_real,f_real,u_log,f_log
#def main():
dname = "/Users/tiwanyan/ANTs/Images"
t1_input = "/Raw/Q_0001_T1.nii.gz"

ft1 = dname + t1_input

t1 = nib.load(ft1).get_data()
affine_t1 = nib.load(ft1).affine

t1_slic = t1[:,:,10]
u_real, f_real, u_log, f_log = Initial_estimate(t1_slic)

u_real1,f_real1,u_log1,f_log1 = iterative_step(u_real,f_real,u_log,f_log,t1_slic)

u_real2,f_real2,u_log2,f_log2 = iterative_step(u_real1,f_real1,u_log1,f_log1,t1_slic)

u_real3,f_real3,u_log3,f_log3 = iterative_step(u_real2,f_real2,u_log2,f_log2,t1_slic)

u_real4,f_real4,u_log4,f_log4 = iterative_step(u_real3,f_real3,u_log3,f_log3,t1_slic)

for i in range(12):
    u_real4,f_real4,u_log4,f_log4 = iterative_step(u_real4,f_real4,u_log4,f_log4,t1_slic)


#if __name__ == "__main__":
#    main()
