"""
Three-dimensional spike localization and improved motion correction for Neuropixels recordings

Code for image-based motion estimation

Input: {x, y, z} localization estimate, spike times, amplitudes, geometry
Output: image-based motion estimation
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, notebook
import torch
import os

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.registration import phase_cross_correlation

from numpy.fft import fft2, ifft2, fftshift, ifftshift # Python DFT
import pywt

from scipy.stats import norm
from scipy.ndimage import shift
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import lsqr
from scipy.stats import zscore
from scipy.sparse import csr_matrix

def estimate_motion(locs, 
                    times, 
                    amps, 
                    geomarray, 
                    direction, # 'x', 'y', 'z'
                    do_destripe=False,
                    do_denoise=True, # Poisson denoising
                    sigma=0.1,
                    h=1.,
                    patch_size=5,
                    patch_distance=5,
                    reg_win_num=10, # set to 1 for rigid
                    reg_block_num=100, # set to 1 for rigid
                    iteration_num=2,
                    fast_mode=True,
                    robust_mode=True,
                    subsampling_rate=1.,
                    error_sigma=0.1,
                    time_sigma=1,
                    robust_regression_sigma=1):
    """
    function that wraps around all functions for motion estimation
    generates raster plot -> poisson denoising (optional) -> decentralized registration

    """
            
    raster = gen_raster(locs, times, amps, geomarray, direction)
    
    if do_destripe:
        destriped = destripe(raster)
    else:
        destriped = raster
    
    if do_denoise:
        denoised = poisson_denoising(destriped,sigma=sigma,h=h,patch_size=patch_size,patch_distance=patch_distance)
    else:
        denoised = destriped
    
    # decentralized registration
    total_shift = decentralized_registration(denoised, fast_mode=fast_mode, robust_mode=robust_mode,
                                             win_num=reg_win_num,
                                             reg_block_num=reg_block_num,
                                             iter_num=iteration_num,
                                             subsampling_rate=subsampling_rate,
                                             error_sigma=error_sigma, 
                                             time_sigma=time_sigma,
                                             robust_regression_sigma = robust_regression_sigma)
    
    return total_shift

def gen_raster(locs, times, amps, geom, direction):
    """
    generates a raster plot from given localization estimates, spike times, amplitudes

    """
    max_t = np.ceil(times.max()).astype(int)

    if direction == 'z':
        D_max = max(locs.max(), geom[:,1].max())
        D_min = min(locs.min(), geom[:,1].min())
        D = np.ceil(D_max - D_min).astype(int)
    elif direction == 'x':
        D_max = max(locs.max(), geom[:,0].max())
        D_min = min(locs.min(), geom[:,0].min())
        D = np.ceil(D_max - D_min).astype(int)+1
    elif direction == 'y':
        D_max = locs.max()
        D_min = locs.min()
        D = np.ceil(D_max - D_min).astype(int)
    
    raster = np.zeros((D,max_t))
    raster_count = np.zeros((D,max_t))
    for i in tqdm(range(max_t)):
        idx = np.intersect1d(np.where(times > i)[0], np.where(times < i+1)[0])

        for j in idx:
            loc = int(np.floor(locs[j] - D_min))
            amp = amps[j]
            raster[loc,i] += amp
            raster_count[loc,i] += 1

    raster_count[np.where(raster_count == 0)] = 1
            
    return raster/raster_count


def poisson_denoising(z, sigma=0.1, h=1., scale=5, 
                      estimate_sig=False, fast_mode=True, multichannel=False,
                     patch_size=5,patch_distance=5):
    """
    Poisson denoising (Anscombe transformation -> Gaussin denoising -> inverse transformation)
    Change sigma and h to adjust denoising

    """
    minmax = (z - z.min()) / (z.max() - z.min()) # scales data to 0-1
    
    # Gaussianizing Poissonian data
    z_anscombe = 2. * np.sqrt(minmax + (3. / 8.))
    
    if estimate_sig:
        sigma = np.mean(estimate_sigma(z_anscombe, multichannel=multichannel)) * scale
        print("estimated sigma: {}".format(sigma))
    # Gaussian denoising
    z_anscombe_denoised = denoise_nl_means(z_anscombe, h=h*sigma, sigma=sigma,
                                           fast_mode=fast_mode, patch_size=patch_size,
                                           patch_distance=patch_distance) # NL means denoising

    z_inverse_anscombe = (z_anscombe_denoised / 2.)**2 + 0.25 * np.sqrt(1.5) * z_anscombe_denoised**-1 - (11. / 8.) * z_anscombe_denoised**-2 +(5. / 8.) * np.sqrt(1.5) * z_anscombe_denoised**-3 - (1. / 8.)
    
    z_inverse_anscombe_scaled = ((z.max() - z.min()) * z_inverse_anscombe) + z.min()
    
    return z_inverse_anscombe_scaled

def destripe(raster):
    D, W = raster.shape
    LL0 = raster
    wlet = 'db5'
    coeffs = pywt.wavedec2(LL0, wlet)
    L = len(coeffs)
    for i in range(1,L):
        HL = coeffs[i][1]    
        Fb = fft2(HL)   
        Fb = fftshift(Fb)
        mid = Fb.shape[0]//2
        Fb[mid,:] = 0
        Fb[mid-1,:] /= 3
        Fb[mid+1,:] /= 3
        Fb = ifftshift(Fb)   
        coeffs[i]= (coeffs[i][0], np.real(ifft2(Fb)), coeffs[i][2] )
    LL = pywt.waverec2(coeffs, wlet)
    LL = LL[:D,:W]
    
    destriped = np.zeros_like(raster)
    destriped[:D,:W] = LL
    return destriped

def get_gaussian_window(height, width, loc, scale=1):
    """
    Gets Gaussian windows along the depth of the raster

    """
    window = np.zeros((height,width))
    for i in range(height):
        window[i] = norm.pdf(i, loc=loc, scale=scale)
    return window / window.max()

def calc_displacement_matrix_raster(raster, nbins=1, disp = 400, step_size = 1, batch_size = 1):
    """
    Calculates the displacement matrix needed for decentralized registration

    """
    T = raster.shape[0]
    possible_displacement = np.arange(-disp, disp + step_size, step_size)
    raster = torch.from_numpy(raster).cuda().float()
    c2d = torch.nn.Conv2d(in_channels = 1, out_channels = T, kernel_size = [nbins, raster.shape[-1]], stride = 1, padding = [0, possible_displacement.size//2], bias = False).cuda()
    c2d.weight[:,0] = raster
    displacement = np.zeros([T, T])
    for i in notebook.tqdm(range(T//batch_size)):
        res = c2d(raster[i*batch_size:(i+1)*batch_size,None])[:,:,0,:].argmax(2)
        displacement[i*batch_size:(i+1)*batch_size] = possible_displacement[res.cpu()]
        del res
    del c2d
    del raster
    torch.cuda.empty_cache()
    return displacement

def calc_displacement_matrix_raster_dft(w_raster, S):
    
    D, T = w_raster.shape
    displacement_matrix = np.zeros((T,T))
    error_mat = np.zeros((T,T))
    
    for i in notebook.tqdm(range(T)):
        for j in range(T):
            if S[i,j] == 0:
                continue
            
            shift, error, diffphase = phase_cross_correlation(w_raster[:,i], w_raster[:,j])
            displacement_matrix[i,j] = shift
            error_mat[i,j] = error
            
    return displacement_matrix, error_mat

def calc_displacement(displacement, n_iter = 1000):
    """
    Calculates the displacement estimate given the displacement matrix (decentralized registration)

    """
    p = torch.zeros(displacement.shape[0]).cuda()
    displacement = torch.from_numpy(displacement).cuda().float()
    n_batch = displacement.shape[0]
    pprev = p.clone()
    for i in notebook.tqdm(range(n_iter)):
        repeat1 = p.repeat_interleave(n_batch).reshape((n_batch, n_batch))
        repeat2 = p.repeat_interleave(n_batch).reshape((n_batch, n_batch)).T
        mat_norm = displacement + repeat1 - repeat2
        p += 2*(torch.sum(displacement-torch.diag(displacement), dim=1) - (n_batch-1)*p)/torch.norm(mat_norm)
        del mat_norm
        del repeat1
        del repeat2
        if torch.allclose(pprev, p):
            break
        else:
            del pprev
            pprev = p.clone()
    disp = np.asarray(p.cpu())
    del p
    del pprev
    del displacement
    torch.cuda.empty_cache()
    return disp

def calc_displacement_robust(displacement_matrix, 
                             error_mat, 
                             S, 
                             error_sigma, 
                             time_sigma, 
                             robust_regression_sigma, n_iter=20):
    error_mat_S = error_mat[np.where(S != 0)]
    W1 = np.exp(-((error_mat_S-error_mat_S.min())/(error_mat_S.max()-error_mat_S.min()))/error_sigma)

    W2 = np.exp(-squareform(pdist(np.arange(error_mat.shape[0])[:,None]))/time_sigma)
    W2 = W2[np.where(S != 0)]

    W = (W2*W1)[:,None]
    
    I, J = np.where(S != 0)
    V = displacement_matrix[np.where(S != 0)]
    M = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),I)))
    N = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]),J)))
    A = M - N
    idx = np.ones(A.shape[0]).astype(bool)
    for i in notebook.tqdm(range(n_iter)):
        p = lsqr(A[idx].multiply(W[idx]), V[idx]*W[idx][:,0])[0]
        idx = np.where(np.abs(zscore(A@p-V)) <= robust_regression_sigma)
    return p

def shift_x(x, shift_amt):
    """
    Shifts (registers) the raster plot x by shift_amt

    """
    shifted = np.zeros_like(x)
    for t in range(x.shape[1]):
        col = x[:,t]
        sh = shift_amt[t]
        shifted[:,t] = shift(col, sh)

    return shifted

def register_raster(raster, total_shift, blocks):
    """
    Shifts (registers) the raster plot x by shift_amt
    internally calls shift_x

    """
    raster_sh = np.zeros_like(raster)
    for k in notebook.tqdm(range(1, blocks.shape[0])):
        cur = blocks[k]
        prev = blocks[k-1]
        sh = np.mean(-total_shift[prev:cur], axis=0)
        roi = np.zeros_like(raster)
        roi[prev:cur] = raster[prev:cur]
        raster_sh += shift_x(roi, sh)
    return raster_sh

def save_registered_raster(raster_sh, i, output_directory):
    """
    Saves registered raster plot

    """
    fname = os.path.join(output_directory, "raster_{}.png".format(str(i+1).zfill(6)))
    print('plotting...')
    plt.figure(figsize=(16, 10))
    plt.imshow(raster_sh, vmin=0, vmax=30, aspect="auto", cmap=plt.get_cmap('inferno'))
    plt.ylabel("depth", fontsize=16)
    plt.xlabel("time", fontsize=16)
    plt.savefig(fname,bbox_inches='tight')
    plt.close()
    
def get_subsampling_matrix(rate, T):
    T = int(T) # make sure it's int
    S = np.zeros((T,T))
    p = np.log(T)/T
    while S.sum() < T*T*p:
        r = np.random.permutation(np.arange(T))
        S[np.arange(T),r] = 1
    S = np.bitwise_or(S.astype(bool), S.T.astype(bool)).astype(int)
    return S
    
def decentralized_registration(raster, 
                               win_num=1, 
                               reg_block_num=1, 
                               iter_num=4, 
                               fast_mode=True,
                               robust_mode=True,
                               subsampling_rate=1.,
                               error_sigma=0.1,
                               time_sigma=1,
                               robust_regression_sigma=1):
    """
    Image-based Decentralized registration
    Input: raster plot (poisson denoised)
    Output: displacement estimate

    """
    D, T = raster.shape
    
    # get windows
    window_list = []
    if win_num == 1:
        window_list.append(np.ones_like(raster))
    else:
        space = int(D//(win_num+1))
        locs = np.linspace(space, D-space, win_num, dtype=np.int32)
        for i in range(win_num):
            window = get_gaussian_window(D, T, locs[i], scale=D/(0.5*win_num))
            window_list.append(window)
    window_sum = np.sum(np.asarray(window_list), axis=0)
    
    # get subsampling matrix, if rate < 1
    if subsampling_rate < 1:
        S = get_subsampling_matrix(subsampling_rate, T)

    shifts = np.zeros((win_num, T))
    total_shift = np.zeros_like(raster)
    
    raster_i = raster.copy()
    
    reg_block_num += 1
    blocks = np.linspace(0, D, reg_block_num, dtype=np.int64)
    
    output_directory = os.path.join('.', "image_based_registered_raster")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    save_registered_raster(raster, -1, output_directory)
    
    for i in notebook.tqdm(range(iter_num)):
        
        shift_amt = np.zeros_like(raster)
        
        for j, w in enumerate(window_list):
            w_raster = w*raster_i
            if fast_mode:
                displacement_matrix = calc_displacement_matrix_raster((w_raster).T[:,np.newaxis,:])
            else:
                displacement_matrix, error_mat = calc_displacement_matrix_raster_dft(w_raster, S)
                            
            if robust_mode:
                disp = calc_displacement_robust(displacement_matrix, error_mat, 
                                                S, error_sigma,
                                                time_sigma, robust_regression_sigma)
            else:
                disp = calc_displacement(displacement_matrix)   
                
            shift_amt += w * disp[np.newaxis,:]
            shifts[j] += disp
            
        total_shift += (shift_amt / window_sum)

        raster_sh = register_raster(raster, total_shift, blocks)
        
        raster_i = raster_sh.copy()
        
        save_registered_raster(raster_sh, i, output_directory)
        
    return total_shift
