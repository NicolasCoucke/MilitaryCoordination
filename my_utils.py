
import numpy as np



def extract_trials(events):
    trialial_info = []
    event_values = events[:, 2]
    event_samples = events[:,0]
    print(event_values)

    # find indices of trial start within the event matrix
    trial_start_indices = np.where(event_values < 10)[0]
    trial_counter = 1
    trials = np.empty(5,)
    for trial_start in trial_start_indices:
        j = trial_start
        end_trial = 0
        condition = event_values[trial_start]
        # don't care about tutorial trials
        if condition == 1:
            continue

        # j remains at start of trial while k goes to the end
        k = j

        while end_trial == 0:
            if event_values[k] == 150: # if player one finishes
                # check if the trial is a real succes
                jk = k
                secondfinished = False
                gameover = False

                # as long as the next trial doesnt start
                while event_values[jk] > 10:
                    # does the second player also finish?
                    if event_values[jk] == 250:
                        secondfinished = True
                    elif ( np.remainder(event_values[jk], 140 + condition) == 0 ) or ( np.remainder(event_values[jk], 240 + condition) == 0 ):
                        gameover = True
                    jk += 1
                    if jk > len(event_samples):
                        break

                if (secondfinished == True) and (gameover == False):
                    success = True 
                else:
                    success = False

                end_trial = 1

                # if this was a real trial than add it, otherwise remove
            elif event_values[k] == 250:
                jk = k
                secondfinished = False
                gameover = False
                while event_values[jk] > 10:
                    # does the second player also finish?
                    if event_values[jk] == 150:
                        secondfinished = True
                    elif ( np.remainder(event_values[jk], 140 + condition) == 0 ) or ( np.remainder(event_values[jk], 240 + condition) == 0 ):
                        gameover = True
                    jk += 1
                    if jk > len(event_samples):
                        break

                if (secondfinished == True) and (gameover == False):
                    success = True
                else:
                    success = False
                end_trial = 1
            # if one of them gets game over then they are both gameove
            elif ( np.remainder(event_values[k], 140 + condition) == 0 ) or ( np.remainder(event_values[k], 240 + condition) == 0 ):
                success = False
                end_trial = 1

            elif trial_start == trial_start_indices[-1]:
                end_trial = 1
            else:
                # if no gameover or finish then continue searching
                end_trial = 0
            
            k += 1
            if k > len(event_samples):
                break
        
        if k > len(event_samples):
            break

        trial_begin = event_samples[j]
        trial_end = event_samples[k]

        new_trial = [trial_begin, trial_end, trial_counter, condition, success]
        trials = np.vstack((trials, np.array(new_trial)))
        trial_counter +=1
    return trials



def create_sub_epochs(trials, sfreq):
    events = np.empty(3,)
    event_successes = np.empty(3,)
    for trial_index in range(np.size(trials,0)):
        trial_start = trials[trial_index, 0]
        trial_end = trials[trial_index, 1]
        condition = trials[trial_index, 3]
        success = trials[trial_index, 4]

        epoch_start = trial_start 
        while (epoch_start + sfreq ) < trial_end:
            event = [epoch_start, 0, condition]
            events = np.vstack((events, np.array(event)))
            
            event_success = [epoch_start, 0, success]
            event_successes = np.vstack((event_successes, event_success))

            # sliding window with 0.5s interval
            epoch_start += 0.5 * sfreq
    events = (np.rint(events)).astype(int)
    event_successes = (np.rint(event_successes)).astype(int)

    return events, event_successes




def dss_line(X, fline, sfreq, nremove=1, nfft=1024, nkeep=None, blocksize=None,
             show=False):
    """Apply DSS to remove power line artifacts.
    Implements the ZapLine algorithm described in [1]_.
    Parameters
    ----------
    X : data, shape=(n_samples, n_chans, n_trials)
        Input data.
    fline : float
        Line frequency (normalized to sfreq, if ``sfreq`` == 1).
    sfreq : float
        Sampling frequency (default=1, which assymes ``fline`` is normalised).
    nremove : int
        Number of line noise components to remove (default=1).
    nfft : int
        FFT size (default=1024).
    nkeep : int
        Number of components to keep in DSS (default=None).
    blocksize : int
        If not None (default), covariance is computed on blocks of
        ``blocksize`` samples. This may improve performance for large datasets.
    show: bool
        If True, show DSS results (default=False).
    Returns
    -------
    y : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    artifact : array, shape=(n_samples, n_chans, n_trials)
        Artifact
    Examples
    --------
    Apply to X, assuming line frequency=50Hz and sampling rate=1000Hz, plot
    results:
    >>> dss_line(X, 50/1000)
    Removing 4 line-dominated components:
    >>> dss_line(X, 50/1000, 4)
    Truncating PCs beyond the 30th to avoid overfitting:
    >>> dss_line(X, 50/1000, 4, nkeep=30);
    Return cleaned data in y, noise in yy, do not plot:
    >>> [y, artifact] = dss_line(X, 60/1000)
    References
    ----------
    .. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to
       remove power line artifacts [Preprint]. https://doi.org/10.1101/782029
    """
    if X.shape[0] < nfft:
        print('Reducing nfft to {}'.format(X.shape[0]))
        nfft = X.shape[0]
    n_samples, n_chans, _ = theshapeof(X)
    if blocksize is None:
        blocksize = n_samples

    # Recentre data
    X = demean(X, inplace=True)

    # Cancel line_frequency and harmonics + light lowpass
    X_filt = smooth(X, sfreq / fline)

    # X - X_filt results in the artifact plus some residual biological signal
    X_noise = X - X_filt

    # Reduce dimensionality to avoid overfitting
    if nkeep is not None:
        cov_X_res = tscov(X_noise)[0]
        V, _ = pca(cov_X_res, nkeep)
        X_noise_pca = X_noise @ V
    else:
        X_noise_pca = X_noise.copy()
        nkeep = n_chans

    # Compute blockwise covariances of raw and biased data
    n_harm = np.floor((sfreq / 2) / fline).astype(int)
    c0 = np.zeros((nkeep, nkeep))
    c1 = np.zeros((nkeep, nkeep))
    for X_block in sliding_window_view(X_noise_pca, (blocksize, nkeep),
                                       axis=(0, 1))[::blocksize, 0]:
        # if n_trials>1, reshape to (n_samples, nkeep, n_trials)
        if X_block.ndim == 3:
            X_block = X_block.transpose(1, 2, 0)

        # bias data
        c0 += tscov(X_block)[0]
        c1 += tscov(gaussfilt(X_block, sfreq, fline, fwhm=1, n_harm=n_harm))[0]

    # DSS to isolate line components from residual
    todss, _, pwr0, pwr1 = dss0(c0, c1)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(pwr1 / pwr0, '.-')
        plt.xlabel('component')
        plt.ylabel('score')
        plt.title('DSS to enhance line frequencies')
        plt.show()

    # Remove line components from X_noise
    idx_remove = np.arange(nremove)
    X_artifact = matmul3d(X_noise_pca, todss[:, idx_remove])
    X_res = tsr(X_noise, X_artifact)[0]  # project them out

    # reconstruct clean signal
    y = X_filt + X_res

    # Power of components
    p = wpwr(X - y)[0] / wpwr(X)[0]
    print('Power of components removed by DSS: {:.2f}'.format(p))
    # return the reconstructed clean signal, and the artifact
    return y, X - y


# helper function
def _multiply_conjugate(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Helper function to compute the product of a complex array and its conjugate.
    It is designed specifically to collapse the last dimension of a four-dimensional array.
    Arguments:
        real: the real part of the array.
        imag: the imaginary part of the array.
        transpose_axes: axes to transpose for matrix multiplication.
    Returns:
        product: the product of the array and its complex conjugate.
    """
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def _multiply_conjugate_time(real: np.ndarray, imag: np.ndarray, transpose_axes: tuple) -> np.ndarray:
    """
    Helper function to compute the product of a complex array and its conjugate.
    Unlike _multiply_conjugate, this doenst collapse the last dimension of a 
    four-dimensional array. Useful when computing some connectivity metrics 
    (e.g., wpli), since it preserves the product values across e.g., time.
    
    Arguments:
        real: the real part of the array.
        imag: the imaginary part of the array.
        transpose_axes: axes to transpose for matrix multiplication.
    Returns:
        product: the product of the array and its complex conjugate.
    """
    formula = 'jilm,jimk->jilkm'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
              np.einsum(formula, imag, imag.transpose(transpose_axes)) - 1j * \
              (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
               np.einsum(formula, imag, real.transpose(transpose_axes)))
    
    return product

def compute_sync(complex_signal: np.ndarray, mode: str, epochs_average: bool = True, save_memory: bool = False) -> np.ndarray:
    """
    Computes frequency- or time-frequency-domain connectivity measures from analytic signals.
    Arguments:
        complex_signal:
            shape = (2, n_epochs, n_channels, n_freq_bins, n_times).
            Analytic signals for computing connectivity between two participants.
        mode:
            Connectivity measure. Options in the notes.
        epochs_average:
            option to either return the average connectivity across epochs (collapse across time) or preserve epoch-by-epoch connectivity, boolean.
            If False, PSD won't be averaged over epochs (the time course is maintained).
            If True, PSD values are averaged over epochs.
        save_memory:
            option to create connectivity matrix epoch per epoch, rather than with all epochs at once.
            is slower but prevents running out of memory

    Returns:
        con:
            Connectivity matrix. The shape is either
            (n_freq, n_epochs, 2*n_channels, 2*n_channels) if time_resolved is False,
            or (n_freq, 2*n_channels, 2*n_channels) if time_resolved is True.
            To extract inter-brain connectivity values, slice the last two dimensions of con with [0:n_channels, n_channels: 2*n_channels].
    Note:
        **supported connectivity measures**
          - 'envelope_corr': envelope correlation
          - 'pow_corr': power correlation
          - 'plv': phase locking value
          - 'ccorr': circular correlation coefficient
          - 'coh': coherence
          - 'imaginary_coh': imaginary coherence
          - 'pli': phase lag index
          - 'wpli': weighted phase lag index
    """

    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
                                    complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal_full = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    transpose_axes = (0, 1, 3, 2)


    if save_memory:
        # loops through each epoch once
        epoch_intervals = range(n_epoch) 
    else:
        # does one iteration that includes all epochs
        epoch_intervals = [range(n_epoch)]


    for epoch_range in epoch_intervals:

        # take either full signal or one epoch
        complex_signal = complex_signal_full[epoch_range, :, :, :].reshape(len(epoch_range), n_freq, 2 * n_ch, n_samp)

        if mode.lower() == 'plv':
            phase = complex_signal / np.abs(complex_signal)
            c = np.real(phase)
            s = np.imag(phase)
            dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
            con = abs(dphi) / n_samp

        elif mode.lower() == 'envelope_corr':
            env = np.abs(complex_signal)
            mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
            env = env - mu_env
            con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

        elif mode.lower() == 'pow_corr':
            env = np.abs(complex_signal) ** 2
            mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
            env = env - mu_env
            con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

        elif mode.lower() == 'coh':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            amp = np.abs(complex_signal) ** 2
            dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
            con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                np.nansum(amp, axis=3)))

        elif mode.lower() == 'imaginary_coh':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            amp = np.abs(complex_signal) ** 2
            dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
            con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                            np.nansum(amp, axis=3)))

        elif mode.lower() == 'ccorr':
            angle = np.angle(complex_signal)
            mu_angle = circmean(angle, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
            angle = np.sin(angle - mu_angle)

            formula = 'nilm,nimk->nilk'
            con = np.einsum(formula, angle, angle.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3), np.sum(angle ** 2, axis=3)))
            
        elif mode.lower() == 'pli':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            dphi = _multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
            con = abs(np.mean(np.sign(np.imag(dphi)), axis=4))
            
        elif mode.lower() == 'wpli':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            dphi = _multiply_conjugate_time(c, s, transpose_axes=transpose_axes)
            con_num = abs(np.mean(abs(np.imag(dphi)) * np.sign(np.imag(dphi)), axis=4))
            con_den = np.mean(abs(np.imag(dphi)), axis=4)      
            con_den[con_den == 0] = 1 
            con = con_num / con_den        

        else:
            ValueError('Metric type not supported.')

        if save_memory:
            if epoch_range == 0:
                aggregate_con = con
            else:
                aggregate_con = np.stack((aggregate_con, con), axis = 0)
        
    if save_memory:
        con = aggregate_con
    
    con = con.swapaxes(0, 1)  # n_freq x n_epoch x 2*n_ch x 2*n_ch
    if epochs_average:
        con = np.nanmean(con, axis=1)

    return con
