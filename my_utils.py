
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