"""
Tools and helper methods for cross-correlation of the NICER data of MAXIJ1535
"""
import numpy as np
import itertools
from astropy.io import fits
from astropy.table import Table, Column
from scipy.stats import binned_statistic
import os


__author__ = 'Abigail Stevens <abigailstev@gmail.com>'
__year__ = "2017-2021"


def find_nearest(array, value):
    """
    Thanks StackOverflow!

    Parameters
    ----------
    array : np.array of ints or floats
        1-D array of numbers to search through. Should already be sorted
        from low values to high values.

    value : int or float
        The value you want to find the closest to in the array.

    Returns
    -------
    array[idx] : int or float
        The array value that is closest to the input value.

    idx : int
        The index of the array of the closest value.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx == len(array) or np.fabs(value - array[idx - 1]) < \
        np.fabs(value - array[idx]):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx


def clock_to_mjd(clocktime, mjdrefi=56658, timezero=0):
    """
    Converts a NICER clock timestamp to MJD. From:
    https://heasarc.gsfc.nasa.gov/docs/nicer/mission_guide/

    Parameters
    ----------
    clocktime: float
        The timestamp in the event list data that you want to convert to MJD.

    mjdrefi: int
        From the NICER EVENTS HDU fits header; all of them seem to be 56658?

    timezero: float
        From the NICER EVENTS HDU fits header; for some its zero, for some
        it's -1 if it's before one of the recent leap seconds

    Returns
    -------
    mjd: float
        The Modified Julian Date (in UTC) conversion of the clocktime.
    """
    return mjdrefi + (timezero + clocktime + 2) / 86400.


def make_2Dlightcurve(time, energy, n_bins, detchans, seg_start, seg_end):
    """
    Populates a segment of a light curve with photons from the event list.

    Parameters
    ----------
    time : np.array of floats
        1-D array of times at which a photon is detected (assumes these times 
        are the front of the timebin?).
    
    energy : np.array of ints
        1-D array of the energy channel in which the photon is detected.
    
    n_bins : int
        Number of bins per segment of light curve.
        
    detchans : int
        Number of detector energy channels.
    
    seg_start : float
        Start time of the segment, in the same units as the time array.
    
    seg_end : float
        End time of the segment, in the same units as the time array.

    Returns
    -------
    lightcurve_2d : np.array of ints
        2-D array of the populated 2-dimensional light curve, with time as one 
        axis and energy channel as the other. In units of count rate.

    """
    ## Ranges need to be amount+1 here, because of how 'histogram2d' bins the 
    ## values
    ## Defining time bin edges
    t_bin_seq = np.linspace(seg_start, seg_end, num=n_bins+1)
    dt = t_bin_seq[1]-t_bin_seq[0]
    e_bin_seq = np.arange(detchans + 1)
    lightcurve_2d, t_bin_edges, e_bin_edges = np.histogram2d(time, energy,
        bins=[t_bin_seq, e_bin_seq], normed=False)
    ## Need counts/dt to have units of count rate
    ## Doing it by multiplying by 1/dt, to keep it as an int and not get
    ## typecasting errors.
    dt_inv_int = np.int64(1./dt)
    lightcurve_2d *= dt_inv_int
    return lightcurve_2d


def make_1Dlightcurve(time, n_bins, seg_start, seg_end):
    """
    Populates a segment of a light curve with photons from the event list.

    Parameters
    ----------
    time : np.array of floats
        1-D array of times at which a photon is detected (assumes these times
        are the front of the timebin?).

    n_bins : int
        Number of bins per segment of light curve.

    seg_start : float
        Start time of the segment, in the same units as the time array.

    seg_end : float
        End time of the segment, in the same units as the time array.

    Returns
    -------
    lightcurve_1d : np.array of ints
        1-D array of the populated light curve, with time bins along the axis.
        In units of count rate.

    """
    ## Ranges need to be amount+1 here, because of how 'histogram' bins the
    ## values
    ## Defining time bin edges
    t_bin_seq = np.linspace(seg_start, seg_end, num=n_bins+1)
    dt = np.median(np.diff(t_bin_seq))
    assert (1./dt).is_integer, "1/dt is not an integer: %.3g" % (1./dt)

    lightcurve_1d, t_bin_edges = np.histogram(time, bins=t_bin_seq,
                                              normed=False)
    ## Need counts/dt to have units of count rate
    ## Doing it by multiplying by 1/dt, to keep it as an int and not get
    ## typecasting errors.
    dt_inv_int = np.int64(1./dt)
    lightcurve_1d *= dt_inv_int
    return lightcurve_1d


def psd_norm(psd, mean_rate, dt, n_bins, norm="frac", noisy=False):
    """
    Normalize a power spectrum by absolute rms^2, fractional rms^2, or
    Leahy normalization. If noisy=True, also subtracts the computed Poisson
    noise level for the specified normalization.
    """
    assert str(norm).lower() in ["abs", "frac", "leahy"], \
        "Invalid normalization was specified."
    assert isinstance(noisy, bool), "`noisy` must be a boolean."
    assert isinstance(psd[1], float), "`psd` must be a 1-D np.array of " \
                                      "floats."
    # print("Shape psd in _psd_norm: "+str(np.shape(psd)))
    # print("Shape mean rate: in _psd_norm "+str(np.shape(mean_rate)))
    if len(np.shape(psd)) == 1:  # if ref psd is already averaged over n_seg
        if not isinstance(mean_rate, float):
            mean_rate = np.mean(
                mean_rate)  # then we want one number for the mean rate
    else:  # ci psds and/or psds for each segment
        print("Multi-dimensional psd. TODO: check that the array " \
              "broadcasting works.")
        assert np.shape(psd)[-1] == np.shape(mean_rate)[0]
    if str(norm).lower() == "abs":
        psd *= 2 * dt / n_bins
        if noisy:
            psd -= 2 * mean_rate
    elif str(norm).lower() == "frac":
        psd *= 2 * dt / n_bins / mean_rate ** 2
        if noisy:
            psd -= 2 / mean_rate
    elif str(norm).lower() == "leahy":
        psd *= 2 * dt / n_bins / mean_rate
        if noisy:
            psd -= 2.
    else:
        print("Invalid normalization was specified.")
    return psd


def lin_rb(freq, power, n_bins, new_n_bins):
    """

    :param freq:
    :param power:
    :param n_bins:
    :param new_n_bins:
    :return:
    """
    p_freq = np.abs(freq[0:int(n_bins / 2 + 1)])

    lin_rb_psd, f_bin_edges, something = binned_statistic(p_freq, power,
                                                          statistic='mean',
                                                          bins=new_n_bins)
    new_df = np.median(np.diff(f_bin_edges))
    new_freq = f_bin_edges[0:-1] + 0.5 * new_df  # so that the freq is mid-bin
    return new_freq, new_df, lin_rb_psd


def geom_rb(freq, power, err_power, rebin_const=1.02):
    """

    :param freq:
    :param power:
    :param err_power:
    :param rebin_const:
    :return:
    """
    assert rebin_const >= 1.0

    ## Initialize variables
    rb_power = np.asarray([])  # List of re-binned power
    rb_freq = np.asarray([])  # List of re-binned frequencies
    rb_err = np.asarray([])  # List of error in re-binned power
    real_index = 1.0  # The unrounded next index in power
    int_index = 1  # The int of real_index, added to current_m every
    #  iteration
    current_m = 1  # Current index in power
    prev_m = 0  # Previous index m
    freq_min = np.asarray([])
    freq_max = np.asarray([])

    ## Loop through the length of the array power, new bin by new bin, to
    ## compute the average power and frequency of that new geometric bin.
    ## Equations for frequency, power, and error are from A. Ingram's PhD thesis
    while current_m < len(power):
        ## Determine the range of indices this specific geometric bin covers
        ## The range of un-binned bins covered by this re-binned bin
        bin_range = np.absolute(current_m - prev_m)
        ## Want mean power of data points contained within one geometric bin
        ## The power of the current re-binned bin
        bin_power = np.mean(power[prev_m:current_m])
        ## Compute error in bin -- equation from Adam Ingram's thesis
        ## The error squared on 'bin_power'
        err_bin_power2 = np.sqrt(np.sum(err_power[prev_m:current_m] ** 2)) / \
                         float(bin_range)

        ## Compute the mean frequency of a geometric bin
        ## The frequency of the current re-binned bin
        bin_freq = np.mean(freq[prev_m:current_m])

        ## Append values to arrays
        rb_power = np.append(rb_power, bin_power)
        rb_freq = np.append(rb_freq, bin_freq)
        rb_err = np.append(rb_err, err_bin_power2)
        freq_min = np.append(freq_min, freq[prev_m])
        freq_max = np.append(freq_max, freq[current_m])

        ## Increment for the next iteration of the loop
        ## Since the for-loop goes from prev_m to current_m-1 (since that's how
        ## the range function and array slicing works) it's ok that we set
        ## prev_m = current_m here for the next round. This will not cause any
        ## double-counting bins or skipping bins.
        prev_m = current_m
        real_index *= rebin_const
        int_index = int(round(real_index))
        current_m += int_index

    return rb_freq, rb_power, rb_err, freq_min, freq_max


def geom_rb_return_bins(freq, power, rebin_const=1.02):
    """

    :param freq:
    :param power:
    :param rebin_const:
    :return:
    """
    assert rebin_const >= 1.0

    ## Initialize variables
    rb_power = np.asarray([])  # List of re-binned power
    rb_freq = np.asarray([])  # List of re-binned frequencies
    real_index = 1.0  # The unrounded next index in power
    int_index = 1  # The int of real_index, added to current_m every
    #  iteration
    current_m = 1  # Current index in power
    prev_m = 0  # Previous index m
    rb_bins = np.asarray([])

    ## Loop through the length of the array power, new bin by new bin, to
    ## compute the average power and frequency of that new geometric bin.
    ## Equations for frequency, power, and error are from A. Ingram's PhD thesis
    while current_m < len(power):
        ## Determine the range of indices this specific geometric bin covers
        ## The range of un-binned bins covered by this re-binned bin
        bin_range = np.absolute(current_m - prev_m)
        ## Want mean power of data points contained within one geometric bin
        ## The power of the current re-binned bin
        bin_power = np.mean(power[prev_m:current_m])

        ## Compute the mean frequency of a geometric bin
        ## The frequency of the current re-binned bin
        bin_freq = np.mean(freq[prev_m:current_m])

        ## Append values to arrays
        rb_power = np.append(rb_power, bin_power)
        rb_freq = np.append(rb_freq, bin_freq)
        rb_bins = np.append(rb_bins, bin_range)

        ## Increment for the next iteration of the loop
        ## Since the for-loop goes from prev_m to current_m-1 (since that's how
        ## the range function and array slicing works) it's ok that we set
        ## prev_m = current_m here for the next round. This will not cause any
        ## double-counting bins or skipping bins.
        prev_m = current_m
        real_index *= rebin_const
        int_index = int(round(real_index))
        current_m += int_index

    return rb_freq, rb_power, rb_bins


def chbin_to_rsp(chbinfile):
    """
    Reads a channel binning file and returns keV mins and maxes for energy
    channels, as though it were the energy information from a response matrix.
    Parameters
    ----------
    chbinfile : str
        Path name of the channel binning file that would be passed to the
        FTOOL rbnpha.

    Returns
    -------
    rsp_tab : astropy.table.Table
        Table with E_MIN and E_MAX for each energy channel.
    """
    assert os.path.isfile(chbinfile), "Channel bin file doesn't exist: %s" % chbinfile
    chbintab = np.loadtxt(chbinfile)
    min_E = np.asarray([])
    max_E = np.asarray([])
    for erow in chbintab:
        if erow[-1] <= 0:
            pass
        else:
            temp = erow[0]
            while temp < erow[1]:
                min_E = np.append(min_E, temp)
                max_E = np.append(max_E, temp + erow[-1])
                temp += erow[-1]

    min_E /= 10  # converts it from PI to keV
    max_E /= 10  # converts it from PI to keV
    rsp_tab = Table()
    rsp_tab.add_column(Column(data=min_E, name='E_MIN'))
    rsp_tab.add_column(Column(data=max_E, name='E_MAX'))
    return rsp_tab


def make_binned_lc(time, energy, n_bins, chan_bins, seg_start, seg_end):
    """
    Populates a segment of a light curve with photons from the event list.

    Parameters
    ----------
    time : np.array of floats
        1-D array of times at which a photon is detected (assumes these times
        are the front of the timebin?).

    energy : np.array of ints
        1-D array of the energy channel in which the photon is detected.

    n_bins : int
        Number of bins per segment of light curve.

    chan_bins : np.array of ints
        The channel binning

    seg_start : float
        Start time of the segment, in the same units as the time array.

    seg_end : float
        End time of the segment, in the same units as the time array.

    Returns
    -------
    lightcurve_2d : np.array of ints
        2-D array of the populated 2-dimensional light curve, with time as one
        axis and (binned) energy channel as the other. In units of count rate.

    """
    ## Ranges need to be amount+1 here, because of how 'histogram2d' bins the
    ## values
    ## Defining time bin edges
    t_bin_seq = np.linspace(seg_start, seg_end, num=n_bins + 1)
    dt = t_bin_seq[1] - t_bin_seq[0]
    lc_2d, t_bin_edges, e_bin_edges = np.histogram2d(time, energy,
                                                     bins=[t_bin_seq,
                                                           chan_bins],
                                                     normed=False)
    ## Need counts/dt to have units of count rate
    ## Doing it by multiplying by 1/dt, to keep it as an int and not get
    ## typecasting errors.
    dt_inv_int = np.int64(1. / dt)
    lc_2d *= dt_inv_int
    return lc_2d


def get_key_val(fits_file, ext, keyword):
    """
    Gets the value of a keyword from a FITS header. Keyword does not seem to be
    case-sensitive.

    Parameters
    ----------
    fits_file : str
        The full path of the FITS file.

    ext : int
        The FITS extension in which to search for the given keyword.

    keyword : str
        The keyword for which you want the associated value.

    Returns
    -------
    any type
        Value of the given keyword.

    Raises
    ------
    IOError if the input file isn't actually a FITS file.

    """

    ext = np.int8(ext)
    assert (ext >= 0 and ext <= 2)
    keyword = str(keyword)

    try:
        hdulist = fits.open(fits_file)
    except IOError:
        print("\tERROR: File does not exist: %s" % fits_file)
        exit()

    key_value = hdulist[ext].header[keyword]
    hdulist.close()

    return key_value


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    From https://docs.python.org/2/library/itertools.html#recipes
    Used when reading lines in the file so I can peek at the next line.

    Parameters
    ----------
    an iterable, like a list or an open file

    Returns
    -------
    The next two items in an iterable, like in the example a few lines above.

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


class Energy_lags(object):
    """
    Compute the lag-energy spectrum. Assumes incoming power spectra are
    abs rms^2 normalized and NOT Poisson-noise-subtracted.
    """

    def __init__(self, in_tab, low_freq_bound=3.,
                 high_freq_bound=9., debug=False):
        assert isinstance(low_freq_bound, float), "`low_freq_bound` should be a float."
        assert isinstance(high_freq_bound, float), "`high_freq_bound` should be a float."
        assert isinstance(debug, bool), "`debug` should be a boolean."
        assert isinstance(in_tab, Table), "`in_tab` should be an astropy Table object."
        assert low_freq_bound < high_freq_bound, "`low_freq_bound` must be less than `high_freq_bound`."
        self.debug = debug

        if self.debug:
            print("L f bound: " + str(low_freq_bound))
            print("H f bound: " + str(high_freq_bound))

        assert len(np.shape(in_tab['PSD_REF'])) == 1
        assert len(np.shape(in_tab['PSD_CI'])) == 2
        assert len(np.shape(in_tab['CROSS'])) == 2
        self.n_bins = in_tab.meta['N_BINS']
        self.n_chans = in_tab.meta['N_CHANS']
        self.n_seg = in_tab.meta['N_SEG']
        self.dt = in_tab.meta['DT']
        self.rate_ref = in_tab.meta['RATE_REF']
        self.rate_ci = np.asarray(in_tab.meta['RATE_CI'][1:-1].split(","),
                                  dtype='float')

        freq = np.abs(in_tab['FREQUENCY'][0:int(self.n_bins / 2 + 1)])
        ## Using the unfiltered (but still shifted) cs & psd
        cs = in_tab['CROSS'][0:int(self.n_bins / 2 + 1), :]
        psd_ci = in_tab['PSD_CI'][0:int(self.n_bins / 2 + 1), :]
        psd_ref = in_tab['PSD_REF'][0:int(self.n_bins / 2 + 1)]

        a, f_mask_low = find_nearest(freq, low_freq_bound)
        a, f_mask_high = find_nearest(freq, high_freq_bound)
        if self.debug:
            print("Low freq: " + str(freq[f_mask_low]))
            print("High freq: " + str(freq[f_mask_high]))

        f_span = f_mask_high - f_mask_low + 1  ## including both ends
        mean_f = np.mean(freq[f_mask_low:f_mask_high + 1])

        temp_tab = Table()
        tt_f = np.repeat(mean_f, self.n_chans)
        temp_tab['FREQUENCY'] = Column(tt_f, dtype=np.float32, unit='Hz')
        tt_cs = np.mean(cs[f_mask_low:f_mask_high + 1, ], axis=0)
        temp_tab['CROSS'] = Column(tt_cs, dtype=np.complex128, unit='---')
        tt_pc = np.mean(psd_ci[f_mask_low:f_mask_high + 1, ], axis=0)
        temp_tab['POWER_CI'] = Column(tt_pc, dtype=np.float64, unit='---')
        tt_pr = np.repeat(np.mean(psd_ref[f_mask_low:f_mask_high + 1]), self.n_chans)
        temp_tab['POWER_REF'] = Column(tt_pr, dtype=np.float64, unit='---')
        if self.debug:
            print(temp_tab.info)

        energy_tab = Table()
        energy_tab['CHANNEL'] = Column(np.arange(int(self.n_chans)),
                                       description='Energy channel of interest',
                                       dtype=np.int, unit='chan')
        energy_tab['PHASE_LAG'] = Column(-np.arctan2(temp_tab['CROSS'].imag,
                                                     temp_tab['CROSS'].real),
                                         unit='rad', description='Phase lag',
                                         dtype=np.float64)
        energy_tab['PHASE_ERR'] = Column(self._phase_err(temp_tab, f_span),
                                         unit='rad', dtype=np.float64,
                                         description='Error on phase lag')
        energy_tab['TIME_LAG'] = Column(self._phase_to_tlags(energy_tab['PHASE_LAG'],
                                                             temp_tab['FREQUENCY']),
                                        unit='s', dtype=np.float64, description='Time lag')
        energy_tab['TIME_ERR'] = Column(self._phase_to_tlags(energy_tab['PHASE_ERR'],
                                                             temp_tab['FREQUENCY']),
                                        unit='s', dtype=np.float64, description='Error on time lag')

        if self.debug:
            print(energy_tab.info)

        self.energy_tab = energy_tab

    def _phase_err(self, tab, n_range=1):
        """
        Compute the error on the complex phase (in radians) via the coherence.
        Power is assumed to be abs rms^2 units and NOT Poisson-noise-subtracted.

        Parameters
        ----------
        tab : Astropy table

        n_range : int or np.array of ints
            Number of frequency bins averaged over per new frequency bin for
            lags. For energy lags, this is the number of frequency bins averaged
            over. For frequency lags not re-binned in frequency, this is 1.
            Same as K in equations in Section 2 of Uttley et al. 2014.

        Returns
        -------
        phase_err : np.array of floats
            1-D array of the error on the phase of the lag.
        """
        #         if self.debug:
        #             print("Phase err")
        coherence = self._comp_coherence(tab, n_range)
        coherence[coherence == 0] = 1e-14
        phase_err = np.sqrt(np.abs(1 - coherence) / \
                            (2 * coherence * n_range * self.n_seg))
        return phase_err

    def _phase_to_tlags(self, phase, f):
        """
        Convert a complex-plane cross-spectrum phase (in radians) to a time lag
        (in seconds).
        """
        #         if self.debug:
        #             print("Phase to time lags")
        f[f == 0] = 1e-14
        tlags = phase / (2.0 * np.pi * f)
        return tlags

    def _comp_coherence(self, tab, n_range):
        """
        Compute the raw coherence of the cross spectrum. Coherence equation from
        Uttley et al 2014 eqn 11, bias term equation from footnote 4 on same
        page.
        Assuming that the power spectra have abs rms^2 normalization and do NOT
        have Poisson noise subtracted.

        Parameters
        ----------
        tab : astropy Table

        Returns
        -------
        coherence : np.array of floats
            The raw coherence of the cross spectrum. (Uttley et al 2014, eqn 11)
            Size = n_chans.
        """
        #         if self.debug:
        #             print("Compute coherence")
        #             print("nrange: "+str(n_range))
        #         cs_bias = self._bias_term(tab, n_range)
        ## Setting bias to 0 since i'm using filtered cs and psds for the computations.
        cs_bias = 0  ## Reasonable assumption, most of the time.
        if self.debug:
            print("WARNING: Assuming bias term for coherence is 0.")
        temp_2 = (np.abs(tab['CROSS']) * (2 * self.dt / self.n_bins)) ** 2 - cs_bias
        powers = tab['POWER_CI'] * tab['POWER_REF']
        powers[powers == 0] = 1e-14
        coherence = temp_2 / powers
        return np.real(coherence)

    def _bias_term(self, tab, n_range):
        """
        Compute the bias term to be subtracted off the cross spectrum to compute
        the covariance spectrum. Equation in Equation in footnote 4 (section 2.1.3,
        page 12) of Uttley et al. 2014.

        Assumes power spectra are abs rms^2 normalized but NOT Poisson-noise-
        subtracted.

        Parameters
        ----------
        tab : astropy.table.Table

        n_range : int
            Number of frequency bins averaged over per new frequency bin for lags.
            For energy lags, this is the number of frequency bins averaged over. For
            frequency lags not re-binned in frequency, this is 1. For frequency lags
            that have been re-binned, this is a 1-D array with ints of the number of
            old bins in each new bin. Same as K in equations in Section 2 of
            Uttley et al. 2014. Default=1

        Returns
        -------
        n_squared : float
            The bias term to be subtracted off the cross spectrum for computing the
            covariance spectrum. Equation in footnote 4 (section 2.1.3, page 12) of
            Uttley et al. 2014.

        """
        ## Compute the Poisson noise level in absolute rms units
        Pnoise_ref = self.rate_ref * 2.0
        Pnoise_ci = self.rate_ci * 2.0

        #         ## Normalizing power spectra to absolute rms normalization
        #         ## Not subtracting the noise (yet)!
        #         abs_ci = tab['POWER_CI'] * (2.0 * self.dt / n_range)
        #         abs_ref = tab['POWER_REF'] * (2.0 * self.dt / n_range)

        temp_a = (tab['POWER_REF'] - Pnoise_ref) * Pnoise_ci
        temp_b = (tab['POWER_CI'] - Pnoise_ci) * Pnoise_ref
        temp_c = Pnoise_ref * Pnoise_ci

        n_squared = np.asarray((temp_a + temp_b + temp_c) / (n_range * self.n_seg))
        return n_squared

    # def _phase_err(self, tab, n_range=1):
    #     """
    #     Compute the error on the complex phase (in radians) via the coherence.
    #     Power is assumed to be abs rms^2 units and NOT Poisson-noise-subtracted.
    #
    #     Parameters
    #     ----------
    #     tab : Astropy table
    #
    #     n_range : int or np.array of ints
    #         Number of frequency bins averaged over per new frequency bin for
    #         lags. For energy lags, this is the number of frequency bins averaged
    #         over. For frequency lags not re-binned in frequency, this is 1.
    #         Same as K in equations in Section 2 of Uttley et al. 2014.
    #
    #     Returns
    #     -------
    #     phase_err : np.array of floats
    #         1-D array of the error on the phase of the lag.
    #     """
    #     if self.debug:
    #         print("Phase err")
    #     coherence = self._comp_coherence(tab, n_range)
    #     #         print("Shape coh:"+str(np.shape(coherence)))
    #     coherence[coherence == 0] = 1e-14
    #     phase_err = np.sqrt(np.abs(1 - coherence) / \
    #                         (2 * coherence * n_range * self.n_seg))
    #     return phase_err
    #
    # def _phase_to_tlags(self, phase, f):
    #     """
    #     Convert a complex-plane cross-spectrum phase (in radians) to a time lag
    #     (in seconds).
    #     """
    #     if self.debug:
    #         print("Phase to time lags")
    #     f[f == 0] = 1e-14
    #     #         print("Shape phase: "+str(np.shape(phase)))
    #     #         print("Shape f: "+str(np.shape(f)))
    #     tlags = phase / (2.0 * np.pi * f)
    #     return tlags
    #
    # def _comp_coherence(self, tab, n_range):
    #     """
    #     Compute the raw coherence of the cross spectrum. Coherence equation from
    #     Uttley et al 2014 eqn 11, bias term equation from footnote 4 on same
    #     page.
    #     Assuming that the power spectra have abs rms^2 normalization and do NOT
    #     have Poisson noise subtracted.
    #
    #     Parameters
    #     ----------
    #     tab : astropy Table
    #
    #     Returns
    #     -------
    #     coherence : np.array of floats
    #         The raw coherence of the cross spectrum. (Uttley et al 2014, eqn 11)
    #         Size = detchans.
    #     """
    #     if self.debug:
    #         print("Compute coherence")
    #         print("nrange: " + str(n_range))
    #     # cs_bias = self._bias_term(tab, n_range)
    #     ## Setting bias to 0 since i'm using filtered cs and psds for the computations.
    #     cs_bias = 0  ## Reasonable assumption, most of the time.
    #     if self.debug:
    #         print("WARNING: Assuming bias term for coherence is 0.")
    #     temp_2 = (np.abs(tab['CROSS']) * (
    #     2 * self.dt / self.n_bins)) ** 2 - cs_bias
    #     powers = tab['POWER_CI'] * tab['POWER_REF']
    #     powers[powers == 0] = 1e-14
    #     coherence = temp_2 / powers
    #     #         print(coherence)
    #     return np.real(coherence)
    #
    # def _bias_term(self, tab, n_range):
    #     """
    #     Compute the bias term to be subtracted off the cross spectrum to compute
    #     the covariance spectrum. Equation in Equation in footnote 4 (section 2.1.3,
    #     page 12) of Uttley et al. 2014.
    #
    #     Assumes power spectra are abs rms^2 normalized but NOT Poisson-noise-
    #     subtracted.
    #
    #     Parameters
    #     ----------
    #     tab : astropy.table.Table
    #
    #     n_range : int
    #         Number of frequency bins averaged over per new frequency bin for lags.
    #         For energy lags, this is the number of frequency bins averaged over. For
    #         frequency lags not re-binned in frequency, this is 1. For frequency lags
    #         that have been re-binned, this is a 1-D array with ints of the number of
    #         old bins in each new bin. Same as K in equations in Section 2 of
    #         Uttley et al. 2014. Default=1
    #
    #     Returns
    #     -------
    #     n_squared : float
    #         The bias term to be subtracted off the cross spectrum for computing the
    #         covariance spectrum. Equation in footnote 4 (section 2.1.3, page 12) of
    #         Uttley et al. 2014.
    #
    #     """
    #     ## Compute the Poisson noise level in absolute rms units
    #     Pnoise_ref = self.rate_ref * 2.0
    #     Pnoise_ci = self.rate_ci * 2.0
    #
    #     #         ## Normalizing power spectra to absolute rms normalization
    #     #         ## Not subtracting the noise (yet)!
    #     #         abs_ci = tab['POWER_CI'] * (2.0 * self.dt / n_range)
    #     #         abs_ref = tab['POWER_REF'] * (2.0 * self.dt / n_range)
    #
    #     temp_a = (tab['POWER_REF'] - Pnoise_ref) * Pnoise_ci
    #     temp_b = (tab['POWER_CI'] - Pnoise_ci) * Pnoise_ref
    #     temp_c = Pnoise_ref * Pnoise_ci
    #
    #     n_squared = np.asarray(
    #         (temp_a + temp_b + temp_c) / (n_range * self.n_seg))
    #     # print(n_squared)
    #     return n_squared