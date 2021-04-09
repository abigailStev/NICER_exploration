import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import scipy.fftpack as fftpack
# from scipy.stats import binned_statistic
import os
import gc
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.colors as colors
from xcor_tools_nicer import make_binned_lc, make_1Dlightcurve, find_nearest, geom_rb

__author__ = "Abigail Stevens <abigailstev@gmail.com>"
__year__ = "2018-2021"

class SegQPO(object):
    """
    Generic QPO class. Used for each segment.
    """
    def __init__(self, lc_seg):

        ## Computing Fourier transform
        fft_seg, self.rate_seg = self._fft(lc_seg)

        ## Computing PSD
        self.psd_seg = self._power(fft_seg).real

        ## Check values
        assert np.isfinite(self.psd_seg).any(), "psd_seg has infinite value(s)."
        assert not np.isnan(self.psd_seg).any(), "psd_seg has NaN value(s)."
        assert np.isfinite(self.rate_seg), \
            "rate_seg has infinite value(s)."
        assert not np.isnan(self.rate_seg), \
            "rate_seg has NaN value(s)."

    def _fft(self, lc):
        """
        Subtract the mean from a light curve and take the Fourier transform of
        the mean-subtracted light curve. Assumes that the time bins are along
        axis=0 and that the light curve is in units of photon counts per second
        (count rate).
        """
        means = np.mean(lc, axis=0)
        # print("Shape means: "+str(np.shape(means)))
        # print("Shape lc: "+str(np.shape(lc)))
        if len(np.shape(lc)) == 2:
            lc_sub_mean = lc - means[np.newaxis, :]
        elif len(np.shape(lc)) == 1:
            lc_sub_mean = lc - means
        else:
            print(
            "WARNING: Light curve array does not have expected dimensions. " 
            "Do not assume the mean count rate was subtracted correctly " 
            "before FFT.")
            lc_sub_mean = lc - means
        return fftpack.fft(lc_sub_mean, axis=0), means

    def _power(self, fft):
        """
        Take the power spectrum of a Fourier transform.
        Tested in trying_multiprocessing.ipynb, and this is faster than
        multiprocessing with mapping or joblib Parallel.
        """
        return np.multiply(fft, np.conj(fft))


def each_file(out_file_base, obj_name, in_file, gti_file, n_bins, dt, df,
              n_seconds, band_le, band_he, nyquist, rebin_by):
    """

    :param out_file:
    :param obj_name:
    :param in_file:
    :param gti_file:
    :param n_seg:
    :param n_bins:
    :param dt:
    :param df:
    :param n_seconds:
    :param band_le:
    :param band_he:
    :param nyquist:
    :return:
    """
    psd = np.zeros((n_bins, 1))
    psds_per_gti = np.zeros((n_bins, 1))
    rate = np.asarray([])
    rates_per_gti = np.asarray([])
    n_seg = 0
    n_gti = 0
    first_start_time = 0
    past_first_start_time = False

    try:
        fits_hdu = fits.open(in_file, memmap=True)
        time = fits_hdu['EVENTS'].data.field('TIME')  ## ext 1
        energy = fits_hdu['EVENTS'].data.field('PI')
        det = fits_hdu['EVENTS'].data.field('DET_ID')
        if gti_file:
            gti_tab = Table.read(gti_file)
            gti_starttimes = gti_tab['START']
            gti_stoptimes = gti_tab['STOP']
        else:
            gti_starttimes = fits_hdu['GTI'].data.field('START')  ## ext 2
            gti_stoptimes = fits_hdu['GTI'].data.field('STOP')
        fits_hdu.close()
    except IOError:
        print("\tERROR: File does not exist: %s" % in_file)
        return n_seg, 0, 0, 0, [0]

    if len(time) > 0:
        start_time = time[0]
        final_time = time[-1]
        print("Number of GTIs in this file: %d" % len(gti_starttimes))
        if not past_first_start_time:
            first_start_time = start_time
        ## Removing the damaged FPMs, 11, 20, 22, and 60, and
        ## the 'bad' FPMs, 14, 34, and 54
        badFPM_mask = (det != 11) & (det != 14) & (det != 20) & \
                      (det != 22) & (det != 34) & (det != 54) & \
                      (det != 60)
        time = time[badFPM_mask]
        energy = energy[badFPM_mask]
        det = det[badFPM_mask]
        n_events = len(time)
        print("Time in file: %.2f" % (final_time - start_time))
        print("Number of events in file: %d" % n_events)

        for (start_gti, stop_gti) in zip(gti_starttimes, gti_stoptimes):
            if start_time <= start_gti:
                start_time = start_gti
            end_time = start_time + n_seconds

            ## Mask out the events that are before the 1st good start time
            dont_want = time < start_time
            time = time[~dont_want]
            energy = energy[~dont_want]
            det = det[~dont_want]

            psd_gti = np.zeros(n_bins)
            segs_per_gti = 0
            rate_gti = 0

            if (stop_gti - start_gti) > float(n_seconds):
                ############################
                ## Looping through segments
                ############################
                while end_time <= stop_gti and end_time <= final_time:

                    ## Getting all the events that belong to this time
                    ## segment
                    seg_mask = time < end_time
                    time_seg = time[seg_mask]
                    energy_seg = energy[seg_mask]
                    det_seg = det[seg_mask]

                    ## All MPUs, energy range
                    band_mask = (energy_seg >= int(band_le * 100)) & \
                                (energy_seg <= int(band_he * 100))
                    time_band = time_seg[band_mask]

                    ## Keep the stuff that isn't in this segment for next
                    ## time
                    time = time[~seg_mask]
                    energy = energy[~seg_mask]
                    det = det[~seg_mask]

                    ## Making populated LC
                    lc_band = make_1Dlightcurve(np.asarray(time_band),
                                                n_bins,
                                                start_time, end_time)
                    thing = SegQPO(lc_band)
                    del lc_band
                    rate = np.append(rate, thing.rate_seg)
                    psd = np.append(psd, thing.psd_seg[:, np.newaxis], axis=1)
                    psd_gti += thing.psd_seg
                    rate_gti += thing.rate_seg
                    if debug:
                        print(np.shape(psd))
                    del thing

                    ## Increment for next segment
                    n_seg += 1
                    segs_per_gti += 1
                    start_time = end_time
                    end_time = start_time + n_seconds
                    if n_seg % 50 == 0 and n_seg != 0:
                        print("\t%d" % n_seg)
                        gc.collect()

                    if debug and n_seg >= 5:
                        break

                ## Done with a GTI, just doing this for ones with
                ## events in the GTI
                if debug:
                    print("Segs per gti: %d" % segs_per_gti)
                if segs_per_gti == 0:
                    psd_gti = np.zeros(n_bins)
                    rate_gti = 1.
                else:
                    psd_gti /= segs_per_gti
                    rate_gti /= segs_per_gti
                psds_per_gti = np.append(psds_per_gti,
                                         psd_gti[:, np.newaxis], axis=1)
                rates_per_gti = np.append(rates_per_gti, rate_gti)
                n_gti += 1

            ## GTI finished
            if debug and n_seg >= 5:
                break
        print("File finished! Total segs in file: %d" % n_seg)

    else:
        print("WARNING: No events in file %s" % in_file)

    ## Chopping off the initializing zeros
    psd = psd[:, 1:]
    psds_per_gti = psds_per_gti[:, 1:]
    exposure = n_seconds * n_seg

    # print("Exposure: " + str(exposure))
    # print("Shape psd: " + str(np.shape(psd)))
    assert np.shape(psd)[-1] == len(rate), "Axes for psd & rate don't line up."

    ## Setting up for re-binning in frequency
    tmp0 = np.ones(int(n_bins / 2 + 1))
    tmp1, tmp2, tmp3, tmp4, tmp5 = geom_rb(tmp0, tmp0, tmp0,
                                           rebin_const=rebin_by)
    new_f_n_bins = int(len(tmp1))
    dyn_psd = np.zeros((new_f_n_bins, int(n_seg)))
    dyn_gtipsd = np.zeros((new_f_n_bins, int(n_gti)))
    p_freq = freq[0:int(n_bins / 2)]

    ## Normalizing and re-binning the dynamical power spectra
    for i in range(n_seg):
        n_psd = psd[0:int(n_bins/2), i]*2*dt / n_bins / rate[i]**2
        rb_freq, dyn_psd[:, i], rb_err, f_min, f_max = geom_rb(p_freq, n_psd,
                                                    tmp0, rebin_const=rebin_by)

    ## Normalizing, re-binning, and plotting the average power spectrum of
    ## each GTI
    font_prop = font_manager.FontProperties(size=14)
    gtipsd_list = []
    for i in range(n_gti):
        n_psd = psds_per_gti[0:int(n_bins/2), i]*2*dt / n_bins / rates_per_gti[i]**2
        rb_freq, dyn_gtipsd[:, i], rb_err, f_min, f_max = geom_rb(p_freq, n_psd,
                                                    tmp0, rebin_const=rebin_by)
        plt.plot(rb_freq, dyn_gtipsd[:, i], lw=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(rb_freq[1], rb_freq[-1])
        plt.xticks(ticks=[0.1, 1, 10], labels=["0.1", "1", "10"])
        plt.ylim(1E-4, 8E-1)
        plt.xlabel("Frequency (Hz)", fontproperties=font_prop)
        plt.ylabel(r'Power [(rms/mean)$^{2}$/Hz]',
                   fontproperties=font_prop)
        plt.tick_params(axis='x', labelsize=14, bottom=True, top=True,
                        labelbottom=True, labeltop=False, direction="in")
        plt.tick_params(axis='y', labelsize=14, left=True, right=True,
                        labelleft=True, labelright=False, direction="in")
        gtipsd_file = "%s_gti%d.png" % (out_file_base, i)
        gtipsd_file = gtipsd_file.replace(obj_name, "%s/psds" % obj_name)
        plt.savefig(gtipsd_file)
        plt.close()
        gtipsd_list.append(gtipsd_file)

    #######################################################
    ## Saving the dynamical power spectrum to a FITS table
    #######################################################
    out_tab = Table()
    out_tab.add_column(Column(data=rb_freq, name="FREQUENCY", unit="Hz"))
    out_tab.add_column(Column(data=dyn_psd, name="PSD"))
    out_tab.add_column(Column(data=dyn_gtipsd, name="PSD_PER_GTI"))
    out_tab.meta['OBJECT'] = obj_name
    out_tab.meta['INST'] = "NICER"
    out_tab.meta['TODAY'] = str(datetime.now())
    out_tab.meta['INFILE'] = in_file
    out_tab.meta['GTI_FILE'] = gti_file
    out_tab.meta['CLOCKTIM'] = first_start_time
    out_tab.meta['N_SEG'] = n_seg
    out_tab.meta['N_GTI'] = n_gti
    out_tab.meta['N_SEC'] = n_seconds
    out_tab.meta['OLD_NBIN'] = n_bins
    out_tab.meta['OLD_DF'] = df
    out_tab.meta['NEW_NBIN'] = new_f_n_bins
    out_tab.meta['REBIN'] = rebin_by
    out_tab.meta['NYQUIST'] = nyquist
    out_tab.meta['EXPOSURE'] = exposure
    out_tab.meta['DT'] = dt
    out_tab.meta['RANGE_B1'] = "%.2f-%.2f-keV" % (band_le, band_he)
    rb_out_file = out_file_base +"_dynpsd.fits"
    out_tab.write(rb_out_file, overwrite=True)

    # print(n_seg)
    # print(rb_freq)
    # print(dyn_psd)
    # print(rate)
    # print(gtipsd_list)
    return n_seg, rb_freq, dyn_psd, rate, gtipsd_list

# noinspection PyInterpreter
if __name__ == "__main__":
    ##################
    ## Getting set up
    ##################

    homedir = os.path.expanduser("~")
    exe_dir = homedir + "/Documents/Research/NICER_exploration"
    obj_name = "GX_339-4"
    obj_prefix = "gx339-2021"
    data_dir = homedir + "/Reduced_data/%s" % obj_name

    dt = 1 / 64.
    n_seconds = 32  # length of light curve segment, in seconds
    rebin_by = 1.03
    debug = False
    # debug = True
    # overwrite = False
    overwrite = True
    band_le = 2.
    band_he = 12.

    out_list_file = exe_dir + "/out/%s/%s_dynpsd-list.txt" % (obj_name,
                                                                obj_prefix)
    ## Need to have already made this file with the list of local filenames
    ## in data_dir
    input_list = exe_dir + "/in/%s_evtlists.txt" % obj_prefix
    ## Need to have already made this file in make_GTIs.ipynb
    gti_list = exe_dir + "/in/%s_32sGTIlists.txt" % obj_prefix

    ###########################################################################
    ###########################################################################

    print("\tDebugging? %s!" % str(debug))
    print("\tOverwriting? %s!" % str(overwrite))

    ## For making a light curve of each detector (to check for flares)
    detid_bin_file = exe_dir + "/in/detectors.txt"
    ## Could otherwise use n_chans = detchans FITS keyword in rsp matrix, and
    ## chan_bins=np.arange(detchans+1)  (need +1 for how histogram does ends)
    detID_bins = np.loadtxt(detid_bin_file, dtype=np.int)

    #################
    ## And it begins
    #################
    n_bins = int(n_seconds / dt)
    freq = fftpack.fftfreq(n_bins, d=dt)
    df = np.median(np.diff(freq))
    nyquist = 1.0 / (2.0 * dt)
    n_bins = int(n_seconds / dt)
    assert np.allclose(df, 1. / n_seconds)

    print("df: " + str(df))
    print("Nyquist: " + str(nyquist))
    print("n_bins: " + str(n_bins))
    print("dt: " + str(dt))

    print("List of event files: %s" % input_list)
    assert os.path.isfile(input_list)

    ## Input_file is a list of eventlists, so get each of those files
    data_files = [line.strip() for line in open(input_list)]
    if not data_files:  ## If data_files is an empty list
        raise Exception("ERROR: No files in the list of event lists: %s"
                        % input_list)
    ## Same with GTI files.
    gti_files = [line.strip() for line in open(gti_list)]
    if not gti_files:  ## If gti_files is an empty list
        raise Exception("ERROR: No files in the list of GTI files: %s"
                        % gti_list)

    ## Initializations for things we want to keep track of across all the files
    n_files = 1
    out_list = []
    gtipsd_list = np.asarray([])
    all_rate = np.asarray([])
    n_seg = 0
    file_segs = [0]

    ## Because we want to plot a binned dynamical power spectrum
    tmp0 = np.ones(int(n_bins / 2 + 1))
    tmp1, tmp2, tmp3, tmp4, tmp5 = geom_rb(tmp0, tmp0, tmp0,
                                           rebin_const=rebin_by)
    new_f_n_bins = int(len(tmp1))
    dyn_psd = np.zeros((new_f_n_bins, 1))

    ##################################
    ## Looping through the data files
    ##################################
    for (in_file, gti_file) in zip(data_files, gti_files):
        if in_file[0] == '.':
            in_file = exe_dir + in_file[1:]
        else:
            in_file = data_dir + "/" + in_file
            gti_file = data_dir + "/" + gti_file

        print("\nInput file %d/%d: %s" % (n_files, len(data_files), in_file))

        end_num = in_file.split('/')[-1].split('.')[0].split('-')[-1]
        try:
            filenum = int(end_num)
        except TypeError or ValueError:
            filenum = n_files
        if debug:
            out_file_base = "%s/out/%s/debug_%s-%s_%dsec_%ddt" % \
                        (exe_dir, obj_name, obj_prefix, str(filenum),
                         n_seconds, int(1 / dt))
        else:
            out_file_base = "%s/out/%s/%s-%s_%dsec_%ddt" % \
                        (exe_dir, obj_name, obj_prefix, str(filenum), n_seconds,
                        int(1 / dt))

        out_file = out_file_base + "_dynpsd.fits"
        if debug:
            print(out_file)
            print("Is file: ", os.path.isfile(out_file))
            print("Overwrite: ", overwrite)

        if overwrite or ((not overwrite) and (not os.path.isfile(out_file))):
            file_n_seg, p_freq, file_rb_psd, file_rate, \
                file_gtipsd_list = each_file(out_file_base, obj_name, in_file,
                                              gti_file, n_bins, dt, df,
                                              n_seconds, band_le, band_he,
                                              nyquist, rebin_by)
            n_seg += file_n_seg
            dyn_psd = np.append(dyn_psd, file_rb_psd, axis=1)
            all_rate = np.append(all_rate, file_rate)
            gtipsd_list = np.append(gtipsd_list, file_gtipsd_list, axis=0)
            file_segs.append(n_seg)
        else:
            print("File has been processed previously. I hope it was with the "
                  "same energy bands! Moving on.")
        out_list.append(out_file)
        n_files += 1

    print("Finished processing all files in list.")
    if n_seg == 0:
        print("WARNING: No files have been processed. Re-run with new data or "
              "with overwrite=True.")
        exit()

    ## Chopping off initializing zeroes
    dyn_psd = dyn_psd[:, 1:]
    assert len(file_segs) == n_files, "Don't have correct segment separators for files."

    ## Prepping for output of whole shebang
    out_file_base = "%s/out/%s/%s_%dsec_%ddt" % \
                    (exe_dir, obj_name, obj_prefix, n_seconds, int(1 / dt))
    plot_file = "%s_dynpsd_rb.png" % out_file_base
    gtipsd_outfile = "%s_gtipsdlist.txt" % out_file_base
    psd_gif_file = "%s_psd.gif" % out_file_base

    ## Saving all the gti psd plots to a list
    with open(gtipsd_outfile, 'w') as f:
        [f.write("%s\n" % gtipsd_file) for gtipsd_file in gtipsd_list]

    #########################################
    ## Plotting the dynamical power spectrum
    #########################################
    lf = int(find_nearest(p_freq, 0.1)[1])
    uf = int(find_nearest(p_freq, 20)[1])
    amp_min = 5E-4
    amp_max = 5E-1
    seg_num = np.arange(0, n_seg+1, dtype=int)
    font_prop = font_manager.FontProperties(size=20)
    fig, ax = plt.subplots(1, 1, figsize=(13.5, 6.75), dpi=300)

    plt.pcolor(seg_num, p_freq[lf-1:uf+1], dyn_psd[lf-1:uf+1,:],
               shading='auto', cmap='inferno',
               norm=colors.LogNorm(vmin=amp_min, vmax=amp_max))
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label(r'Power [(rms/mean)$^{2}$/Hz]',
                   fontproperties=font_prop)
    cb_ax = cbar.ax
    cb_ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Frequency (Hz)', fontproperties=font_prop)
    ax.set_yscale('log')
    ax.set_ylim(p_freq[lf], p_freq[uf])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    for file_sep in file_segs:
        ax.axvline(file_sep, c='black', lw=1)
    # ax.set_xlim(1, lfqpo.n_seg+1)
    ax.set_xlabel(r'Elapsed time ($\times$ %d s)' % n_seconds,
                  fontproperties=font_prop)
    ax.set_title("%.0f-%.0f keV dynamical power spectrum" % (band_le, band_he),
                 fontproperties=font_prop)
    ## Setting the axes' minor ticks. It's complicated.
    if debug:
        xLocator = MultipleLocator(1)
        ax.set_xticks(np.arange(0, n_seg, 5))
    else:
        xLocator = MultipleLocator(100)
        ax.set_xticks(np.arange(0, n_seg, 500))
    ax.xaxis.set_minor_locator(xLocator)
    ax.tick_params(axis='both', labelsize=20)
    ax.tick_params(which='major', width=1.5, length=7)
    ax.tick_params(which='minor', width=1.5, length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    # Save this monstrosity before you lose it, and tell the user where it is
    plt.savefig(plot_file)
    print("Dynamical power spectrum: %s" % plot_file)
    if debug:
        subprocess.call(['open', plot_file])

    ## Making a gif of all the gti psd plots
    print("GIF command:")
    print("convert -delay 35 @%s %s" % (gtipsd_outfile, psd_gif_file))
    try:
        cmd = ['convert', '-delay', '35', '@%s' % gtipsd_outfile, psd_gif_file]
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        output, error = child.communicate()
        if child.returncode != 0:
            raise Exception("Oops, GIF wasn't made.")
    except:
        print("Internal problem making the GIF")
    else:
        print("GIF of power spectra for each GTI: %s" % psd_gif_file)
        subprocess.call(['open', '-a', 'Firefox', psd_gif_file])
