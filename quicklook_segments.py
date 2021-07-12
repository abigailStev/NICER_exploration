#!/usr/bin/env python
"""
Gets segment information for a lot of data, saves it to a table, also makes
power spectra of each GTI (averaging segments in there). Uses GTIs from
make_GTIs.ipynb in the same directory.

How to call at the command line:
python quicklook_segments.py

or to see how long it runs, type:
time python quicklook_segments.py

"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
import scipy.fftpack as fftpack
from datetime import datetime
import os
import gc
from xcor_tools_nicer import make_1Dlightcurve, find_nearest
from fast_histogram import histogram1d
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.fitting import LevMarLSQFitter

__author__ = "Abigail Stevens <abigailstev@gmail.com>"
__year__ = "2018-2021"


class PSD(object):
    """
    Generic class to make a power spectrum. Used for each segment.
    """
    def __init__(self, lc):

        ## Computing Fourier transform
        fft, self.rate = self._fft(lc)

        ## Computing PSD
        self.psd = self._power(fft).real

        ## Check values
        assert np.isfinite(self.psd).all(), "psd has infinite value(s)."
        assert not np.isnan(self.psd).all(), "psd has NaN value(s)."
        assert np.isfinite(self.rate), "rate has infinite value(s)."
        assert not np.isnan(self.rate), "rate has NaN value(s)."

    def _fft(self, lc):
        """
        Subtract the mean from a light curve and take the Fourier transform of
        the mean-subtracted light curve. Assumes that the time bins are along
        axis=0 and that the light curve is in units of photon counts per second
        (count rate).
        """
        means = np.mean(lc, axis=0)
        lc_sub_mean = lc - means
        return fftpack.fft(lc_sub_mean, axis=0), means

    def _power(self, fft):
        """
        Take the power spectrum of a Fourier transforms.
        Tested in trying_multiprocessing.ipynb, and this is faster than
        multiprocessing with mapping or joblib Parallel.
        """
        return np.multiply(fft, np.conj(fft))

def eq_width(before_feline, feline, after_feline, x_cont, x_fe):
    """

    :param before_feline:
    :param feline:
    :param after_feline:
    :return:
    """
    cont = np.append(before_feline[0:-1], after_feline[0:-1])
    feline = feline[0:-1]
    assert len(cont) == len(x_cont), "ERROR: Continuum bins don't have same number."
    assert len(feline) == len(x_fe), "ERROR: Iron bins don't have same number."

    for i in range(0, 2):
        pl_init = PowerLaw1D(amplitude=50, x_0=1000., alpha=4.)
        fit_pl = LevMarLSQFitter()
        pl = fit_pl(pl_init, x_cont, cont)

    ratios = feline/pl(x_fe)
    # if debug:
    #     print(ratios)
    return ratios


def per_file(out_file, obj_name, in_file, gti_file, n_seg, n_bins, dt, df,
             n_seconds, broad_le, broad_he, soft_le, soft_he, hard_le, hard_he,
             rms_lf, rms_hf, x_cont, x_fe, nyquist):
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
    :param broad_le:
    :param broad_he:
    :param soft_le:
    :param soft_he:
    :param hard_le:
    :param hard_he:
    :param rms_lf:
    :param rms_hf:
    x_cont
    x_fe
    :return:
    """
    print(out_file)
    if not os.path.isdir(os.path.dirname(out_file)):
        print(os.path.dirname(out_file))
        os.makedirs(os.path.dirname(out_file))

    with open(out_file, 'w') as f:
        f.write("# OBJECT = %s\n" % obj_name)
        f.write("# INST = NICER\n")
        f.write("# TODAY = %s\n" % str(datetime.now()))
        f.write("# INFILE = %s\n" % in_file)
        f.write("# GTIFILE = %s\n" % gti_file)
        f.write("# N_BINS = %d\n" % n_bins)
        f.write("# DT = %.9f s\n" % dt)
        f.write("# DF = %.9f Hz\n" % df)
        f.write("# NYQUIST = %.2f Hz\n" % nyquist)
        f.write("# N_SECOND = %d\n" % n_seconds)
        f.write("# BROAD_BAND: %.2f-%.2f keV\n" % (broad_le, broad_he))
        f.write("# Count rates have been scaled to 49 FPMs\n")
        f.write("# SOFT_BAND: %.2f-%.2f keV\n" % (soft_le, soft_he))
        f.write("# HARD_BAND: %.2f-%.2f keV\n" % (hard_le, hard_he))
        f.write("# RMS_RANGE: %.2f-%.2f Hz\n" % (rms_lf, rms_hf))
        f.write("# FE_RATIO: 6.2, 6.3, 6.4, 6.5, 6.6 keV\n")

        f.write("# \n")
        f.write("# ni+obsID start_time end_time total_rate broad_rate rms "
                "hard_rate soft_rate hardness fe_ratio\n")
        f.write("# \n")
        # print(file_info)
    print("Output saving to: %s \n" % out_file)

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
        print("\tERROR: File does not exist: %s; %s" % (in_file, gti_file))
        return n_seg

    if len(time) > 0:
        # print("Events in file: ", len(time))
        # try:
        #     local_infilename = in_file.split('/')[5]
        #     if '_' in local_infilename:
        #         obsID = local_infilename.split('_')[0][2:]
        #     else:
        #         obsID = local_infilename[2:13]
        #     obsID = int(obsID)
        # except TypeError or ValueError:
        # obsID = n_files
        obsID = 666
        # print("ObsID: ", obsID)

        print("Number of GTIs in this file: %d" % len(gti_starttimes))
        ## The things i want to keep track of for every segment
        file_obsID = []
        file_start_time = []
        file_end_time = []
        file_seg_rate = []
        file_broad_rate = []
        file_rms = []
        file_hard_rate = []
        file_soft_rate = []
        file_hardness = []
        file_fe_ratio = np.zeros((5,1), dtype=float)

        start_time = time[0]
        final_time = time[-1]

        ## Removing the damaged FPMs, 11, 20, 22, and 60, and
        ## the frequently 'bad' FPMs, 14, 34, and 54
        badFPM_mask = (det != 11) & (det != 14) & (det != 20) & \
                      (det != 22) & (det != 34) & (det != 54) & \
                      (det != 60)
        time = time[badFPM_mask]
        energy = energy[badFPM_mask]
        det = det[badFPM_mask]

        for (start_gti, stop_gti) in zip(gti_starttimes, gti_stoptimes):
            # print('GTI Start time: %.15f' % start_gti)
            # print('GTI Stop time: %.15f' % stop_gti)
            if start_time <= start_gti:
                start_time = start_gti
            end_time = start_time + n_seconds

            # print('Time in GTI: %.6f' % (stop_gti - start_gti))
            # print('n_seg: %d' % n_seg)

            ## Mask out the events that are before the 1st good start time
            dont_want = time < start_time
            time = time[~dont_want]
            energy = energy[~dont_want]
            det = det[~dont_want]
            num_fpms = len(np.unique(det))
            if debug:
                ## this prints each GTI if debugging until you have 5 segments
                print("Number of FPMs: ", num_fpms)

            if (stop_gti - start_gti) > float(n_seconds):
                ############################
                ## Looping through segments
                ############################
                while end_time <= stop_gti and end_time <= final_time:

                    seg_mask = time < end_time
                    time_seg = time[seg_mask]
                    energy_seg = energy[seg_mask]
                    # print("Events in here: %d" % len(time_seg))

                    ## For all MPUs, broad 0.25-10 keV
                    broad_mask = (energy_seg >= int(broad_le * 100)) & \
                                 (energy_seg <= int(broad_he * 100))
                    time_broad = time_seg[broad_mask]

                    ## Soft band is 1-2 keV (all MPUs)
                    soft_mask = (energy_seg >= int(soft_le * 100)) & \
                                (energy_seg <= int(soft_he * 100))
                    time_soft = time_seg[soft_mask]

                    ## Hard band is 7-10 keV (all MPUs)
                    hard_mask = (energy_seg >= int(hard_le * 100)) & \
                                (energy_seg <= int(hard_he * 100))
                    time_hard = time_seg[hard_mask]

                    ## Iron line equivalent width stuff
                    before_mask = (energy_seg >= 500) & (energy_seg <= 560)
                    before_feline = histogram1d(energy_seg[before_mask],
                                                range=[500,560], bins=6)
                    line_mask = (energy_seg >= 620) & (energy_seg <= 680)
                    feline = histogram1d(energy_seg[line_mask],
                                         range=[620,680], bins=6)
                    after_mask = (energy_seg >= 750) & (energy_seg <= 810)
                    after_feline = histogram1d(energy_seg[after_mask],
                                               range=[750,810], bins=6)
                    # print(max(before_feline), max(feline), max(after_feline))

                    ## Keep the stuff that isn't in this segment for next time
                    time = time[~seg_mask]
                    energy = energy[~seg_mask]
                    det = det[~seg_mask]

                    ## 'Populating' all the discrete events into a continuous
                    ## lightcurve
                    lc_seg = make_1Dlightcurve(np.asarray(time_seg), n_bins,
                                               start_time, end_time)
                    lc_broad = make_1Dlightcurve(np.asarray(time_broad),
                                                 n_bins, start_time, end_time)
                    lc_hard = make_1Dlightcurve(np.asarray(time_hard), n_bins,
                                                start_time, end_time)
                    lc_soft = make_1Dlightcurve(np.asarray(time_soft), n_bins,
                                                start_time, end_time)

                    seg_rate = np.mean(lc_seg) * (49. / num_fpms)
                    hard_rate = np.mean(lc_hard) * (49. / num_fpms)
                    soft_rate = np.mean(lc_soft) * (49. / num_fpms)
                    broad_rate = np.mean(lc_broad) * (49. / num_fpms)

                    del lc_hard
                    del lc_seg
                    del lc_soft

                    fe_ratios = eq_width(before_feline, feline, after_feline,
                                        x_cont, x_fe)

                    ## Compute hardness ratio
                    if soft_rate != 0:
                        hardness = hard_rate / soft_rate
                    else:
                        hardness = -666.

                    psd_rms = PSD(lc_broad)
                    # print(np.shape(psd_rms.psd))
                    # print(np.shape(psd_rms.rate))

                    ## Compute the integrated rms in the broad band.
                    ## Compute Poisson noise level from >30 Hz.
                    temp_psd = np.asarray(psd_rms.psd)
                    temp_fracpsd = temp_psd * 2 * dt / n_bins / (psd_rms.rate ** 2)
                    noise_level = np.mean(temp_fracpsd[hf:int(n_bins / 2)])
                    # print("%.3g  %.3g" % (noise_level, 2./psd_rms.rate))
                    temp_fracpsd -= noise_level
                    var = np.sum(temp_fracpsd[lf:uf] * df)
                    if var >= 0:
                        rms = np.sqrt(var)
                    else:
                        rms = 666
                    # print(rms)

                    ## Saving
                    file_obsID.append(obsID)
                    file_start_time.append(start_time)
                    file_end_time.append(end_time)
                    file_seg_rate.append(seg_rate)
                    file_broad_rate.append(broad_rate)
                    file_rms.append(rms)
                    file_hard_rate.append(hard_rate)
                    file_soft_rate.append(soft_rate)
                    file_hardness.append(hardness)
                    file_fe_ratio = np.append(file_fe_ratio,
                                              fe_ratios[:,np.newaxis], axis=1)
                    # print(np.shape(file_fe_ratio))
                    # print(file_fe_ratio)

                    del psd_rms
                    del lc_broad

                    ## Increment for next segment
                    n_seg += 1
                    start_time = end_time
                    end_time = start_time + n_seconds

                    if n_seg % 50 == 0:
                        print("\t%d" % n_seg)
                        gc.collect() ## collect garbage to help memory

                    if debug and n_seg >= 5:
                        break

            ## Done with a GTI
            # print("new GTI")
            if debug and n_seg >= 5:
                break

        ## Make sure everything is the right data type
        file_obsID = np.array(file_obsID, dtype='int32')
        file_start_time = np.array(file_start_time, dtype='float64')
        file_end_time = np.array(file_end_time, dtype='float64')
        file_seg_rate = np.array(file_seg_rate, dtype='float64')
        file_broad_rate = np.array(file_broad_rate, dtype='float64')
        file_rms = np.array(file_rms, dtype='float64')
        file_hard_rate = np.array(file_hard_rate, dtype='float64')
        file_soft_rate = np.array(file_soft_rate, dtype='float64')
        file_hardness = np.array(file_hardness, dtype='float64')
        file_fe_ratio = file_fe_ratio[:, 1:]
        # print(np.shape(file_fe_ratio))


        ## Done with a file
        print("Total segs in file: %d" % n_seg)

        file_info = np.stack((file_obsID, file_start_time,
                              file_end_time, file_seg_rate,
                              file_broad_rate, file_rms,
                              file_hard_rate, file_soft_rate,
                              file_hardness, file_fe_ratio[0,:],
                              file_fe_ratio[1,:], file_fe_ratio[2,:],
                              file_fe_ratio[3,:], file_fe_ratio[4,:]), axis=1)
        # print(file_info)
        ## Saving the output!
        with open(out_file, 'ab') as f:
            np.savetxt(f, file_info, fmt='%d %.9f %.9f %.6f %.6f %.9f '
                                         '%.6f %.6f %.8f %.4f %.4f %.4f %.4f '
                                         '%.4f')
    else:
        print("\tWARNING: No events in this file: %s" % in_file)

    return n_seg


# noinspection PyInterpreter
if __name__ == "__main__":

    ##########
    ## SET UP
    ##########

    obj_name = "Swift_J1728.9-3613"
    obj_prefix = "SwiftJ1728"

    homedir = os.path.expanduser("~")
    data_dir = "%s/Reduced_data/%s" % (homedir, obj_name)
    n_seconds = int(16)  # length of light curve segment, in seconds
    dt = 1./128.  # length of time bin, in seconds
    # debug = True
    debug = False
    # overwrite = True
    overwrite = False
    broad_le = 2.  # keV
    broad_he = 12.  # keV
    soft_le = 1.  # keV
    soft_he = 3.  # keV
    hard_le = 4.  # keV
    hard_he = 12.  # keV
    rms_lf = 1.  # Hz
    rms_hf = 15.  # Hz

    # obj_dir = "%s/Documents/Research/NICER_exploration" % (homedir)
    # out_list_file = "%s/out/%s/%s_seg-info-list.txt" % (obj_dir, obj_name,
    #                                                             obj_prefix)
    obj_dir = "%s/Documents/Research/%s" % (homedir, obj_prefix)
    out_list_file = "%s/out/%s_seg-info-list.txt" % (obj_dir, obj_prefix)

    ## Need to have already made this file with the list of local filenames
    ## in data_dir
    input_list = "%s/in/%s_evtlists.txt" % (obj_dir, obj_prefix)
    ## Need to have already made this file in make_GTIs.ipynb
    gti_list = "%s/in/%s_16sGTIlists.txt" % (obj_dir, obj_prefix)


    ###########################################################################
    ###########################################################################

    if debug:
        out_file_base = "%s/out/%s/debug_%s_seg-info" % (obj_dir, obj_name,
                                                                obj_prefix)
    else:
        out_file_base = "%s/out/%s/%s_seg-info" % (obj_dir, obj_name,
                                                           obj_prefix)

    # rsp_matrix_file = obj_dir + "/nicer_v1.02rbn-2.rsp"
    # rsp_hdu = fits.open(rsp_matrix_file)
    # detchans = np.int(rsp_hdu['EBOUNDS'].header['DETCHANS'])

    print("\tDebugging? %s!" % str(debug))
    print("\tOverwriting? %s!" % str(overwrite))


    #################
    ## And it begins
    #################
    # print("* Compute Fourier frequencies and df")
    n_bins = int(n_seconds/dt)
    freq = fftpack.fftfreq(n_bins, d=dt)
    df = np.median(np.diff(freq))
    nyquist = np.abs(freq[int(n_bins/2)+1])
    print("df: "+str(df))
    print("nyquist: %.2f" % nyquist)
    assert np.allclose(df, 1./n_seconds)

    ## Frequency bounds for computing the rms of the power spectrum:
    lf = int(find_nearest(freq[0:int(n_bins/2+1)], rms_lf)[1])
    uf = int(find_nearest(freq[0:int(n_bins/2+1)], rms_hf)[1])
    hf = int(find_nearest(freq[0:int(n_bins/2+1)], 30)[1])

    print("List of event files: %s" % input_list)
    assert os.path.isfile(input_list)

    ## Input_file is a list of eventlists, so get each of those files
    data_files = [line.strip() for line in open(input_list)]
    if not data_files:  ## If data_files is an empty list
        raise Exception("ERROR: No files in the list of event lists: "
                        "%s" % input_list)
    gti_files = [line.strip() for line in open(gti_list)]

    print("n bins: "+str(n_bins))
    print("dt: "+str(dt))
    print("n seconds: "+str(n_seconds))
    n_seg = 0

    # For power-law fitting for a hacky equivalent width
    x_cont = np.append(np.arange(500,560,10)[0:-1], np.arange(750,810,10)[0:-1])
    x_fe = np.arange(620,680,10)[0:-1]

    print("* Loop through files")

    n_files = 1
    out_list = []

    ## Looping through the data files to read the light curves
    for (in_file,gti_file)in zip(data_files, gti_files):
        if in_file[0] == '.':
            in_file = obj_dir + in_file[1:]
            gti_file = obj_dir + gti_file[1:]

        else:
            in_file = data_dir + "/" + in_file
            gti_file = data_dir + "/" + gti_file
        print("\nInput file %d/%d: %s" % (n_files, len(data_files), in_file))
        if "ni" in os.path.basename(in_file):
            filenum = os.path.basename(in_file).split('_')[0][-3:]
        else:
            end_num = in_file.split('/')[-1].split('.')[0].split('-')[-1]
            try:
                filenum = int(end_num)
            except TypeError or ValueError:
                filenum = n_files
        print(filenum)
        out_file = out_file_base + "-"+str(filenum)+".dat"
        # print("Is file: ", os.path.isfile(out_file))
        # print("Overwrite: ", overwrite)

        if overwrite or ((not overwrite) and (not os.path.isfile(out_file))):
            n_seg = per_file(out_file, obj_name, in_file, gti_file, n_seg,
                             n_bins, dt, df, n_seconds, broad_le, broad_he,
                             soft_le, soft_he, hard_le, hard_he, rms_lf,
                             rms_hf, x_cont, x_fe, nyquist)
        else:
            print("File has been processed previously. I hope it was with the "
                  "same energy bands! Moving on.")
        out_list.append(out_file)
        n_files += 1

    ## Done with reading in all the files
    print("Total number of segments: %d" % n_seg)

    with open(out_list_file, 'w') as f:
        [f.write("%s\n" % out_file) for out_file in out_list]
    print("Output files saved to %s" % out_list_file)

print("\nDone!\n")

os.system('\a')
