import numpy as np
from astropy.table import Table
from astropy.time import Time
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.colors as colors
from xcor_tools_nicer import clock_to_mjd

__author__ = "Abigail Stevens <abigailstev@gmail.com>"
__year__ = "2018-2021"

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


if __name__ == "__main__":

    filename = "./out/GX_339-4/gx339-2021_32sec_64dt_dynpsd_rb"

    dyn_psd = Table.read(filename+".fits")
    print(dyn_psd.info)
    lf, lf_idx = find_nearest(dyn_psd['FREQUENCY'], 0.1)
    uf, uf_idx = find_nearest(dyn_psd['FREQUENCY'], 20)
    ls = 0
    # us = 120
    v_min = 1E-3
    v_max = 5E-1
    us = dyn_psd.meta['N_SEG']
    seg_num = np.arange(0, dyn_psd.meta['N_SEG']+1, dtype=int)

    #######################################################
    ## Plotting the dynamical power spectrum and saving it
    #######################################################
    font_prop = font_manager.FontProperties(size=20)
    fig, ax = plt.subplots(1, 1, figsize=(13.5, 6.75), dpi=300,
                           tight_layout=True)
    freqs = dyn_psd['FREQUENCY'][lf_idx:uf_idx+1][:,np.newaxis]
    # print(freqs.shape)
    psds = dyn_psd['PSD'][lf_idx:uf_idx+1,ls:us]
    plt.pcolor(seg_num[ls:us], freqs, psds, cmap='inferno', shading='auto',
               norm=colors.LogNorm(vmin=v_min, vmax=v_max))

    cbar = plt.colorbar(pad=0.01)
    # cbar.set_label(r'Power $\times$ freq. [(rms/mean)$^2$]',
    #                fontproperties=font_prop)
    cbar.set_label(r'Power [(rms/mean)$^2$/Hz]',
                   fontproperties=font_prop)
    cb_ax = cbar.ax
    cb_ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Frequency (Hz)', fontproperties=font_prop)
    ax.set_ylim(lf, uf)
    ax.set_xlim(ls, us)
    ax.set_yscale('log')
    ax.set_xlabel(r'Elapsed time ($\times$ %d s)' % dyn_psd.meta['N_SEC'],
                  fontproperties=font_prop)

    ## Setting the axes' minor ticks. It's complicated.
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.set_xticks(np.arange(ls, us, 500))
    ## Y ticks
    y_maj_loc = [0.1, 1, 10, 20]
    y_maj_labels = ["0.1", "1", "10", "20"]
    ax.set_yticks(y_maj_loc)
    ax.set_yticklabels(y_maj_labels, rotation='horizontal', fontsize=20)
    y_min_loc = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2,3,4,5,6,7,8,9,11,12,13,14,
                 15,16,17,18,19, 21,22, 23,24,25,26,27,28,29]
    ax.yaxis.set_minor_locator(FixedLocator(y_min_loc))
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both', labelsize=20)
    ax.tick_params(which='major', width=1.5, length=7)
    ax.tick_params(which='minor', width=1.5, length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    plt.savefig(filename+".png")
    print("OUTPUT:")
    print(filename+".png")
    subprocess.call(['open', filename+".png"])