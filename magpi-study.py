#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""HMF plots"""

import numpy as np
import common
import os

##################################
h0 = 0.6751

mlow = -2.0
mupp = 3.0
dm = 0.25
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

def plot_numbercounts(plt, outdir, obsdir, ncounts):

    #for cumulative number counts
    xlf_obs  = xlf
 
    xtit="$\\rm log_{10}(S/mJy)$"
    ytit="$\\rm log_{10}(N/dlog_{10}S/A\, deg^{-2} dex^{-1})$"

    xmin, xmax, ymin, ymax = -2, 2, 0 , 3.0
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,5))

    ax = fig.add_subplot(111)

    labels= ('Band-8', 'Band-7', 'Band-6', 'Band-4')
    colors  = ('purple','green','orange','red') 
    bands = (28, 29, 30, 32)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(2, 2, 1, 1))

    p = 0
    for b in bands:
        ind = np.where(ncounts[b,:] != 0)
        y = ncounts[b,ind]
        ax.plot(xlf_obs[ind], y[0], linestyle='solid', color=colors[p], linewidth=3, label=labels[p])
        p = p + 1

    x = [1.5,1.5]
    y = [0,3]
    ax.plot(np.log10(x),y,color='green',linestyle='dotted')
    common.prepare_legend(ax, colors, loc='lower left')
    common.savefig(outdir, fig, "number-counts-magpi-targets.pdf")


def prepare_data(phot_data, phot_dataab, ids_sed, hdf5_data, subvols, lightcone_dir,  nbands, ncounts):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd) = hdf5_data
   
    #(SCO, id_cos) = co_hdf5_data
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_bulge = phot_data[1]
    SEDs_dustab   = phot_dataab[0]
    SEDs_dust_bulgeab = phot_dataab[1]

    mstartot = np.log10((mstarb+mstard)/h0)
    sfrtot = np.log10((sfrb+sfrd)/h0/1e9)
    re = (rsb*mstarb + mstard*rsd) / (mstarb+mstard)
    BT = mstarb / (mstarb+mstard)

    ind = np.where((mstartot > 10.5) & (zobs > 0.3) & (zobs < 0.35))
    SEDs_dust_magpi = SEDs_dust[:,ind]
    print SEDs_dust_magpi.shape

    p = 0
    for i in range(0,nbands):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust_magpi[i,0,:] > 0) & (SEDs_dust_magpi[i,0,:] < 40))
        m = np.log10(10.0**(SEDs_dust_magpi[i,0,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[p,:] = ncounts[p,:] + H
        p = p + 1

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/waves-g23/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/waves-g23/Plots/'
    obsdir= '/home/clagos/shark/data/'

    subvols = (9,36) #,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-eagle-rr14"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 50.58743129433447 #107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}
    fields_absed = {'SED/ab_dust': ('total', 'bulge_t')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    ids_sed, sedsab = common.read_photometry_data_hdf5(lightcone_dir, fields_absed, subvols, sed_file)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent','rstar_disk_apparent')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    ncounts = np.zeros(shape = (nbands, len(mbins)))
    prepare_data(seds, sedsab, ids_sed, hdf5_data, subvols, lightcone_dir, nbands, ncounts)

    ncounts = ncounts/areasub/dm
    # Take logs
    ind = np.where(ncounts > 1e-5)
    ncounts[ind] = np.log10(ncounts[ind])

    plot_numbercounts(plt, outdir, obsdir, ncounts)


if __name__ == '__main__':
    main()
