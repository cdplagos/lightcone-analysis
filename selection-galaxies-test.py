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

# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s

# Mass function initialization

mlow = 0
mupp = 30
dm = 0.5
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

mlow2 = -2
mupp2 = 2
dm2 = 0.15
mbins2 = np.arange(mlow2,mupp2,dm2)
xlf2   = mbins2 + dm2/2.0

def plot_numbercounts(plt, outdir, obsdir, ncounts, ncounts_cum_850):

    xlf_obs  = xlf
 
    xtit="$\\rm app\, mag (AB)$"
    ytit="$\\rm log_{10}(N/{\\rm 0.5 mag}^{-1}/A\, deg^2)$"

    xmin, xmax, ymin, ymax = 10, 30, -2 , 5.5
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(12,12))

    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    bands = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 24)
    labels= ('GALEX FUV', 'GALEX NUV', 'SDSS u', 'SDSS g', 'SDSS r', 'SDSS i', 'SDSS z', 'VISTA Y', 'VISTA J')#, 'VISTA H', 'VISTA K', 'IRAC 3.6', 'IRAC 4.5', 'WISE 1', 'IRAC 5.8', 'IRAC 8', 'WISE 2', 'WISE 3', 'WISE 4', 'P70', 'P100', 'P160', 'S250', 'S350', 'S500')
    obs_start = (0, 32,98, 194,272,387,497,656,756)#,865,969, 1031,1061,1089,1106,1129,1146,1163,1232,1273,1336,1416,1446,1472)
    obs_end   = (31,97,193,271,386,496,655,755,864)#,968,1030,1060,1088,1105,1128,1145,1162,1177,1272,1335,1415,1445,1471,1495)

    file = obsdir+'/lf/numbercounts/Driver16_numbercounts.data'
    lm,p,dp = np.loadtxt(file,usecols=[2,3,4],unpack=True)
    yobs = np.log10(p)
    ydn  = np.log10(p-dp)
    yup  = np.log10(p+dp)

    for subplot, idx, b in zip(subplots, idx, bands):

        ax = fig.add_subplot(subplot)
        if (idx == 0 or idx == 3 or idx == 6):
            ytitplot = ytit
        else:
            ytitplot = ' '
        if (idx >= 6):
            xtitplot = xtit
        else:
            xtitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytitplot, locators=(2, 2, 1, 1))
        ax.text(xleg,yleg, labels[idx])

        ax.errorbar(lm[obs_start[idx]:obs_end[idx]], yobs[obs_start[idx]:obs_end[idx]], yerr=[yobs[obs_start[idx]:obs_end[idx]]-ydn[obs_start[idx]:obs_end[idx]],yup[obs_start[idx]:obs_end[idx]]-yobs[obs_start[idx]:obs_end[idx]]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='o')

        #Predicted LF
        ind = np.where(ncounts[4,b,:] != 0)
        y = ncounts[4,b,ind]
        ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)

        ind = np.where(ncounts[3,b,:] != 0)
        y = ncounts[3,b,ind]
        ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
        ind = np.where(ncounts[2,b,:] != 0)
        y = ncounts[2,b,ind]
        ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')

    common.savefig(outdir, fig, "number-counts-deep-lightcone-optical.pdf")

    fig = plt.figure(figsize=(12,12))

    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    bands = (9, 10, 11, 12, 13, 14, 16, 17, 18)#, 18, 19, 20, 21, 22, 23, 24)
    labels= ('VISTA H', 'VISTA K', 'WISE 1','IRAC 3.6', 'IRAC 4.5', 'WISE 2', 'IRAC 8',  'WISE 3', 'WISE 4')#, 'P70', 'P100', 'P160', 'S250', 'S350', 'S500')
    obs_start = (865,969 ,1031,1048,1095,1078,1140,1123,1163)
    obs_end   = (968,1030,1047,1077,1122,1094,1162,1139,1177)


    for subplot, idx, b in zip(subplots, idx, bands):

        ax = fig.add_subplot(subplot)
        if (idx == 0 or idx == 3 or idx == 6):
            ytitplot = ytit
        else:
            ytitplot = ' '
        if (idx >= 6):
            xtitplot = xtit
        else:
            xtitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytitplot, locators=(2, 2, 1, 1))
        ax.text(xleg,yleg, labels[idx])

        ax.errorbar(lm[obs_start[idx]:obs_end[idx]], yobs[obs_start[idx]:obs_end[idx]], yerr=[yobs[obs_start[idx]:obs_end[idx]]-ydn[obs_start[idx]:obs_end[idx]],yup[obs_start[idx]:obs_end[idx]]-yobs[obs_start[idx]:obs_end[idx]]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='o')

        #Predicted LF
        ind = np.where(ncounts[4,b,:] != 0)
        y = ncounts[4,b,ind]
        ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)

        ind = np.where(ncounts[3,b,:] != 0)
        y = ncounts[3,b,ind]
        ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
        ind = np.where(ncounts[2,b,:] != 0)
        y = ncounts[2,b,ind]
        ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')

    common.savefig(outdir, fig, "number-counts-deep-lightcone-IR.pdf")


    xmin, xmax, ymin, ymax = 8, 24, -2 , 5.5
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(12,12))

    subplots = (331, 332, 333, 334, 335, 336, 337)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
    idx = (0, 1, 2, 3, 4, 5, 6)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    bands = (19, 20, 21, 22, 23, 25, 26)
    labels= ('P70', 'P100', 'P160', 'S250', 'S350', 'S500', 'JCTM850')
    obs_start = (1232,1273,1336,1416,1446,1472)
    obs_end   = (1272,1335,1415,1445,1471,1495)

    for subplot, idx, b in zip(subplots, idx, bands):

        ax = fig.add_subplot(subplot)
        if (idx == 0 or idx == 3 or idx == 6):
            ytitplot = ytit
        else:
            ytitplot = ' '
        if (idx >= 4):
            xtitplot = xtit
        else:
            xtitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytitplot, locators=(2, 2, 1, 1))
        ax.text(xleg,yleg, labels[idx])

        if (idx < 6):
            ax.errorbar(lm[obs_start[idx]:obs_end[idx]], yobs[obs_start[idx]:obs_end[idx]], yerr=[yobs[obs_start[idx]:obs_end[idx]]-ydn[obs_start[idx]:obs_end[idx]],yup[obs_start[idx]:obs_end[idx]]-yobs[obs_start[idx]:obs_end[idx]]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='o')

        #Predicted LF
        ind = np.where(ncounts[4,b,:] != 0)
        y = ncounts[4,b,ind]
        ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)

        ind = np.where(ncounts[3,b,:] != 0)
        y = ncounts[3,b,ind]
        ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
        ind = np.where(ncounts[2,b,:] != 0)
        y = ncounts[2,b,ind]
        ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')

    common.savefig(outdir, fig, "number-counts-deep-lightcone-FIR.pdf")

    xtit="$\\rm log_{10}(S_{\\rm JCMT\,850/mJy)$"
    ytit="$\\rm log_{10}(N(>S)/deg^2)$"

    xmin, xmax, ymin, ymax = -2, 2, -1 , 6
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(2, 2, 1, 1))

    file = obsdir+'/lf/numbercounts/ncts850_Karim13.data'
    lmK13,pK13,dpupK13,dpdnK13 = np.loadtxt(file,usecols=[4,5,6,7],unpack=True)
    yobs = np.log10(pK13)
    ydn  = np.log10(pK13-dpdnK13)
    yup  = np.log10(pK13+dpupK13)
    ax.errorbar(np.log10(lmK13),yobs,yerr=[yobs-ydn,yup-yobs], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s')

    file = obsdir+'/lf/numbercounts/ncts850_knudsen08.data'
    lmK08,pK08,dpupK08,dpdnK08 = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
    yobs = np.log10(pK08)
    ydn  = np.log10(pK08-dpdnK08)
    yup  = np.log10(pK08+dpupK08)
    ax.errorbar(np.log10(lmK08),yobs,yerr=[yobs-ydn,yup-yobs], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='*')

    #Predicted LF
    ind = np.where(ncounts_cum_850[4,:] != 0)
    y = ncounts_cum_850[4,ind]
    ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)

    ind = np.where(ncounts_cum_850[3,:] != 0)
    y = ncounts_cum_850[3,ind]
    ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
    ind = np.where(ncounts_cum_850[2,:] != 0)
    y = ncounts_cum_850[2,ind]
    ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')

    common.savefig(outdir, fig, "number-counts-deep-lightcone-850.pdf")

def prepare_data(phot_data, ids_sed, sed_nodust, hdf5_data, subvols, lightcone_dir):

    (dec, ra, zobs, idgal) = hdf5_data
   
    #(SCO, id_cos) = co_hdf5_data
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_nodust = sed_nodust[0]

    #print len(SEDs_dust)
    #filte_to_write = lightcone_dir
    #fname = os.path.join(lightcone_dir, 'split', 'matched_population_driver19_test.dat')

    bands = (2, 4, 10)
    print idgal[0:10],ids_sed[0:10]
    with open('/mnt/su3ctm/clagos/Lightcones/DriverNumberCounts/Shark/Shark-Deep-comp-Lightcone.txt', 'wb') as f:
         f.write("#Galaxies from Shark (Lagos et al. 2018) in the GAMA-deep lightcone (r<32)\n")
         #f.write("#area: 79.935deg^2 (-4<dec<4 and 175<ra<185)\n")
         f.write("#dec ra z u r K u_nodust r_nodust\n")
         for a,b,c,d,e,g,h,i in zip(dec, ra, zobs, SEDs_dust[2], SEDs_dust[4], SEDs_dust[10], SEDs_nodust[2], SEDs_nodust[4]):
             if(e < 32):
                f.write("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n" % (a,b,c,d,e,g,h,i))

def main():

    lightcone_dir = '/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    outdir= '/home/clagos/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/git/shark/data/'

    subvols = (0, 2) 
    #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    fields_sed = {'SED/ap_dust': ('total', 'disk')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)
 
    fields_sed = {'SED/ap_nodust': ('total', 'disk')}

    ids_sed_nodust, seds_nodust = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    #fields_co = {'galaxies': ('SCO','id_galaxy_sky')}

    #co_hdf5_data = common.read_co_hdf5(lightcone_dir, fields_co, subvols)
 
    prepare_data(seds, ids_sed, seds_nodust, hdf5_data, subvols, lightcone_dir)

if __name__ == '__main__':
    main()
