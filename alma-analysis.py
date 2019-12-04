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
import functools


import common
import utilities_statistics as us
import os

##################################
flux_threshs = np.array([1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0]) #in milli Jy
flux_threshs_log = np.log10(flux_threshs[:])

# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s

# Mass function initialization

mlow = -2.0
mupp = 3.0
dm = 0.25
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

mlow2 = -2
mupp2 = 2
dm2 = 0.15
mbins2 = np.arange(mlow2,mupp2,dm2)
xlf2   = mbins2 + dm2/2.0

zlow = 0
zupp = 6
dz = 0.2
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0

dzobs = 0.5
zbinsobs = np.arange(zlow,zupp,dzobs)
xzobs   = zbinsobs + dzobs/2.0


def plot_redshift(plt, outdir, obsdir, zdist, zdist_flux_cuts):
    xtit="$\\rm redshift$"
    ytit="$\\rm N(S>5 mJy)/dz/area [deg^{-2}]$"

    xmin, xmax, ymin, ymax = 0, 6, 0, 200
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 20, 20))
    plt.subplots_adjust(left=0.2, bottom=0.15)
    file = obsdir+'/lf/numbercounts/SMG_z_LESS_wardlow2011_table.data'
    sw11,sw11err,zw11 = np.loadtxt(file,usecols=[1,2,4],unpack=True)
    ind = np.where(sw11 >= 5)
    zdistw11 = np.zeros(shape = (2,len(zbinsobs)))

    H, bins_edges = np.histogram(zw11[ind],bins=np.append(zbinsobs,zupp))
    zdistw11[0,:] = zdistw11[0,:] + H
    zdistw11[1,:] = np.sqrt(zdistw11[0,:])
    areaobs = 0.5*0.5
    yobs = zdistw11[0,:]/areaobs/dzobs
    yup  = (zdistw11[0,:]+zdistw11[1,:])/areaobs/dzobs
    ydn  = (zdistw11[0,:]-zdistw11[1,:])/areaobs/dzobs
    ind = np.where(yobs > 0)
    ax.errorbar(xzobs[ind],yobs[ind],yerr=[yobs[ind]-ydn[ind],yup[ind]-yobs[ind]], ls='None', mfc='None', ecolor = 'MediumBlue', mec='MediumBlue',marker='D',label='Wardlow+11')


    file = obsdir+'/lf/numbercounts/ncts870_Dudzeviciute19.data'
    zd19,nd19,nd19errdn,nd19errup = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
    ax.errorbar(zd19,nd19,yerr=[nd19-nd19errdn,nd19errup-nd19], ls='None', mfc='None', ecolor = 'Orange', mec='Orange',marker='s',label='Dudzeviciute+19')

    bands = (0, 3)
    colors  = ('MediumBlue','Orange','DarkRed')
    p = 0
    for j in bands:
        ax.plot(flux_threshs,zdist_flux_cuts[j,0,:],color=colors[p])
        p = p + 1

    common.prepare_legend(ax, ['MediumBlue','Orange'], loc="upper left")

    common.savefig(outdir, fig, "zdistribution-FIR-5mJy.pdf")


    #thresholds
    ytit="$\\rm redshift$"
    xtit="$\\rm S_{\\rm threshold}/mJy$"

    xmin, xmax, ymin, ymax = 1e-3, 10, 0.5, 6
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 1, 1))
    plt.xscale('log')
    plt.subplots_adjust(left=0.2, bottom=0.15)

    bands = (1, 2, 3, 4)
    colors  = ('Navy','DarkCyan','Orange','Firebrick')
    labels = ('band-8','band-7','band-6','band-4')
    lines = ('solid','dashed','dotted','dashdot')
    p = 0
    for j in bands:
        ind = np.where(zdist_flux_cuts[j,0,:] != 0)
        y = zdist_flux_cuts[j,0,ind]
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts[j,2,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.2,interpolate=True)
        ax.plot(flux_threshs[ind],y[0],linestyle=lines[p],color=colors[p],label=labels[p])
        p = p + 1

    common.prepare_legend(ax, colors, loc="upper left")

    common.savefig(outdir, fig, "zdistribution-fluxcut-ALMAbands.pdf")

def plot_numbercounts(plt, outdir, obsdir, ncounts):

    #for cumulative number counts
    xlf_obs  = xlf-dm*0.5
 
    xtit="$\\rm log_{10}(S/mJy)$"
    ytit="$\\rm log_{10}(N(>S)/A\, deg^{-2})$"

    xmin, xmax, ymin, ymax = -2, 2, 0 , 6.5
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(12,9))

    subplots = (231, 232, 233, 234, 235, 236)
    idxs = (0, 1, 2, 3, 4)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    bands = (3, 4, 5, 6, 8)#1, 2, 3, 4, 5)#, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 24)
    labels= ('Band-9','Band-8', 'Band-7', 'Band-6', 'Band-4')
    #(24, 25, 26, 27, 28, 29, 30, 31, 32)
    for subplot, idx, b in zip(subplots, idxs, bands):

        ax = fig.add_subplot(subplot)
        if (idx == 0 or idx == 3 or idx == 6):
            ytitplot = ytit
        else:
            ytitplot = ' '
        if (idx >= 2):
            xtitplot = xtit
        else:
            xtitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytitplot, locators=(2, 2, 1, 1))
        ax.text(xleg,yleg, labels[idx])

        if(idx == 2):
            file = obsdir+'/lf/numbercounts/ncts850_Karim13.data'
            lmg17,pg17,dpg17up,dpg17dn = np.loadtxt(file,usecols=[4,5,6,7],unpack=True)
            ax.errorbar(np.log10(lmg17),np.log10(pg17),yerr=[np.log10(pg17) - np.log10(pg17-dpg17dn), np.log10(pg17+dpg17up) - np.log10(pg17)], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='o', label='Karim+2013')
            file = obsdir+'/lf/numbercounts/ncts-band7_Oteo16.data'
            lmu17, pu17, dpu17dn, dpu17up = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
            ax.errorbar(np.log10(lmu17),np.log10(pu17),yerr=[np.log10(pu17)-np.log10(pu17-dpu17dn), np.log10(dpu17up+pu17)-np.log10(pu17)], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='v',label='Oteo+2016')

        if(idx == 1):
            file = obsdir+'/lf/numbercounts/ncts-band8_Klitsch20.data'
            lmu17, pu17, dpu17up, dpu17dn = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
            ax.errorbar(np.log10(lmu17),np.log10(pu17),yerr=[np.log10(pu17)-np.log10(dpu17dn), np.log10(dpu17up)-np.log10(pu17)], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='D',label='Klitsch+2020')

        if(idx == 3):
            file = obsdir+'/lf/numbercounts/ncts1p2mm_Fujimoto16.data'
            lmf16, pf16, dpf16dn, dpf16up = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
            ax.errorbar(np.log10(lmf16),pf16,yerr=[dpf16dn,dpf16up], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',label='Fujimoto+2016')
            file = obsdir+'/lf/numbercounts/ncts1p2mm_Umehata17.data'
            lmu17, pu17, dpu17up, dpu17dn = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
            ax.errorbar(np.log10(lmu17),np.log10(pu17),yerr=[np.log10(pu17)-np.log10(pu17-dpu17dn), np.log10(pu17+dpu17up)-np.log10(pu17)], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='d',label='Umehata+2017')
            file = obsdir+'/lf/numbercounts/ncts1p2mm_Htsukade18.data'
            lmf16, pf16, dpf16dn, dpf16up = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)
            ax.errorbar(np.log10(lmf16),np.log10(pf16),yerr=[np.log10(pf16)-np.log10(pf16-dpf16dn), np.log10(pf16+dpf16up)-np.log10(pf16)], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='^',label='Hatsukade+2018')
            file = obsdir+'/lf/numbercounts/ncts1p2mm_Gonzalez-Lopez19.data'
            lmf19, pf19, dpf19 = np.loadtxt(file,usecols=[0,1,2],unpack=True)
            ax.errorbar(lmf19,np.log10(pf19),yerr=[np.log10(pf19)-np.log10(pf19-dpf19), np.log10(pf19+dpf19)-np.log10(pf19)], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='*',label='Gonzalez-Lopez+2020')

        #Predicted LF
        if(idx == 4):
            ind = np.where(ncounts[0,b,:] != 0)
            y = ncounts[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=3, label='Shark total')
 
            ind = np.where(ncounts[1,b,:] != 0)
            y = ncounts[1,b,ind]
            ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted', label='disks')
            ind = np.where(ncounts[2,b,:] != 0)
            y = ncounts[2,b,ind]
            ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed', label='bulges')
        else:
            ind = np.where(ncounts[0,b,:] != 0)
            y = ncounts[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)
 
            ind = np.where(ncounts[1,b,:] != 0)
            y = ncounts[1,b,ind]
            ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
            ind = np.where(ncounts[2,b,:] != 0)
            y = ncounts[2,b,ind]
            ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')
        if (idx == 4):
            common.prepare_legend(ax, ['k','b','r','grey','grey','grey'], bbox_to_anchor=[1.1,0.1])
        if (idx == 1 or idx == 2 or idx == 3):
            common.prepare_legend(ax, ['grey','grey','grey','grey'], loc='lower left')

    common.savefig(outdir, fig, "number-counts-deep-FIR-ALMA.pdf")

def prepare_data(phot_data, ids_sed, hdf5_data, subvols, lightcone_dir, ncounts, nbands, zdist, bands, zdist_flux_cuts):

    (dec, ra, zobs, idgal) = hdf5_data
 
    bin_it = functools.partial(us.wmedians_cum, xbins=flux_threshs_log)
 
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_disk = phot_data[1]
    SEDs_dust_bulge = phot_data[2]
    SEDs_dust_bulge_d = phot_data[3]
    SEDs_dust_bulge_m = phot_data[4]

    #print SEDs_dust(0)
    indices = range(len(bands))
    for i, j in zip(bands, indices):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[0,j,:] = ncounts[0,j,:] + H
        zdist_flux_cuts[j] = bin_it(x=m[0,:],y=zobs[ind])

        #contribution from disks and bulges
        ind = np.where((SEDs_dust_disk[i,:] > 0) & (SEDs_dust_disk[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust_disk[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[1,j,:] = ncounts[1,j,:] + H

        ind = np.where((SEDs_dust_bulge[i,:] > 0) & (SEDs_dust_bulge[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust_bulge[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[2,j,:] = ncounts[2,j,:] + H

        ind = np.where((SEDs_dust_bulge_d[i,:] > 0) & (SEDs_dust_bulge_d[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust_bulge_d[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[3,j,:] = ncounts[3,j,:] + H

        ind = np.where((SEDs_dust_bulge_m[i,:] > 0) & (SEDs_dust_bulge_m[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust_bulge_m[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[4,j,:] = ncounts[4,j,:] + H

        #redshift distribution for sources brighter than 5mJy
        ind = np.where((SEDs_dust[i,:] > -10) & (SEDs_dust[i,:] < 14.6525))
        H, bins_edges = np.histogram(zobs[ind],bins=np.append(zbins,zupp))
        zdist[j,:] = zdist[j,:] + H

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14-testmmbands"
    subvols = range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2

    bands = (24, 25, 26, 27, 28, 29, 30, 31, 32) #(26, 27, 28, 29, 30, 31, 32)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    ncounts = np.zeros(shape = (5, len(bands), len(mbins)))
    ncounts_cum = np.zeros(shape = (5, len(bands), len(mbins)))

    zdist = np.zeros(shape = (len(bands), len(zbins)))
    zdist_flux_cuts = np.zeros(shape = (len(bands), 3,len(flux_threshs)))

    prepare_data(seds, ids_sed, hdf5_data, subvols, lightcone_dir, ncounts, nbands, zdist, bands, zdist_flux_cuts)

    if(totarea > 0.):
        for b in range(0,len(bands)):
            for j in range(len(mbins)):
                ncounts_cum[0,b,j] = np.sum(ncounts[0,b,j:len(mbins)])
                ncounts_cum[1,b,j] = np.sum(ncounts[1,b,j:len(mbins)])
                ncounts_cum[2,b,j] = np.sum(ncounts[2,b,j:len(mbins)])
            #print 'band', b
            #for m,a,b,c in zip(xlf,ncounts_cum[0,b,:],ncounts_cum[1,b,:],ncounts_cum[2,b,:]):
            #    print m-dm*0.5,a/areasub,b/areasub,c/areasub

        ncounts   = ncounts/areasub/dm
        ncounts_cum = ncounts_cum/areasub
        zdist = zdist/areasub

    # Take logs
    ind = np.where(ncounts > 1e-5)
    ncounts[ind] = np.log10(ncounts[ind])
    ind = np.where(ncounts_cum > 1e-5)
    ncounts_cum[ind] = np.log10(ncounts_cum[ind])

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_numbercounts(plt, outdir, obsdir, ncounts_cum)
    plot_redshift(plt, outdir, obsdir, zdist, zdist_flux_cuts)

if __name__ == '__main__':
    main()
