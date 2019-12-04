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

zlow = 0
zupp = 6
dz = 0.2
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0

dzobs = 0.5
zbinsobs = np.arange(zlow,zupp,dzobs)
xzobs   = zbinsobs + dzobs/2.0


def plot_redshift(plt, outdir, obsdir):
    file = 'zDistribution-Deep-Optical-eagle-rr14-steep.data'
    data_eagle_rr14_steep = np.loadtxt(file)
    file = 'zDistribution-Deep-Optical-eagle-rr14.data'
    data_eagle_rr14 = np.loadtxt(file)
    file = 'zDistribution-Deep-Optical-eagle-const.data'
    data_eagle_const = np.loadtxt(file)
    file = 'zDistribution-Deep-Optical-CF00.data'
    data_cf00 = np.loadtxt(file)


    xtit="$\\rm redshift$"
    ytit="$\\rm N(S_{\\rm 850 \\mu m}>5 mJy)/dz/area [deg^{-2}]$"

    xmin, xmax, ymin, ymax = 0, 6, 0, 200
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 20, 20))
    plt.subplots_adjust(left=0.2, bottom=0.15)
    #file = obsdir+'/lf/numbercounts/ncts870_Dudzeviciute19.data'
    #zd19,nd19,nd19errdn,nd19errup = np.loadtxt(file,usecols=[0,1,2,3],unpack=True)

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
    ax.errorbar(xzobs[ind],yobs[ind],yerr=[yobs[ind]-ydn[ind],yup[ind]-yobs[ind]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='D',label='Wardlow+11')
    #ax.errorbar(zd19,nd19,yerr=[nd19-nd19errdn,nd19errup-nd19], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='s',label='Dudzeviciute+19')

    ax.plot(data_eagle_rr14_steep[:,0],data_eagle_rr14_steep[:,1],'Indigo', linewidth=4, alpha=0.5)
    ax.plot(data_eagle_rr14[:,0],data_eagle_rr14[:,1],'Indigo', linewidth=4, linestyle='dotted')
    ax.plot(data_eagle_const[:,0],data_eagle_const[:,1],'Indigo', linestyle='dashed', linewidth=4, alpha=0.4)
    ax.plot(data_cf00[:,0],data_cf00[:,1],'Indigo', linestyle='dashdot', linewidth=4, alpha=0.3)

    common.prepare_legend(ax, ['grey','grey'], loc="upper left")

    common.savefig(outdir, fig, "zdistribution-850microns-5mJy.pdf")

def plot_numbercounts(plt, outdir, obsdir):

    file = 'NumberCounts-Deep-Optical-eagle-rr14-steep.data'
    datam = np.loadtxt(file)
    lm_eagle_rr14_steep = datam[:,0]
    num_eagle_rr14_steep = datam[:,1:10]

    file = 'NumberCounts-Deep-Optical-eagle-rr14.data'
    datam = np.loadtxt(file)
    lm_eagle_rr14 = datam[:,0]
    num_eagle_rr14 = datam[:,1:10]

    file = 'NumberCounts-Deep-Optical-eagle-const.data'
    datam = np.loadtxt(file)
    lm_eagle_const = datam[:,0]
    num_eagle_const = datam[:,1:10]

    file = 'NumberCounts-Deep-Optical-CF00.dat'
    datam = np.loadtxt(file)
    lm_cf00 = datam[:,0]
    num_cf00 = datam[:,1:10]


    xtit="$\\rm app\, mag (AB)$"
    ytit="$\\rm log_{10}(N/{\\rm 0.5 mag}^{-1}/A\, deg^2)$"

    xmin, xmax, ymin, ymax = 8, 30, -2 , 5.5
    xmin2, xmax2 = 8, 24
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(12,12))

    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)

    labels= ('GALEX NUV', 'SDSS r', 'VISTA Y', 'IRAC 3.6', 'IRAC 8', 'P70', 'P100', 'S500', 'JCTM850')
    bands = (1, 4, 7, 12, 16, 19, 20, 25, 26)
    obs_start = (32, 272, 656, 1031, 1106, 1232, 1273, 1472)
    obs_end   = (98, 387, 756, 1061, 1129, 1273, 1336, 1496)

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
        if(idx <= 5):
            xminin, xmaxin= xmin, xmax
        else:
            xminin, xmaxin= xmin2, xmax2

        common.prepare_ax(ax, xminin, xmaxin, ymin, ymax, xtitplot, ytitplot, locators=(2, 2, 1, 1))
        ax.text(xleg,yleg, labels[idx])

        if(idx <= 7):
           ax.errorbar(lm[obs_start[idx]:obs_end[idx]], yobs[obs_start[idx]:obs_end[idx]], yerr=[yobs[obs_start[idx]:obs_end[idx]]-ydn[obs_start[idx]:obs_end[idx]],yup[obs_start[idx]:obs_end[idx]]-yobs[obs_start[idx]:obs_end[idx]]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='o')
        else:
            file = obsdir+'/lf/numbercounts/ncts850_Geach2017.data'
            lmg17,pg17,dpg17 = np.loadtxt(file,usecols=[0,1,4],unpack=True)
            bing17  = 1.0 #mJy
            xobsg17 = -2.5 * np.log10(lmg17*1e-3) + 8.9 #in AB mag
            yobsg17 = np.zeros(shape = (3,len(xobsg17)))
            for o in range(0,len(xobsg17)):
                dmo = abs(-2.5 * np.log10((lmg17[o]-bing17/2.0)*1e-3) + 2.5 * np.log10((lmg17[o]+bing17/2.0)*1e-3))
                yobsg17[0,o] = np.log10(pg17[o]*bing17 / dmo)
                yobsg17[1,o] = np.log10((pg17[o]-dpg17[o])*bing17 / dmo)
                yobsg17[2,o] = np.log10((pg17[o]+dpg17[o])*bing17 / dmo)
            ax.errorbar(xobsg17,yobsg17[0,:] + np.log10(0.5),yerr=[yobsg17[0,:]-yobsg17[1,:],yobsg17[2,:]-yobsg17[0,:]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='*')

        #Predicted LF
        if(idx < 5):
           x = lm_eagle_rr14_steep[240:305]
           y = num_eagle_rr14_steep[240:305,idx]
           ind = np.where(y != 0)
           ax.plot(x[ind],y[ind],'Indigo', linewidth=1)

        x = lm_eagle_rr14_steep[0:60]
        y = num_eagle_rr14_steep[0:60,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'Indigo', linewidth=4, alpha=0.5)

        x = lm_eagle_rr14_steep[60:120]
        y = num_eagle_rr14_steep[60:120,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'blue', linewidth=2, linestyle='dotted', label='disks')
        x = lm_eagle_rr14_steep[120:180]
        y = num_eagle_rr14_steep[120:180,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'LightSalmon', linewidth=2, linestyle='dashdot', label='SBs diskins')
        x = lm_eagle_rr14_steep[180:240]
        y = num_eagle_rr14_steep[180:240,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'red', linewidth=2, linestyle='dashed', label='SBs mergers')

        x = lm_eagle_rr14
        y = num_eagle_rr14[:,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'Indigo', linewidth=4, linestyle='dotted')
        x = lm_eagle_const
        y = num_eagle_const[:,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'Indigo', linewidth=4, alpha=0.4, linestyle='dashed')
        x = lm_cf00
        y = num_cf00[:,idx]
        ind = np.where(y != 0)
        ax.plot(x[ind],y[ind],'Indigo', linewidth=4, alpha=0.3, linestyle='dashdot')

#        ind = np.where(LFs_dust2[z,4,band,:] < 0.)
#        y = LFs_dust2[z,4,band,ind]+volcorr-np.log10(dm)
#        ax.plot(xlf_obs[ind],y[0],'Indigo', linewidth=3, linestyle='dotted',label='$\\rm EAGLE-\\tau\, RR14$')
#        ind = np.where(LFs_dust3[z,4,band,:] < 0.)
#        y = LFs_dust3[z,4,band,ind]+volcorr-np.log10(dm)
#        ax.plot(xlf_obs[ind],y[0],'Indigo', linewidth=3, alpha=0.4, linestyle='dashed', label='$\\rm EAGLE-\\tau\,f_{\\rm dust}\, const$')
#        ind = np.where(LFs_dust4[z,4,band,:] < 0.)
#        y = LFs_dust4[z,4,band,ind]+volcorr-np.log10(dm)
#        ax.plot(xlf_obs[ind],y[0],'Indigo', linewidth=3, alpha=0.3, linestyle='dashdot', label='CF00')

        if (idx == 5):
            common.prepare_legend(ax, ['b','LightSalmon','red'], loc='lower right')
    common.savefig(outdir, fig, "number-counts-deep-lightcone-selected.pdf")

def main():

    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    outdir = os.path.join(outdir, 'allmodels')

    plt = common.load_matplotlib()

    plot_numbercounts(plt, outdir, obsdir)
    plot_redshift(plt, outdir, obsdir)

if __name__ == '__main__':
    main()
