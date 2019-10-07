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


def plot_redshift(plt, outdir, obsdir, zdist):
    xtit="$\\rm redshift$"
    ytit="$\\rm N(S_{\\rm 850 \\mu m}>5 mJy)/dz/area [deg^{-2}]$"

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
    ax.errorbar(xzobs[ind],yobs[ind],yerr=[yobs[ind]-ydn[ind],yup[ind]-yobs[ind]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='D',label='Wardlow+11')
    ax.plot(xz,zdist/dz,'k')
    #for x,y in zip(xz,zdist/dz):
    #    print x,y

    common.prepare_legend(ax, ['grey'], loc="upper left")

    common.savefig(outdir, fig, "zdistribution-850microns-5mJy.pdf")

def plot_numbercounts(plt, outdir, obsdir, ncounts, ncounts_nod):

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
        if(idx == 8):
            ind = np.where(ncounts[0,b,:] != 0)
            y = ncounts[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=3, label='Shark total')
 
            ind = np.where(ncounts_nod[0,b,:] != 0)
            y = ncounts_nod[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=1, label='no dust')
 
            ind = np.where(ncounts[1,b,:] != 0)
            y = ncounts[1,b,ind]
            ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted', label='disks')
            ind = np.where(ncounts[2,b,:] != 0)
            y = ncounts[2,b,ind]
            ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed', label='bulges')
        if(idx < 8):
            ind = np.where(ncounts[0,b,:] != 0)
            y = ncounts[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)
 
            ind = np.where(ncounts_nod[0,b,:] != 0)
            y = ncounts_nod[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=1)
 
            ind = np.where(ncounts[1,b,:] != 0)
            y = ncounts[1,b,ind]
            ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
            ind = np.where(ncounts[2,b,:] != 0)
            y = ncounts[2,b,ind]
            ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')

        if (idx == 8):
            common.prepare_legend(ax, ['k','k','b','r'], loc='lower right')
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
        ind = np.where(ncounts[0,b,:] != 0)
        y = ncounts[0,b,ind]
        ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)

        ind = np.where(ncounts[1,b,:] != 0)
        y = ncounts[1,b,ind]
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
            ax.errorbar(xobsg17,yobsg17[0,:]+np.log10(0.5),yerr=[yobsg17[0,:]-yobsg17[1,:],yobsg17[2,:]-yobsg17[0,:]], ls='None', mfc='None', ecolor = 'grey', mec='grey',marker='*')
 
        #Predicted LF
        ind = np.where(ncounts[0,b,:] != 0)
        y = ncounts[0,b,ind]
        ax.plot(xlf_obs[ind],y[0],'k', linewidth=3)
    
        ind = np.where(ncounts[1,b,:] != 0)
        y = ncounts[1,b,ind]
        ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted')
        ind = np.where(ncounts[2,b,:] != 0)
        y = ncounts[2,b,ind]
        ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed')
 

    common.savefig(outdir, fig, "number-counts-deep-lightcone-FIR.pdf")

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
    obs_end   = (98, 387, 756, 1061, 1129, 1373, 1336, 1496)

    file = obsdir+'/lf/numbercounts/Driver16_numbercounts.data'
    lm,p,dp = np.loadtxt(file,usecols=[2,3,4],unpack=True)
    yobs = np.log10(p)
    ydn  = np.log10(p-dp)
    yup  = np.log10(p+dp)

    #for x,y1,y2,y3,y4,y5,y6,y7,y8,y9 in zip(xlf_obs,ncounts[0,bands[0],:],ncounts[0,bands[1],:],ncounts[0,bands[2],:],ncounts[0,bands[3],:],ncounts[0,bands[4],:],ncounts[0,bands[5],:],ncounts[0,bands[6],:],ncounts[0,bands[7],:],ncounts[0,bands[8],:]):
    #    print x,y1,y2,y3,y4,y5,y6,y7,y8,y9

    #for x,y1,y2,y3,y4,y5,y6,y7,y8,y9 in zip(xlf_obs,ncounts_nod[0,bands[0],:],ncounts_nod[0,bands[1],:],ncounts_nod[0,bands[2],:],ncounts_nod[0,bands[3],:],ncounts_nod[0,bands[4],:],ncounts_nod[0,bands[5],:],ncounts_nod[0,bands[6],:],ncounts_nod[0,bands[7],:],ncounts_nod[0,bands[8],:]):
    #    print x,y1,y2,y3,y4,y5,y6,y7,y8,y9

    #for x,y1,y2,y3,y4,y5,y6,y7,y8,y9 in zip(xlf_obs,ncounts[1,bands[0],:],ncounts[1,bands[1],:],ncounts[1,bands[2],:],ncounts[1,bands[3],:],ncounts[1,bands[4],:],ncounts[1,bands[5],:],ncounts[1,bands[6],:],ncounts[1,bands[7],:],ncounts[1,bands[8],:]):
    #    print x,y1,y2,y3,y4,y5,y6,y7,y8,y9

    #for x,y1,y2,y3,y4,y5,y6,y7,y8,y9 in zip(xlf_obs,ncounts[3,bands[0],:],ncounts[3,bands[1],:],ncounts[3,bands[2],:],ncounts[3,bands[3],:],ncounts[3,bands[4],:],ncounts[3,bands[5],:],ncounts[3,bands[6],:],ncounts[3,bands[7],:],ncounts[3,bands[8],:]):
    #    print x,y1,y2,y3,y4,y5,y6,y7,y8,y9

    #for x,y1,y2,y3,y4,y5,y6,y7,y8,y9 in zip(xlf_obs,ncounts[4,bands[0],:],ncounts[4,bands[1],:],ncounts[4,bands[2],:],ncounts[4,bands[3],:],ncounts[4,bands[4],:],ncounts[4,bands[5],:],ncounts[4,bands[6],:],ncounts[4,bands[7],:],ncounts[4,bands[8],:]):
    #    print x,y1,y2,y3,y4,y5,y6,y7,y8,y9

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
            ind = np.where(ncounts[0,b,:] != 0)
            y = ncounts[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=3, label='Shark total')
            ind = np.where(ncounts_nod[0,b,:] != 0)
            y = ncounts_nod[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=1, label='no dust')
 
            ind = np.where(ncounts[1,b,:] != 0)
            y = ncounts[1,b,ind]
            ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted', label='disks')
            ind = np.where(ncounts[4,b,:] != 0)
            y = ncounts[4,b,ind]
            ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed', label='bulges mergers')
            ind = np.where(ncounts[3,b,:] != 0)
            y = ncounts[3,b,ind]
            ax.plot(xlf_obs[ind],y[0],'LightSalmon', linewidth=2, linestyle='dashdot', label='bulges diskins')
        else:
            ind = np.where(ncounts[0,b,:] != 0)
            y = ncounts[0,b,ind]
            ax.plot(xlf_obs[ind],y[0],'k', linewidth=3, label='Shark total')

            ind = np.where(ncounts[1,b,:] != 0)
            y = ncounts[1,b,ind]
            ax.plot(xlf_obs[ind],y[0],'b', linewidth=2, linestyle='dotted', label='disks')
            ind = np.where(ncounts[4,b,:] != 0)
            y = ncounts[4,b,ind]
            ax.plot(xlf_obs[ind],y[0],'r', linewidth=2, linestyle='dashed', label='bulges mergers')
            ind = np.where(ncounts[3,b,:] != 0)
            y = ncounts[3,b,ind]
            ax.plot(xlf_obs[ind],y[0],'LightSalmon', linewidth=2, linestyle='dashdot', label='bulges diskins')

        if (idx == 5):
            common.prepare_legend(ax, ['k','b','r','LightSalmon'], loc='lower right')
    common.savefig(outdir, fig, "number-counts-deep-lightcone-selected.pdf")

def prepare_data(phot_data, phot_data_nod, ids_sed, hdf5_data, subvols, lightcone_dir, ncounts, nbands, ncounts_nodust, zdist):

    (dec, ra, zobs, idgal) = hdf5_data
   
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

    SEDs_nodust   = phot_data_nod[0]

    #print SEDs_dust(0)
    for i in range(0,nbands):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust[i,ind],bins=np.append(mbins,mupp))
        ncounts[0,i,:] = ncounts[0,i,:] + H

        ind = np.where((SEDs_nodust[i,:] > 0) & (SEDs_nodust[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_nodust[i,ind],bins=np.append(mbins,mupp))
        ncounts_nodust[0,i,:] = ncounts_nodust[0,i,:] + H

        #contribution from disks and bulges
        ind = np.where((SEDs_dust_disk[i,:] > 0) & (SEDs_dust_disk[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust_disk[i,ind],bins=np.append(mbins,mupp))
        ncounts[1,i,:] = ncounts[1,i,:] + H

        ind = np.where((SEDs_dust_bulge[i,:] > 0) & (SEDs_dust_bulge[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust_bulge[i,ind],bins=np.append(mbins,mupp))
        ncounts[2,i,:] = ncounts[2,i,:] + H

        ind = np.where((SEDs_dust_bulge_d[i,:] > 0) & (SEDs_dust_bulge_d[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust_bulge_d[i,ind],bins=np.append(mbins,mupp))
        ncounts[3,i,:] = ncounts[3,i,:] + H

        ind = np.where((SEDs_dust_bulge_m[i,:] > 0) & (SEDs_dust_bulge_m[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust_bulge_m[i,ind],bins=np.append(mbins,mupp))
        ncounts[4,i,:] = ncounts[4,i,:] + H

        #redshift distribution for 850microns sources brighter than 5mJy
        if(i == 26):
           ind = np.where((SEDs_dust[i,:] > -10) & (SEDs_dust[i,:] < 14.6525))
           H, bins_edges = np.histogram(zobs[ind],bins=np.append(zbins,zupp))
           zdist[:] = zdist[:] + H

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/waves-g23/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/waves-g23//Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14"
    subvols = (0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  50.58743129433447 #107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2

    fields_sed = {'SED/ap_nodust': ('total', 'disk', 'bulge_t')}

    ids_sed_ab, seds_nod = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    ncounts = np.zeros(shape = (5, nbands, len(mbins)))
    ncounts_nodust = np.zeros(shape = (3, nbands, len(mbins)))
    zdist = np.zeros(shape = (len(zbins)))

    prepare_data(seds, seds_nod, ids_sed, hdf5_data, subvols, lightcone_dir, ncounts, nbands, ncounts_nodust, zdist)

    if(totarea > 0.):
        ncounts   = ncounts/areasub
        ncounts_nodust = ncounts_nodust/areasub
        zdist = zdist/areasub

    # Take logs
    ind = np.where(ncounts > 1e-5)
    ncounts[ind] = np.log10(ncounts[ind])
    ind = np.where(ncounts_nodust > 1e-5)
    ncounts_nodust[ind] = np.log10(ncounts_nodust[ind])


    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_numbercounts(plt, outdir, obsdir, ncounts, ncounts_nodust)
    plot_redshift(plt, outdir, obsdir, zdist)

if __name__ == '__main__':
    main()
