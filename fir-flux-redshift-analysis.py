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
flux_threshs = np.array([1e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 3.5e-1, 4.5e-1, 5.5e-1, 6.5e-1, 7.5e-1, 8.5e-1, 9.5e-1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]) #in milli Jy
flux_threshs_log = np.log10(flux_threshs[:])

flux_threshs_compact = np.array([1e-2, 1e-1, 1.0])
flux_threshs_compactab = -2.5 * np.log10(flux_threshs_compact*1e-3 / 3631.0)

temp_screen = 22.0
temp_bc = 57.0 #49.0 #37.0 

mlowab = 0
muppab = 30
dmab = 0.5
mbinsab = np.arange(mlowab,muppab,dmab)
xlfab   = mbinsab + dmab/2.0

zlow = 0
zupp = 7.0
dz = 0.25
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0

flow = 0
fupp = 50.0
df = 7.5
fbins = np.arange(flow,fupp,df)
xf   = fbins + df/2.0


# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s
h0 = 0.6751
zsun = 0.0189
min_metallicity = zsun * 1e-3

# Function initialization
mlow = -3.0
mupp = 3.0
dm = 0.25
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

mlow2 = -2
mupp2 = 3
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

mslow = 7.0
msupp = 13.0
dms   = 0.35
msbins = np.arange(mslow,msupp,dms)
xmsf   = msbins + dms/2.0

msfrlow = -3.0
msfrupp = 3.0
dmsfr   = 0.35
msfrbins = np.arange(msfrlow,msfrupp,dmsfr)
xmsfr   = msfrbins + dmsfr/2.0



#choose dust model between mm14, rr14 and constdust
m14 = False
rr14 = True
constdust = False
rr14xcoc = False

def dust_mass(mz, mg, h0):
    md = np.zeros(shape = len(mz))
    ind = np.where((mz > 0) & (mg > 0))
    XHd = np.log10(mz[ind]/mg[ind]/zsun)
    if(m14 == True):
        DToM = (polyfit_dm[0] * XHd**4.0 + polyfit_dm[1] * XHd**3.0 + polyfit_dm[2] * XHd**2.0 + polyfit_dm[3] * XHd + polyfit_dm[4])/corrfactor_dm
        DToM = np.clip(DToM, 1e-6, 0.5)
        md[ind] = mz[ind]/h0 * DToM
        DToM_MW = polyfit_dm[4]/corrfactor_dm
    elif(rr14 == True):
         y = np.zeros(shape = len(XHd))
         highm = np.where(XHd > -0.59)
         y[highm] = 10.0**(2.21 - XHd[highm]) #gas-to-dust mass ratio
         lowm = np.where(XHd <= -0.59)
         y[lowm] = 10.0**(0.96 - (3.1) * XHd[lowm]) #gas-to-dust mass ratio
         DToM = 1.0 / y / (mz[ind]/mg[ind])
         DToM = np.clip(DToM, 1e-6, 1)
         md[ind] = mz[ind]/h0 * DToM
         DToM_MW = 1.0 / (10.0**(2.21)) / zsun
    elif(rr14xcoc == True):
         y = np.zeros(shape = len(XHd))
         highm = np.where(XHd > -0.15999999999999998)
         y[highm] = 10.0**(2.21 - XHd[highm]) #gas-to-dust mass ratio
         lowm = np.where(XHd <= -0.15999999999999998)
         y[lowm] = 10.0**(1.66 - 4.43 * XHd[lowm]) #gas-to-dust mass ratio
         DToM = 1.0 / y / (mz[ind]/mg[ind])
         DToM = np.clip(DToM, 1e-6, 1)
         md[ind] = mz[ind]/h0 * DToM
         DToM_MW = 1.0 / (10.0**(2.21)) / zsun
    elif(constdust == True):
         md[ind] = 0.33 * mz[ind]/h0
         DToM_MW = 0.33

    return (md, DToM_MW)


def plot_observations(ax):

    # Add in points from Bethermin (he sent me on 15 May 2019)
    ax.plot(9.0, 0.52, 'o', mfc='black',markersize=9) #Berta+11
    ax.plot(20.0, 0.97, 'o', mfc='purple',markersize=9) #Bethermin+12b
    ax.plot(5.0, 1.4, 'o', mfc='blue',markersize=9) #Geach+13
    ax.plot(13.0, 1.95, 'o', mfc='blue',markersize=9) #Casey+13
    ax.plot(4.1, 2.5, 'o', mfc='aqua',markersize=9) #Wardlow+11
    ax.plot(3.0, 2.2, 'o', mfc='aqua',markersize=9) #Chapman+05
    ax.plot(4.2, 3.1, 'x', mec='green',markersize=9,mew=3) #Smolcic+12
    ax.plot(1.0, 2.2, 'o', mfc='green',markersize=9) #Michalowski+12
    ax.plot(2.0, 2.6, 'o', mfc='green',markersize=9) #Yun+12
    ax.plot(0.25, 2.91, 'o', mfc='red',markersize=9) #Staguhn+14
    # Add on more recent points for observational constraints (interferometric only?)
    ax.plot(3.6, 2.7, 'x', mec='aqua',markersize=9,mew=3)  # da Cunha+15
    ax.plot(3.6, 2.3, 'x', mec='aqua',markersize=9,mew=3)  # Simpson+14
    ax.plot(8.0, 2.65, 'x', mec='aqua',markersize=9,mew=3) # Simpson+17
    ax.plot(2.25, 2.74, 'x', mec='aqua',markersize=9, mew=3)    # Cowie+18
    ax.plot(5.0, 2.8, 'x', mec='aqua',markersize=9, mew=3)    # Dudzeviciute+19 
    ax.plot(3.5, 2.48, 'x', mec='green',markersize=9,mew=3)  # Brisbin+17
    ax.plot(3.3, 3.1, 'x', mec='green',markersize=9,mew=3) # Miettinen15 as revised by Brisbin17
    ax.plot(25.0, 3.9, 's', mfc='none',mec='orange',markersize=9,mew=3) # Strandet+16


def plot_redshift(plt, outdir, obsdir, zdist_flux_cuts, zdist_flux_cuts_scatter, zdist_cosmicvar):

    #thresholds
    ytit="$\\rm Median redshift$"
    xtit="$\\rm Flux\, density\, cut\, (mJy)$"

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xscale('log')
    plt.ylim([0,7])
    plt.xlim([0.2,120]) # OTherwise the solid curves go funky at higher fluxes
    plt.xlabel('Flux density cut (mJy)',size=16)
    plt.ylabel('Median redshift',size=16)

    bands = (0, 1, 2, 4, 5, 6, 7)
    colors  = ('Black','purple','blue','aqua','green','orange','red')
    labels = ('100$\mu$m','250$\mu$m','450$\mu$m','850$\mu$m','1100$\mu$m','1400$\mu$m','2000$\mu$m')
    p = 0
    for j in bands:
        ind = np.where(zdist_flux_cuts[j,0,:] != 0)
        y = zdist_flux_cuts[j,0,ind]
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts[j,1,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.2,interpolate=True)
        ax.plot(flux_threshs[ind],y[0],linestyle='solid',color=colors[p])
        p = p + 1

    plot_observations(ax)
    plt.text(16.0,6.5,'100$\mu$m',size=12,color='black')
    plt.text(16.0,6.25,'250$\mu$m',size=12,color='purple')
    plt.text(16.0,6.0,'450$\mu$m',size=12,color='blue')
    plt.text(16.0,5.75,'850$\mu$m',size=12,color='aqua')
    plt.text(40.0,6.5,'1100$\mu$m',size=12,color='green')
    plt.text(40.0,6.25,'1400$\mu$m',size=12,color='orange')
    plt.text(40.0,6.0,'2000$\mu$m',size=12,color='red')

    plt.text(0.22,6.7,'Lagos et al. 2019 model',size=12,color='black')
    ax.plot([0.25,0.5],[6.5,6.5],color='black',linewidth=2)
    plt.text(0.6,6.4,'Unlensed',size=12,color='black')

    plt.tight_layout()  # This makes sure labels dont go off page
    common.savefig(outdir, fig, "zdistribution-fluxcut-AllIRbands.pdf")
    
    #separating different bands to make it more readable
    #thresholds
    ytit="$\\rm Median redshift$"
    xtit="$\\rm Flux\, density\, cut\, (mJy)$"

    fig = plt.figure(figsize=(10,7))
    subplots = (231, 232, 233, 234, 235, 236)

    bands = (0, 1, 2, 4, 5, 7)
    colors  = ('Black','purple','blue','CadetBlue','green','red')
    labels = ('100$\mu$m','250$\mu$m','450$\mu$m','850$\mu$m','1100$\mu$m','2000$\mu$m')
    p = 0
    for j in bands:
        ax = fig.add_subplot(subplots[p])
        plt.xscale('log')
        plt.ylim([0,5])
        plt.xlim([1e-2,120]) # OTherwise the solid curves go funky at higher fluxes
        if(p == 0 or p == 3):
           plt.ylabel('Median redshift',size=12)
        if(p >= 3):
           plt.xlabel('Flux density cut (mJy)',size=12)
        
        ind = np.where(zdist_flux_cuts[j,0,:] != 0)
        y = zdist_flux_cuts[j,0,ind]
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts[j,1,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.7,interpolate=True)
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts_scatter[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts_scatter[j,2,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.3,interpolate=True)
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_cosmicvar[j,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_cosmicvar[j,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.7,interpolate=True)

        ax.plot(flux_threshs[ind],y[0],linestyle='solid',color=colors[p],label=labels[p])

        if(p == 0):
           ax.plot(9.0, 0.52, 'o', mfc='black',markersize=9) #Berta+11
        elif (p == 1):
           ax.plot(20.0, 0.97, 'o', mfc='purple',markersize=9) #Bethermin+12b
        elif (p == 2):
            ax.plot(5.0, 1.4, 'o', mfc='blue',markersize=9) #Geach+13
            ax.plot(13.0, 1.95, 'o', mfc='blue',markersize=9) #Casey+13
        elif (p == 3):
            ax.plot(4.1, 2.5, 'o', mfc='CadetBlue',markersize=9) #Wardlow+11
            ax.plot(3.0, 2.2, 'o', mfc='CadetBlue',markersize=9) #Chapman+05
            ax.plot(3.6, 2.7, 'D', mec='CadetBlue',markersize=9,mew=2)  # da Cunha+15
            ax.plot(3.6, 2.3, 'D', mec='CadetBlue',markersize=9,mew=2)  # Simpson+14
            ax.plot(8.0, 2.65, 'D', mec='CadetBlue',markersize=9,mew=2) # Simpson+17
            ax.plot(2.25, 2.74, 'D', mec='CadetBlue',markersize=9, mew=2)    # Cowie+18
            ax.plot(5.0, 2.8, 'D', mec='CadetBlue',markersize=9, mew=3)    # Dudzeviciute+19 
        elif (p == 4):
            ax.plot(4.2, 3.1, 'D', mec='green',markersize=9,mew=3) #Smolcic+12
            ax.plot(1.0, 2.2, 'o', mfc='green',markersize=9) #Michalowski+12
            ax.plot(2.0, 2.6, 'o', mfc='green',markersize=9) #Yun+12
            ax.plot(3.5, 2.48, 'D', mec='green',markersize=9,mew=3)  # Brisbin+17
            ax.plot(3.3, 3.1, 'D', mec='green',markersize=9,mew=3) # Miettinen15 as revised by Brisbin17
        elif (p == 5): 
            ax.plot(0.25, 2.91, 'o', mfc='red',markersize=9) #Staguhn+14

        common.prepare_legend(ax, 'k', loc='upper left')
        p = p + 1
  
    plt.tight_layout()  # This makes sure labels dont go off page
    common.savefig(outdir, fig, "zdistribution-fluxcut-AllIRbands-splitpanels.pdf")

def plot_number_counts_smgs(plt, outdir, obsdir,  ncounts_optical_ir_smgs, bands_of_interest_for_smgs, ncounts_all):


    def plot_number_counts(alma_band, name):
        xlf_obs  = xlfab
  
        xtit="$\\rm app\, mag (AB)$"
        ytit="$\\rm log_{10}(N/dex/A\, deg^2)$"
 
        xmin, xmax, ymin, ymax = 10, 30, -2 , 5.5
        xleg = xmax - 0.1 * (xmax-xmin)
        xleg2 = xmin + 0.2 * (xmax-xmin)
        yleg = ymax - 0.1 * (ymax-ymin)
 
        fig = plt.figure(figsize=(12,12))
 
        subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
        idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
        labels= ('u', 'g', 'r', 'J', 'H', 'K', '$4.6\\mu m$', '$12\\mu m$', '$22\\mu m$')
        #"FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
        #"z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
        #"W2_WISE", "W3_WISE", "W4_WISE"
        #bands_of_interest_for_smgs = (0,1,2,3,4,5,6,7,8,10,11,14,17,18)
        bands = (2, 3, 4, 8, 9, 10, 12, 13, 14)

        labelsflux = ('>0.01mJy', '>0.1mJy', '>1mJy')
        colors = ('Indigo','YellowGreen','Orange')
  
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
            if (idx == 0):
                ax.text(xleg,yleg, labels[idx])
            else:
                ax.text(xleg2,yleg, labels[idx])
 
            if (idx == 1):
               ax.text(15,5.7,name,fontsize=17)    

            for f in range(0,len(labelsflux)):
                #Predicted LF
                ind = np.where(ncounts_all[b,:] != 0)
                y = ncounts_all[b,ind]
                ax.plot(xlf_obs[ind],y[0],linewidth=3, linestyle='solid', color='grey')
                ind = np.where(ncounts_optical_ir_smgs[alma_band,b,f,:] != 0)
                y = ncounts_optical_ir_smgs[alma_band,b,f,ind] 
                if(idx == 0):
                    ax.plot(xlf_obs[ind],y[0],linewidth=3, linestyle='solid', color=colors[f], label=labelsflux[f])
        
                if(idx > 0):
                    ax.plot(xlf_obs[ind],y[0],linewidth=3, linestyle='solid', color=colors[f])
 
            if (idx == 0):
                common.prepare_legend(ax, colors, loc='upper left')

        namefig = "number-counts-smgs-" + name + ".pdf"
        common.savefig(outdir, fig, namefig)


    #Define bands
    alma_bands = [0, 1, 2]
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    #plot number counts for all ALMA band selected galaxies
    for j in alma_bands:
        plot_number_counts(j, names_bands[j])

def plot_magnitudes_z_smgs(plt, outdir, obsdir, mags_vs_z_flux_cuts):

    def plot_mags(alma_band, name):
        xlf_obs  = xlfab
  
        xtit="$\\rm redshift$"
        ytit="$\\rm AB\\,  mag$"
 
        xmin, xmax, ymin, ymax = 0, 6, 12, 35
        xleg = xmax - 0.18 * (xmax-xmin)
        yleg = ymin + 0.1 * (ymax-ymin)
 
        fig = plt.figure(figsize=(12,10))
 
        subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
        idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
        labels= ('u', 'g', 'r', 'J', 'H', 'K', '$4.6\\mu m$', '$12\\mu m$', '$22\\mu m$')
        #"FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
        #"z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
        #"W2_WISE", "W3_WISE", "W4_WISE"
        #bands_of_interest_for_smgs = (0,1,2,3,4,5,6,7,8,10,11,14,17,18)
        bands = (2, 3, 4, 8, 9, 10, 12, 13, 14)
        labelsflux = ('>0.01mJy', '>0.1mJy', '>1mJy')
        colors = ('Indigo','Green','Orange')
  
        limits = np.array([28, 25, 25, 30, 26, 27, 28, 30, 28.59, 28.0, 28.4, 25.7, 27.23, 24.0, 21.7])
        idxlims = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8])
        labelslims = np.array(['HUDF','CFHT','HSC','HUDF','HSC wide',' ','HSC udeep','HUDF','JWST NIR','JWST NIR','JWST NIR','A-LESS','JWST NIR','JWST MIR','JWST MIR'])
        xlim = np.array([0,6])
        ylim = np.array([0,0])

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
            common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtitplot, ytitplot, locators=(2, 2, 3, 3))
            ax.text(xleg,yleg, labels[idx])
 
            if (idx == 1):
               ax.text(1.5,36,name,fontsize=17)    
 
            #selec limits to plot
            inpanel = np.where(idxlims == idx)
            for l, labs in zip(limits[inpanel], labelslims[inpanel]):
                ylim[0:2] = l
                ax.plot(xlim, ylim, linestyle='dotted',color='k')
                ax.text(0.2,ylim[0]+0.2,labs,fontsize=12)

            for f in range(0,len(labelsflux)):
                #Predicted LF
                ind = np.where(mags_vs_z_flux_cuts[alma_band,b,f,0,:] != 0)
                y = mags_vs_z_flux_cuts[alma_band,b,f,0,ind]    
                yerrdn = y - mags_vs_z_flux_cuts[alma_band,b,f,1,ind] 
                yerrup = y + mags_vs_z_flux_cuts[alma_band,b,f,2,ind] 
                ax.fill_between(xz[ind],yerrdn[0], yerrup[0], facecolor=colors[f], alpha=0.2,interpolate=True)

                if(idx == 0):
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f], label=labelsflux[f])
        
                if(idx > 0):
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f])

 
            if (idx == 0):
                common.prepare_legend(ax, colors, loc='lower left')


        namefig = "mag_vs_redshift-smgs-" + name + ".pdf"
        common.savefig(outdir, fig, namefig)


    #Define bands
    alma_bands = [0, 1, 2]
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    #plot number counts for all ALMA band selected galaxies
    for j in alma_bands:
        plot_mags(j, names_bands[j])

def plot_props_z_smgs(plt, outdir, obsdir, props_vs_z_flux_cuts, ms_z):

    xtit="$\\rm redshift$"
    ytits=["$\\rm log_{\\rm 10}(M_{\\star}/M_{\\odot})$", "$\\rm log_{\\rm 10}(sSFR/Gyr^{-1})$", "$\\rm log_{\\rm 10}(M_{\\rm halo}/M_{\\odot})$"]
    xmin, xmax = 0, 6 
    ymin = [8, -1, 10]
    ymax = [12, 2, 14]

    xleg = xmax - 0.7 * (xmax-xmin)
 
    fig = plt.figure(figsize=(12,10))
 
    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    labelsflux = ('>0.01mJy', '>0.1mJy', '>1mJy')
    colors = ('Indigo','Green','Orange')
  
    p = 0
    for prop in range(0,3):


        if (p >= 6):
            xtitplot = xtit
        else:
            xtitplot = ' '

        yleg = ymax[prop] - 0.1 * (ymax[prop]-ymin[prop])

        for b in range(0,3):
            ax = fig.add_subplot(subplots[p])
            if b == 0:
               ytitplot = ytits[prop]
            else:
               ytitplot = ' '

            common.prepare_ax(ax, xmin, xmax, ymin[prop], ymax[prop], xtitplot, ytitplot, locators=(2, 2, 0.5, 0.5))
            ax.text(xleg,yleg, names_bands[b])

            if (p == 0):
               ax.errorbar([2.6], [11.079], yerr=[[0.3],[0.25]], xerr=[[0.8],[0.8]],color='k', marker='o')

            if (p == 3):
               ax.errorbar([2.6], [0.49776802469730641], yerr=[[0.5],[0.7]], xerr=[[0.8],[0.8]],color='k', marker='o')

            if(prop == 1):
                ind = np.where(ms_z[0,0,:] != 0)
                y = ms_z[0,0,ind]
                ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)

            for f in range(0,len(labelsflux)):
                #Predicted LF
                ind = np.where(props_vs_z_flux_cuts[b,prop,f,0,:] != 0)
                y = props_vs_z_flux_cuts[b,prop,f,0,ind]
                yerrdn = y - props_vs_z_flux_cuts[b,prop,f,1,ind] 
                yerrup = y + props_vs_z_flux_cuts[b,prop,f,2,ind] 
                ax.fill_between(xz[ind],yerrdn[0], yerrup[0], facecolor=colors[f], alpha=0.2,interpolate=True)
                if(p == 2):
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f], label=labelsflux[f])
                else:
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f])
            if (p == 2):
                common.prepare_legend(ax, colors, loc='lower left')

            p = p + 1

    namefig = "props_vs_redshift-smgs.pdf"
    common.savefig(outdir, fig, namefig)

    #second page of plots
    ytits=["$\\rm log_{\\rm 10}(M_{\\rm dust}/M_{\\odot})$", "Rest-frame $\\rm A_{\\rm V}$", "$\\rm T_{\\rm dust}$"]
    xmin, xmax = 0, 6 
    ymin = [6, 0, 30]
    ymax = [11, 4, 55]

    xleg = xmax - 0.7 * (xmax-xmin)
 
    fig = plt.figure(figsize=(12,10))
 
    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
 
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    labelsflux = ('>0.01mJy', '>0.1mJy', '>1mJy')
    colors = ('Indigo','Green','Orange')
    props = (3,4,5)
    p = 0
    for pn, prop in enumerate(props):
        if (p >= 6):
            xtitplot = xtit
        else:
            xtitplot = ' '

        yleg = ymax[pn] - 0.1 * (ymax[pn]-ymin[pn])

        for b in range(0,3):
            ax = fig.add_subplot(subplots[p])

            if (p == 0):
               ax.errorbar([2.6], [8.83], xerr=[[0.8],[0.8]], yerr=[[0.4],[0.26]],color='k', marker='o')

            if (p == 6):
               #ax.errorbar([2.5], [30.4], yerr=[[4.5],[7.3]], xerr=[[0.8],[0.8]],color='k', marker='o')
               ax.errorbar([2.5], [38.912], yerr=[[4.5],[7.3]], xerr=[[0.8],[0.8]],color='k', marker='o',fill=None)
               #ind = np.where(ms_z[1,0,:] != 0)
               #y = ms_z[1,0,ind]
               #ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)
               #ind = np.where(ms_z[2,0,:] != 0)
               #y = ms_z[2,0,ind]
               #ax.plot(xz[ind],y[0], linestyle='dashed', color='k', linewidth=5)

            if b == 0:
               ytitplot = ytits[pn]
            else:
               ytitplot = ' '
            if(pn == 0):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 1, 1))
            if(pn == 1):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 1, 1))
            if(pn == 2):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 5, 5))

            ax.text(xleg,yleg, names_bands[b])

            for f in range(0,len(labelsflux)):
                #Predicted LF
                ind = np.where(props_vs_z_flux_cuts[b,prop,f,0,:] != 0)
                y = props_vs_z_flux_cuts[b,prop,f,0,ind]
                yerrdn = y - props_vs_z_flux_cuts[b,prop,f,1,ind] 
                yerrup = y + props_vs_z_flux_cuts[b,prop,f,2,ind] 
                ax.fill_between(xz[ind],yerrdn[0], yerrup[0], facecolor=colors[f], alpha=0.2,interpolate=True)
                if(p == 8):
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f], label=labelsflux[f])
                else:
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f])
            if (p == 8):
                common.prepare_legend(ax, colors, loc=4)

            p = p + 1

    namefig = "props_vs_redshift-smgs-secondpage.pdf"
    common.savefig(outdir, fig, namefig)

    #third page of plots
    ytits=["$\\rm f_{SFR,burst}$", "$\\rm r_{\\rm gal}/kpc$", "$\\rm log_{10}(L_{\\rm CO(1-0)}/K km s^{-1} pc^2)$"]
    xmin, xmax = 0, 6 
    ymin = [0, 0.5, 7]
    ymax = [1, 10, 12]

    xleg = xmax - 0.7 * (xmax-xmin)
 
    fig = plt.figure(figsize=(12,10))
 
    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
 
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    labelsflux = ('>0.01mJy', '>0.1mJy', '>1mJy')
    colors = ('Indigo','Green','Orange')
    props = (6,7,8)
    p = 0
    for pn, prop in enumerate(props):
        if (p >= 6):
            xtitplot = xtit
        else:
            xtitplot = ' '

        yleg = ymax[pn] - 0.1 * (ymax[pn]-ymin[pn])

        for b in range(0,3):
            ax = fig.add_subplot(subplots[p])

            if b == 0:
               ytitplot = ytits[pn]
            else:
               ytitplot = ' '
            if(pn == 0):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 0.2, 0.2))
            if(pn == 1):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 2, 2))
            if(pn == 2):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 1, 1))

            ax.text(xleg,yleg, names_bands[b])

            for f in range(0,len(labelsflux)):
                #Predicted LF
                ind = np.where(props_vs_z_flux_cuts[b,prop,f,0,:] != 0)
                y = props_vs_z_flux_cuts[b,prop,f,0,ind]
                yerrdn = y - props_vs_z_flux_cuts[b,prop,f,1,ind] 
                yerrup = y + props_vs_z_flux_cuts[b,prop,f,2,ind] 
                ax.fill_between(xz[ind],yerrdn[0], yerrup[0], facecolor=colors[f], alpha=0.2,interpolate=True)
                if(p == 5):
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f], label=labelsflux[f])
                else:
                    ax.plot(xz[ind],y[0],linewidth=3, linestyle='solid', color=colors[f])
            if (pn  == 0):
                xin = [0,6]
                yin = [0.5,0.5]
                ax.plot(xin, yin, linestyle='dotted', color='k')
            if (p == 5):
                common.prepare_legend(ax, colors, loc='lower right')

            p = p + 1

    namefig = "props_vs_redshift-smgs-thirdpage.pdf"
    common.savefig(outdir, fig, namefig)


def plot_temp_mainseq (plt, outdir, temp_ms_sfr, zinterst):

    bin_it = functools.partial(us.wmedians, xbins=xmsf)

    #plot temperature evolution in main sequence
    xtit="$\\rm log_{10}(M_{\star}/M_{\\odot})$"
    ytit="$\\rm log_{10}(SFR/M_{\\odot}\\, yr^{-1})$"

    xmin, xmax, ymin, ymax = 9, 12, -4, 3
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(16,4.5))

    subplots = (141, 142, 143, 144)
    idx = (0, 1, 2, 3)
   
    for subplot, idx, z in zip(subplots, idx, zinterst):
        ax = fig.add_subplot(subplot)
        if idx == 0:
            ytitplot = ytit
        else:
            ytitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(2, 2, 3, 3))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.15
        zhigh = z + 0.15

        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >= -4))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        z = temp_ms_sfr[2,ind]
        meds = bin_it(x = x, y = y)
        im = ax.hexbin(x[0], y[0], z[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='magma', mincnt=4)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Temperature [K]')
    namefig = "temperature_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)

    #plot dust surface density in SFR-stellar mass plane
    fig = plt.figure(figsize=(16,4.5))
    idx = (0, 1, 2, 3)
   
    for subplot, idx, z in zip(subplots, idx, zinterst):
        ax = fig.add_subplot(subplot)
        if idx == 0:
            ytitplot = ytit
        else:
            ytitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(2, 2, 3, 3))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.15
        zhigh = z + 0.15

        ind = np.where((temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >= -4) & (temp_ms_sfr[5,:] > 0))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        z = temp_ms_sfr[5,ind]
        meds = bin_it(x = x, y = y)
        im = ax.hexbin(x[0], y[0], z[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='magma', mincnt=4)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('$\\rm log_{10}(\\Sigma_{\\rm dust}/M_{\\odot}\\, kpc^{-2})$')
    namefig = "dustsurfacedensity_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)

    #plot SFR surface density in SFR-stellar mass plane
    fig = plt.figure(figsize=(16,4.5))
    idx = (0, 1, 2, 3)
   
    for subplot, idx, z in zip(subplots, idx, zinterst):
        ax = fig.add_subplot(subplot)
        if idx == 0:
            ytitplot = ytit
        else:
            ytitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(2, 2, 3, 3))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.15
        zhigh = z + 0.15

        ind = np.where((temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >= -4))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        z = temp_ms_sfr[6,ind]
        meds = bin_it(x = x, y = y)
        im = ax.hexbin(x[0], y[0], z[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='magma', mincnt=4)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('$\\rm log_{10}(\\Sigma_{\\rm SFR}/M_{\\odot}\\, yr^{-1}\\, kpc^{-2})$')
    namefig = "sfrsurfacedensity_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)
  
    #plot alpha_co evolution in main sequence
    fig = plt.figure(figsize=(16,4.5))
    idx = (0, 1, 2, 3)

    for subplot, idx, z in zip(subplots, idx, zinterst):
        ax = fig.add_subplot(subplot)
        if idx == 0:
            ytitplot = ytit
        else:
            ytitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(2, 2, 3, 3))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.1
        zhigh = z + 0.1
        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >= -4))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        z = temp_ms_sfr[4,ind]
        meds = bin_it(x = x, y = y)
        im = ax.hexbin(x[0], y[0], z[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='Spectral', mincnt=4)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('$\\alpha_{\\rm CO(1-0)}$')
    namefig = "alphaco10_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)

def plot_colors_smgs(plt, outdir, cols_smgs, zinterst):


    threshs = [-1, 0, 100]
    plotsnames = ['faint','bright']

    #plot evolution of SMGs in the UVJ plane
    xtit="$\\rm r-J$"
    ytit="$\\rm u-r$"

    xmin, xmax, ymin, ymax = 0, 2.5, 0.5, 3
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)
    subplots = (121, 122)

    zint = [2,3]
    for p in range(2):
        fig = plt.figure(figsize=(8,4.5))
 
        idx = (0, 1, 2, 3)
        for subplot, idx, z in zip(subplots, idx, zint):
            ax = fig.add_subplot(subplot)
            if idx == 0:
                ytitplot = ytit
            else:
                ytitplot = ' '
            common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(0.5, 0.5, 0.5, 0.5))
            ax.text(xleg,yleg, "z=%s" % str(z))
            zlow = z - 0.5
            zhigh = z + 0.5
 
            ind = np.where((cols_smgs[2,:] > zlow) & (cols_smgs[2,:] < zhigh) & (cols_smgs[0,:] >= 0) & (cols_smgs[1,:] >= 0) & (cols_smgs[3,:] > threshs[p]) & (cols_smgs[3,:] <= threshs[p+1]))
            x = cols_smgs[1,ind]
            y = cols_smgs[0,ind]
            im = ax.hexbin(x[0], y[0], xscale='linear', yscale='linear', gridsize=(20,20), cmap='Spectral', mincnt=4)
            xin = [0,0.8]
            yin = [1.8,1.8] 
            ax.plot(xin,yin,linestyle='dotted',color='k')
            xin = np.array([0.8,2.5])
            ax.plot(xin,xin+1.0,linestyle='dotted',color='k')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel('Number')
        namefig = "optical_colors_smgs_"+plotsnames[p]+".pdf"
        common.savefig(outdir, fig, namefig)

def prepare_data(phot_data, phot_data_nodust, phot_data_ab, phot_data_ab_nodust, ids_sed, hdf5_data, hdf5_co_data, cont_bc, subvols, lightcone_dir, nbands, bands, zdist_flux_cuts, 
                 zdist_flux_cuts_scatter, ncounts_optical_ir_smgs, bands_of_interest_for_smgs, selec_alma,
                 ncounts_all, mags_vs_z_flux_cuts, props_vs_z_flux_cuts, ms_z, n_highz_500microns, zdist_cosmicvar, area):

    (dec, ra, zobs, idgal, msb, msd, mhalo, sfrb, sfrd, typeg, mgd, mgb, mmold, mmolb, zd, zb, dc, rgd, rgb) = hdf5_data
    (SCO, SCO_peak) = hdf5_co_data

    LCO = np.zeros(shape = (len(SCO[:,0]),len(SCO[0,:])))
    dgal = 4.0 * PI * pow((1.0+zobs) * dc/h0, 2.0)
    for l in range(0,len(SCO[0,:])):
        LCO[:,l] = SCO[:,l] * dgal[:]
    LCO10 = LCO[:,0] * 3.25e7 / (115.2712)**2.0 / (4.0*PI) #in K km/s pc^2
    alphaco10 = (mmold + mmolb)/h0/LCO10

    ngals = len(zobs)
    id_order = np.arange(0,ngals,1,  dtype=np.int32)
 
    ind = np.where(zd <= 0)
    zd[ind] = min_metallicity
    ind = np.where(zb <= 0)
    zb[ind] = min_metallicity

    mdd, dtog  = dust_mass(zd * mgd, mgd, h0)
    mdb, dtog  = dust_mass(zb * mgb, mgb, h0)
    mdust = mdd + mdb

    bin_it = functools.partial(us.medians_cum_err, xbins=flux_threshs_log)
    bin_it_scatter = functools.partial(us.wmedians_cum, xbins=flux_threshs_log)

    bin_it_z = functools.partial(us.wmedians, xbins=xz)

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

    SEDs_nodust = phot_data_nodust[0]
    SEDs_ab = phot_data_ab[0]
    SEDs_ab_nodust = phot_data_ab_nodust[0]

    #Av and colours computed from the rest-frame magnitudes
    #band ACS wfc_f555w
    Vnodust = SEDs_ab_nodust[4,:] + 0.4424 * (SEDs_ab_nodust[3,:] - SEDs_ab_nodust[4,:]) + 0.028 
    Vdust = SEDs_ab[4,:] + 0.4424 * (SEDs_ab[3,:] - SEDs_ab[4,:]) + 0.028
    Av = Vdust - Vnodust

    #apparent colours
    UVcol = SEDs_ab[2,:] - SEDs_ab[4,:] 
    VJcol = SEDs_ab[4,:] - SEDs_ab[8,:]

    Contribution_bc = cont_bc[0]
    Contribution_bc = Contribution_bc[0,:]
    temp_total = temp_bc * Contribution_bc + (1.0 - Contribution_bc) * temp_screen
 
    indices = range(len(bands))
    #calculate appropriate number of resamplings 
    bins_var = max(30, np.floor(area/0.5))

    for i, j in zip(bands, indices):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        zdist_flux_cuts[j] = bin_it(x=m[0,:],y=zobs[ind])
        zdist_flux_cuts_scatter[j] = bin_it_scatter(x=m[0,:],y=zobs[ind])
        zdist_cosmicvar[j] = us.compute_cosmic_variance_redshifts(m[0,:], zobs[ind], flux_threshs_log, int(bins_var))
        if (i == 25):
            ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40) & (zobs > 4.0))
            m = (10.0**(SEDs_dust[i,ind]/(-2.5))*3631.0*1e3) #in mJy
            H, bins_edges = np.histogram(m,bins=np.append(fbins,fupp))
            n_highz_500microns[:] =  n_highz_500microns[:] + H
     
    for j, i in enumerate(selec_alma):
        for tn,t in enumerate(flux_threshs_compactab):
            ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < t))
            #np.savetxt('ids_selec_%s_%s.txt' % (str(i), str(t)), id_order[ind]) 

            SEDs_dust_smgs = SEDs_dust[:,ind]
            SEDs_dust_smgs = SEDs_dust_smgs[:,0,:]
            z_smgs = zobs[ind]
            mdust_smgs =  np.log10(mdust[ind])
            ms_smgs = np.log10((msb[ind] + msd[ind])/h0) 
            ssfr_smgs = np.log10((sfrb[ind] + sfrd[ind])/(msb[ind] + msd[ind]))
            mhalo_smgs = np.log10(mhalo[ind]/h0)
            Av_smgs = Av[ind]
            td_smgs = temp_total[ind]
            sfr_ratio_smgs = sfrb[ind] / (sfrb[ind] + sfrd[ind])
            rg_smgs = (rgd[ind] * sfrd[ind] + rgb[ind] * sfrb[ind]) / (sfrb[ind] + sfrd[ind]) * 1e3/h0
            LCO10_smgs = np.log10(LCO10[ind])
            props_vs_z_flux_cuts[j,0,tn,:] = bin_it_z(x=z_smgs, y=ms_smgs)
            props_vs_z_flux_cuts[j,1,tn,:] = bin_it_z(x=z_smgs, y=ssfr_smgs)
            props_vs_z_flux_cuts[j,2,tn,:] = bin_it_z(x=z_smgs, y=mhalo_smgs)
            props_vs_z_flux_cuts[j,3,tn,:] = bin_it_z(x=z_smgs, y=mdust_smgs)
            props_vs_z_flux_cuts[j,4,tn,:] = bin_it_z(x=z_smgs, y=Av_smgs)
            props_vs_z_flux_cuts[j,5,tn,:] = bin_it_z(x=z_smgs, y=td_smgs)
            props_vs_z_flux_cuts[j,6,tn,:] = bin_it_z(x=z_smgs, y=sfr_ratio_smgs)
            props_vs_z_flux_cuts[j,7,tn,:] = bin_it_z(x=z_smgs, y=rg_smgs)
            props_vs_z_flux_cuts[j,8,tn,:] = bin_it_z(x=z_smgs, y=LCO10_smgs)
          
            for bn, b in enumerate(bands_of_interest_for_smgs):
                 ind = np.where((SEDs_dust_smgs[b,:] > 0) & (SEDs_dust_smgs[b,:] < 40))
                 H, bins_edges = np.histogram(SEDs_dust_smgs[b,ind],bins=np.append(mbinsab,muppab))
                 ncounts_optical_ir_smgs[j,bn,tn,:] = ncounts_optical_ir_smgs[j,bn,tn,:] + H
                 mags_selec = SEDs_dust_smgs[b,ind]
                 mags_vs_z_flux_cuts[j,bn,tn,:] = bin_it_z(x=z_smgs[ind], y=mags_selec[0])

    for bn, b in enumerate(bands_of_interest_for_smgs):
        ind = np.where((SEDs_dust[b,:] > 0) & (SEDs_dust[b,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust[b,ind],bins=np.append(mbinsab,muppab))
        ncounts_all[bn,:] = ncounts_all[bn,:] + H

    #selection of centrals in the mass range below to compute main sequence
    ind = np.where(((msb+msd) > 3e9) & ((msb+msd) < 1e10) & (typeg <= 0) & (sfrb + sfrd > 0))
    ms_z[0,:] = bin_it_z(x=zobs[ind], y=np.log10((sfrb[ind] + sfrd[ind])/(msb[ind] + msd[ind])))
    ms_z[1,:] = bin_it_z(x=zobs[ind], y=temp_total[ind])

    #fit to main sequence
    ms_fit = np.polyfit(np.log10(1.0 + zobs[ind]), np.log10((sfrb[ind] + sfrd[ind])/(msb[ind] + msd[ind])), 2)
    #use fit to main sequence to compute temperautre evolution of galaxies in the main sequence and starbursts
    ssfr_gal = (sfrb + sfrd) / (msb + msd)
    main_seq_position = np.log10(ssfr_gal) - (ms_fit[0] * np.log10(1.0 + zobs)**2.0 + ms_fit[1]* np.log10(1.0 + zobs) +  ms_fit[2])

    # main sequence
    ind = np.where((main_seq_position > -0.6) & (main_seq_position < 0.6) & ((msb+msd) > 1e10))
    #np.savetxt('ids_mainseq.txt', id_order[ind]) 
 
    # starbursts
    ind = np.where((main_seq_position > 1) & (main_seq_position < 10.0) & ((msb+msd) > 1e9))
    ms_z[2,:] = bin_it_z(x=zobs[ind], y=temp_total[ind])
    #np.savetxt('ids_sbs.txt', id_order[ind]) 

    # temperature distribution in the mstellar-SFR plane
    ind = np.where(( (msb + msd) > 1e7) & ( (sfrb + sfrd) > 0))
    rg = (rgd[ind] * sfrd[ind] + rgb[ind] * sfrb[ind]) / (sfrb[ind] + sfrd[ind]) * 1e3/h0
    temp_ms_sfr = np.zeros(shape = (7, len(sfrd[ind])))
    temp_ms_sfr[0,:] = np.log10(msb[ind] + msd[ind])
    temp_ms_sfr[1,:] = np.log10((sfrb[ind] + sfrd[ind]) / h0) - 9.0
    temp_ms_sfr[2,:] = temp_total[ind]
    temp_ms_sfr[3,:] = zobs[ind]
    temp_ms_sfr[4,:] = alphaco10[ind]
    temp_ms_sfr[5,:] = np.log10(mdust[ind] / (3.1416 * rg**2.0)) #in Msun/kpc^2
    temp_ms_sfr[6,:] = np.log10((sfrb[ind] + sfrd[ind]) / 1e9/h0 / (3.1416 * rg**2.0)) #in Msun/yr/kpc^2

    #check out a few relevant numbers of A-LESS-like galaxies
    ind = np.where((SEDs_dust[29,:] > 0) & (SEDs_dust[29,:] < 16.514459348683918))
    H, bins_edges = np.histogram(zobs[ind],bins=np.append(zbins,zupp))
    mags = SEDs_dust[29,ind] 
    print (len(mags[0]), len(zobs))
    nsmgs = len(zobs[ind])
    print("Number of A-LESS galaxies %d" % nsmgs)
    ind = np.where((SEDs_dust[29,:] > 0) & (SEDs_dust[29,:] < 16.514459348683918) & (SEDs_dust[10,:] < 25.7))
    nsmgs_kbanddec = len(zobs[ind])
    print("Number of A-LESS galaxies with K-band < 25.7 %d" % nsmgs_kbanddec)

    #band 6 sources with S>0.1mJy
    ind = np.where((SEDs_dust[30,:] > 0) & (SEDs_dust[30,:] < 18.900065622282231))
    cols_smgs = np.zeros(shape = (4, len(sfrd[ind])))
    cols_smgs[0,:] = UVcol[ind]
    cols_smgs[1,:] = VJcol[ind]
    cols_smgs[2,:] = zobs[ind]
    cols_smgs[3,:] = np.log10(10.0**(SEDs_dust[30,ind] / (-2.5)) * 3631.0 * 1e3) #in mJy

    return (temp_ms_sfr, cols_smgs)

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14"

    subvols = [0] #,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 60, 61, 62, 63]#,11,12,13,14,15,16,17,18,19,20,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #(0,1) #range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2
    print ("Area of survey in deg2 %f" % areasub)
    #100, 250, 450, 850, band-7, band-6, band-5, band-4
    bands = (21, 23, 25, 26, 29, 30, 31, 32)

    fields_sed = {'SED/ab_dust': ('total', 'disk')}
    ids_sed, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    fields_sed = {'SED/ab_nodust': ('total', 'disk')}
    ids_sed, seds_ab_nodust = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}
    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields_sed = {'SED/ap_nodust': ('total', 'disk')}
    ids_sed_nodust, seds_nodust = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields_sed = {'SED/lir_dust_contribution_bc': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}
    ids_sed, cont_bc = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky', 'mstars_bulge', 'mstars_disk', 
                           'mvir_hosthalo', 'sfr_burst', 'sfr_disk', 'type',
                           'mgas_disk','msgas_bulge','mmol_disk','mmol_bulge',
                           'zgas_bulge','zgas_disk', 'dc', 'rgas_disk_intrinsic', 'rgas_bulge_intrinsic')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)
  
    fields = {'galaxies': ('SCO','SCO_peak')}
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])


   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    #"FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
    #"z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
    #"W2_WISE", "W3_WISE", "W4_WISE"
    bands_of_interest_for_smgs = (0,1,2,3,4,5,6,7,8,9,10,11,14,17,18)
    selec_alma = (29, 30, 32)
    zinterst = [1.0, 2.0, 3.0, 4.0]

    #define arrays
    #redshift distributions
    zdist_flux_cuts = np.zeros(shape = (len(bands), 2,len(flux_threshs)))
    zdist_flux_cuts_scatter = np.zeros(shape = (len(bands), 3,len(flux_threshs)))
    zdist_cosmicvar = np.zeros(shape = (len(bands), len(flux_threshs)))

    #number counts
    ncounts_optical_ir_smgs = np.zeros(shape = (len(selec_alma), len(bands_of_interest_for_smgs), len(flux_threshs_compact), len(mbinsab)))
    ncounts_all =  np.zeros(shape = (len(bands_of_interest_for_smgs), len(mbinsab)))
    #magnitude and other galaxy propertiesvs redshift
    mags_vs_z_flux_cuts = np.zeros(shape =  (len(selec_alma), len(bands_of_interest_for_smgs),  len(flux_threshs_compact), 3, len(zbins)))
    props_vs_z_flux_cuts = np.zeros(shape =  (len(selec_alma), 9,  len(flux_threshs_compact), 3, len(zbins)))

    #main sequence evolution
    ms_z = np.zeros(shape =  (3, 3, len(zbins)))
    #number counts of 500microns sources at z>4
    n_highz_500microns = np.zeros(shape =  (len(fbins)))

    #process data
    (temp_ms_sfr, cols_smgs) = prepare_data(seds, seds_nodust, seds_ab, seds_ab_nodust, ids_sed, hdf5_data, hdf5_co_data, cont_bc, subvols, lightcone_dir, nbands, bands, zdist_flux_cuts, zdist_flux_cuts_scatter,
                                            ncounts_optical_ir_smgs, bands_of_interest_for_smgs, selec_alma, ncounts_all, mags_vs_z_flux_cuts, 
                                            props_vs_z_flux_cuts, ms_z, n_highz_500microns, zdist_cosmicvar, areasub)
    if(totarea > 0.):
        ncounts_optical_ir_smgs   = ncounts_optical_ir_smgs/areasub/dmab
        ncounts_all = ncounts_all/areasub/dmab
        n_highz_500microns = n_highz_500microns/areasub/dm

    # Take logs
    ind = np.where(ncounts_optical_ir_smgs > 0)
    ncounts_optical_ir_smgs[ind] = np.log10(ncounts_optical_ir_smgs[ind])
    ind = np.where(ncounts_all > 0)
    ncounts_all[ind] = np.log10(ncounts_all[ind])
    ind = np.where(n_highz_500microns > 0)
    n_highz_500microns[ind] = np.log10(n_highz_500microns[ind])
    #for a,b in zip(xf, n_highz_500microns):
    #    print a,b 

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_temp_mainseq(plt, outdir, temp_ms_sfr, zinterst)
    #plot_colors_smgs(plt, outdir, cols_smgs, zinterst)

    #plot_redshift(plt, outdir, obsdir, zdist_flux_cuts, zdist_flux_cuts_scatter, zdist_cosmicvar)
    #plot_number_counts_smgs(plt, outdir, obsdir,  ncounts_optical_ir_smgs, 
    #                        bands_of_interest_for_smgs, ncounts_all)
    #plot_magnitudes_z_smgs(plt, outdir, obsdir, mags_vs_z_flux_cuts)
    #plot_props_z_smgs(plt, outdir, obsdir, props_vs_z_flux_cuts, ms_z)

if __name__ == '__main__':
    main()
