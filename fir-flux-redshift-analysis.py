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


def sample_spt_2020(seds, z):

    fluxes = 10.0**(seds[32,:]/(-2.5)) * 3631.0 * 1e3 #mJy
    sptsources = np.loadtxt('S2mm_SPT.dat')
    tot_gals = np.sum(sptsources[:,2])
    nbins = len(sptsources[:,0])
    Nsampling = 50

    zmeds = np.zeros(shape = (Nsampling, len(xz)))
    for s in range(0,Nsampling):
        g = 0
        z_gals =  np.zeros(shape = (int(tot_gals)))
        for i in range(0,nbins):
            ind = np.where((fluxes[:] > sptsources[i,0]) & (fluxes[:] < sptsources[i,1]))
            nselected = int(sptsources[i,2])
            if(nselected > len(fluxes[ind])):
               nselected = len(fluxes[ind])
            if(nselected > 0):
               fluxesin = fluxes[ind]
               zin = z[ind]
               ids = np.arange(len(fluxes[ind]))
               selected = np.random.choice(ids, size=int(sptsources[i,2]))
               for j in range(0,len(selected)):
                   z_gals[g+j] = zin[selected[j]]
               g = g + nselected
        ind = np.where(z_gals != 0)
        H, bins_edges = np.histogram(z_gals[ind],bins=np.append(zbins,zupp))
        zmeds[s,:] = H

    zmed_spt = np.zeros(shape = (3, len(xz)))
    for i in range(0,len(xz)):
        zmed_spt[:,i] = us.gpercentiles(zmeds[:,i])
        zmed_spt[0,i] = np.max(zmeds[:,i])

    print(zmed_spt[0,:], xz, np.sum(zmed_spt[0,:]))
    return (zmed_spt)

def sample_aless(seds, sedsbm, sedsbd, sedsd, ms, sfr, mh, md, td, av, z, rg):

    bin_it_z = functools.partial(us.medians, xbins=xz)

    #check out a few relevant numbers of A-LESS-like galaxies
    fluxes = 10.0**(seds[29,:]/(-2.5)) * 3631.0 * 1e3 #mJy
    aless_sources = np.loadtxt('ALESSsourcedistribution.dat')
    nbins = len(aless_sources[:,0])
    tot_gals = np.sum(aless_sources[:,2])
    Nsampling = 30
    seds_mjy = 10.0**(seds/(-2.5)) * 3631.0 * 1e3 #mJy
    seds_bm_mjy = 10.0**(sedsbm/(-2.5)) * 3631.0 * 1e3 #mJy
    seds_bd_mjy = 10.0**(sedsbd/(-2.5)) * 3631.0 * 1e3 #mJy
    seds_d_mjy = 10.0**(sedsd/(-2.5)) * 3631.0 * 1e3 #mJy

    seds_survey = np.zeros(shape = (4, 3, Nsampling, len(seds[:,0])))
    props_survey = np.zeros(shape = (Nsampling, 7, 2, len(zbins)))

    for s in range(0,Nsampling):
        g = 0
        seds_gals =  np.zeros(shape = (4,len(seds[:,0]), int(tot_gals)))
        props_gals = np.zeros(shape = (8, int(tot_gals)))

        for i in range(0,nbins):
            ind = np.where((fluxes[:] > aless_sources[i,0]) & (fluxes[:] < aless_sources[i,1]))
            if(len(fluxes[ind]) >= int(aless_sources[i,2])):
               ids = np.arange(len(fluxes[ind]))
               seds_in = seds_mjy[:,ind]
               seds_inbm = seds_bm_mjy[:,ind]
               seds_inbd = seds_bd_mjy[:,ind]
               seds_ind = seds_d_mjy[:,ind]

               ms_in = ms[ind]
               sfr_in = sfr[ind]
               mh_in = mh[ind]
               md_in = md[ind]
               td_in = td[ind]
               av_in = av[ind]
               z_in = z[ind]
               rg_in = rg[ind]
               selected = np.random.choice(ids, size=int(aless_sources[i,2]))
               for j in range(0,len(selected)):
                   seds_gals[0,:,g+j] = seds_in[:,0,selected[j]]
                   seds_gals[1,:,g+j] = seds_inbm[:,0,selected[j]]
                   seds_gals[2,:,g+j] = seds_inbd[:,0,selected[j]]
                   seds_gals[3,:,g+j] = seds_ind[:,0,selected[j]]
                   props_gals[0,g+j] = ms_in[selected[j]]
                   props_gals[1,g+j] = sfr_in[selected[j]]-ms_in[selected[j]] + 9.0
                   props_gals[2,g+j] = mh_in[selected[j]]
                   props_gals[3,g+j] = md_in[selected[j]]
                   props_gals[4,g+j] = av_in[selected[j]]
                   props_gals[5,g+j] = td_in[selected[j]]
                   props_gals[6,g+j] = rg_in[selected[j]]
                   props_gals[7,g+j] = z_in[selected[j]]
               g = g + int(aless_sources[i,2])
        for p in range(0,7):
            props_survey[s,p,:] = bin_it_z(x=props_gals[7,:], y=props_gals[p,:])

        for b in range(0,len(seds[:,0])):
            for c in range(0,4):
                seds_survey[c,:,s,b] = us.gpercentiles(np.nan_to_num(seds_gals[c,b,:]))

    seds_ave = np.zeros(shape = (len(seds[:,0]), 4, 3))
    props_ave = np.zeros(shape = (7, len(zbins)))
    for b in range(0,len(seds[:,0])):
        for c in range(0,4):
            seds_ave[b,c,0] = np.mean(seds_survey[c,0,:,b])
            seds_ave[b,c,1] = seds_ave[b,c,0] - np.mean(seds_survey[c,1,:,b])
            seds_ave[b,c,2] = seds_ave[b,c,0] + np.mean(seds_survey[c,2,:,b])
    for p in range(0,7):
        for j in range(0,len(zbins)):
            ind = np.where(props_survey[:,p,0,j] != 0)
            props_ave[p,j] = np.median(props_survey[ind,p,0,j])

    return (np.log10(seds_ave), np.nan_to_num(props_ave[:,:]))
 
def sample_as2uds(seds,zobs,ms,Av,sfr,mdust,temp):

    selec_band7_flux = 16.954687496323121 #17.152640611442184 #16.954687496323121
    selec_band7_flux_bright = 13.566218351356687
    #check out a few relevant numbers of A-LESS-like galaxies
    fluxes = 10.0**(seds[29,:]/(-2.5)) * 3631.0 * 1e3 #mJy
    irac1fluxes = seds[12,:]
    kbandfluxes = seds[10,:]
    limitirac = 23.5
    limitk = 25.7
    au2uds_sources = np.loadtxt('AS2UDSsourcedistribution.dat')
    nbins = len(au2uds_sources[:,0])

    Nsampling = 30
    fractions = np.zeros(shape = Nsampling)
    fractionsk = np.zeros(shape = Nsampling)
    for s in range(0,Nsampling):
        nbright = 0
        nbrightk = 0
        for i in range(0,nbins):
            ind = np.where((fluxes[:] > au2uds_sources[i,0]) & (fluxes[:] < au2uds_sources[i,1]))
            selected = np.random.choice(irac1fluxes[ind], size=int(au2uds_sources[i,2]))
            selectedk = np.random.choice(kbandfluxes[ind], size=int(au2uds_sources[i,2]))
            bright = np.where(selected < limitirac)
            nbright = nbright + len(selected[bright])
            bright = np.where(selectedk < limitk)
            nbrightk = nbrightk + len(selectedk[bright])
    
        fractions[s] = (nbright + 0.0) / np.sum(au2uds_sources[:,2])
        fractionsk[s] = (nbrightk + 0.0) / np.sum(au2uds_sources[:,2])

    print("Fraction of AS2UDS 3.6 IRAC < 23.5 %s error %s" % (str(np.mean(fractions)), str(np.std(fractions))))
    print("Fraction of AS2UDS K-band < 25.7 %s error %s" % (str(np.mean(fractionsk)), str(np.std(fractionsk))))

    #compute median properties
    ssfr = np.log10(sfr/ms)
    ms = np.log10(ms)
    mdust = np.log10(mdust)

    bright = np.where((irac1fluxes < limitirac) & (fluxes > 1))
    faint = np.where((irac1fluxes > limitirac) & (fluxes > 1))
    props_bright = np.zeros(shape = (6, 3))
    props_faint = np.zeros(shape = (6, 3))
    props_bright[0,:] = us.gpercentiles(ms[bright])
    props_bright[1,:] = us.gpercentiles(ssfr[bright])
    props_bright[2,:] = us.gpercentiles(Av[bright])
    props_bright[3,:] = us.gpercentiles(temp[bright])
    props_bright[4,:] = us.gpercentiles(zobs[bright])
    props_bright[5,:] = us.gpercentiles(mdust[bright])
    props_faint[0,:]  = us.gpercentiles(ms[faint])
    props_faint[1,:]  = us.gpercentiles(ssfr[faint])
    props_faint[2,:]  = us.gpercentiles(Av[faint])
    props_faint[3,:]  = us.gpercentiles(temp[faint])
    props_faint[4,:]  = us.gpercentiles(zobs[faint])
    props_faint[5,:]  = us.gpercentiles(mdust[faint])
    print(props_bright, props_faint)
    print("Average Ms, sSFR, Av, T, zobs, Mdust of IRAC bright (%s, %s, %s, %s, %s, %s)" % (str(np.median(ms[bright])), str(np.median(ssfr[bright])), str(np.median(Av[bright])), str(np.median(temp[bright])), str(np.median(zobs[bright])), str(np.median(mdust[bright]))))
    print("Average Ms, sSFR, Av, T, zobs, Mdust of IRAC faint  (%s, %s, %s, %s, %s, %s)" % (str(np.median(ms[faint])), str(np.median(ssfr[faint])), str(np.median(Av[faint])), str(np.median(temp[faint])), str(np.median(zobs[faint])), str(np.median(mdust[faint]))))

def color_color_analysis_smgs(colur, colrj, band6flux, z):
    
    zbins = [1, 2, 3, 4, 5]
    fluxthresh = [0.1, 1]

    for zb in zbins:
        for flux in fluxthresh:
            passiveall = np.where((colur[:] > 1.8) & (colur[:] - colrj[:] > 1) & (z[:] > zb-0.5) & (z[:] < zb + 0.5))
            numpassive = len(colur[passiveall])
            band6fluxpass = band6flux[passiveall]
            smgs = np.where((band6fluxpass > flux) & (band6fluxpass < 1e10))
            numsmgspassive = len(band6fluxpass[smgs])
            if(numpassive > 0):
               print("Redshift %s, percentage of passive galaxies in u-r vs r-J that have band-6 > %s is %s of %s" % (str(zb), str(flux), str((numsmgspassive+0.0)/(numpassive+0.0)*100.0), str(numpassive)))
        
            smgs = np.where((band6flux > flux) & (band6flux < 1e10))
            numsmgs = len(band6flux[smgs])
            colursmg = colur[smgs]
            colrjsmg = colrj[smgs]
            if(numsmgs > 0):
               print("Redshift %s, percentage of band-6 sources with flux>%s that are classified passive galaxies in u-r vs r-J plane is %s of %s" % (str(zb), str(flux), str((numsmgspassive+0.0)/(numsmgs+0.0)*100.0), str(numsmgs)))
    

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
    ax.plot(1.0, 4.1, 'o', mfc='red',markersize=9) #Magnelli+19

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

    bands = (0, 1, 2, 3, 5, 6, 7)
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
            #ax.plot(8.0, 2.65, 'D', mec='CadetBlue',markersize=9,mew=2) # Simpson+17
            ax.plot(1, 2.32, 'D', mec='CadetBlue',markersize=9, mew=2)    # Cowie+18
            ax.plot(1.5, 2.8, 'D', mec='CadetBlue',markersize=9, mew=3)    # Dudzeviciute+19 
        elif (p == 4):
            ax.plot(4.2, 3.1, 'D', mec='green',markersize=9,mew=3) #Smolcic+12
            ax.plot(1.0, 2.2, 'o', mfc='green',markersize=9) #Michalowski+12
            ax.plot(2.0, 2.6, 'o', mfc='green',markersize=9) #Yun+12
            ax.plot(3.5, 2.48, 'D', mec='green',markersize=9,mew=3)  # Brisbin+17
            ax.plot(3.3, 3.1, 'D', mec='green',markersize=9,mew=3) # Miettinen15 as revised by Brisbin17
        elif (p == 5): 
            ax.plot(0.25, 2.91, 'o', mfc='red',markersize=9) #Staguhn+14
            ax.plot(0.4, 3.9, 'D', mfc='red',markersize=9) #Magnelli+19
            #ax.plot(1.0, 3.8, 's', mfc='red',markersize=9) #Reuter+20

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
  
        limits = np.array([28, 25, 26.5, 28.1, 30, 26.1, 27.7, 30, 28.59, 28.0, 28.4, 27.23, 24.0, 21.7])
        idxlims = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8])
        labelslims = np.array(['HUDF','CFHT','HSC wide','HSC udeep','HUDF','HSC wide','HSC udeep','HUDF','JWST NIR','JWST NIR','JWST NIR','JWST NIR','JWST MIR','JWST MIR'])
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

            #for band-7 K-band, plot AS2UDS K-band:
            if (idx == 5):
                if(name == 'Band-7-selected'):
                   file = obsdir+'/lf/kband_evolution_as2uds.dat'
                   zD20, kD20 = np.loadtxt(file,usecols=[0,1],unpack=True)
                   ax.plot(zD20, kD20, linestyle='dashed', color='k',label='AS2UDS')
            if (idx == 0):
                common.prepare_legend(ax, colors, loc='lower left')
            if (idx == 5):
                if(name == 'Band-7-selected'):
                   common.prepare_legend(ax, 'k', loc='lower left')


        namefig = "mag_vs_redshift-smgs-" + name + ".pdf"
        common.savefig(outdir, fig, namefig)


    #Define bands
    alma_bands = [0, 1, 2]
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    #plot number counts for all ALMA band selected galaxies
    for j in alma_bands:
        plot_mags(j, names_bands[j])

def plot_props_z_smgs(plt, outdir, obsdir, props_vs_z_flux_cuts, ms_z, most_massive_z):


    data_aless_like = np.loadtxt('Shark_ALESS_Like.dat')

    xtit="$\\rm redshift$"
    ytits=["$\\rm log_{\\rm 10}(M_{\\star}/M_{\\odot})$", "$\\rm log_{\\rm 10}(sSFR/Gyr^{-1})$", "$\\rm log_{\\rm 10}(M_{\\rm halo}/M_{\\odot})$"]
    xmin, xmax = 0, 6 
    ymin = [8, -1, 11.0]
    ymax = [12, 2, 14.7]

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
               ax.errorbar([2.6], [11.079], yerr=[[0.3],[0.25]], xerr=[[0.8],[0.8]],color='k', marker='s',fill=None, alpha=0.5)
               ax.errorbar([3], [10.95], xerr=[[0.3],[0.8]], yerr=[[0.4],[0.6]],color='k', marker='o',fill=None, alpha=0.5)
               ax.errorbar([2], [10.75], xerr=[[0.8],[0.7]], yerr=[[0.8],[0.8]],color='k', marker='o',fill=None, alpha=0.5)
               ax.plot(data_aless_like[:,0], data_aless_like[:,prop+1], linestyle='dashdot', color='Firebrick')
            if (p == 3):
               ax.errorbar([2.6], [0.49776802469730641], yerr=[[0.5],[0.7]], xerr=[[0.8],[0.8]],color='k', marker='s', fill=None, alpha=0.5,label='AS2UDS')
               ax.errorbar([3], [0.45], xerr=[[0.3],[0.8]], yerr=[[0.5],[0.5]],color='k', marker='o',fill=None,alpha=0.5, label='ALESS')
               ax.errorbar([2], [0.65], xerr=[[0.8],[0.7]], yerr=[[0.7],[0.7]],color='k', marker='o',fill=None,alpha=0.5)
               ax.plot(data_aless_like[:,0], data_aless_like[:,prop+1], linestyle='dashdot', color='Firebrick',label='ALESS-like')
            if (p == 6):
               ax.errorbar([2], [12.8], yerr=[[0.32],[0.37]], xerr=[[0.5],[0.5]],color='k', marker='d', fill=None, alpha=0.5)
               ax.plot(data_aless_like[:,0], data_aless_like[:,prop+1], linestyle='dashdot', color='Firebrick')
            if(prop == 0):
               ax.plot(xz, most_massive_z[0,:], linestyle='dashed', color='k', linewidth=3)

            if(prop == 1):
               ind = np.where(ms_z[0,0,:] != 0)
               y = ms_z[0,0,ind]
               ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)

            if(prop == 2):
               ax.plot(xz, most_massive_z[1,:], linestyle='dashed', color='k', linewidth=3)

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
            if (p == 3):
                common.prepare_legend(ax, ('Firebrick', 'k', 'k'), loc='lower right')

            if (p == 2):
                common.prepare_legend(ax, colors, loc='lower left')

            p = p + 1

    namefig = "props_vs_redshift-smgs.pdf"
    common.savefig(outdir, fig, namefig)

    #second page of plots
    ytits=["$\\rm log_{\\rm 10}(M_{\\rm dust}/M_{\\odot})$", "$\\rm T_{\\rm dust}$", "Rest-frame $\\rm A_{\\rm V}$"]
    xmin, xmax = 0, 6 
    ymin = [6, 30, 0]
    ymax = [11, 55, 4]

    xleg = xmax - 0.7 * (xmax-xmin)
 
    fig = plt.figure(figsize=(12,10))
 
    subplots = (331, 332, 333, 334, 335, 336, 337, 338, 339)#, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525)
 
    idx = (0, 1, 2, 3, 4, 5, 6, 7, 8)#, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    names_bands = ['Band-7-selected', 'Band-6-selected', 'Band-4-selected']

    labelsflux = ('>0.01mJy', '>0.1mJy', '>1mJy')
    colors = ('Indigo','Green','Orange')
    props = (3,5,4)
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
               ax.errorbar([3], [8.65], xerr=[[0.3],[0.8]], yerr=[[0.3],[0.4]],color='k', marker='o',fill=None, alpha=0.5)
               ax.errorbar([2], [8.75], xerr=[[0.8],[0.7]], yerr=[[0.5],[0.3]],color='k', marker='o',fill=None, alpha=0.5)

               ax.plot(data_aless_like[:,0], data_aless_like[:,pn+4], linestyle='dashdot', color='Firebrick')

            if (p == 6):
               ax.errorbar([3], [2.4], xerr=[[0.3],[0.8]], yerr=[[1.5],[1.5]],color='k', marker='o',fill=None, alpha=0.5)
               ax.errorbar([2], [1.6], xerr=[[0.8],[0.7]], yerr=[[0.8],[0.8]],color='k', marker='o',fill=None, alpha=0.5)
               ax.plot(data_aless_like[:,0], data_aless_like[:,pn+4], linestyle='dashdot', color='Firebrick')

            if (p == 3):
               ax.errorbar([3], [43.0], xerr=[[0.3],[0.8]], yerr=[[10],[10]],color='k', marker='o',fill=None, alpha=0.5,  label='ALESS')
               ax.errorbar([2], [38.0], xerr=[[0.8],[0.7]], yerr=[[10],[10]],color='k', marker='o',fill=None, alpha=0.5)

               ax.plot(data_aless_like[:,0], data_aless_like[:,pn+4], linestyle='dashdot', color='Firebrick',  label='ALESS-like')

            if(prop == 3):
               ax.plot(xz, most_massive_z[2,:], linestyle='dashed', color='k', linewidth=3)
               ax.plot(xz, ms_z[4,0,:], linestyle='dotted', color='k', linewidth=5)

            if(prop == 4):
               ind = np.where(ms_z[3,0,:] != 0)
               y = ms_z[3,0,ind]
               ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)

            if b == 0:
               ytitplot = ytits[pn]
            else:
               ytitplot = ' '
            if(pn == 0):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 1, 1))
            if(pn == 2):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 1, 1))
            if(pn == 1):
               common.prepare_ax(ax, xmin, xmax, ymin[pn], ymax[pn], xtitplot, ytitplot, locators=(2, 2, 5, 5))

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
            if (p == 3):
                common.prepare_legend(ax, ('Firebrick', 'k'), loc='lower right')
            if (p == 5):
                common.prepare_legend(ax, colors, loc=4)

            p = p + 1

    namefig = "props_vs_redshift-smgs-secondpage.pdf"
    common.savefig(outdir, fig, namefig)

    #third page of plots
    ytits=["$\\rm f_{SFR,burst}$", "$\\rm r_{\\rm gal}/kpc$", "$\\rm log_{10}(L^{\\prime}_{\\rm CO(1-0)}/K km s^{-1} pc^2)$"]
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
            if(prop == 7):
               plt.yscale('log')
 
            if(prop != 7):
               ax.text(xleg,yleg, names_bands[b])
            else:
               ax.text(0.2,8, names_bands[b])

            if(prop == 6):
               ind = np.where(ms_z[5,0,:] != 0)
               y = ms_z[5,0,ind]
               ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)
            if(prop == 7):
               ind = np.where(ms_z[6,0,:] != 0)
               y = ms_z[6,0,ind]
               ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)
               if(b == 0):
                  ax.plot(data_aless_like[:,0], data_aless_like[:,7], linestyle='dashdot', color='Firebrick',label='ALESS-like')
                  ax.errorbar([2.1], [2.4], xerr=[[0.8],[0.8]], yerr=[[0.2],[0.2]],color='k', marker='D',fill=None, alpha=0.5,label='Simpson+15')
            if(prop == 8):
               ind = np.where(ms_z[7,0,:] != 0)
               y = ms_z[7,0,ind]
               ax.plot(xz[ind],y[0], linestyle='dotted', color='k', linewidth=5)

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
            if (p == 3):
                common.prepare_legend(ax, ['Firebrick','k'], loc='upper right')

            if (p == 5):
                common.prepare_legend(ax, colors, loc='upper right')

            p = p + 1

    namefig = "props_vs_redshift-smgs-thirdpage.pdf"
    common.savefig(outdir, fig, namefig)


def plot_temp_mainseq (plt, outdir, temp_ms_sfr, zinterst, obsdir):

    bin_it = functools.partial(us.wmedians, xbins=xmsf)
    bin_it_ssfr = functools.partial(us.wmedians, xbins=xlf)

    def plot_ms(z, ax):
        m = xmsf - 9
        r = np.log10(1 + z)
        yoff =  m - 0.36 - 2.5 * r
        ind=np.where(yoff < 0)
        yoff[ind] = 0
        ax.plot(xmsf, m - 0.5 + 1.5 * r - 0.3 * yoff**2.0,linestyle='dashed', color='grey')


    #plot temperature evolution in main sequence
    xtit="$\\rm log_{10}(M_{\star}/M_{\\odot})$"
    ytit="$\\rm log_{10}(SFR/M_{\\odot}\\, yr^{-1})$"

    xmin, xmax, ymin, ymax = 9, 12, -3, 3
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
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(1, 1, 1, 1))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.15
        zhigh = z + 0.15

        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] > -3) & (temp_ms_sfr[1,:]-temp_ms_sfr[0,:]+9 > -1.25 + 0.5*np.log10(z+1)))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        meds = bin_it(x = x, y = y)

        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >-3) & (temp_ms_sfr[10,:] > 5) & (temp_ms_sfr[1,:] != 0))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        a = temp_ms_sfr[2,ind]
        im = ax.hexbin(x[0], y[0], a[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='magma', mincnt=3)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')
        plot_ms(z, ax)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Temperature [K]')
    namefig = "temperature_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)

    #plot dust mass evolution in main sequence
    fig = plt.figure(figsize=(16,4.5))

    idx = (0, 1, 2, 3)
    for subplot, idx, z in zip(subplots, idx, zinterst):
        ax = fig.add_subplot(subplot)
        if idx == 0:
            ytitplot = ytit
        else:
            ytitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(1, 1, 1, 1))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.15
        zhigh = z + 0.15

        #change values below 5 to 5 and above 12 to 12 (for ease of plotting the range of dust masses)
        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] > -3) & (temp_ms_sfr[1,:]-temp_ms_sfr[0,:]+9 > -1.25 + 0.5*np.log10(z+1)))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        meds = bin_it(x = x, y = y)

        temp_ms_sfr[10,:] =  np.clip(temp_ms_sfr[10,:], 5, 12)
        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >-3) & (temp_ms_sfr[10,:] > 5) & (temp_ms_sfr[1,:] != 0))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        a = temp_ms_sfr[10,ind]
        im = ax.hexbin(x[0], y[0], a[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='magma', mincnt=4)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')
        plot_ms(z, ax)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('$\\rm log_{10}(M_{\\rm dust}/M_{\\odot})$')
    namefig = "mdust_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)

    #plot LCO(1-0) evolution in main sequence
    fig = plt.figure(figsize=(16,4.5))

    idx = (0, 1, 2, 3)
    for subplot, idx, z in zip(subplots, idx, zinterst):
        ax = fig.add_subplot(subplot)
        if idx == 0:
            ytitplot = ytit
        else:
            ytitplot = ' '
        common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytitplot, locators=(1, 1, 1, 1))
        ax.text(xleg,yleg, "z=%s" % str(z))
        zlow = z - 0.15
        zhigh = z + 0.15

        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] > -3) & (temp_ms_sfr[1,:]-temp_ms_sfr[0,:]+9 > -1.25 + 0.5*np.log10(z+1)))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        meds = bin_it(x = x, y = y)

        #change values below 5 to 5 and above 12 to 12 (for ease of plotting the range of dust masses)
        temp_ms_sfr[10,:] =  np.clip(temp_ms_sfr[10,:], 5, 12)
        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >-3) & (temp_ms_sfr[10,:] > 5) & (temp_ms_sfr[1,:] != 0) & (temp_ms_sfr[12,:] > 0))
        x = temp_ms_sfr[0,ind]
        y = temp_ms_sfr[1,ind]
        a = temp_ms_sfr[12,ind]
        im = ax.hexbin(x[0], y[0], a[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='jet', mincnt=4)
        ind = np.where(meds[0,:] != 0)
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xmsf[ind], y[0], 'k', linestyle='solid')
        ax.plot(xmsf[ind], ydn[0], 'k', linestyle='dotted')
        ax.plot(xmsf[ind], yup[0], 'k', linestyle='dotted')
        plot_ms(z, ax)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('$\\rm log_{10}(L^{\\prime}_{\\rm CO(1-0)}/K\\, km\\, s^{-1} \\, pc^{2})$')
    namefig = "lco10_mstar_sfr.pdf"
    common.savefig(outdir, fig, namefig)


    #plot temperature vs sSFR evolution
    xtit="$\\rm log_{10}(sSFR_{\star}/Gyr^{-1})$"
    ytit="$\\rm Dust\\, temperature/K$"

    xmin, xmax, ymin, ymax = -3, 2.1, 20, 60
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4.5))
    idx = (0, 1, 2, 3)
    colors= ['Firebrick', 'YellowGreen', 'Teal', 'DarkMagenta']
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 5, 5))
  
    for idx, z in zip(idx, zinterst):
        zlow = z - 0.15
        zhigh = z + 0.15

        ind = np.where( (temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] - temp_ms_sfr[0,:] > -12) & (temp_ms_sfr[6,:] != 0))
        x = temp_ms_sfr[1,ind] - temp_ms_sfr[0,ind] + 9.0
        y = temp_ms_sfr[2,ind]
        p = temp_ms_sfr[11,ind]
        meds = bin_it_ssfr(x = x, y = y) 
        #if(idx == 1):
        #   im = ax.hexbin(x[0], y[0], p[0], xscale='linear', yscale='linear', gridsize=(15,15), cmap='coolwarm', mincnt=5)
        ind = np.where((meds[0,:] != 0) & (meds[3,:] > 9))
        y =  meds[0,ind]
        ydn = meds[0,ind] - meds[1,ind]
        yup = meds[0,ind] + meds[2,ind]
        ax.plot(xlf[ind], y[0], color=colors[idx], linestyle='solid', label="z=%s" % str(z))
        ax.plot(xlf[ind], ydn[0], color=colors[idx], linestyle='dotted')
        ax.plot(xlf[ind], yup[0], color=colors[idx], linestyle='dotted')
    fig.subplots_adjust(bottom=0.15,right=0.85)
    #cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
    #cbar = fig.colorbar(im, cax=cbar_ax)
    #cbar.ax.set_ylabel('$\\rm SFR_{\\rm bursts}/SFR_{\\rm total}$')
    common.prepare_legend(ax, colors, loc='lower right')
    namefig = "temperature_sSFR_dustsurfacedensity.pdf"
    common.savefig(outdir, fig, namefig)


    #plot dust surface density in SFR-stellar mass plane
    xtit="$\\rm log_{10}(M_{\star}/M_{\\odot})$"
    ytit="$\\rm log_{10}(SFR/M_{\\odot}\\, yr^{-1})$"

    xmin, xmax, ymin, ymax = 9, 12, -3, 3
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    ids_matrix = [5, 7, 8]
    names = ['total', 'disk', 'bulge']
    for p in range(0,len(ids_matrix)):
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
    
            ind = np.where((temp_ms_sfr[3,:] > zlow) & (temp_ms_sfr[3,:] < zhigh) & (temp_ms_sfr[0,:] >= 9.0) & (temp_ms_sfr[1,:] >= -4) & (temp_ms_sfr[ids_matrix[p],:] > 0))
            x = temp_ms_sfr[0,ind]
            y = temp_ms_sfr[1,ind]
            z = temp_ms_sfr[ids_matrix[p],ind]
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
        namefig = "dustsurfacedensity_mstar_sfr_" + names[p] + ".pdf"
        common.savefig(outdir, fig, namefig)

    #plot SFR surface density in SFR-stellar mass plane
    xtit="$\\rm log_{10}(M_{\star}/M_{\\odot})$"
    ytit="$\\rm log_{10}(SFR/M_{\\odot}\\, yr^{-1})$"

    xmin, xmax, ymin, ymax = 9, 12, -3, 3
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

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
    labelsflux = ['$\\rm 0.1<S_{\\rm band-7}<1$','$\\rm S_{\\rm band-7} > 1$']
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
            ax.text(1.3,yleg-0.2, labelsflux[p])

            zlow = z - 0.5
            zhigh = z + 0.5
 
            ind = np.where((cols_smgs[2,:] > zlow) & (cols_smgs[2,:] < zhigh) & (cols_smgs[0,:] >= 0) & (cols_smgs[1,:] >= 0) & (cols_smgs[3,:] > threshs[p]) & (cols_smgs[3,:] <= threshs[p+1]))
            x = cols_smgs[1,ind]
            y = cols_smgs[0,ind]
            z = np.zeros(shape = (len(x[0])))
            z[:] = 1.0/(len(x[0]) + 0.0) 
            im = ax.hexbin(x[0], y[0], z, xscale='linear', yscale='linear', gridsize=(20,20), cmap='Spectral', mincnt=4, reduce_C_function=np.sum)
            xin = [0,0.8]
            yin = [1.8,1.8] 
            ax.plot(xin,yin,linestyle='dotted',color='k')
            xin = np.array([0.8,2.5])
            ax.plot(xin,xin+1.0,linestyle='dotted',color='k')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.86, 0.15, 0.025, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel('$\\rm N/N_{\\rm total}$')
        namefig = "optical_colors_smgs_"+plotsnames[p]+".pdf"
        common.savefig(outdir, fig, namefig)

def plot_seds_smgs(plt, outdir, obsdir, SEDs_dust_smgs, zsmgs, seds_aless): 

    #plot evolution of SMGs in the UVJ plane
    xtit="$\\rm log_{10}(\lambda/Ang\, (obs-frame))$"
    ytit="$\\rm log_{10}(f/mJy)$"
    #wavelength in angstroms.
    file = 'Shark_SED_bands.dat'
    lambda_bands = np.loadtxt(file,usecols=[0],unpack=True)
    lambda_bands = np.log10(lambda_bands)

    xmin, xmax, ymin, ymax = 3.0, 7.35, -5, 1.3
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,7))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 1, 1))
    ax.text(6, -4.5, '$\\rm S_{\\rm band-7}>1$ mJy')
    zlist = [1.5, 2.5, 3.5, 4.5, 5.5]
    colors = plt.cm.jet(np.linspace(0,1,len(zlist)))
    for iz, z in enumerate(zlist):
        ind = np.where((zsmgs > z - 0.5) & (zsmgs < z + 0.5))
        nselected = len(zsmgs[ind])
        nrandom = 5
        if(nselected > nrandom):
           SEDs_dust_smgs_in = SEDs_dust_smgs[:,0,ind]
           ave_sed = np.zeros(shape = (3,len(lambda_bands)))
           for b in range(0,len(lambda_bands)):
               mags = SEDs_dust_smgs_in[b,0,:]
               incal = np.where((mags > 0) & (mags < 100))
               ave_sed[:,b] = us.gpercentiles(np.log10(10.0**(mags[incal]/(-2.5)) * 3631.0 * 1e3))
           ax.fill_between(lambda_bands,ave_sed[0,:]+ave_sed[2,:], ave_sed[0,:]-ave_sed[1,:],facecolor=colors[iz], alpha=0.3,interpolate=True)
           ax.plot(lambda_bands, ave_sed[0,:], linewidth=1, linestyle='solid', color=colors[iz], label='z=%s' % str(z))

    common.prepare_legend(ax, colors, loc='upper left')
    namefig = "example_seds_smgs_band7.pdf"
    common.savefig(outdir, fig, namefig)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, 2.5, xtit, ytit, locators=(1, 1, 1, 1))
    fig.subplots_adjust(bottom=0.3)

    xobs, yobs = np.loadtxt(obsdir + 'lf/CSED/SED_ALESS_daCunha15.dat', usecols=[0,1], unpack = True)
    ax.plot(np.log10(xobs * 1e4 * 3.2), yobs, linestyle='dotted', linewidth = 5, color='k', label='da Cunha+15')
    ax.fill_between(lambda_bands,seds_aless[:,0,1],seds_aless[:,0,2],facecolor='red', alpha=0.3,interpolate=True)
    ax.plot(lambda_bands, seds_aless[:,0,0], linestyle='solid', color='red', label='Shark total')
    #bulge_m
    ax.plot(lambda_bands, seds_aless[:,1,0], linestyle='dashed', color='PaleVioletRed', label='bulges by mergers')
    #bulge_d
    ax.plot(lambda_bands, np.log10(10.0** seds_aless[:,0,0] - (10.0**seds_aless[:,1,0] + 10.0**seds_aless[:,3,0])), linestyle='dashed', color='DarkCyan', label='bulges by DI')
    #disk
    ax.plot(lambda_bands, seds_aless[:,3,0], linestyle='dashed', color='blue',label='disks')

    common.prepare_legend(ax, ['k','red','PaleVioletRed','DarkCyan','blue','k'], loc='upper left')
    namefig = "comparison_aless_band7.pdf"
    common.savefig(outdir, fig, namefig)

def plot_props_z_spt(plt, outdir, obsdir, zmed_spt, area):

    #plot evolution of SMGs in the UVJ plane
    xtit="$\\rm redshift$"
    ytit="$\\rm N/dz$"

    xmin, xmax, ymin, ymax = 0, 6, 0, max(zmed_spt[0,:])/dz + 1.0
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 5, 5))
    fig.subplots_adjust(left=0.15, bottom=0.15)
    ind = np.where(zmed_spt[0,:] != 0)
    x = xz[ind]
    y = zmed_spt[0,ind] / dz
    yerr_dn = zmed_spt[1,ind]
    yerr_up = zmed_spt[2,ind]

    ax.fill_between(x,y[0]-yerr_dn[0], y[0]+yerr_up[0],facecolor='red', alpha=0.5,interpolate=True)
    ax.plot(x,y[0], linestyle='solid', color='red')

    namefig = "spt_redshift_distribution.pdf"
    common.savefig(outdir, fig, namefig)

def prepare_data(phot_data, phot_data_nodust, phot_data_ab, phot_data_ab_nodust, ids_sed, hdf5_data, hdf5_co_data, hdf5_attenuation, cont_bc, subvols, lightcone_dir, nbands, bands, zdist_flux_cuts, 
                 zdist_flux_cuts_scatter, ncounts_optical_ir_smgs, bands_of_interest_for_smgs, selec_alma,
                 ncounts_all, mags_vs_z_flux_cuts, props_vs_z_flux_cuts, ms_z, n_highz_500microns, zdist_cosmicvar, most_massive_z, area):

    (dec, ra, zobs, idgal, msb, msd, mhalo, sfrb, sfrd, typeg, mgd, mgb, mmold, mmolb, zd, zb, dc, rgd, rgb) = hdf5_data
    (SCO, SCO_peak) = hdf5_co_data
    (pow_disk, pow_bulge) = hdf5_attenuation

    pow_screen = (sfrd * pow_disk + sfrb * pow_bulge) / (sfrd + sfrb)
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
    pow_total = (-0.7) * Contribution_bc + (1.0 - Contribution_bc) * pow_screen 
    indices = range(len(bands))
    #calculate appropriate number of resamplings 
    bins_var = max(30, np.floor(area/0.5))

    #analysis spt 
    (zmed_spt) = sample_spt_2020(SEDs_dust, zobs)

    #analyse sample of as2uds-like galaxies
    sample_as2uds(SEDs_dust,zobs,msb+msd,Av,sfrb+sfrd,mdust,temp_total)
    (seds_aless, props_aless) = sample_aless(SEDs_dust, SEDs_dust_bulge_m, SEDs_dust_bulge_d, SEDs_dust_disk, np.log10((msb+msd)/h0), np.log10((sfrb+sfrd)/h0/1e9), np.log10(mhalo/h0), np.clip(np.log10(mdust), 5.0, 12.0), temp_total, Av, zobs, (rgd * sfrd + rgb * sfrb) / (sfrb + sfrd) * 1e3/h0)

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
            mdust_smgs =  np.clip(np.log10(mdust[ind]), 5.0, 12.0)
            ms_smgs = np.log10((msb[ind] + msd[ind])/h0) 
            ssfr_smgs = np.log10((sfrb[ind] + sfrd[ind])/(msb[ind] + msd[ind]))
            sfr_smgs = np.log10((sfrb[ind] + sfrd[ind])/h0/1e9)
            mhalo_smgs = np.log10(mhalo[ind]/h0)
            Av_smgs = Av[ind]
            td_smgs = temp_total[ind]
            sfr_ratio_smgs = sfrb[ind] / (sfrb[ind] + sfrd[ind])
            rg_smgs = (rgd[ind] * sfrd[ind] + rgb[ind] * sfrb[ind]) / (sfrb[ind] + sfrd[ind]) * 1e3/h0
            LCO10_smgs = np.log10(LCO10[ind])
            pow_smgs = pow_total[ind]
            props_vs_z_flux_cuts[j,0,tn,:] = bin_it_z(x=z_smgs, y=ms_smgs)
            props_vs_z_flux_cuts[j,1,tn,:] = bin_it_z(x=z_smgs, y=ssfr_smgs)
            props_vs_z_flux_cuts[j,2,tn,:] = bin_it_z(x=z_smgs, y=mhalo_smgs)
            props_vs_z_flux_cuts[j,3,tn,:] = bin_it_z(x=z_smgs, y=mdust_smgs)
            props_vs_z_flux_cuts[j,4,tn,:] = bin_it_z(x=z_smgs, y=Av_smgs)
            props_vs_z_flux_cuts[j,5,tn,:] = bin_it_z(x=z_smgs, y=td_smgs)
            props_vs_z_flux_cuts[j,6,tn,:] = bin_it_z(x=z_smgs, y=sfr_ratio_smgs)
            props_vs_z_flux_cuts[j,7,tn,:] = bin_it_z(x=z_smgs, y=rg_smgs)
            props_vs_z_flux_cuts[j,8,tn,:] = bin_it_z(x=z_smgs, y=LCO10_smgs)
            props_vs_z_flux_cuts[j,9,tn,:] = bin_it_z(x=z_smgs, y=sfr_smgs)
            props_vs_z_flux_cuts[j,10,tn,:] = bin_it_z(x=z_smgs, y=pow_smgs)
        
            for bn, b in enumerate(bands_of_interest_for_smgs):
                 ind = np.where((SEDs_dust_smgs[b,:] > 0) & (SEDs_dust_smgs[b,:] < 40))
                 H, bins_edges = np.histogram(SEDs_dust_smgs[b,ind],bins=np.append(mbinsab,muppab))
                 ncounts_optical_ir_smgs[j,bn,tn,:] = ncounts_optical_ir_smgs[j,bn,tn,:] + H
                 mags_selec = SEDs_dust_smgs[b,ind]
                 mags_vs_z_flux_cuts[j,bn,tn,:] = bin_it_z(x=z_smgs[ind], y=mags_selec[0])

    print(props_vs_z_flux_cuts[0,10,2,:])

    for bn, b in enumerate(bands_of_interest_for_smgs):
        ind = np.where((SEDs_dust[b,:] > 0) & (SEDs_dust[b,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust[b,ind],bins=np.append(mbinsab,muppab))
        ncounts_all[bn,:] = ncounts_all[bn,:] + H

    #selection of centrals in the mass range below to compute main sequence
    def compute_most_massive_gals(zobs, ms, mh, typeg, md):

        nselec = 100
        mass_average = np.zeros(shape = (3, len(xz)))
        for iz, z in enumerate(xz):
            ind = np.where((zobs > z - dz*0.5) & (zobs < z + dz*0.5))
            msin = ms[ind]
            mdin = md[ind]
            typesin = typeg[ind]
            mhin = mh[ind]
            ids_mass_sorted =  np.argsort(1.0 / msin)
            sorted_m = msin[ids_mass_sorted]
            mass_average[0, iz] = np.log10(np.median(sorted_m[0:nselec]))
            ids_mass_sorted =  np.argsort(1.0 / mdin)
            sorted_m = mdin[ids_mass_sorted]
            mass_average[2, iz] = np.log10(np.median(sorted_m[0:nselec]))
            cens = np.where(typesin == 0)
            mhin = mhin[cens]
            ids_mass_sorted =  np.argsort(1.0 / mhin)
            sorted_m = mhin[ids_mass_sorted]
            mass_average[1, iz] = np.log10(np.median(sorted_m[0:nselec]))
            
        return mass_average
            
    most_massive_z[:] = compute_most_massive_gals(zobs, (msb+msd)/h0, mhalo/h0, typeg, mdust) 
    ind = np.where(((msb+msd) > 3e9) & ((msb+msd) < 1e10) & (typeg <= 0) & (sfrb + sfrd > 0))
    rg = (rgd[ind] * sfrd[ind] + rgb[ind] * sfrb[ind]) / (sfrd[ind] + sfrb[ind]) * 1e3/h0
    ms_z[0,:] = bin_it_z(x=zobs[ind], y=np.log10((sfrb[ind] + sfrd[ind])/(msb[ind] + msd[ind])))
    ms_z[3,:] = bin_it_z(x=zobs[ind], y=Av[ind])
    ms_z[4,:] = bin_it_z(x=zobs[ind], y=np.log10(mdust[ind]))
    ms_z[5,:] = bin_it_z(x=zobs[ind], y=sfrb[ind]/(sfrb[ind] + sfrd[ind]))
    ms_z[6,:] = bin_it_z(x=zobs[ind], y=rg)
    ms_z[7,:] = bin_it_z(x=zobs[ind], y=np.log10(LCO10[ind]))

    #fit to main sequence
    ms_fit = np.polyfit(np.log10(1.0 + zobs[ind]), np.log10((sfrb[ind] + sfrd[ind])/(msb[ind] + msd[ind])), 2)
    #use fit to main sequence to compute temperautre evolution of galaxies in the main sequence and starbursts
    ssfr_gal = (sfrb + sfrd) / (msb + msd)
    main_seq_position = np.log10(ssfr_gal) - (ms_fit[0] * np.log10(1.0 + zobs)**2.0 + ms_fit[1]* np.log10(1.0 + zobs) +  ms_fit[2])

    # starbursts
    ind = np.where((main_seq_position > 1) & (main_seq_position < 10.0) & ((msb+msd) > 1e9))
    ms_z[2,:] = bin_it_z(x=zobs[ind], y=temp_total[ind])
    #np.savetxt('ids_sbs.txt', id_order[ind]) 

    ind = np.where(((msb+msd) > 5e10) & ((msb+msd) < 1e12) &  (typeg <= 0) & (sfrb + sfrd > 0))
    ms_z[1,:] = bin_it_z(x=zobs[ind], y=temp_total[ind])

    # temperature distribution in the mstellar-SFR plane
    ind = np.where(( (msb + msd) > 1e7) & ( (sfrb + sfrd) > 0))
    rg = (rgd[ind] * mdd[ind] + rgb[ind] * mdb[ind]) / (mdd[ind] + mdb[ind]) * 1e3/h0
    disk_sfr = np.nan_to_num(sfrd[ind] / (3.1416 * (rgd[ind] * 1e3/h0)**2.0))
    bulge_sfr = np.nan_to_num(sfrb[ind] / (3.1416 * (rgb[ind] * 1e3/h0)**2.0))
    disk_dust = np.nan_to_num(mdd[ind] / (3.1416 * (rgd[ind] * 1e3/h0)**2.0))
    bulge_dust = np.nan_to_num(mdb[ind] / (3.1416 * (rgb[ind] * 1e3/h0)**2.0))
    temp_ms_sfr = np.zeros(shape = (13, len(sfrd[ind])))
    temp_ms_sfr[0,:] = np.log10(msb[ind] + msd[ind])
    temp_ms_sfr[1,:] = np.log10((sfrb[ind] + sfrd[ind]) / h0) - 9.0
    temp_ms_sfr[2,:] = temp_total[ind]
    temp_ms_sfr[3,:] = zobs[ind]
    temp_ms_sfr[4,:] = alphaco10[ind]
    temp_ms_sfr[6,:] = np.log10((disk_sfr + bulge_sfr)/ 1e9/h0) #in Msun/yr/kpc^2
    temp_ms_sfr[7,:] = np.log10(mdd[ind] / (3.1416 * (rgd[ind] * 1e3/h0)**2.0)) #in Msun/kpc^2
    temp_ms_sfr[8,:] = np.log10(mdb[ind] / (3.1416 * (rgb[ind] * 1e3/h0)**2.0)) #in Msun/kpc^2
    temp_ms_sfr[5,:] = np.log10(disk_dust + bulge_dust) #in Msun/kpc^2
    temp_ms_sfr[9,:] = np.log10((zd[ind] * sfrd[ind] + sfrb[ind] * zb[ind])/(sfrd[ind] + sfrb[ind])/zsun) #metallicity in Zsun
    temp_ms_sfr[10,:] = np.log10(mdust[ind])
    temp_ms_sfr[11,:] = sfrb[ind] / ((sfrb[ind] + sfrd[ind]))
    temp_ms_sfr[12,:] = np.log10(LCO10[ind])
    #aavoid nan and inf
    temp_ms_sfr = np.nan_to_num(temp_ms_sfr)

    #band 7 sources with S>0.1mJy
    ind = np.where((SEDs_dust[9,:] > 27) & (SEDs_dust[29,:] < 18.900065622282231))
    cols_smgs = np.zeros(shape = (4, len(sfrd[ind])))
    cols_smgs[0,:] = UVcol[ind]
    cols_smgs[1,:] = VJcol[ind]
    cols_smgs[2,:] = zobs[ind]
    cols_smgs[3,:] = np.log10(10.0**(SEDs_dust[29,ind] / (-2.5)) * 3631.0 * 1e3) #in mJy
    color_color_analysis_smgs(UVcol, VJcol, 10.0**(SEDs_dust[29,:] / (-2.5)) * 3631.0 * 1e3, zobs)

    #SEDs band-7 sources with S>1mJy
    ind = np.where((SEDs_dust[29,:] > 0) & (SEDs_dust[29,:] < 16.400065622282231))
    SEDs_dust_smgs = SEDs_dust[:,ind]
    zsmgs = zobs[ind]

    #select extreme H-dropouts to test Wang+19 number density
    ind = np.where((SEDs_dust[9,:] > 27) & (SEDs_dust[13,:] < 24))
    print("Extreme H-dropouts have median zobs", np.median(zobs[ind]), " and are #", len(zobs[ind]))

    return (temp_ms_sfr, cols_smgs, SEDs_dust_smgs, zsmgs, seds_aless, zmed_spt)

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'


    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14"

    subvols = [0,1,2,3,4,5] #,6,7,8,9,10] #range(64) #[0,1,2,3,4,5] #,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
#[0] #range(64) #[0, 1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 59, 60, 61, 62]
    #[0,1]#,3,4,5,6,7,8,9,10] #range(64) #[0,1,2,3,4,5,6,7,8,9,10] #,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 60, 61, 62, 63]#,11,12,13,14,15,16,17,18,19,20,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #(0,1) #range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
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

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mocksky")
  
    fields = {'galaxies': ('SCO','SCO_peak')}
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)


    fields = {'galaxies': {'pow_screen_disk', 'pow_screen_bulge'}}
    hdf5_attenuation = common.read_attenuation(lightcone_dir, fields, subvols, 'eagle-rr14')
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
    mags_vs_z_flux_cuts = np.zeros(shape =  (len(selec_alma), len(bands_of_interest_for_smgs),  len(flux_threshs_compact), 4, len(zbins)))
    props_vs_z_flux_cuts = np.zeros(shape =  (len(selec_alma), 11,  len(flux_threshs_compact), 4, len(zbins)))

    #main sequence evolution
    ms_z = np.zeros(shape =  (8, 4, len(zbins)))
    #most massive systems lightcone
    most_massive_z = np.zeros(shape =  (3, len(zbins)))

    #number counts of 500microns sources at z>4
    n_highz_500microns = np.zeros(shape =  (len(fbins)))

    #process data
    (temp_ms_sfr, cols_smgs, SEDs_dust_smgs, zsmgs, seds_aless, zmed_spt) = prepare_data(seds, seds_nodust, seds_ab, seds_ab_nodust, ids_sed, hdf5_data, hdf5_co_data, hdf5_attenuation, cont_bc, subvols, 
                                                                       lightcone_dir, nbands, bands, zdist_flux_cuts, zdist_flux_cuts_scatter,
                                                                       ncounts_optical_ir_smgs, bands_of_interest_for_smgs, selec_alma, ncounts_all, mags_vs_z_flux_cuts, 
                                                                       props_vs_z_flux_cuts, ms_z, n_highz_500microns, zdist_cosmicvar, most_massive_z, areasub)

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

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_temp_mainseq(plt, outdir, temp_ms_sfr, zinterst, obsdir)
    plot_colors_smgs(plt, outdir, cols_smgs, zinterst)
    plot_seds_smgs(plt, outdir, obsdir, SEDs_dust_smgs, zsmgs, seds_aless)

    plot_redshift(plt, outdir, obsdir, zdist_flux_cuts, zdist_flux_cuts_scatter, zdist_cosmicvar)
    plot_number_counts_smgs(plt, outdir, obsdir,  ncounts_optical_ir_smgs, 
                            bands_of_interest_for_smgs, ncounts_all)
    plot_magnitudes_z_smgs(plt, outdir, obsdir, mags_vs_z_flux_cuts)
    plot_props_z_smgs(plt, outdir, obsdir, props_vs_z_flux_cuts, ms_z, most_massive_z)
    plot_props_z_spt(plt, outdir, obsdir, zmed_spt, areasub)

if __name__ == '__main__':
    main()
