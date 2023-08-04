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

import functools

import numpy as np
import os
import h5py
import math

import common
import utilities_statistics as us

import mbb_emcee
##################### constants #####################

c_light = 299792458.0 #m/s
beta = 2.0
PI = 3.1416

##################################
file = 'Shark_SED_bands.dat'
lambda_bands = np.loadtxt(file,usecols=[0],unpack=True)
freq_bands   = c_light / (lambda_bands * 1e-10) #in Hz
lambda_bands = lambda_bands/1e4 #in microns


llow = 1.3 
lupp = 3.0
dl = 0.025
lbins = np.arange(llow,lupp,dl)

mlowab = -30
muppab = -16
dmab = 0.5
mbinsab = np.arange(mlowab,muppab,dmab)
xlfab   = mbinsab + dmab/2.0

zlow = 0
zupp = 7.0
dz = 0.5
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0


def plot_temperature(plt, outdir, temp_to_lum):

    xtit="$\\rm M_{\\rm 850\\mu m}$"
    ytit="$\\rm T/K$"

    xmin, xmax, ymin, ymax = -30, -16, 20, 150
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 20, 20))

    ind = np.where(temp_to_lum[0,:] != 0)
    y = temp_to_lum[0,ind]
    yerrdn = temp_to_lum[1,ind]
    yerrup = temp_to_lum[2,ind]

    ax.errorbar(xlfab[ind],y[0],yerr=[yerrdn[0],yerrup[0]], ls='None', mfc='None', ecolor = 'MediumBlue', mec='MediumBlue')

    common.savefig(outdir, fig, "TdustToLum850.pdf")
  
def analyse_data(subv, outdir): 
 
    bin_it_z = functools.partial(us.wmedians, xbins=xz)

    files_to_process = ['temp_ids_selec_29_16.4000656223', 'temp_ids_selec_30_16.4000656223', 'temp_ids_selec_32_16.4000656223', 'temp_ids_sbs', 'temp_ids_mainseq']

    for i,s in enumerate(subv):

        fileread = np.loadtxt(outdir + files_to_process[0] + '_' + str(s) + '.txt')
        if i == 0:
           temp_band_29 = fileread
        else:
           temp_band_29 = np.concatenate([temp_band_29, fileread])

        fileread = np.loadtxt(outdir + files_to_process[1] + '_' + str(s) + '.txt')
        if i == 0:
           temp_band_30 = fileread
        else:
           temp_band_30 = np.concatenate([temp_band_30, fileread])

        fileread = np.loadtxt(outdir + files_to_process[2] + '_' + str(s) + '.txt')
        if i == 0:
           temp_band_32 = fileread
        else:
           temp_band_32 = np.concatenate([temp_band_32, fileread])
        
        fileread = np.loadtxt(outdir + files_to_process[3] + '_' + str(s) + '.txt')
        if i == 0:
           temp_band_sb = fileread
        else:
           temp_band_sb = np.concatenate([temp_band_sb, fileread])
        fileread = np.loadtxt(outdir + files_to_process[4] + '_' + str(s) + '.txt')
        if i == 0:
           temp_band_ms = fileread
        else:
           temp_band_ms = np.concatenate([temp_band_ms, fileread])

        #fileread = np.loadtxt(outdir + files_to_process[4] + '_' + str(s) + '.txt')
        #temp_band_ms = temp_band_ms.concatenate(fileread)

    t_vs_z_29 = bin_it_z(x=temp_band_29[:,2], y=temp_band_29[:,1])
    t_vs_z_30 = bin_it_z(x=temp_band_30[:,2], y=temp_band_30[:,1])
    t_vs_z_32 = bin_it_z(x=temp_band_32[:,2], y=temp_band_32[:,1])
    t_vs_z_sb = bin_it_z(x=temp_band_sb[:,2], y=temp_band_sb[:,1])
    t_vs_z_ms = bin_it_z(x=temp_band_ms[:,2], y=temp_band_ms[:,1])
    print t_vs_z_29,t_vs_z_30,t_vs_z_32,t_vs_z_sb,t_vs_z_ms

def prepare_data(ids_sed, seds, hdf5_data, model_dir, subvol, lambda_to_fit, plt, file_to_process, outdir):


    (zobs, idgal) = hdf5_data

    bin_it = functools.partial(us.wmedians, xbins=xlfab)

    bin_it_z = functools.partial(us.wmedians, xbins=xz)

    SEDs_ab = seds[0]

    numgals = len(SEDs_ab[0,:])
    minl = min(lambda_to_fit)
    maxl = max(lambda_to_fit)


    g = 0
    lambdain = lambda_bands[lambda_to_fit]

    ids_band7 = np.loadtxt(outdir + file_to_process)
    ids_band7 = ids_band7.astype(int)
    temperature_select = np.zeros( shape = (len(ids_band7)))
    z_select = np.zeros( shape = (len(ids_band7)))

    for n,g in enumerate(ids_band7):
        SEDsfit = SEDs_ab[lambda_to_fit,g]
        fir_sed = 10.0**(SEDsfit / (-2.5)) * 3631.0 * 1e3 / (4.0 * PI * ( 229.433*1e5)**2.0) # in mJy - move the galaxy to z=0.05
        err = np.zeros(shape = len(fir_sed))
        err = 0.02 * fir_sed #in mJy - 5% of flux assumed
        
        ind = np.where(fir_sed != np.inf)
        lambdain = lambdain[ind]
        table_input = np.zeros(shape = (len(lambdain), 3))
        table_input[:,0] = lambdain
        table_input[:,1] = fir_sed[ind]
        table_input[:,2] = err[ind]
        np.savetxt(file_to_process + 'test.txt', table_input) 
 
        table_input = open(file_to_process + 'test.txt', 'r')
 
        results = mbb_emcee.mbb_fitter(nwalkers=100, photfile='test.txt')
        results.set_uplim('T', 100)
        results.set_lowlim('T', 0.1)
        results.set_uplim('beta', 2.5)
        results.set_lowlim('beta', 1.5)
 
        results.set_gaussian_prior('Beta', 2, 0.3)
        p0init = np.array([20, 2, 200, 0.3, 5.0])
        p0sig = np.array([20, 2, 200, 0.3, 5.0])
        p0 = results.generate_initial_values(p0init, p0sig)
        results.run(10, 90, p0)
 
        res = mbb_emcee.mbb_results(fit=results, redshift=0.05, cosmo_type='Planck13')
 
        if(g == 0 or g == 1 or g == 100000 or g == 500000 or g == 1000000 or g == 1200000):
           p_wave = np.linspace(lambda_bands.min() * 0.5, lambda_bands.max() * 1.5, 200)
           fig = plt.figure(figsize=(5,4))
           ax = fig.add_subplot(111)
           x = lambda_bands[lambda_to_fit]
           y = fir_sed
           yerr = err
           ax.errorbar(x, y, yerr=yerr, fmt='ro')
           ax.plot(p_wave, res.best_fit_sed(p_wave), color='blue')
           ax.set_xlabel('Wavelength [um]')
           ax.set_ylabel('Flux Density [mJy]')
           common.savefig(outdir, fig, "SEDFittingExample_%s.pdf" % str(g))
 
        temperature_select[n] = res.par_central_values[0][0]
        z_select[n] = zobs[g]
   
    with open(outdir + 'temp_' + file_to_process, 'wb') as fil:
         fil.write("#id_position_file temp_mbb zobs\n")
         for a,b,c in zip(ids_band7, temperature_select, z_select):
             fil.write("%10.0f %5.10f %5.10f\n" % (a,b,c))
        

    ind = np.where((temperature_select > 0) & (temperature_select < 100))
    temp_bin = bin_it_z(x=temperature_select[ind], y=temperature_select[ind])
    print temp_bin
    
    write = False

    if write: 
       # will write the hdf5 files with temperatures
       file_to_write = os.path.join(model_dir, 'split', 'temperature-mbb-eagle-rr14_%02d.hdf5' % subvol)
       print ('Will write dust temperatures to %s' % file_to_write)
       hf = h5py.File(file_to_write, 'w')
    
       hf.create_dataset('galaxies/id_galaxy_sky', data=ids_sed)
       hf.create_dataset('galaxies/temperature_mbb', data=temperature)
       hf.close()

    temp_to_lum = 0
    return temp_to_lum

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/split/'
    obsdir= '/home/clagos/shark/data/'

    subvols = [0,1,60] #0,1,60] #,1) #range(64)
    sed_file = "Sting-SED-eagle-rr14"
    file_to_process = 'ids_selec_32_16.4000656223_' + str(subvols[0]) + '.txt'

   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    lambda_to_fit = [19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32]

    plt = common.load_matplotlib()
    fields_sed = {'SED/ab_dust': ('total', 'disk')}
    fields = {'galaxies': ('zobs','id_galaxy_sky')}

    ComputeT = False
    if ComputeT :
       for subv in subvols:
           ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, [subv], sed_file)
           hdf5_data = common.read_lightcone(lightcone_dir, fields, [subv])
           temp_to_lum = prepare_data(ids_sed, seds, hdf5_data, lightcone_dir, subv, lambda_to_fit, plt, file_to_process, outdir)
    else :
       analyse_data(subvols, outdir) 
    
    #plot_temperature(plt, outdir, temp_to_lum)
 
if __name__ == '__main__':
    main()
