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

import numpy as np
import scipy.optimize as so

def wmedians_2sigma(x=None, y=None, xbins=None):

    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    dx = xbins[1] - xbins[0]
    result = np.zeros(shape = (3, nbins))

    for i in range (0,nbins):
        xlow = xbins[i]-dx/2.0
        xup  = xbins[i]+dx/2.0
        ind  = np.where((x > xlow) & (x< xup))
        if(len(x[ind]) > 9):

            obj_bin = len(x[ind])
            ybin    = y[ind]
            result[0, i] = np.median(ybin)
            #sort array on 1/y because we want it to sort from the smallest to the largest item, and the default of argsort is to order from the largest to the smallest.
            IDs = np.argsort(ybin,kind='quicksort')
            ID16th = int(np.floor(obj_bin*0.025))+1   #take the lower edge.
            ID84th = int(np.floor(obj_bin*0.975))-1   #take the upper edge.
            result[1, i] = np.abs(result[0, i] - ybin[IDs[ID16th]])
            result[2, i] = np.abs(ybin[IDs[ID84th]] - result[0, i])

    return result


def medians(x=None, y=None, xbins=None):

    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    dx = xbins[1] - xbins[0]
    result = np.zeros(shape = (2, nbins))

    for i in range (0,nbins):
        xlow = xbins[i]-dx/2.0
        xup  = xbins[i]+dx/2.0
        ind  = np.where((x > xlow) & (x< xup))
        ybin = y[ind]
        result[1, i] = len(x[ind])
        result[0, i] = np.median(ybin)

    return result



def wmedians(x=None, y=None, xbins=None, low_numbers=False):

    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    dx = xbins[1] - xbins[0]
    result = np.zeros(shape = (4, nbins))

    for i in range (0,nbins):
        xlow = xbins[i]-dx/2.0
        xup  = xbins[i]+dx/2.0
        ind  = np.where((x > xlow) & (x< xup))
        result[3, i] = len(x[ind])
        if(len(x[ind]) > 29):

            obj_bin = len(x[ind])
            ybin    = y[ind]
            result[0:3, i] = gpercentiles(ybin)
        elif(low_numbers and len(x[ind]) > 0):
            ybin    = y[ind]
            result[0, i] = np.median(ybin)
            result[1, i] = np.abs(result[0, i] - np.min(y[ind]))
            result[2, i] = np.abs(np.max(y[ind]) - result[0, i])

    return result

def gpercentiles(y=None):

    obj_bin = len(y)
    result = np.zeros(shape = 3)
    result[0] = np.median(y)
    #sort array on 1/y because we want it to sort from the smallest to the largest item, and the default of argsort is to order from the largest to the smallest.
    IDs = np.argsort(y,kind='quicksort')
    ID16th = int(np.floor(obj_bin*0.16))+1   #take the lower edge.
    ID84th = int(np.floor(obj_bin*0.84))-1   #take the upper edge.
    result[1] = np.abs(result[0] - y[IDs[ID16th]])
    result[2] = np.abs(y[IDs[ID84th]] - result[0])

    return result

def wmedians_cum(x=None, y=None, xbins=None, low_numbers=False):

    #return the median and error on the median
    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    result = np.zeros(shape = (3, nbins))

    for i in range (0,nbins):
        ind  = np.where(x >= xbins[i])
        if(len(x[ind]) > 9):
            obj_bin = len(x[ind])
            ybin    = y[ind]
            result[0, i] = np.median(ybin)
            #sort array on 1/y because we want it to sort from the smallest to the largest item, and the default of argsort is to order from the largest to the smallest.
            IDs = np.argsort(ybin,kind='quicksort')
            ID16th = int(np.floor(obj_bin*0.16))+1   #take the lower edge.
            ID84th = int(np.floor(obj_bin*0.84))-1   #take the upper edge.
            result[1, i] = np.abs(result[0, i] - ybin[IDs[ID16th]])
            result[2, i] = np.abs(ybin[IDs[ID84th]] - result[0, i])
        elif(low_numbers and len(x[ind]) > 0):
            ybin    = y[ind]
            result[0, i] = np.median(ybin)
            result[1, i] = np.abs(result[0, i] - np.min(y[ind]))

    return result

def medians_cum_err(x=None, y=None, xbins=None, low_numbers=False):

    #return the median and error on the median
    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    result = np.zeros(shape = (2, nbins))

    for i in range (0,nbins):
        ind  = np.where(x >= xbins[i])
        if(len(x[ind]) > 9):
            obj_bin = len(x[ind])
            ybin    = y[ind]
            result[0, i] = np.median(ybin)
            #sort array on 1/y because we want it to sort from the smallest to the largest item, and the default of argsort is to order from the largest to the smallest.
            result[1, i] = jackknife(ybin)
        elif(low_numbers and len(x[ind]) > 0):
            ybin    = y[ind]
            result[0, i] = np.median(ybin)
            result[1, i] = np.abs(result[0, i] - np.min(y[ind]))

    return result


def jackknife(x, iterations=10):

    medians = np.zeros(shape = iterations)
    len_subsample = len(x) / iterations
    for i in range(iterations):
         subsample = np.random.choice(x, size=int(np.floor(len_subsample)))
         medians[i] = np.median(subsample)

    return np.std(medians)

def stacking(x=None, y=None, xbins=None, low_numbers=False):

    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    dx = xbins[1] - xbins[0]
    result = np.zeros(shape = (nbins))

    for i in range (0,nbins):
        xlow = xbins[i]-dx/2.0
        xup  = xbins[i]+dx/2.0
        ind  = np.where((x > xlow) & (x< xup))
        if(len(x[ind]) > 0):
            ybin    = y[ind]
            result[i] = np.log10(np.mean(ybin))

    return result


def fractions(x=None, y=None, xbins=None, ythresh=None):

    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    dx = xbins[1]-xbins[0]
    result = np.zeros(shape = (nbins))

    for i in range (0,nbins):
        xlow = xbins[i]-dx/2.0
        xup  = xbins[i]+dx/2.0
        ind  = np.where((x > xlow) & (x< xup))
        if(len(x[ind]) > 9):
            ngalaxies = len(x[ind])
            ybin      = y[ind]
            above     = np.where(ybin > ythresh)
            nabove    = len(ybin[above])
            result[i] = (nabove+ 0.0)/(ngalaxies+0.0)
        else:
            result[i] = -1

    return result

def fractional_contribution(x=None, y=None, xbins=None):

    nbins = len(xbins)
    #define size of bins, assuming bins are all equally spaced.
    dx = xbins[1]-xbins[0]
    result = np.zeros(shape = (nbins))

    for i in range (0,nbins):
        xlow = xbins[i]-dx/2.0
        xup  = xbins[i]+dx/2.0
        ind  = np.where((x > xlow) & (x< xup))
        if(len(x[ind]) > 4):
            mtotal    = sum(x[ind])
            mbulge_tot= sum(x[ind]*y[ind])
            result[i] = mbulge_tot/mtotal
        else:
            result[i] = -1

    return result



def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level



def density_contour(ax, xdata, ydata, nbins_x, nbins_y):
    """ Create a density contour plot.
    Parameters
    ----------
    ax : matplotlib.Axes
        Plot the contour to this axis
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    low_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.01))
    twenty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.2))
    thirty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.3))
    forty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.4))
    fifty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.5))
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    eighty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.8))
    ninty_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.9))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    levels = [three_sigma, two_sigma, ninty_sigma, eighty_sigma, one_sigma, fifty_sigma, forty_sigma, thirty_sigma, twenty_sigma, low_sigma]

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as col

    # The viridis colormap is only available since mpl 1.5
    extra_args = {}
    if tuple(mpl.__version__.split('.')) >= ('1', '5'):
        extra_args['cmap'] = plt.get_cmap('viridis')

    return ax.contourf(X, Y, Z, levels=levels, origin="lower", alpha=0.75,
                      norm=col.Normalize(vmin=0, vmax=0.01), **extra_args)

def twodim_plot(xdata, ydata, prop, xbins, ybins):

    dx = abs(xbins[0] - xbins[1]) / 2.0
    dy = abs(ybins[0] - ybins[1]) / 2.0

    cout = np.zeros(shape = (len(xbins), len(ybins)))
    for xi,x in enumerate(xbins):
        for yi,y in enumerate(ybins):
            ind = np.where((xdata > x - dx) & (xdata <= x + dx) & (ydata > y - dy) & (ydata <= y + dy))
            if len(prop[ind]) >= 5:
               cout[xi,yi] = np.median(prop[ind])


    return cout


def look_back_time(z, h=0.6751, omegam=0.3121, omegal=0.6879):

	"""Calculates the look back time of an array of redshifts
	Parameters
	---------
	z: array of redshifts
	h: hubble parameter
	omegam: omega matter
	omegal: omega lambda
	"""

	#define some constants:
	H0100=100.0
	KM2M=1.0e3
	GYR2S=3.15576e16
	MPC2M=3.0856775807e22
	H0100PGYR=H0100*KM2M*GYR2S/MPC2M

	#calculate the expansion parameters
	a = 1.0 / (1.0 + z)

	#The Hubble time for H_0=100km/s/Mpc
	Hubble_Time=1.0/H0100PGYR
	t0= Hubble_Time*(2/(3*h*np.sqrt(1-omegam)))*np.arcsinh(np.sqrt((1.0/omegam-1.0)*1.0)*1.0)
	t = Hubble_Time*(2/(3*h*np.sqrt(1-omegam)))*np.arcsinh(np.sqrt((1.0/omegam-1.0)*a)*a)

	return t0-t


def redshift(lbt, h=0.6751, omegam=0.3121, omegal=0.6879):

	"""Calculates the look back time of an array of redshifts
	Parameters
	---------
	z: array of redshifts
	h: hubble parameter
	omegam: omega matter
	omegal: omega lambda
	"""

	#define some constants:
	H0100=100.0
	KM2M=1.0e3
	GYR2S=3.15576e16
	MPC2M=3.0856775807e22
	H0100PGYR=H0100*KM2M*GYR2S/MPC2M

	#calculate the expansion parameters
	#a = 1.0 / (1.0 + z)

	#The Hubble time for H_0=100km/s/Mpc
	Hubble_Time=1.0/H0100PGYR
	t0= Hubble_Time*(2/(3*h*np.sqrt(1-omegam)))*np.arcsinh(np.sqrt((1.0/omegam-1.0)*1.0)*1.0)
	age = t0 - lbt

	a = pow(np.sinh(age/(Hubble_Time*(2/(3*h*np.sqrt(1-omegam))))) / np.sqrt(1.0/omegam-1.0), 2.0/3.0)
	
	z = 1.0 / a - 1.0
	for i in range (0,len(z)):
		z[i] = round(z[i], 2)

	return z

def compute_cosmic_variance_redshifts(flux, z, flux_threshs_log, n_rand):

    len_subsample = len(flux) / n_rand
    std_redshifts= np.zeros(shape = (len(flux_threshs_log)))
    for ft in range(0,len(flux_threshs_log)):
        bright = np.where(flux > flux_threshs_log[ft])
        if(len(flux[bright]) > 0):
           z_selec = z[bright]
           zsamples = np.zeros(shape = (n_rand))
           for i in range(0,n_rand):
               subsample = np.random.choice(z_selec, size=int(np.floor(len_subsample)))
               zsamples[i] = np.median(subsample)

           pos = np.where(zsamples > 0) 
           std_redshifts[ft] = np.std(zsamples[pos])
  
    #catch nans
    return np.nan_to_num(std_redshifts)

def compute_cosmic_variance_number_counts(area_total, flux, xbins, n_rand):

    std_counts= np.zeros(shape = (len(xbins)))
    len_subsample = len(flux) / n_rand
    counts_cum = np.zeros(shape = (n_rand,len(xbins)))
    area_sub = area_total/n_rand

    for i in range(n_rand):
        subsample = np.random.choice(flux, size=int(np.floor(len_subsample)))
        for ft in range(0,len(xbins)):
            bright = np.where(subsample > xbins[ft])
            if(len(subsample[bright]) > 0):
               counts_cum[i,ft] = (len(subsample[bright]) + 0.0) / area_sub

    #now compute standard deviation
    for ft in range(0,len(xbins)):              
        pos = np.where(counts_cum[:,ft] > 0)  
        std_counts[ft] = np.std(np.log10(counts_cum[pos,ft]))
    #catch nans
    return np.nan_to_num(std_counts)
