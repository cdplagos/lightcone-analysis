import numpy as np
import common

s24, s250 = np.loadtxt("/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-AtLAST-FIR_p3.txt", unpack = True, usecols = [0,4])

s24 = np.log10(s24)
s250 = np.log10(s250)

plt = common.load_matplotlib()

xtit="$\\rm log_{10}(S_{24}/mJy)$"
ytit="$\\rm log_{10}(S_{250}/mJy)$"

xmin, xmax, ymin, ymax = -4, 2, -4, 2

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 1, 1))

im = ax.hexbin(s24, s250, xscale='linear', yscale='linear', gridsize=(25,25), cmap='magma', mincnt=10)

namefig = "s24_s250_correlation_table.pdf"

common.savefig('/scratch/pawsey0119/clagos/Stingray/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/Plots/', fig, namefig)

