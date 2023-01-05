#!/usr/bin/env python

###########################
###### IMPORT MODULES
###########################

import LCSbase as lb
import LCScommon as lcscommon

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import numpy as np
import os
import scipy.stats as st
from scipy.stats import ks_2samp, anderson_ksamp, binned_statistic,binned_statistic_2d

import argparse# here is min mass = 9.75

from urllib.parse import urlencode
from urllib.request import urlretrieve

from astropy.io import fits, ascii
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit
from astropy.stats import bootstrap,binom_conf_interval
from astropy.utils import NumpyRNGContext
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import median_absolute_deviation as MAD

from PIL import Image

import multiprocessing as mp

import sys
LCSpath = os.path.join(os.getenv("HOME"),'github','LCS','python','')
sys.path.append(LCSpath)
from fit_line_sigma_clip import fit_line_sigma_clip

###########################
##### DEFINITIONS
###########################
homedir = os.getenv("HOME")
plotdir = homedir+'/research/LCS/plots/'

#USE_DISK_ONLY = np.bool(np.float(args.diskonly))#True # set to use disk effective radius to normalize 24um size
USE_DISK_ONLY = True
#if USE_DISK_ONLY:
#    print('normalizing by radius of disk')
minsize_kpc=1.3 # one mips pixel at distance of hercules
#minsize_kpc=2*minsize_kpc

mstarmin=10.#float(args.minmass)
mstarmax=10.8
minmass=mstarmin #log of M*
ssfrmin=-12.
ssfrmax=-9
spiralcut=0.8
truncation_ratio=0.5


zmin = 0.0137
zmax = 0.0433
# this is from fitting a line that is parallel to MS but that intersects
# where gaussians of SF and quiescent cross
MS_OFFSET = 1.5*0.3
NMASSMATCH=30 # number of field to draw for each cluster/infall galaxy
Mpcrad_kpcarcsec = 2. * np.pi/360./3600.*1000.
mipspixelscale=2.45
exterior=.68
colors=['k','b','c','g','m','y','r','sienna','0.5']
shapes=['o','*','p','d','s','^','>','<','v']
#colors=['k','b','c','g','m','y','r','sienna','0.5']

truncated=np.array([113107,140175,79360,79394,79551,79545,82185,166185,166687,162832,146659,99508,170903,18236,43796,43817,43821,70634,104038,104181],'i')

minsize_kpc=1.3 # one mips pixel at distance of hercules
BTkey = '__B_T_r'
#BTkey_lcs = '(B/T)r'
Rdkey = 'Rd_2'


# ellipticity to use in combined tables
# the first one e_1 refers to the ellip of the bulge during B/D decomposition
# this is the same in the LCS and GSWLC catalogs - phew!
ellipkey = 'e_2'

###########################
##### Functions
###########################
# using colors from matplotlib default color cycle
mycolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

colorblind1='#F5793A' # orange
colorblind2 = '#85C0F9' # light blue
colorblind3='#0F2080' # dark blue
darkblue = colorblind3
darkblue = mycolors[1]
lightblue = colorblind2
#lightblue = 'b'
#colorblind2 = 'c'
colorblind3 = 'k'


# my version, binned median for GSWLC galaxies with vr < 15,0000 
t = Table.read(homedir+'/research/APPSS/GSWLC2-median-ssfr-mstar-vr15k.dat',format='ipac')
log_mstar2 = t['med_logMstar']
log_ssfr2 = t['med_logsSFR']


def ratio_error(num,denom):
    ''' return fraction and lower, upper error based on binomial statistics '''

    ratio = num/denom
    if denom > num:
        yerrs = binom_conf_interval(num,denom)#lower, upper

        try:
            # works for arrays
            yerr = np.zeros((2,len(num)))
            yerr[0] = ratio-yerrs[0]
            yerr[1] = yerrs[1]-ratio
        except TypeError:
            yerr = np.zeros((2,1))
            yerr[0] = ratio-yerrs[0]
            yerr[1] = yerrs[1]-ratio
    else:
        # putting in place holder for now for when fractions are greater than 1
        yerr = np.sqrt(num)/denom
    return ratio,yerr
    
def running_median(x,y,nbin,ax,color='k',alpha=.4):
    mybins=np.linspace(st.stats.scoreatpercentile(x,2),st.stats.scoreatpercentile(x,98),nbin)
    ybin,xbin,binnumb = binned_statistic(x,y,statistic='median',bins=mybins)
    yerr,xbin,binnumb = binned_statistic(x,y,statistic='std',bins=mybins)
    nyerr,xbin,binnumb = binned_statistic(x,y,statistic='count',bins=mybins)            
    yerr = yerr/np.sqrt(nyerr)
    dbin = xbin[1]-xbin[0]
    ax.plot(xbin[:-1]+0.5*dbin,ybin,color=color,lw=3)
    ax.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=color,alpha=alpha)              
    

def linear_func(x,m,b):
    return m*x+b

def get_bootstrap_confint(d,bootfunc=np.median,nboot=100):
    bootsamp = bootstrap(d,bootfunc=bootfunc,bootnum=nboot)
    bootsamp.sort()
    # get indices
    ilower = int(((nboot - .68*nboot)/2))
    iupper = nboot-ilower
    return bootsamp[ilower],bootsamp[iupper]

def get_BV_MS(logMstar,MSfit=None):
    ''' 
    get MS fit that BV calculated from GSWLC; 
    
    MSfit = can use with alternate fit function if you provide the function 
    '''
    #return 0.53*logMstar-5.5
    # for no BT cut, e < 0.75
    if MSfit is None:
        return 0.3985*logMstar - 4.2078
    else:
        return MSfit(logMstar)
def get_SFR_cut(logMstar,BTcut=False):
    ''' 
    get min allowable SFR as a function of stellar mass
        
    '''
    #return 0.53*logMstar-5.5
    # for no BT cut, e < 0.75
    #return 0.6*logMstar - 6.11 - 0.6

    # using full GSWLC, just cut to LCS redshift range
    if not(args.cutBT) and not(args.cutN):
        return get_MS(logMstar,BTcut=BTcut) - 0.845
    elif args.cutBT or args.cutN:
        #return get_MS(logMstar) - 0.947
        return get_MS(logMstar,BTcut=BTcut) - 0.947
def get_MS(logMstar,BTcut=False):
    # not BT cut
    # updating this after we expanded the mass range used for fitting MS
    if not(args.cutBT) and not(args.cutN):
        #self.MS_std = 0.22
        #return 0.6*x-6.11

        # using full GSWLC, just cut to LCS redshift
        #return 0.592*logMstar - 6.18

        # using 2nd order polynomial fit
        return -0.1969*logMstar**2 + 4.4187*logMstar -24.607
    elif args.cutBT or args.cutN:
        # you get the same thing from fitting the MS or from fitting peaks of gaussian
        #self.MS_std = 0.16
        #return 0.62*logMstar-6.35
        return -0.0935*logMstar**2 + 2.4325*logMstar -15.107
def get_MS_poly(logMstar):
    # not BT cut
    # updating this after we expanded the mass range used for fitting MS
    if not(args.cutBT):
        #self.MS_std = 0.22
        #return 0.6*x-6.11

        # using full GSWLC, just cut to LCS redshift
        #return 0.592*logMstar - 6.18

        # using 2nd order polynomial fit
        return -0.17476*logMstar**2 + 3.84123*logMstar -21.12097
    elif args.cutBT:
        # you get the same thing from fitting the MS or from fitting peaks of gaussian
        #self.MS_std = 0.16
        #return 0.62*logMstar-6.35
        return -0.01587975*logMstar**2 + 0.829243*logMstar -6.85584377

def get_MS_BTcut(logMstar,MSfit=None,BTcut=False):
    ''' 
    get MS fit with B/T < 0.4 cut
    
    MSfit = can use with alternate fit function if you provide the function 
    '''
    #return 0.53*logMstar-5.5
    # for no BT cut, e < 0.75
    #return 0.4731*logMstar-4.8771
    #return 0.754*logMstar-7.56
    # using the full fit for now
    return get_MS(logMstar,BTcut=BTcut)

def plot_Durbala_MS(ax,ls='-'):
    plt.sca(ax)
    #lsfr = log_mstar2+log_ssfr2
    ##plt.plot(log_mstar2, lsfr, 'w-', lw=4)
    #plt.plot(log_mstar2, lsfr, c='m',ls='-', lw=6, label='Durbala+20')
    x1,x2 = 9.7,11.
    xline = np.linspace(x1,x2,100)
    yline = get_MS_BTcut(xline)
    ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
    ax.plot(xline,yline,c='m',ls=ls,lw=3,label='$B/T < {}$'.format(args.BT))

def plot_MS_lowBT(ax,ls='-'):
    plt.sca(ax)
    #lsfr = log_mstar2+log_ssfr2
    ##plt.plot(log_mstar2, lsfr, 'w-', lw=4)
    #plt.plot(log_mstar2, lsfr, c='m',ls='-', lw=6, label='Durbala+20')
    x1,x2 = 9.7,11.
    xline = np.linspace(x1,x2,100)
    yline = 0.62*xline-6.35
    ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
    ax.plot(xline,yline,c='m',ls=ls,lw=3,label='$B/T < 0.3$')

def plot_BV_MS(ax,color='mediumblue',ls='-'):
    plt.sca(ax)
    plot_Durbala_MS(ax)
    
    x1,x2 = 9.7,11.
    xline = np.linspace(x1,x2,100)
    yline = get_BV_MS(xline)
    ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
    ax.plot(xline,yline,c=color,ls=ls,lw=3,label='MS Fit')

    # scatter around MS fit
    sigma=0.3
    ax.plot(xline,yline-1.5*sigma,c='w',ls='--',lw=4)
    ax.plot(xline,yline-1.5*sigma,c=color,ls='--',lw=3,label='fit-1.5$\sigma$')

def plot_GSWLC_sssfr(ax=None,ls='-'):
    if ax is None:
        ax = plt.gca()

    ssfr = -11.5
    x1,x2 = 9.6,11.15
    xline = np.linspace(x1,x2,100)
    yline = ssfr+xline
    ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
    ax.plot(xline,yline,c='0.5',ls=ls,lw=3,label='log(sSFR)=-11.5')
def plot_sfr_mstar_lines(ax=None,apexflag=False,BTcut=False):
    if ax is not None:
        plt.sca(ax)
    # plot MS fit
    
    x1,x2 = 9,11.1
    xline = np.linspace(x1,x2,100)
    yline = get_MS(xline,BTcut=BTcut)
    #plt.fill_between(xline,yline+.45,yline-.45,color='b',alpha=.15,label=r'$\rm MS \pm 1.5 \sigma$')                
    plt.plot(xline,yline,c='w',ls='-',lw=5,label='_nolegend_')
    plt.plot(xline,yline,c='b',ls='-',lw=4,label='MS Fit')

    plt.plot(xline,yline-.45,c='w',ls='-',lw=4)
    plt.plot(xline,yline-.45,c='b',ls='--',lw=3,label=r'$\rm MS - 1.5 \sigma$')                
    # plot passive cut

    yl = get_SFR_cut(xline,BTcut=args.cutBT)
    plt.plot(xline,yl,'r-',lw=2,label='Passive Cut')
    if not apexflag:
        yl =xline + float(args.minssfr)
        plt.plot(xline,yl,'k--',lw=2,label=r'$\rm sSFR=-11.5$')
        
def colormass(x1,y1,x2,y2,name1,name2, figname, hexbinflag=False,contourflag=False, \
             xmin=7.9, xmax=11.6, ymin=-1.2, ymax=1.2, contour_bins = 40, ncontour_levels=5,\
              xlabel=r'$\rm \log_{10}(M_\star/M_\odot) $', ylabel='$(g-i)_{corrected} $', color1=colorblind3,color2=colorblind2,\
              nhistbin=50, alpha1=.1,alphagray=.1,lcsflag=False,ssfrlimit=None,cumulativeFlag=False,marker2='o'):

    '''
    PARAMS:
    -------
    * x1,y1
    * x2,y2
    * name1
    * name2
    * hexbinflag
    * contourflag
    * xmin, xmax
    * ymin, ymax
    * contour_bins, ncontour_levels
    * color1
    * color2
    * nhistbin
    * alpha1
    * alphagray
    

    '''
    fig = plt.figure(figsize=(8,8))
    plt.subplots_adjust(left=.15,bottom=.15)
    nrow = 4
    ncol = 4
    
    # for purposes of this plot, only keep data within the 
    # window specified by [xmin:xmax, ymin:ymax]
    
    keepflag1 = (x1 >= xmin) & (x1 <= xmax) & (y1 >= ymin) & (y1 <= ymax)
    keepflag2 = (x2 >= xmin) & (x2 <= xmax) & (y2 >= ymin) & (y2 <= ymax)
    
    x1 = x1[keepflag1]
    y1 = y1[keepflag1]
    
    x2 = x2[keepflag2]
    y2 = y2[keepflag2]
    n1 = sum(keepflag1)
    n2 = sum(keepflag2)

    ax1 = plt.subplot2grid((nrow,ncol),(1,0),rowspan=nrow-1,colspan=ncol-1, fig=fig)
    if hexbinflag:
        #t1 = plt.hist2d(x1,y1,bins=100,cmap='gray_r')
        #H, xbins,ybins = np.histogram2d(x1,y1,bins=20)
        #extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        #plt.contour(np.log10(H.T+1),  10, extent = extent, zorder=1,colors='k')
        #plt.hexbin(xvar2,yvar2,bins='log',cmap='Blues', gridsize=100)
        label=name1+' (%i)'%(n1)
        plt.hexbin(x1,y1,bins='log',cmap='gray_r', gridsize=75)
    else:
        label=name1+' (%i)'%(n1)
        if lcsflag:
        
            plt.plot(x1,y1,'ko',color=color1,alpha=alphagray,label=label, zorder=10,mec='k',markersize=8)
        else:
            plt.plot(x1,y1,'k.',color=color1,alpha=alphagray,label=label, zorder=1,markersize=8)        
    if contourflag:
        H, xbins,ybins = np.histogram2d(x2,y2,bins=contour_bins)
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
        plt.contour((H.T), levels=ncontour_levels, extent = extent, zorder=1,colors=color2, label='__nolegend__')
        #plt.legend()
    else:
        label=name2+' (%i)'%(n2)
        plt.plot(x2,y2,'co',color=color2,alpha=alpha1, label=label,markersize=8,mec='k',marker=marker2)
        
        

    plt.legend(loc='upper right')
    #sns.kdeplot(agc['LogMstarTaylor'][keepagc],agc['gmi_corrected'][keepagc])#,bins='log',gridsize=200,cmap='blue_r')
    #plt.colorbar()

    # replaced by other function
    #if ssfrlimit is not None:
    #    xl=np.linspace(xmin,xmax,100)
    #    yl = get_SFR_cut(xl)
    #    plt.plot(xl,yl,'k--')#,label='sSFR=-11.5')        
    #    yl =xl + float(args.minssfr)
    #    plt.plot(xl,yl,'k:')#,label='sSFR=-11.5')                


    plt.axis([xmin,xmax,ymin,ymax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(xlabel,fontsize=26)
    plt.ylabel(ylabel,fontsize=26)
    plt.gca().tick_params(axis='both', labelsize=16)
    #plt.axis([7.9,11.6,-.05,2])
    ax2 = plt.subplot2grid((nrow,ncol),(0,0),rowspan=1,colspan=ncol-1, fig=fig, sharex = ax1, yticks=[])
    print('just checking ...',len(x1),len(x2))
    print(min(x1))
    print(min(x2))
    minx = min([min(x1),min(x2)])
    maxx = max([max(x1),max(x2)])    
    mybins = np.linspace(minx,maxx,nhistbin)
    if cumulativeFlag:
        t = plt.hist(x1, density=True, cumulative=True,bins=len(x1),color=color1,histtype='step',lw=1.5, label=name1+' (%i)'%(n1))
        t = plt.hist(x2, density=True, cumulative=True,bins=len(x2),color=color2,histtype='step',lw=1.5, label=name2+' (%i)'%(n2))
    else:
        t = plt.hist(x1, density=True, bins=mybins,color=color1,histtype='step',lw=1.5, label=name1+' (%i)'%(n1))
        t = plt.hist(x2, density=True, bins=mybins,color=color2,histtype='step',lw=1.5, label=name2+' (%i)'%(n2))

    if hexbinflag:
        plt.legend()
    #leg = ax2.legend(fontsize=12,loc='lower left')
    #for l in leg.legendHandles:
    #    l.set_alpha(1)
    #    l._legmarker.set_alpha(1)
    ax2.xaxis.tick_top()
    ax3 = plt.subplot2grid((nrow,ncol),(1,ncol-1),rowspan=nrow-1,colspan=1, fig=fig, sharey = ax1, xticks=[])
    miny = min([min(y1),min(y2)])
    maxy = max([max(y1),max(y2)])    
    mybins = np.linspace(miny,maxy,nhistbin)

    if cumulativeFlag:
        t=plt.hist(y1, density=True, cumulative=True,orientation='horizontal',bins=len(y1),color=color1,histtype='step',lw=1.5, label=name1)
        t=plt.hist(y2, density=True, cumulative=True,orientation='horizontal',bins=len(y2),color=color2,histtype='step',lw=1.5, label=name2)
    else:
        t=plt.hist(y1, density=True, orientation='horizontal',bins=mybins,color=color1,histtype='step',lw=1.5, label=name1)
        t=plt.hist(y2, density=True, orientation='horizontal',bins=mybins,color=color2,histtype='step',lw=1.5, label=name2)

    plt.yticks(rotation='horizontal')
    ax3.yaxis.tick_right()
    ax3.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    #ax3.set_title('$log_{10}(SFR)$',fontsize=20)
    #plt.savefig(figname)

    print('############################################################ ')
    print('KS test comparising galaxies within range shown on the plot')
    print('')
    print('STELLAR MASS')
    t = lcscommon.ks(x1,x2,run_anderson=False)
    t = anderson_ksamp([x1,x2])
    print('Anderson-Darling: ',t)
    print('')
    print('COLOR')
    t = lcscommon.ks(y1,y2,run_anderson=False)
    t = anderson_ksamp([y1,y2])
    print('Anderson-Darling: ',t)
    return ax1,ax2,ax3

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    ''' https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py  ''' 
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins,histtype='step',lw=2,alpha=1)
    ax_histy.hist(y, bins=bins, orientation='horizontal',histtype='step',lw=2,alpha=1)
    
def scatter_hist_wrapper(x,y,fig):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    return ax,ax_histx,ax_histy

def plotsalim07():
    #plot the main sequence from Salim+07 for a Chabrier IMF

    lmstar=np.arange(8.5,11.5,0.1)

    #use their equation 11 for pure SF galaxies
    lssfr = -0.35*(lmstar - 10) - 9.83

    #use their equation 12 for color-selected galaxies including
    #AGN/SF composites.  This is for log(Mstar)>9.4
    #lssfr = -0.53*(lmstar - 10) - 9.87

    lsfr = lmstar + lssfr -.3
    sfr = 10.**lsfr

    plt.plot(lmstar, lsfr, 'w-', lw=4)
    plt.plot(lmstar, lsfr, c='salmon',ls='-', lw=2, label='$Salim+07$')
    plt.plot(lmstar, lsfr-np.log10(5.), 'w--', lw=4)
    plt.plot(lmstar, lsfr-np.log10(5.), c='salmon',ls='--', lw=2)

def mass_match(input_mass,comp_mass,seed,nmatch=10,dm=.15,inputZ=None,compZ=None,dz=.0025,mp=False):
    if mp:
        return_indices =  mass_match_mp(input_mass,comp_mass,seed,nmatch=nmatch,dm=dm,inputZ=inputZ,compZ=compZ,dz=dz)
    else:
        return_indices =  mass_match_linear(input_mass,comp_mass,seed,nmatch=nmatch,dm=dm,inputZ=inputZ,compZ=compZ,dz=dz)
    return return_indices

def mass_match_linear(input_mass,comp_mass,seed,nmatch=10,dm=.15,inputZ=None,compZ=None,dz=.0025):
    '''
    for each galaxy in parent, draw nmatch galaxies in match_mass
    that are within +/-dm

    PARAMS:
    -------
    * input_mass - sample to create mass-matched sample for, ref mass distribution    
    * comp_mass - sample to draw matches from
    * dm - mass offset from which to draw matched galaxies from
    * nmatch = number of matched galaxies per galaxy in the input
    * inputZ = redshift for input sample; need to supply this to do redshift cut 
    * compZ = redshift for comparison sample 

    RETURNS:
    --------
    * indices of the comp_sample

    '''
    # for each galaxy in parent
    # select nmatch galaxies from comp_sample that have stellar masses
    # within +/- dm
    np.random.seed(seed)
    return_index = np.zeros(len(input_mass)*nmatch,'i')

    comp_index = np.arange(len(comp_mass))
    # hate to do this with a loop,
    # but I can't think of a smarter way right now
    for i in range(len(input_mass)):
        # limit comparison sample to mass of i galaxy, +/- dm
        flag = np.abs(comp_mass - input_mass[i]) < dm
        # if redshifts are provided, also restrict based on redshift offset
        if inputZ is not None:
            flag = flag & (np.abs(compZ - inputZ[i]) < dz)
        # select nmatch galaxies randomly from this limited mass range
        # NOTE: can avoid repeating items by setting replace=False
        if sum(flag) < nmatch:
            print('galaxies in slice < # requested',sum(flag),nmatch,input_mass[i])
        if sum(flag) == 0:
            print('\truh roh - doubling mass and redshift slices')
            flag = np.abs(comp_mass - input_mass[i]) < 2*dm
            # if redshifts are provided, also restrict based on redshift offset
            if inputZ is not None:
                flag = flag & (np.abs(compZ - inputZ[i]) < 2*dz)
            if sum(flag) == 0:
                print('\truh roh again - tripling mass and redshift slices')
                flag = np.abs(comp_mass - input_mass[i]) < 4*dm
                # if redshifts are provided, also restrict based on redshift offset
                if inputZ is not None:
                    flag = flag & (np.abs(compZ - inputZ[i]) < 4*dz)
                if sum(flag) == 0:
                    print("can't seem to find a match for mass = ",input_mass[i], i)
                    print("skipping this galaxy")                    
                    continue
        #return_index.append(np.random.choice(comp_index[flag],nmatch,replace=True).tolist())
        return_index[int(i*nmatch):int((i+1)*nmatch)] = np.random.choice(comp_index[flag],nmatch,replace=True)
    return np.array(return_index,'i')
    
def mass_match_mp(input_mass,comp_mass,seed,dm=.15,nmatch=1,inputZ=None,compZ=None,dz=.0025):
    '''
    for each galaxy in parent, draw nmatch galaxies in match_mass
    that are within +/-dm

    PARAMS:
    -------
    * input_mass - sample to create mass-matched sample for, ref mass distribution    
    * comp_mass - sample to draw matches from
    * dm - mass offset from which to draw matched galaxies from
    * nmatch = number of matched galaxies per galaxy in the input
    * inputZ = redshift for input sample; need to supply this to do redshift cut 
    * compZ = redshift for comparison sample 

    RETURNS:
    --------
    * indices of the comp_sample

    '''
    # for each galaxy in parent
    # select nmatch galaxies from comp_sample that have stellar masses
    # within +/- dm
    np.random.seed(seed)
    return_index = np.zeros(len(input_mass)*nmatch,'i')

    # using multiprocessing to increase speed of mass matching
    mm_pool = mp.Pool(mp.cpu_count())
    myresults = [mm_pool.apply_async(mass_match_subroutine,args=(onemass,comp_mass,nmatch)) for onemass in input_mass]
    mm_pool.close()
    mm_pool.join()
    return_indices= [r.get() for r in myresults]
    
    return return_indices

def collect_results_mm(result):

    global results
    massmatch_results.append(result)

def mass_match_subroutine(onemass,comparison_masses,nmatch,dm=0.15):
    flag = np.abs(comparison_masses - onemass) < dm
    
    comp_index = np.arange(len(comparison_masses))
    
    # select nmatch galaxies randomly from this limited mass range
    # NOTE: can avoid repeating items by setting replace=False
    if sum(flag) == 0:
        #print('\truh roh - doubling mass and redshift slices')
        flag = np.abs(comp_mass - input_mass[i]) < 2*dm
    if sum(flag) == 0:
        #print('\truh roh again - tripling mass and redshift slices')
        flag = np.abs(comp_mass - input_mass[i]) < 4*dm
    return np.random.choice(comp_index[flag],nmatch,replace=True)
    
def plotelbaz():
    #plot the main sequence from Elbaz+13
        
    xe=np.arange(8.5,11.5,.1)
    xe=10.**xe

    #I think that this comes from the intercept of the
    #Main-sequence curve in Fig. 18 of Elbaz+11.  They make the
    #assumption that galaxies at a fixed redshift have a constant
    #sSFR=SFR/Mstar.  This is the value at z=0.  This is
    #consistent with the value in their Eq. 13

    #This is for a Salpeter IMF
    ye=(.08e-9)*xe   
        
        
    plt.plot(log10(xe),np.log19(ye),'w-',lw=3)
    plt.plot(log10(xe),np.log10(ye),'k-',lw=2,label='$Elbaz+2011$')
    #plot(log10(xe),(2*ye),'w-',lw=4)
    #plot(log10(xe),(2*ye),'k:',lw=2,label='$2 \ SFR_{MS}$')
    plt.plot(log10(xe),np.log10(ye/5.),'w--',lw=4)
    plt.plot(log10(xe),np.log10(ye/5.),'k--',lw=2,label='$SFR_{MS}/5$')

def getlegacy(ra1,dec1,jpeg=True,getfits=True,imsize=None):
    '''
    imsize is size of desired cutout in arcmin
    '''
    default_image_size = 60
    gra = '%.5f'%(ra1) # accuracy is of order .1"
    gdec = '%.5f'%(dec1)
    galnumber = gra+'-'+gdec
    if imsize is not None:
        image_size=imsize
    else:
        image_size=default_image_size
    cwd = os.getcwd()
    if not(os.path.exists(cwd+'/cutouts/')):
        os.mkdir(cwd+'/cutouts')
    rootname = 'cutouts/legacy-im-'+str(galnumber)+'-'+str(image_size)
    jpeg_name = rootname+'.jpg'

    fits_name = rootname+'.fits'

    # check if images already exist
    # if not download images

    if not(os.path.exists(jpeg_name)):
        print('downloading image ',jpeg_name)
        url='http://legacysurvey.org/viewer/jpeg-cutout?ra='+str(ra1)+'&dec='+str(dec1)+'&layer=dr8&size='+str(image_size)+'&pixscale=1.00'
        urlretrieve(url, jpeg_name)

    if not(os.path.exists(fits_name)):
        print('downloading image ',fits_name)
        url='http://legacysurvey.org/viewer/cutout.fits?ra='+str(ra1)+'&dec='+str(dec1)+'&layer=dr8&size='+str(image_size)+'&pixscale=1.00'
        urlretrieve(url, fits_name)
            
    try:
        t,h = fits.getdata(fits_name,header=True)
    except IndexError:
        print('problem accessing image')
        print(fits_name)
        url='http://legacysurvey.org/viewer/cutout.fits?ra='+str(ra1)+'&dec='+str(dec1)+'&layer=dr8&size='+str(image_size)+'&pixscale=1.00'
        print(url)
        return None
    
    if np.mean(t[1]) == 0:
        return None
    norm = simple_norm(t[1],stretch='asinh',percent=99.5)
    if jpeg:
        t = Image.open(jpeg_name)
        plt.imshow(t,origin='lower')
    else:
        plt.imshow(t[1],origin='upper',cmap='gray_r', norm=norm)
    w = WCS(fits_name,naxis=2)        
    
    return t,w

def sersic(x,Ie,n,Re):
    bn = 1.999*n - 0.327
    return Ie*np.exp(-1*bn*((x/Re)**(1./n)-1))

def plot_models():
    plt.figure(figsize=(8,6))
    plt.subplots_adjust(wspace=.35)

    rmax = 4
    scaleRe = 0.8
    rtrunc = 1.5
    n=1
    # shrink Re
    plt.subplot(2,2,1)
    x = np.linspace(0,rmax,100)
    
    Ie=1
    Re=1
    y = sersic(x,Ie,n,Re)
    plt.plot(x,y,label='sersic n='+str(n),lw=2)
    y2 = sersic(x,Ie,n,scaleRe*Re)
    plt.plot(x,y2,label="Re "+r"$ \rightarrow \ $"+str(scaleRe)+"Re",ls='--',lw=2)
    #plt.legend()
    
    plt.ylabel('Intensity',fontsize=18)
    plt.text(0.1,.85,'(a)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    
    # plot total flux
    plt.subplot(2,2,2)
    dx = x[1]-x[0]
    sum1 = (y*dx*2*np.pi*x)
    sum2 = (y2*dx*2*np.pi*x)
    plt.plot(x,np.cumsum(sum1)/np.max(np.cumsum(sum1)),label='sersic n='+str(n),lw=2)
    plt.plot(x,np.cumsum(sum2)/np.max(np.cumsum(sum1)),label="Re "+r"$ \rightarrow \ $"+str(scaleRe)+"Re",ls='--',lw=2)
    plt.ylabel('Enclosed Flux',fontsize=18)
    plt.grid()
    plt.text(0.1,.85,'(b)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    
    # truncated sersic model
    plt.subplot(2,2,3)
    plt.plot(x,y,label='sersic n='+str(n),lw=2)
    y3 = y.copy()
    flag = x > rtrunc
    y3[flag] = np.zeros(sum(flag))
    plt.plot(x,y3,ls='--',label='Rtrunc =  '+str(rtrunc)+' Re',lw=2)
    plt.legend()
    plt.ylabel('Intensity',fontsize=18)
    #plt.legend()
    #plt.gca().set_yscale('log')
    plt.text(0.1,.85,'(c)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    plt.xlabel('r/Re')
    
    plt.subplot(2,2,4)
    sum3 = (y3*dx*2*np.pi*x)
    plt.plot(x,np.cumsum(sum1)/np.max(np.cumsum(sum1)),label='sersic n='+str(n),lw=2)
    plt.plot(x,np.cumsum(sum3)/np.max(np.cumsum(sum1)),label='Rtrunc =  '+str(rtrunc)+' Re',ls='--',lw=2)
    plt.ylabel('Enclosed Flux',fontsize=18)
    
    plt.grid()
    plt.text(0.1,.85,'(d)',transform=plt.gca().transAxes,horizontalalignment='left')
    plt.legend(fontsize=14)
    plt.xlabel('r/Re')
    plt.savefig(plotdir+'/cartoon-models.png')
    plt.savefig(plotdir+'/cartoon-models.pdf')
    pass



###########################
##### Plot parameters
###########################

# figure setup
plotsize_single=(6.8,5)
plotsize_2panel=(10,5)
params = {'backend': 'pdf',
          'axes.labelsize': 24,
          'font.size': 20,
          'legend.fontsize': 12,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          #'lines.markeredgecolor'  : 'k',  
          #'figure.titlesize': 20,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'text.usetex': True,
          'figure.figsize': plotsize_single}
plt.rcParams.update(params)
#figuredir = '/Users/rfinn/Dropbox/Research/MyPapers/LCSpaper1/submit/resubmit4/'
figuredir='/Users/grudnick/Work/Local_cluster_survey/Papers/Finn_MS/Plots/'
#figuredir = '/Users/grudnick/Work/Local_cluster_survey/Analysis/MS_paper/Plots/'

#colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

########################################
##### STATISTICS TO COMPARE CORE VS EXTERNAL
########################################

test_statistics = lambda x: (np.mean(x), np.var(x), MAD(x), st.skew(x), st.kurtosis(x))
stat_cols = ['mean','var','MAD','skew','kurt']
###########################
##### START OF GALAXIES CLASS
###########################


class gswlc_full():
    '''
    class for full GSWLC catalog.  this will cut on BT and save a trimmed section that
    overlaps LCS region
    '''
    def __init__(self,catalog,cutBT=False,HIdef=False):
        self.cutBT = cutBT
        if catalog.find('.fits') > -1:
            #print('got a fits file')
            self.gsw = Table.read(catalog,format='fits')
            #print(self.gsw.colnames)
            self.redshift_field = 'zobs'
            if cutBT:
                self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap-BTcut.fits'
                print('outfile = ',self.outfile)
            else:
                self.outfile = catalog.split('.fits')[0]+'-LCS-Zoverlap.fits'                        
                print('outfile = ',self.outfile)
        else:
            print('reading ascii')
            self.gsw = ascii.read(catalog)
            if catalog.find('v2') > -1:
                self.redshift_field = 'Z_1'
            else:
                self.redshift_field = 'Z'
            if cutBT:
                self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap-BTcut.fits'
            else:
                self.outfile = catalog.split('.dat')[0]+'-LCS-Zoverlap.fits'
        #print(self.gsw.colnames[0:10],len(self.gsw.colnames))
        self.keepflag = np.ones(len(self.gsw),'bool')
        self.cut_redshift()

        if cutBT:
            self.cut_BT(BT=float(args.BT))
        self.cut_ellip()
        if args.cutN:

            self.cut_nsersic()
        self.save_trimmed_cat()
        #self.get_dsfr
        if HIdef:
            self.save_trimmed_HIdef()
    def cut_redshift(self):
        z1 = zmin
        z2 = zmax
        #print(self.redshift_field)
        #print(self.gsw.colnames[0:10],len(self.gsw.colnames))        
        #print(self.gsw.colnames)
        #print(z1,z2)
        zflag = (self.gsw[self.redshift_field] > z1) & (self.gsw[self.redshift_field] < z2)
        massflag = self.gsw['logMstar'] > 0
        #self.gsw = self.gsw[zflag & massflag]
        self.keepflag = self.keepflag & zflag & massflag
    def cut_BT(self,BT=None):
        btflag = self.gsw[BTkey] <= BT
        #self.gsw = self.gsw[btflag]
        self.keepflag = self.keepflag & btflag        
    def cut_ellip(self):
        ellipflag = self.gsw[ellipkey] <= float(args.ellip)
        #self.gsw = self.gsw[ellipflag]
        self.keepflag = self.keepflag & ellipflag                
    def cut_nsersic(self):
        ''' cut catalog based on the sersic index '''
        # colname is ng for gswlc and ng_1 for LCS
        nsersicflag = self.gsw['ng'] <= float(args.nsersic)
        #self.gsw = self.gsw[ellipflag]
        self.keepflag = self.keepflag & nsersicflag
    def get_dsfr(self):
        ''' get distance from MS '''
        #self.dsfr = self.gsw['logSFR'] - get_BV_MS(self.gsw['logMstar'])
        self.dsfr = self.gsw['logSFR'] - get_MS(self.gsw['logMstar'],BTcut=self.cutBT)
        
    def save_trimmed_cat(self):
        t = Table(self.gsw[self.keepflag])

        t.write(self.outfile,format='fits',overwrite=True)
    def save_trimmed_HIdef(self):
        fname = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-tab3-Tempel-13-2021Jan07-HIdef-2021Aug28'
        HIdef = Table.read(fname+'.fits')
        HIdef_trimmed = HIdef[self.keepflag]
        HIdef_trimmed.write(fname+'-trimmed.fits',format='fits',overwrite=True)

class gswlc_base():
    ''' functions that will be applied to LCS-GSWLC catalog and GSWLC catalog   '''
    def base_init(self):
        # selecting agn in a different way - from dr10 catalog
        #self.get_agn()
        # cut catalog to remove agn
        #self.remove_agn()

        self.calc_ssfr()
        #self.get_dsfr()
        #self.get_lowsfr_flag()

    def get_agn(self):
        self.specflag = ~np.isnan(self.cat['O3FLUX'])
        self.AGNKAUFF= (np.log10(self.cat['O3FLUX']/self.cat['HBFLUX']) > (.61/(np.log10(self.cat['N2FLUX']/self.cat['HAFLUX']-.05))+1.3)) | (np.log10(self.cat['N2FLUX']/self.cat['HAFLUX']) > 0.)  #& (self.s.HAEW > 0.)
        # add calculations for selecting the sample
        
        self.wiseagn=(self.cat['W1MAG_3'] - self.cat['W2MAG_3']) > 0.8
        self.agnflag = (self.AGNKAUFF & self.specflag) | self.wiseagn
        print('fraction of AGN = %.3f (%i/%i)'%(sum(self.agnflag)/len(self.agnflag),sum(self.agnflag),len(self.agnflag)))
    def remove_agn(self):
        print('REMOVING AGN')
        self.cat = self.cat[~self.agnflag]
        
    def calc_ssfr(self):
        self.ssfr = self.cat['logSFR'] - self.cat['logMstar']
    def get_dsfr(self):
        ''' get distance from MS '''
        #self.dsfr = self.cat['logSFR'] - get_BV_MS(self.cat['logMstar'])
        self.dsfr = self.cat['logSFR'] - get_MS(self.cat['logMstar'],BTcut=self.cutBT)
    def get_lowsfr_flag(self):
        self.lowsfr_flag = (self.dsfr < -1*MS_OFFSET)

    def plot_ms(self,plotsingle=True,outfile1=None,outfile2=None):
        if plotsingle:
            plt.figure(figsize=(8,6))
        x = self.cat['logMstar']
        y = self.cat['logSFR']
        plt.plot(x,y,'k.',alpha=.1)
        #plt.hexbin(x,y,gridsize=30,vmin=5,cmap='gray_r')
        #plt.colorbar()
        xl=np.linspace(8,12,50)
        ssfr_limit = -11.5
        plt.plot(xl,xl+ssfr_limit,'c-')
        plotsalim07()
        plt.xlabel('logMstar',fontsize=20)
        plt.ylabel('logSFR',fontsize=20)
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/sfms-ssfr-cut.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/sfms-ssfr-cut.png')
        else:
            plt.savefig(outfile2)
        
    def plot_positions(self,plotsingle=True, filename=None):
        if plotsingle:
            plt.figure(figsize=(14,8))

        plt.scatter(self.cat['RA'],selfgsw['DEC'],c=np.log10(self.densNN*u.Mpc),vmin=-.5,vmax=1.,s=10)
        plt.colorbar(label='$\log_{10}(N/d_N)$')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.gca().invert_xaxis()
        plt.axhline(y=5)
        plt.axhline(y=60)
        plt.axvline(x=135)
        plt.axvline(x=225)
        #plt.axis([130,230,0,65])
        if filename is not None:
            plt.savefig(filename)
class gswlc(gswlc_base):
    ''' class for GSWLC field catalog    '''
    def __init__(self,catalog,cutBT=False,HIdef_file=None,args=None):
        self.cat = fits.getdata(catalog) 
        self.base_init()
        #self.calc_local_density()
        self.get_field1()
        if HIdef_file is not None:
            self.HIdef = Table.read(HIdef_file)

    def fit_MS(self,flag=None,plotFlag=False,linearFit=True):
        ''' fit the SF main sequence for current sample (using same BT, mass, ellipticity cut as sample)  '''
        x = self.cat['logMstar']
        y = self.cat['logSFR']        
        if flag is not None:
            x = x[flag]
            y = y[flag]

        # using curve fit with
        if linearFit:
            fitted_line,mask = fit_line_sigma_clip(x,y,sigma=3,slope=.6,intercept=-7.2)
            self.MS_line = fitted_line
        
            self.MS_slope = fitted_line.slope.value
            self.MS_intercept = fitted_line.intercept.value
        else:
            fitted_line,mask = fit_line_sigma_clip(x,y,sigma=3,slope=.6,intercept=-7.2,linearFit=False)
            self.MS_line = fitted_line
            print('MS fit = ',fitted_line)
            
        self.MS_std = np.std(np.abs(y[~mask]-fitted_line(x[~mask])))
        print('STD of pruned data = {:.2f}'.format(self.MS_std))
        self.MS_std = np.std(np.abs(y-fitted_line(x)))
        print('STD of full data = {:.2f}'.format(self.MS_std))        
        
        print('##################################')
        print('### FITTING WITH SIGMA CLIPPING ')
        print('##################################')        
        print('Best-fit slope = {:.2f}'.format(self.MS_slope))
        print('Best-fit inter = {:.2f}'.format(self.MS_intercept))
        print('Width of the MS = {:.2f} (unclipped data)'.format(self.MS_std))
        #reset scatter in MS to 0.3 for consistency with text
        #self.MS_std = 0.3
        
        # using curve fit after sigma clipping
        print('###################################')
        print('### FITTING MS AFTER SIGMA CLIPPING ')
        print('###################################')        
              
        popt,pcov = curve_fit(linear_func,x[~mask],y[~mask])
        perr = np.sqrt(np.diag(pcov))
        print('Best-fit slope = {:.2f}+/-{:.2f}'.format(popt[0],perr[0]))
        print('Best-fit inter = {:.2f}+/-{:.2f}'.format(popt[1],perr[1]))
        if plotFlag:
            plt.figure()
            xmin,xmax = 8,11
            ymin,ymax = -3,2
            plt.hexbin(x,y,extent=(xmin,xmax,ymin,ymax),cmap='gray_r',gridsize=50,vmin=0,vmax=200)
            xline = np.linspace(xmin,xmax,100)
            y_MS = self.MS_slope*xline+self.MS_intercept
            plt.plot(xline,y_MS,'b-',lw=2,label='MS fit')            
            plt.plot(xline,y_MS-self.MS_std,'b--',lw=2,label='MS-std')
            plt.plot(xline,y_MS+self.MS_std,'b--',lw=2,label='MS+std')

            # plot filtered data
            yclipped = np.ma.masked_array(y, mask=~mask)
            plt.plot(x,yclipped,'r,',label='rejected data')
            plt.xlabel('$\log_{10}(M_\star/M_\odot)$')
            plt.ylabel('$\log_{10}(SFR)$')
            plt.legend()
            
            

            
            
    def plot_MS_old(self,ax,color='mediumblue',ls='-'):
        '''
        plot our MS fit
        '''
        plot_Durbala_MS(ax)
        x1,x2 = 9.7,11.
        xline = np.linspace(x1,x2,100)
        yline = get_MS(xline,BTcut=self.cutBT)
        ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
        ax.plot(xline,yline,c=color,ls=ls,lw=3,label='MS Fit')

        # scatter around MS fit
        sigma=self.MS_std
        ax.plot(xline,yline-1.5*sigma,c='w',ls='--',lw=4)
        ax.plot(xline,yline-1.5*sigma,c=color,ls='--',lw=3,label='fit-1.5$\sigma$')
        
    def plot_MS(self,ax,color='mediumblue',ls='-'):
        '''
        plot our MS fit
        '''
        #plot_MS_lowBT(ax)
        x1,x2 = 9.7,11.
        xline = np.linspace(x1,x2,100)
        yline = get_MS(xline,BTcut=self.cutBT)
        ax.plot(xline,yline,c='w',ls=ls,lw=4,label='_nolegend_')
        ax.plot(xline,yline,c=color,ls=ls,lw=3,label='MS Fit')

        # scatter around MS fit
        ax.plot(xline,yline-MS_OFFSET,c='w',ls='--',lw=4)
        ax.plot(xline,yline-MS_OFFSET,c=color,ls='--',lw=3,label='$low\ SFR$')
        
    def calc_local_density(self,NN=10):
        try:
            redshift = self.cat['Z']
        except KeyError:
            redshift = self.cat['Z_1']
        pos = SkyCoord(ra=self.cat['RA']*u.deg,dec=self.cat['DEC']*u.deg, distance=redshift*3.e5/70*u.Mpc,frame='icrs')

        idx, d2d, d3d = pos.match_to_catalog_3d(pos,nthneighbor=NN)
        self.densNN = NN/d3d
        self.sigmaNN = NN/d2d
    def get_field1(self):
        ramin=135
        ramax=225
        decmin=5
        decmax=60
        gsw_position_flag = (self.cat['RA'] > ramin) & (self.cat['RA'] < ramax) & (self.cat['DEC'] > decmin) & (self.cat['DEC'] < decmax)
        #gsw_density_flag = np.log10(self.densNN*u.Mpc) < .2
        #self.field1 = gsw_position_flag & gsw_density_flag
    def plot_field1(self,figname1=None,figname2=None):
        plt.figure(figsize=(14,8))
        plt.scatter(self.cat['RA'],self.cat['DEC'],c=np.log10(self.densNN*u.Mpc),vmin=-.5,vmax=1.,s=10)
        plt.colorbar(label='$\log_{10}(N/d_N)$')
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.gca().invert_xaxis()
        plt.axhline(y=5)
        plt.axhline(y=60)
        plt.axvline(x=135)
        plt.axvline(x=225)
        plt.title('Entire GSWLC (LCS z range)')
        plt.axis([130, 230, 0, 65])
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)
    def plot_dens_hist(self,figname1=None,figname2=None):
        plt.figure()
        t = plt.hist(np.log10(self.densNN*u.Mpc), bins=20)
        plt.xlabel('$ \log_{10} (N/d_N)$')

        print('median local density =  %.3f'%(np.median(np.log10(self.densNN*u.Mpc))))
        plt.axvline(x = (np.median(np.log10(self.densNN*u.Mpc))),c='k')
        plt.title('Distribution of Local Density')
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)

#######################################################################
#######################################################################
########## NEW CLASS: LCSGSW
#######################################################################
#######################################################################
class lcsgsw(gswlc_base):
    '''
    functions to operate on catalog that is cross-match between
    LCS and GSWLC 
    '''
    def __init__(self,catalog,sigma_split=600,cutBT=False,args=None):
        # read in catalog of LCS matched to GSWLC
        #self.cat = fits.getdata(catalog)
        self.cat = Table.read(catalog)
        print('number of lines in LCS cat = ',len(self.cat))
        '''
        c1 = Column(self.agnflag,name='agnflag')
        c2 = Column(self.wiseagn,name='wiseagn')
        c3 = Column(self.AGNKAUFF,name='kauffagn')
        c4 = Column(self.cat['N2FLUX']/self.cat['HAFLUX'],name='n2ha')
        c5 = Column(self.cat['O3FLUX']/self.cat['HBFLUX'],name='o3hb')        
        tabcols = [c1,c2,c3,c4,c5,self.cat['HAFLUX'],self.cat['N2FLUX'],self.cat['HBFLUX'],self.cat['O3FLUX'],self.cat['W1MAG_3'],self.cat['W2MAG_3'],self.cat['AGNKAUFF']]
        tabnames = ['agnflag','wiseagn','kauffagn','N2/HA','O3/HB','HAFLUX','N2FLUX','HBFLUX','O3FLUX','W1MAG_3','W2MAG_3','AGNKAUF']
        newtable = Table(data=tabcols,names=tabnames)
        newtable.write(homedir+'/research/LCS/tables/lcs-gsw-noagn.fits',overwrite=True)
        '''

        self.base_init()


        # this doesn't work after adding agn cut
        # getting error that some array lengths are not the same
        # I am not actually using this data file in the simulation
        # so commenting it out for now.
        # I'm sure the error will pop up somewhere else...
        #self.write_file_for_simulation()

        if cutBT:
            self.cut_BT(BT=float(args.BT))
        self.cut_ellip()
        if args.cutN:
            self.cut_nsersic()
        
        nsaflag = ~self.cat['NSAID'].mask
        indices = np.arange(len(self.cat))[nsaflag]
        self.nsadict=dict((a,b) for a,b in zip(self.cat['NSAID'][nsaflag],indices))
        #self.get_dsfr()
        #self.get_lowsfr_flag()

        # set up an ID number for the LCS catalog
        self.lcsid = ['LCS{:04d}'.format(i) for i in range(len(self.cat))]
        
        self.get_membflag()
        self.get_infallflag()
        
        self.get_DA()
        self.get_sizeflag()
        self.get_sbflag()
        self.get_galfitflag()
        self.get_sampleflag()

        self.calculate_sizeratio()
        if args.cutBT:
            self.write_file_for_simulation()        
        self.group = self.cat['CLUSTER_SIGMA'] < sigma_split
        self.cluster = self.cat['CLUSTER_SIGMA'] > sigma_split
        self.get_NUV24()
        #self.get_dsfr()

    def get_mass_match(self):
        # better to run this once than to implement mass matching in each plot separately
        # get indices of mass-matched sample to LCS

        # get indices of mass-matched sample to LCS core

        # get indices of mass-matched sample to LCS infall

        pass

        
        
    def get_NUV24(self):
        self.NUVr=self.cat['ABSMAG'][:,1] - self.cat['ABSMAG'][:,4]
        self.NUV = 22.5 - 2.5*np.log10(self.cat['NMGY'][:,1])
        self.MAG24 = 2.5*np.log10(3631./(self.cat['FLUX24']*1.e-6))
        self.NUV24 =self.NUV-self.MAG24

        #lcspath = homedir+'/github/LCS/'
        #self.lcsbase = lb.galaxies(lcspath)

    def get_DA(self):
        # stole this from LCSbase.py
        #print(self.cat.colnames)
        self.DA=np.zeros(len(self.cat))
        for i in range(len(self.DA)):
            if self.membflag[i]:
                self.DA[i] = cosmo.angular_diameter_distance(self.cat['CLUSTER_REDSHIFT'][i]).value*Mpcrad_kpcarcsec
            else:
                self.DA[i] = cosmo.angular_diameter_distance(self.cat['ZDIST'][i]).value*Mpcrad_kpcarcsec
    def get_sizeflag(self):
        ''' calculate size flag '''
        self.sizeflag=(self.cat['SERSIC_TH50']*self.DA > minsize_kpc)
    def get_sbflag(self):
        '''  surface brightness flag '''
        self.sb_obs = 999*np.ones(len(self.cat))
        mipsflag = self.cat['FLUX24'] > 0
        self.sb_obs[mipsflag]=(self.cat['fcmag1'][mipsflag] + 2.5*np.log10(np.pi*((self.cat['fcre1'][mipsflag]*mipspixelscale)**2)*self.cat['fcaxisratio1'][mipsflag]))
        self.sbflag = self.sb_obs < 20.
        print('got sb flag')
    def get_gim2dflag(self):
        # don't need this b/c the catalog has already been match to the simard table
        self.gim2dflag=(self.cat['SERSIC_TH50']*self.DA > minsize_kpc)
    def get_galfitflag(self):
        ''' calculate galfit flag '''
        
        self.galfitflag = (self.cat['fcmag1'] > .01)  & ~self.cat['fcnumerical_error_flag24']
        
        
        galfit_override = [70588,70696,43791,69673,146875,82170, 82182, 82188, 82198, 99058, 99660, 99675, 146636, 146638, 146659, 113092, 113095, 72623,72631,72659, 72749, 72778, 79779, 146121, 146130, 166167, 79417, 79591, 79608, 79706, 80769, 80873, 146003, 166044,166083, 89101, 89108,103613,162792,162838, 89063,99509,72800,79381,10368]
        for id in galfit_override:
            try:
                self.galfitflag[self.nsadict[int(id)]] = True
                #print('HEY! found a match, just so you know, with NSAID ',id,'\n\tCould be from sources that were removed with AGN/BT/GSWLC matches')
            except KeyError:
                pass
                #print('got a key error, just so you know, with NSAID ',id,'\n\tCould be from sources that were removed with AGN/BT/GSWLC matches')
            except IndexError:
                pass
                #print('WARNING: got an index error in nsadict for NSAID', id,'\n\tCould be from sources that were removed with AGN/BT/GSWLC matches')
        #self.galfitflag = self.galfitflag

        try:
            self.galfitflag[self.nsadict[79378]] = False
        except KeyError:
            pass
        # bringing this over from LCSbase.py
        self.badfits=np.zeros(len(self.cat),'bool')
        nearbystar=[142655, 143485, 99840, 80878] # bad NSA fit; 24um is ok
        #nearbygalaxy=[103927,143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        # checked after reworking galfit
        nearbygalaxy=[143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        #badNSA=[166185,142655,99644,103825,145998]
        #badNSA = [
        badfits= nearbygalaxy#+nearbystar+nearbygalaxy
        badfits=np.array(badfits,'bool')
        for gal in badfits:
            flag = self.cat['NSAID'] == gal
            if sum(flag) == 1:
                self.badfits[flag]  = True

        # fold badfits into galfit flag
        self.galfitflag = self.galfitflag & ~self.badfits
    def get_membflag(self):
        self.membflag = (np.abs(self.cat['DV_SIGMA']) < (-4./3.*self.cat['DR_R200'] + 2))
    def get_infallflag(self):
        self.infallflag = (np.abs(self.cat['DV_SIGMA']) < 3) & ~self.membflag
    def get_sampleflag(self):
        #print('in get_sampleflag')
        #print(len(self.galfitflag),len(self.sbflag),len(self.cat['lirflag']),len(self.sizeflag))
        self.sampleflag=  self.sbflag & self.cat['lirflag'] & self.sizeflag  & self.galfitflag
        self.uvirflag = (self.cat['flag_uv'] > 0) & (self.cat['flag_midir'] > 0)        
        self.uvirsampleflag = self.sampleflag & self.uvirflag
        #& self.cat['galfitflag2'] #& &  &    #~self.cat['AGNKAUFF'] #&  # #& self.cat['gim2dflag'] ##& ~self.cat['fcnumerical_error_flag24'] 

    def calculate_sizeratio(self):
        # all galaxies in the catalog have been matched to simard table 1
        # does this mean they all have a disk scale length?
        self.gim2dflag = np.ones(len(self.cat),'bool')#  self.cat['matchflag'] & self.cat['lirflag'] & self.cat['sizeflag'] & self.cat['sbflag']
        # stole this from LCSbase.py
        self.SIZE_RATIO_DISK = np.zeros(len(self.cat))
        a =  self.cat['fcre1'][self.gim2dflag]*mipspixelscale # fcre1 = 24um half-light radius in mips pixels
        b = self.DA[self.gim2dflag]
        c = self.cat[Rdkey][self.gim2dflag] # gim2d half light radius for disk in kpc

        # this is the size ratio we use in paper 1
        self.SIZE_RATIO_DISK[self.gim2dflag] =a*b/c
        self.SIZE_RATIO_DISK_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK_ERR[self.gim2dflag] = self.cat['fcre1err'][self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.cat[Rdkey][self.gim2dflag]

        self.sizeratio = self.SIZE_RATIO_DISK
        self.sizeratioERR=self.SIZE_RATIO_DISK_ERR
        # size ratio corrected for inclination 
        #self.size_ratio_corr=self.sizeratio*(self.cat.faxisratio1/self.cat.SERSIC_BA)
        self.size_flag = self.sizeratio > 0.
    def write_file_for_simulation(self):
        # need to calculate size ratio
        # for some reason, I don't have this in my table
        # what was I thinking???

        # um, nevermind - looks like it's in the file afterall
        
        self.get_DA()
        #self.calculate_sizeratio()
        # write a file that contains the
        # sizeratio, error, SFR, sersic index, membflag
        # we are using GSWLC SFR
        c1 = Column(self.sizeratio,name='sizeratio')
        c2 = Column(self.sizeratioERR,name='sizeratio_err')
        # using all simard values of sersic fit
        tabcols = [c1,c2,self.membflag,self.cat[BTkey],self.cat['ng_1'],self.cat['logSFR'],self.cat['logMstar'],self.cat['fcre1'],self.cat['fcnsersic1'],self.cat['Rd_1'],self.cat['DELTA_V']]
        tabnames = ['sizeratio','sizeratio_err','membflag','B_T_r','ng','logSFR','logMstar','fcre1','fcnsersic1','Rd','DELTA_V']

        newtable = Table(data=tabcols,names=tabnames)
        newtable = newtable[self.sampleflag]
        newtable.write(homedir+'/research/LCS/tables/LCS-simulation-data.fits',format='fits',overwrite=True)

    def cut_BT(self,BT=None):
        print('inside LCS cut_BT, BT = ',BT)
        print('inside LCS cut_BT, BTkey = ',BTkey)        
        btflag = (self.cat[BTkey] <= BT) #& (self.cat['matchflag'])
        self.cat = self.cat[btflag]
        self.ssfr = self.ssfr[btflag]
        #self.sizeratio = self.sizeratio[btflag]
        #self.sizeratioERR = self.sizeratioERR[btflag]
    def cut_ellip(self):
        ellipflag = (self.cat[ellipkey] <= float(args.ellip))
        self.cat = self.cat[ellipflag]
        self.ssfr = self.ssfr[ellipflag]
    def cut_nsersic(self):
        ''' cut catalog based on sersic index '''

        # colname is 'ng_1' for LCS
        nsersicflag = (self.cat['ng_1'] <= float(args.nsersic))
        self.cat = self.cat[nsersicflag]
        self.ssfr = self.ssfr[nsersicflag]
        
    def get_mstar_limit(self,rlimit=17.7):
        
        print(rlimit,zmax)
        # assume hubble flow
        
        dmax = zmax*3.e5/70
        # abs r
        # m - M = 5logd_pc - 5
        # M = m - 5logd_pc + 5
        ## r = 22.5 - np.log10(lcsgsw['NMGY'][:,4])
        Mr = rlimit - 5*np.log10(dmax*1.e6) +5
        print(Mr)
        ssfr = self.cat['logSFR'] - self.cat['logMstar']
        flag = (self.cat['logMstar'] > 0) & (ssfr > -11.5)
        Mr = self.cat['ABSMAG'][:,4]+5*np.log10(.7)
        plt.figure()
        plt.plot(Mr[flag],self.cat['logMstar'][flag],'bo',alpha=.2,markersize=3)
        plt.axvline(x=-18.6)
        plt.axhline(y=10)
        plt.axhline(y=9.7,ls='--')        

        plt.xlabel('Mr')
        plt.ylabel('logMstar GSWLC')
        plt.grid(True)        
    def plot_dvdr(self, figname1=None,figname2=None,plotsingle=True):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)
        xmin,xmax,ymin,ymax = 0,3.5,0,3.5
        if plotsingle:
            plt.figure(figsize=(8,6))
            ax=plt.gca()
            plt.subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
            plt.ylabel('$ \Delta v/\sigma $',fontsize=26)
            plt.xlabel('$ \Delta R/R_{200}  $',fontsize=26)
            plt.legend(loc='upper left',numpoints=1)

        if USE_DISK_ONLY:
            clabel=['$R_{24}/R_d$','$R_{iso}(24)/R_{iso}(r)$']
        else:
            clabel=['$R_e(24)/R_e(r)$','$R_{iso}(24)/R_{iso}(r)$']
        
        x=(self.cat['DR_R200'])
        y=abs(self.cat['DELTA_V'])
        plt.hexbin(x,y,extent=(xmin,xmax,ymin,ymax),cmap='gray_r',gridsize=50,vmin=0,vmax=10)
        xl=np.arange(0,2,.1)
        plt.plot(xl,-4./3.*xl+2,'k-',lw=3,color='b')
        props = dict(boxstyle='square', facecolor='0.8', alpha=0.8)
        plt.text(.1,.1,'CORE',transform=plt.gca().transAxes,fontsize=18,color=colorblind3,bbox=props)
        plt.text(.6,.6,'INFALL',transform=plt.gca().transAxes,fontsize=18,color=colorblind3,bbox=props)        
        #plt.plot(xl,-3./1.2*xl+3,'k-',lw=3)

        # plot low sfr galaxies
        flag = self.lowsfr_flag & self.sampleflag & self.membflag
        plt.plot(x[flag],y[flag],'ko',color=darkblue,mec='k',markersize=10)

        
        plt.axis([xmin,xmax,ymin,ymax])
        
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)
            
    def compare_sfrs(self,shift=None,masscut=None,nbins=20):
        '''
        plot distribution of LCS external and core SFRs

        '''
        if masscut is None:
            masscut = self.masscut
        flag = (self.cat['logSFR'] > -99) & (self.cat['logSFR']-self.cat['logMstar'] > -11.5) & (self.cat['logMstar'] > masscut)
        sfrcore = self.cat['logSFR'][self.cat['membflag'] & flag] 
        sfrext = self.cat['logSFR'][~self.cat['membflag']& flag]
        plt.figure()
        mybins = np.linspace(-2.5,1.5,nbins)

        plt.hist(sfrext,bins=mybins,histtype='step',label='External',lw=3)
        if shift is not None:
            plt.hist(sfrext+np.log10(1-shift),bins=mybins,histtype='step',label='Shifted External',lw=3)
        plt.hist(sfrcore,bins=mybins,histtype='step',label='Core',lw=3)            
        plt.legend()
        plt.xlabel('SFR')
        plt.ylabel('Normalized Counts')
        print('CORE VS EXTERNAL')
        t = lcscommon.ks(sfrcore,sfrext,run_anderson=False)

    def plot_ssfr_sizeratio(self,outfile1='plot.pdf',outfile2='plot.png'):
        '''
        GOAL: 
        * compare sSFR vs sizeratio for galaxies in the paper 1 sample

        INPUT: 
        * outfile1 - usually pdf name for output plot
        * outfile2 - usually png name for output plot

        OUTPUT:
        * plot of sSFR vs sizeratio
        '''

        plt.figure(figsize=(8,6))
        flag = self.sampleflag
        plt.scatter(self.ssfr[flag],self.sizeratio[flag],c=self.cat[BTkey][flag])
        cb = plt.colorbar(label='B/T')
        plt.ylabel('$R_e(24)/R_e(r)$',fontsize=20)
        plt.xlabel('$sSFR / (yr^{-1})$',fontsize=20)
        t = lcscommon.spearmanr(self.sizeratio[flag],self.ssfr[flag])
        print(t)
        plt.savefig(outfile1)
        plt.savefig(outfile2)

    def plot_HIdef(self):
        ''' compare delta sfr and HI def of field, core and infall  '''
        pass

#######################################################################
#######################################################################
########## NEW CLASS: comp_lcs_gsw
#######################################################################
#######################################################################

class comp_lcs_gsw():
    '''
    class that combines the LCS and GSWLC catalogs as self.lcs.xx and self.gsw.xx
    - used to compare LCS with field sample constructed from SDSS with GSWLC SFR and Mstar values

    '''
    def __init__(self,lcs,gsw,minmstar = 10, minssfr = -11.5,cutBT=False):
        self.lcs = lcs
        self.gsw = gsw
        self.masscut = minmstar
        self.ssfrcut = minssfr

        #self.lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut) & (self.lcs.lowsfr_flag)        
        self.cutBT = cutBT
        
        #self.lcs_mass_sfr_flag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        #self.gsw_mass_sfr_flag = (self.gsw.cat['logMstar']> self.masscut)  & (self.gsw.ssfr > self.ssfrcut)        
        self.lcs_mass_sfr_flag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.cat['logSFR'] > get_SFR_cut(self.lcs.cat['logMstar'],BTcut=cutBT)) & (self.lcs.ssfr > self.ssfrcut)

        self.gsw_mass_sfr_flag = (self.gsw.cat['logMstar']> self.masscut)  & (self.gsw.cat['logSFR'] >  get_SFR_cut(self.gsw.cat['logMstar'],BTcut=cutBT))& (self.gsw.ssfr > self.ssfrcut)        

        self.gsw_uvir_flag = (self.gsw.cat['flag_uv']> 0)  & (self.gsw.cat['flag_midir'] > 0)
        self.lcs_uvir_flag = (self.lcs.cat['flag_uv']> 0)  & (self.lcs.cat['flag_midir'] > 0)
        #self.fit_gsw_ms()
        self.get_lowsfr()
        self.get_HIobserved_flag()
    def plot_full_ms(self,apexflag=False):
        ''' plot full ms and show division between passive and SF galaxies '''
        plt.figure()

        plt.hexbin(self.gsw.cat['logMstar'],self.gsw.cat['logSFR'],\
                   bins='log',cmap='gray_r',gridsize=75)
        # show the mass cut
        if not apexflag:
            plt.axvline(x=float(args.minmass),ls=':',color=mycolors[2],lw=2.5,label="Mass Limit")
        plot_sfr_mstar_lines(apexflag=apexflag)

        plt.ylabel(r'$\rm \log_{10}(SFR/(M_\odot/yr))$',fontsize=24)


        plt.xlabel(r'$\rm \log_{10}(M_\star/M_\odot)$',fontsize=24)
        plt.xlim(8.7,11.4)
        plt.legend()        
        plt.savefig('ms_passive_cut.png')
        plt.savefig('ms_passive_cut.pdf')        
        
    def get_lowsfr(self):
        self.gsw_dsfr = self.gsw.cat['logSFR'] - get_MS(self.gsw.cat['logMstar'],BTcut=self.cutBT)
        self.gsw_lowsfr_flag = self.gsw_dsfr < -1*MS_OFFSET
        self.gsw.lowsfr_flag = self.gsw_dsfr < -1*MS_OFFSET 
        self.lcs_dsfr = self.lcs.cat['logSFR'] - get_MS(self.lcs.cat['logMstar'],BTcut=self.cutBT)
        self.lcs_lowsfr_flag = self.lcs_dsfr < -1*MS_OFFSET
        self.lcs.lowsfr_flag = self.lcs_dsfr < -1*MS_OFFSET
        
    def get_HIobserved_flag(self):

        # for field sample, choose conservative limits to make life easier
        # field region
        # 140 < RA  < 230
        # 2 < DEC < 32
        # region covered by ALFALFA
        ramin=140.
        ramax=230.
        decmin=2.
        decmax=32
        
        # calculate the fraction of suppressed galaxies in the field
        self.gsw_HIobs_flag = (self.gsw.cat['RA'] > ramin) & (self.gsw.cat['RA'] < ramax) & \
            (self.gsw.cat['DEC'] > decmin) & (self.gsw.cat['DEC'] < decmax)

        # for cluster, use a finer range

        self.lcs_HIobs_flag = np.ones(len(self.lcs.cat),'bool')

        # bad regions
        # ra > 230 and  17.7 < dec < 24
        bad1_flag = (self.lcs.cat['RA_1'] > 230) & (self.lcs.cat['DEC_1'] > 17.7) & (self.lcs.cat['DEC_1'] < 24)
        # ra > 230 and  dec > 32
        bad2_flag = (self.lcs.cat['RA_1'] > 230) & (self.lcs.cat['DEC_1'] > 32) 
        bad_flag = bad1_flag | bad2_flag
        
        self.lcs_HIobs_flag[bad_flag] = np.zeros(np.sum(bad_flag),'bool')
    def get_HIfrac_SFR_env(self,BTcut=None,plotsingle=True):
        # plot the fraction of normal and suppressed galaxies as a function of core/infall/field
        slimit = -1*MS_OFFSET

        # number in SF core sample with HI observations
        cflag = self.lcs.membflag &  (self.lcs_mass_sfr_flag) & self.lcs_HIobs_flag
        iflag = self.lcs.infallflag &  (self.lcs_mass_sfr_flag) & self.lcs_HIobs_flag
        fflag = (self.gsw_mass_sfr_flag) & self.gsw_HIobs_flag        
        if BTcut is not None:
            cflag = cflag & (self.lcs.cat[BTkey] < BTcut)
            iflag = iflag & (self.lcs.cat[BTkey] < BTcut)            
            fflag = fflag & (self.gsw.cat[BTkey] < BTcut)

        labels = ['Core','Infall','Field']
        flag0 = [cflag,iflag,fflag]
        lowsfrflag = [~self.lcs_lowsfr_flag,~self.lcs_lowsfr_flag,~self.gsw_lowsfr_flag]
        HIflag = [ self.lcs.cat['HIdef_flag'], self.lcs.cat['HIdef_flag'], self.gsw.HIdef['HIdef_flag']]
        

        
        xvars = [2,1,0]
        colors = [darkblue,lightblue,'.5']
        #colors = [mycolors[1],lightblue,'.5']        
        labels = ['$Core$','$Infall$','$Field$']
        orders = [1,3,2]
        lws = [3,4,3]
        alphas = [.8,.8,.8]
        markers = ['o','s','o']
        markersizes = 20*np.ones(3)
        if plotsingle:
            plt.figure()

        for i in xvars:
            for j in range(2):
                    
                # number of low SFR galaxies
                if j == 0:
                    ntot = np.sum(flag0[i] & (lowsfrflag[i]))
                    nlow_HI = np.sum(flag0[i] & (lowsfrflag[i]) & HIflag[i])
                else:
                    ntot = np.sum(flag0[i] & ~(lowsfrflag[i]))
                    nlow_HI = np.sum(flag0[i] & ~(lowsfrflag[i]) & HIflag[i])
                
                y = (nlow_HI/ntot)
                yerrs = binom_conf_interval(nlow_HI,ntot)#lower, upper
            
                yerr = np.zeros((2,1))
                yerr[0] = y-yerrs[0]
                yerr[1] = yerrs[1]-y

                if j == 0:
                    markerfacecolor = colors[i]
                    marker = 'o'
                    mylabel = labels[i]+'$\ norm$'
                else:
                    markerfacecolor = colors[i]
                    marker = 'v'
                    mylabel = labels[i]+'$\ low$'

                print('{}: {:.3f} + {:.3f} - {:.3f}'.format(mylabel,y,yerr[0][0],yerr[1][0]))
                    
                plt.errorbar(np.array(xvars[i]),np.array(y),yerr=yerr,\
                             color=colors[i],alpha=alphas[i],zorder=orders[i],\
                             markersize=markersizes[i],fmt=marker,label=mylabel,markerfacecolor=markerfacecolor)
        if plotsingle:
            plt.xticks(np.arange(0,3),[r'$\rm Field$',r'$\rm Infall$',r'$\rm Core$'],fontsize=20)
            #plt.xlabel('$Environment$',fontsize=20)
            plt.ylabel(r'$\rm Frac\ of \ SF \ Gals \ with \ HI \ Detection $',fontsize=20)
            plt.legend([r'$\rm Normal \ SFR $',r'$\rm Low \ SFR$'],fontsize=16,markerscale=.8)
        plt.xlim(-0.4,2.4)
            

        pass

    def write_tables_for_SFR_sim(self):
        '''  write out field and LCS core samples to use in SFR simulation '''

        # LCS sample
        # above ssfr and mstar limits, and core galaxy (membflag)
        lcs = self.lcs.cat[self.lcs_mass_sfr_flag & self.lcs.membflag]

        # create new table with Mstar and SFR
        newtab = Table([lcs['logMstar'],lcs['logSFR'],lcs[BTkey]],names=['logMstar','logSFR','BT'])
        
        # write out table with Mstar and SFR
        if args.cutBT:
            outfile = os.path.join(homedir,'research/LCS/tables/','lcs-sfr-sim-BTcut.fits')
        else:
            outfile = os.path.join(homedir,'research/LCS/tables/','lcs-sfr-sim.fits')
        newtab.write(outfile,overwrite=True,format='fits')

        
        # FIELD SAMPLE
        # above ssfr and mstar limits
        gsw = self.gsw.cat[self.gsw_mass_sfr_flag]
        

        # create new table with Mstar and SFR
        newtab = Table([gsw['logMstar'],gsw['logSFR'],gsw[BTkey]],names=['logMstar','logSFR','BT'])
        # write out table with Mstar and SFR
        if args.cutBT:
            outfile = os.path.join(homedir,'research/LCS/tables/','gsw-sfr-sim-BTcut.fits')
        else:
            outfile = os.path.join(homedir,'research/LCS/tables/','gsw-sfr-sim.fits')
        newtab.write(outfile,overwrite=True,format='fits')


        # REPEAT BUT INCLUDE ALL SFRS IN CASE WE WANT TO MODEL PASSIVE POPULATION AS WELL
        # JUST CUT ON STELLAR MASS BUT NOT SFR
        
        # LCS sample
        # above ssfr and mstar limits, and core galaxy (membflag)
        flag = self.lcs.membflag & (self.lcs.cat['logMstar'] > float(args.minmass))
        lcs = self.lcs.cat[flag]

        # create new table with Mstar and SFR
        newtab = Table([lcs['logMstar'],lcs['logSFR'],lcs[BTkey]],names=['logMstar','logSFR','BT'])
        
        # write out table with Mstar and SFR
        if args.cutBT:
            outfile = os.path.join(homedir,'research/LCS/tables/','lcs-allsfr-sim-BTcut.fits')
        else:
            outfile = os.path.join(homedir,'research/LCS/tables/','lcs-allsfr-sim.fits')
        newtab.write(outfile,overwrite=True,format='fits')

        
        # FIELD SAMPLE
        # above ssfr and mstar limits
        flag = (self.gsw.cat['logMstar'] > float(args.minmass))        
        gsw = self.gsw.cat[flag]
        

        # create new table with Mstar and SFR
        newtab = Table([gsw['logMstar'],gsw['logSFR'],gsw[BTkey]],names=['logMstar','logSFR','BT'])
        # write out table with Mstar and SFR
        if args.cutBT:
            outfile = os.path.join(homedir,'research/LCS/tables/','gsw-allsfr-sim-BTcut.fits')
        else:
            outfile = os.path.join(homedir,'research/LCS/tables/','gsw-allsfr-sim.fits')
        newtab.write(outfile,overwrite=True,format='fits')

        
    def fit_gsw_ms(self):
        #self.gsw.fit_MS(flag=self.gsw_mass_sfr_flag)
        massFlag = (self.gsw.cat['logMstar'] > 8.75) & (self.gsw.cat['logMstar'] < 10.)
        flag =  self.gsw_mass_sfr_flag & massFlag #& (self.gsw.cat[BTkey]< .5) 
        self.gsw.fit_MS(flag=flag,plotFlag=False)#flag=
        print('fitting MS with parabola')
        self.gsw.fit_MS(flag=flag,plotFlag=False,linearFit=False)#flag=        
    def plot_sfr_mstar(self,lcsflag=None,label='Core',outfile1=None,outfile2=None,coreflag=True,massmatch=True,hexbinflag=False,marker2='o',alpha1='0.1',plotMS=True,plotlegend=True):
        '''
        PURPOSE:
        * compares ssfr vs mstar of lcs and gswlc field samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:
        * lcsflag 
          - if None, this will select LCS members
          - you can use this to select other slices of the LCS sample, like external, or uv/ir detected
        
        * label
          - name of the LCS subsample being plotted
          - default is 'LCS core'

        * outfile1, outfile2
          - figure name to save as
          - I'm using two to save a png and pdf version

        * massmatch
          - set to True to draw a mass-matched sample from GSWLC field sample

        * hexbinflag
          - set to True to use hexbin to plot field sample
          - do this if NOT drawing mass-matched sample b/c number of points is large


        OUTPUT:
        * creates a plot of sfr vs mstar
        * creates histograms above x and y axes to compare two samples
        * prints out the KS test statistics
        '''
        
        if lcsflag is None:
            #lcsflag = self.lcs.cat['membflag']
            lcsflag = self.lcs.membflag            
        
        flag1 = lcsflag &  self.lcs_mass_sfr_flag
        # removing field1 cut because we are now using Tempel catalog that only
        # includes galaxies in halo masses logM < 12.5
        flag2 = self.gsw_mass_sfr_flag  #& self.gsw.field1
        print('number in lcs sample = ',sum(flag1))
        print('number in gsw sample = ',sum(flag2))
        # GSWLC sample
        x1 = self.gsw.cat['logMstar'][flag2]
        y1 = self.gsw.cat['logSFR'][flag2]
        z1 = self.gsw.cat['Z_1'][flag2]
        # LCS sample (forgive the switch in indices)
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1]
        z2 = self.lcs.cat['Z_1'][flag1]
        # get indices for mass-matched gswlc sample
        if massmatch:
            #keep_indices = mass_match(x2,x1,inputZ=z2,compZ=z1,dz=.002)
            keep_indices = mass_match(x2,x1,3124,nmatch=NMASSMATCH)
            #print("Printing return from mass_match: ",keep_indices)
            # if keep_indices == False
            # remove redshift constraint
            if (len(keep_indices) == 1):
                if (keep_indices == False):
                    print("WARNING: Removing the redshift constraint from the mass-matched sample")
                    keep_indices = mass_match(x2,x1,74832,nmatch=NMASSMATCH)
                
            x1 = x1[keep_indices]
            y1 = y1[keep_indices]
        
        if coreflag:
            color2=darkblue
        else:
            color2=lightblue
        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'GSWLC',label,'sfr-mstar-gswlc-field.pdf',ymin=-1.5,ymax=1.6,xmin=9.5,xmax=11.5,nhistbin=10,ylabel=r'$\rm \log_{10}(SFR)$',contourflag=False,alphagray=.1,hexbinflag=hexbinflag,color2=color2,color1='0.2',alpha1=1,ssfrlimit=-11.5,marker2=marker2)
        # add marker to figure to show galaxies with size measurements

        #self.plot_lcs_size_sample(ax1,memb=lcsmemb,infall=lcsinfall,ssfrflag=False)
        #ax1.legend(loc='upper left')
        if not hexbinflag and plotlegend:
            ax1.legend(loc='upper left')
        #plot_BV_MS(ax1)
        if plotMS:
            #self.gsw.plot_MS(ax1)
            plot_sfr_mstar_lines(ax=ax1)
        if plotlegend:
            ax1.legend(loc='lower right')
        plt.subplots_adjust(left=.15)
        #mybins=np.linspace(min(x2),max(x2),8)
        #ybin,xbin,binnumb = binned_statistic(x2,y2,statistic='median',bins=mybins)
        #yerr,xbin,binnumb = binned_statistic(x2,y2,statistic='std',bins=mybins)
        #nyerr,xbin,binnumb = binned_statistic(x2,y2,statistic='count',bins=mybins)            
        #yerr = yerr/np.sqrt(nyerr)
        #dbin = xbin[1]-xbin[0]
        #ax1.plot(xbin[:-1]+0.5*dbin,ybin,color=color2,lw=3)
        #ax1.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=color2,alpha=.4)              
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-sfms.png')
        else:
            plt.savefig(outfile2)
        return ax1
    def plot_ssfr_mstar(self,lcsflag=None,outfile1=None,outfile2=None,label='Core',nbins=20,coreflag=True,massmatch=True,hexbinflag=True,lcsmemb=False,lcsinfall=False):
        """
        OVERVIEW:
        * compares ssfr vs mstar of lcs and gswlc field samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:
        * lcsflag 
        - if None, this will select LCS members
        - you can use this to select other slices of the LCS sample, like external
        
        * label
        - name of the LCS subsample being plotted
        - default is 'LCS core'

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        flag1 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag2 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        
        x1 = self.gsw.cat['logMstar'][flag2]
        y1 = self.gsw.ssfr[flag2]
        z1 = self.gsw.cat['Z_1'][flag2]

        # LCS sample (forgive the switch in indices)
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1] - self.lcs.cat['logMstar'][flag1] 
        z2 = self.lcs.cat['Z_1'][flag1]
        
        if coreflag:
            color2=darkblue
        else:
            color2=lightblue
            
        # get indices for mass-matched gswlc sample
        if massmatch:
            keep_indices = mass_match(x2,x1,inputZ=z2,compZ=z1,dz=.002)
            x1 = x1[keep_indices]
            y1 = y1[keep_indices]
        
            print('AFTER MASS MATCHING')
            print('number of gswlc = ',len(x1))
            print('number of lcs = ',len(x2))

        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'GSWLC',label,'sfr-mstar-gswlc-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.5,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(sSFR)$',\
                                contourflag=False,alphagray=.15,hexbinflag=hexbinflag,color1='0.5',color2=color2,alpha1=1)

        #self.plot_lcs_size_sample(ax1,memb=lcsmemb,infall=lcsinfall,ssfrflag=True)
        ax1.legend(loc='upper left')

        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)

    def plot_sfr_mstar_lcs(self,outfile1=None,outfile2=None,nbins=10):
        """
        OVERVIEW:
        * compares ssfr vs mstar within lcs samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        
        baseflag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag1 = baseflag & self.lcs.cat['membflag'] 
        flag2 = baseflag & ~self.lcs.cat['membflag']  & (self.lcs.cat['DELTA_V'] < 3.)
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        
        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'Core','Infall','sfr-mstar-lcs-core-field.pdf',ymin=-2,ymax=1.5,xmin=9.5,xmax=11.25,nhistbin=nbins,ylabel='$\log_{10}(SFR)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True,ssfrlimit=-11.5)

        self.plot_lcs_size_sample(ax1,memb=True,infall=True)
        ax1.legend(loc='upper left')
        #plot_BV_MS(ax1)
        self.gsw.plot_MS(ax1)
        ax1.legend(loc='lower right')
        plt.subplots_adjust(left=.15)
        
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-external-sfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-external-sfrmstar.png')
        else:
            plt.savefig(outfile2)

    def plot_ssfr_mstar_lcs(self,outfile1=None,outfile2=None,nbins=20):
        """
        OVERVIEW:
        * compares ssfr vs mstar within lcs samples
        * applies stellar mass and ssfr cuts that are specified at beginning of program

        INPUT:

        * outfile1, outfile2
        - figure name to save as
        - I'm using two to save a png and pdf version

        OUTPUT:
        * creates a plot of ssfr vs mstar
        * creates histograms above x and y axes to compare two samples

        """
        
        baseflag = (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        flag1 = baseflag & self.lcs.cat['membflag'] 
        flag2 = baseflag & ~self.lcs.cat['membflag']  & (self.lcs.cat['DELTA_V'] < 3.)
        print('number in core sample = ',sum(flag1))
        print('number in external sample = ',sum(flag2))
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.ssfr[flag1]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.ssfr[flag2]


        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'Core','Infall','sfr-mstar-lcs-core-field.pdf',ymin=-11.6,ymax=-8.75,xmin=9.5,xmax=11.5,nhistbin=nbins,ylabel='$\log_{10}(sSFR)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True)


        #self.plot_lcs_size_sample(ax1,memb=True,infall=True,ssfrflag=True)
        ax1.legend(loc='upper left')
        
        plt.subplots_adjust(left=.15)
        
        if outfile1 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.pdf')
        else:
            plt.savefig(outfile1)
        if outfile2 is None:
            plt.savefig(homedir+'/research/LCS/plots/lcscore-gsw-ssfrmstar.png')
        else:
            plt.savefig(outfile2)
    def plot_lcs_size_sample(self,ax,memb=True,infall=True,ssfrflag=False):
        cmemb = colorblind1
        cmemb = 'darkmagenta'
        cinfall = lightblue
        cinfall = 'blueviolet'
        cinfall = 'darkmagenta'
        if ssfrflag:
            y = self.lcs.cat['logSFR'] - self.lcs.cat['logMstar']
        else:
            y = self.lcs.cat['logSFR']
        baseflag = self.lcs.sampleflag& (self.lcs.cat['logMstar'] > self.masscut)
        if memb:
            flag = self.lcs.sampleflag & self.lcs.cat['membflag']  & (self.lcs.cat['logMstar'] > self.masscut)
            #ax.plot(self.lcs.cat['logMstar'][flag],y[flag],'ks',color=cmemb,alpha=.5,markersize=8,label='LCS memb w/size ('+str(sum(flag))+')')
        if infall:
            flag = self.lcs.sampleflag & ~self.lcs.cat['membflag']  & (self.lcs.cat['DELTA_V'] < 3.) & (self.lcs.cat['logMstar'] > self.masscut)
            #ax.plot(self.lcs.cat['logMstar'][flag],y[flag],'k^',color=cinfall,alpha=.5,markersize=10,label='LCS infall w/size ('+str(sum(flag))+')')
        flag = baseflag& (self.lcs.cat['DELTA_V'] < 3.)
        ax.plot(self.lcs.cat['logMstar'][flag],y[flag],'ks',alpha=.6,zorder=1,markersize=10,lw=2,label='LCS w/size ('+str(sum(flag))+')')
    def plot_dsfr_hist(self,nbins=15,outfile1=None,outfile2=None,massmatch=False):
        #lcsflag = self.lcs.cat['membflag']
        
        flag1 = self.lcs.membflag &  self.lcs_mass_sfr_flag #(self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        # removing field1 cut because we are now using Tempel catalog that only
        # includes galaxies in halo massses logM < 12.5
        flag2 = self.gsw_mass_sfr_flag #(self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        # GSWLC
        if massmatch:

            keep_indices = mass_match(self.lcs.cat['logMstar'][self.lcs_mass_sfr_flag],\
                                      self.gsw.cat['logMstar'][self.gsw_mass_sfr_flag],\
                                      nmatch=NMASSMATCH,seed=559)

            x1 = self.gsw.cat['logMstar'][self.gsw_mass_sfr_flag][keep_indices]
            y1 = self.gsw.cat['logSFR'][self.gsw_mass_sfr_flag][keep_indices]
        else:
            x1 = self.gsw.cat['logMstar'][flag2]
            y1 = self.gsw.cat['logSFR'][flag2]

        #dsfr1 = y1-get_BV_MS(x1)
        dsfr1 = y1-get_MS(x1,BTcut=self.cutBT)
        #print("min dsfr for field sample = ",np.min(dsfr1))
        #tflag = dsfr1 < -1.5
        #print("logmstar of low dsfr = ",x1[tflag])
        #print("logsfr of low dsfr = ",y1[tflag])
        #print("dsfr of low dsfr = ",dsfr1[tflag])
        
        
        #LCS core
        x2 = self.lcs.cat['logMstar'][flag1]
        y2 = self.lcs.cat['logSFR'][flag1]
        #dsfr2 = y2-get_BV_MS(x2)
        dsfr2 = y2-get_MS(x2,BTcut=self.cutBT)
        #LCS infall
        #(abs(self.cat['DELTA_V']) < 3) & ~self.membflag        
        #lcsflag = ~self.lcs.cat['membflag'] & (np.abs(self.lcs.cat['DELTA_V']) < 3.)
        flag3 = self.lcs.infallflag & self.lcs_mass_sfr_flag
        
        #print('number of infall galaxies, first selection = ',sum(flag3))
        #flag3 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        #print('number of infall galaxies, second selection = ',sum(flag3))              

        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        #dsfr3 = y3-get_BV_MS(x3)
        dsfr3 = y3-get_MS(x3,BTcut=self.cutBT)
        plt.figure(figsize=(8,6))

        mybins = np.linspace(-1.6,1.6,nbins)
        delta_bin = mybins[1]-mybins[0]
        mybins = mybins + 0.5*delta_bin
        dsfrs = [dsfr1,dsfr2,dsfr3]
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        hatches = ['/','\\','|']
        for i in range(len(dsfrs)):
            plt.hist(dsfrs[i],bins=mybins,color=colors[i],density=True,\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=orders[i])#hatch=hatches[i])
            plt.hist(dsfrs[i],bins=mybins,color=colors[i],density=True,\
                     histtype='step',lw=lws[i],zorder=orders[i],label=labels[i])#hatch=hatches[i])
            # calculate median and error from bootstrap resampling

            thismedian = np.median(dsfrs[i])
            # error from bootstrap resampling
            lower_median, upper_median = get_bootstrap_confint(dsfrs[i])
            print()
            print('{}: median = {:.4f}-{:.4f}+{:.4f} (bootstrap)'.format(labels[i],thismedian,thismedian-lower_median,upper_median-thismedian))
            print('{}: mean, std, std_err = {:.4f},{:.4f},{:.4f} '.format(labels[i],np.mean(dsfrs[i]),np.std(dsfrs[i]),np.std(dsfrs[i])/np.sqrt(len(dsfrs[i]))))
            print('')
                  
        #plt.xlabel('$ \log_{10}SFR - \log_{10}SFR_{MS} \ (M_\odot/yr) $',fontsize=20)
        plt.xlabel(r'$\rm \Delta \log_{10}SFR $',fontsize=22)
        plt.ylabel(r'$\rm Normalized \ Distribution$',fontsize=22)
        #plt.yticks([],[])
        # add range considered "normal" SF
        #plt.axvline(x=MS_OFFSET,ls='--',color='b')
        plt.axvline(x=-1*MS_OFFSET,ls='--',color='b')        
        plt.legend()
        if args.cutBT:
            plt.title(r'$ B/T \le 0.3$',fontsize=22)
        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)

        print('KS STATISTICS: FIELD VS CORE')
        print(ks_2samp(dsfr1,dsfr2))
        print(anderson_ksamp([dsfr1,dsfr2]))
        print('')
        print('KS STATISTICS: FIELD VS INFALL')
        print(ks_2samp(dsfr1,dsfr3))
        print(anderson_ksamp([dsfr1,dsfr3]))
        print('')              
        print('KS STATISTICS: CORE VS INFALL')
        print(ks_2samp(dsfr2,dsfr3))
        print(anderson_ksamp([dsfr2,dsfr3]))


    def plot_sfr_morph_hist(self,nbins=15,outfile1=None,outfile2=None):



        # B/T

        # ng - sersic index


        # for LCS, we have NSA, NUV-r
        # 
        lcsflag = self.lcs.cat['membflag']        
        flag1 = lcsflag &  self.lcs.sfsample 
        # removing field1 cut because we are now using Tempel catalog that only
        # includes galaxies in halo masses logM < 12.5
        flag = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        # GSWLC
        #LCS infall
        lcsflag = ~self.lcs.cat['membflag'] & (self.lcs.cat['DELTA_V'] < 3.)
        flag3 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)

        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        #dsfr3 = y3-get_BV_MS(x3)
        dsfr3 = y3-get_MS(x3,BTcut=self.cutBT)        
        plt.figure(figsize=(8,6))

        mybins = np.linspace(-1.5,1.5,nbins)
        delta_bin = mybins[1]-mybins[0]
        mybins = mybins + 0.5*delta_bin
        dsfrs = [dsfr1,dsfr2,dsfr3]
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        hatches = ['/','\\','|']
        for i in range(len(dsfrs)):
            plt.hist(dsfrs[i],bins=mybins,color=colors[i],density=True,\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=orders[i])#hatch=hatches[i])
            plt.hist(dsfrs[i],bins=mybins,color=colors[i],density=True,\
                     histtype='step',lw=lws[i],zorder=orders[i],label=labels[i])#hatch=hatches[i])
            
        plt.xlabel('$ \log_{10}SFR - \log_{10}SFR_{MS} \ (M_\odot/yr) $',fontsize=20)
        plt.ylabel('$Normalized \ Distribution$',fontsize=20)
        plt.legend()
        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)
        

    def plot_dsfr_sizeratio(self,nbins=15,outfile1=None,outfile2=None,sampleflag=None,maxmass=None,apexflag=False,errorbar=False):
        if sampleflag is None:
            sampleflag = self.lcs.sampleflag
        if maxmass is not None:
            sampleflag = sampleflag & (self.lcs.cat['logMstar'] < maxmass)
        print('number in sampleflag = ',sum(sampleflag),len(sampleflag))
        print('number in membflag = ',sum(self.lcs.membflag),len(self.lcs.membflag))
        lcsflag = self.lcs.membflag & sampleflag        
        print('number in both = ',sum(lcsflag))
        print('number in both and in sfr/mstar cut = ',sum(lcsflag & self.lcs_mass_sfr_flag))

        flag2 = lcsflag &  self.lcs_mass_sfr_flag
        #LCS core
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        z2 = self.lcs.sizeratio[flag2]        
        #dsfr2 = y2-get_BV_MS(x2)
        dsfr2 = y2-get_MS(x2,BTcut=self.cutBT)
        print('fraction of core with dsfr below 0.3dex = {:.3f} ({:d}/{:d})'.format(sum(dsfr2 < -0.3)/len(dsfr2),sum(dsfr2 < -0.3),len(dsfr2)))
        
        #LCS infall
        lcsflag = sampleflag  &self.lcs.infallflag
        flag3 = lcsflag &  self.lcs_mass_sfr_flag
        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        z3 = self.lcs.sizeratio[flag3]
        #dsfr3 = y3-get_BV_MS(x3)
        dsfr3 = y3-get_MS(x3,BTcut=self.cutBT)
        print('fraction of core with dsfr below 0.3dex = {:.3f} ({:d}/{:d})'.format(sum(dsfr3 < -0.3)/len(dsfr3),sum(dsfr3 < -0.3),len(dsfr3)) )       


        # make figure
        #plt.figure(figsize=(8,6))
        sizes = [z2,z3]
        dsfrs = [dsfr2,dsfr3]
        colors = [darkblue,lightblue]
        labels = ['Core ({})'.format(sum(flag2)),'Infall ({})'.format(sum(flag3))]
        hatches = ['/','\\','|']
        #for i in range(len(dsfrs)):
        #    plt.plot(sizes[i],dsfrs[i],'bo',c=colors[i],label=labels[i])
            
        #plt.ylabel('$ SFR - SFR_{MS}(M_\star) \ (M_\odot/yr) $',fontsize=20)
        #plt.xlabel('$R_{24}/R_d$',fontsize=20)
        #plt.legend()
        #plt.axhline(y=0,color='k')

        plt.figure()
        nbins=12
        ylab = '$\log_{10}(SFR)-\log_{10}(SFR_{MS})  \ (M_\odot/yr)$'
        ylab = '$\Delta \log_{10}(SFR) $'
        ax1,ax2,ax3 = colormass(z2,dsfr2,z3,dsfr3,'Core','Infall','temp.pdf',ymin=-1.1,ymax=1.,xmin=-.05,xmax=2,nhistbin=nbins,xlabel='$R_{24}/R_d$',ylabel=ylab,contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True)
        var1 = z2.tolist()+z3.tolist()
        var2 = dsfrs[0].tolist()+dsfrs[1].tolist()
        print('')
        print('Spearman rank test, dsfr, sizeratio')
        t = lcscommon.spearmanr(var1,var2)
        print(t)

        if not apexflag:
            # add line to show sb limit from LCS size measurements
            size=np.linspace(0,2)
            sb = .022
            sfr = sb*(4*np.pi*size**2)
            ax1.plot(size,np.log10(sfr),'k--',label='SB limit')
        ax1.axhline(y=MS_OFFSET,ls=':',color='0.5')
        ax1.axhline(y=-1*MS_OFFSET,ls=':',color='0.5')

        if not apexflag:
            # plot all galfit results
            baseflag = self.lcs_mass_sfr_flag & ~sampleflag #& ~self.lcs.cat['agnflag'] 
            flag4 = self.lcs.cat['membflag'] &  baseflag
            x4 = self.lcs.cat['logMstar'][flag4]
            y4 = self.lcs.cat['logSFR'][flag4]
            z4 = self.lcs.sizeratio[flag4]        
            #dsfr4 = y4-get_BV_MS(x4)
            dsfr4 = y4-get_MS(x4,BTcut=self.cutBT)
            ax1.plot(z4,dsfr4,'kx',c=darkblue,markersize=10)


            # plot all galfit results
            flag4 = self.lcs.infallflag&   self.lcs_mass_sfr_flag & ~sampleflag #& ~self.lcs.cat['agnflag'] 
            x4 = self.lcs.cat['logMstar'][flag4]
            y4 = self.lcs.cat['logSFR'][flag4]
            z4 = self.lcs.sizeratio[flag4]        
            #dsfr4 = y4-get_BV_MS(x4)
            dsfr4 = y4-get_MS(x4,BTcut=self.cutBT)        
            ax1.plot(z4,dsfr4,'kx',markersize=10,c=lightblue)
        
        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)

            
        ### COMPARE MEAN SKEW AND KURTOSIS OF THE SIZE and SFR DISTRIBUTIONS

        print('#################################################')
        print('size distributions')
        print('\tMean Core     = {:.2f}; Infall = {:.2f}'.format(np.mean(z2),np.mean(z3)))
        print('\tSkew Core     = {:.2f}; Infall = {:.2f}'.format(st.skew(z2),st.skew(z3)))
        nboot = 100
        # get conf intervale
        coreskew_low,coreskew_up = get_bootstrap_confint(z2,bootfunc=st.skew)
        infallskew_low,infallskew_up = get_bootstrap_confint(z3,bootfunc=st.skew)
        print('Core skew:',coreskew_low,coreskew_up)
        print('\tSkew Core     = {:.2f}+{:.2f}-{:.2f}; Infall = {:.2f}+{:.2f}-{:.2f}'.format(st.skew(z2),coreskew_up-st.skew(z2),st.skew(z2)-coreskew_low,st.skew(z3),infallskew_up-st.skew(z3),st.skew(z3)-infallskew_low))
        print('\tKurtosis Core = {:.2f}; Infall = {:.2f}'.format(st.kurtosis(z2),st.kurtosis(z3)))


        # x2 and x3 are the core and infall mass
        print('#################################################')
        print('dSFR distributions')
        print('\tMean Core     = {:.2f}; Infall = {:.2f}'.format(np.mean(dsfr2),np.mean(dsfr3)))
        print('\tSTD Core      = {:.2f}; Infall = {:.2f}'.format(np.std(dsfr2),np.std(dsfr3)))
        print('\tERR MEAN Core = {:.3f}; Infall = {:.3f}'.format(np.std(dsfr2)/np.sqrt(len(dsfr2)),np.std(dsfr3)/np.sqrt(len(dsfr3))))                
        print('\tSkew Core     = {:.2f}; Infall = {:.2f}'.format(st.skew(dsfr2),st.skew(dsfr3)))
        coreskew_low,coreskew_up = get_bootstrap_confint(dsfr2,bootfunc=st.skew)
        infallskew_low,infallskew_up = get_bootstrap_confint(dsfr3,bootfunc=st.skew)
        print('Core skew:',coreskew_low,coreskew_up)
        print('\tSkew Core     = {:.2f}+{:.2f}-{:.2f}; Infall = {:.2f}+{:.2f}-{:.2f}'.format(st.skew(dsfr2),coreskew_up-st.skew(dsfr2),st.skew(dsfr2)-coreskew_low,st.skew(dsfr3),infallskew_up-st.skew(dsfr3),st.skew(dsfr3)-infallskew_low))
        print('\tKurtosis Core = {:.2f}; Infall = {:.2f}'.format(st.kurtosis(dsfr2),st.kurtosis(dsfr3)))
        
        
        return ax1,ax2,ax3

    def plot_dsfr_HIdef(self,nbins=15,outfile1=None,outfile2=None):
        lcsflag = self.lcs.cat['membflag'] & self.lcs.sampleflag & self.lcs.cat['HIflag']

        flag2 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut) 
        #LCS core
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        z2 = self.lcs.cat['HIDef'][flag2]        
        #dsfr2 = y2-get_BV_MS(x2)
        dsfr2 = y2-get_MS(x2,BTcut=self.cutBT)
        
        #LCS infall
        lcsflag = self.lcs.sampleflag & self.lcs.cat['HIflag'] & ~self.lcs.cat['membflag'] & (self.lcs.cat['DELTA_V'] < 3.)
        flag3 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        x3 = self.lcs.cat['logMstar'][flag3]
        y3 = self.lcs.cat['logSFR'][flag3]
        z3 = self.lcs.cat['HIDef'][flag3]
        #dsfr3 = y3-get_BV_MS(x3)
        dsfr3 = y3-get_MS(x3,BTcut=self.cutBT)

        # make figure
        #plt.figure(figsize=(8,6))
        sizes = [z2,z3]
        dsfrs = [dsfr2,dsfr3]
        colors = [darkblue,lightblue]
        labels = ['LCS Core ({})'.format(sum(flag2)),'Infall ({})'.format(sum(flag3))]
        hatches = ['/','\\','|']
        #for i in range(len(dsfrs)):
        #    plt.plot(sizes[i],dsfrs[i],'bo',c=colors[i],label=labels[i])
            
        #plt.ylabel('$ SFR - SFR_{MS}(M_\star) \ (M_\odot/yr) $',fontsize=20)
        #plt.xlabel('$R_{24}/R_d$',fontsize=20)
        #plt.legend()
        #plt.axhline(y=0,color='k')

        plt.figure()
        nbins=12
        ax1,ax2,ax3 = colormass(z2,dsfr2,z3,dsfr3,'Core','Infall','temp.pdf',ymin=-1,ymax=1.,xmin=-.05,xmax=2,nhistbin=nbins,xlabel='$HI \ Deficiency$',ylabel='$\log_{10}(SFR)-\log_{10}(SFR_{MS})  \ (M_\odot/yr)$',contourflag=False,alphagray=.8,alpha1=1,color1=darkblue,lcsflag=True)
        var1 = z2.tolist()+z3.tolist()
        var2 = dsfrs[0].tolist()+dsfrs[1].tolist()
        t = lcscommon.spearmanr(var1,var2)
        print(t)

        if outfile1 is not None:
            plt.savefig(outfile1)
        if outfile2 is not None:
            plt.savefig(outfile2)
            
    def plot_phasespace_dsfr(self):
        ''' plot phase space diagram, and mark galaxies with SFR < (MS - 1.5sigma)   '''

        # maybe try size that scales with size ratio
        pass

    def print_lowssfr_nsaids(self,lcsflag=None,ssfrmin=None,ssfrmax=-11):
        if ssfrmin is not None:
            ssfrmin=ssfrmin
        else:
            ssfrmin = -11.5
        lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  &\
            (self.lcs.ssfr > ssfrmin) & (self.lcs.ssfr < ssfrmax)

        
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        flag = lcsflag & lowssfr_flag        
        nsaids = self.lcs.cat['NSAID'][flag]
        for n in nsaids:
            print(n)
            
    def get_legacy_images(self,lcsflag=None,ssfrmax=-11,ssfrmin=-11.5):
        if ssfrmin is not None:
            ssfrmin=ssfrmin
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']

        lowssfr_flag = (self.lcs.cat['logMstar']> self.masscut)  & \
            (self.lcs.ssfr > ssfrmin) & (self.lcs.ssfr < ssfrmax)

        flag = lowssfr_flag & lcsflag
        ids = np.arange(len(lowssfr_flag))[flag]
        for i in ids:
            # open figure
            plt.figure()
            # get legacy image
            #print(i,self.lcs.cat['RA_1'][i],self.lcs.cat['DEC_1'][i])
            if np.isnan(self.lcs.cat['RA_1'][i]) |  np.isnan(self.lcs.cat['DEC_1'][i]):
                continue
            d, w = getlegacy(self.lcs.cat['RA_1'][i],self.lcs.cat['DEC_1'][i])
            ssfr = self.lcs.cat['logSFR'][i] - self.lcs.cat['logMstar'][i]

            s = "NSAID {}, sSFR = {:.1f}, dSFR={:.1f}, BT={:.1f}".format(self.lcs.cat['NSAID'][i],ssfr,self.lcs_dsfr[i],self.lcs.cat[BTkey][i])
            #except:
            #    s = "NSID {0}, dSFR={1:.1f}".format(self.lcs.cat['NSAID'][i],self.lcs_dsfr[i],)
            plt.title(s,fontsize=8)
    def get_legacy_images_1flag(self,flag=None,sortbymass=False,sortby=None,nrow=7,ncol=7,titleflag=True,wspace=.02,hspace=.02,apexflag=False):
        if flag is None:
            print('Please provide a flag')
            return


        ids = np.arange(len(flag))[flag]
        if sortbymass:
            mass = self.lcs.cat['logMstar'][flag]
            sorted_indices = np.argsort(mass)
            ids = ids[sorted_indices]

        # allow for sorting by other parameters
        if sortby is not None:
            sorted_indices = np.argsort(sortby)
            ids = ids[sorted_indices]
            
        plt.figure(figsize=(14,14))
        plt.subplots_adjust(hspace=hspace,wspace=wspace)
        nplot=1
        #nrow = 7
        #ncol = 7
        for i in ids:
                   
            # open figure
            if nplot > nrow*ncol:
                print('ran out of space')
                return
            plt.subplot(nrow,ncol,nplot)
            # get legacy image
            #print(i,self.lcs.cat['RA_1'][i],self.lcs.cat['DEC_1'][i])
            if np.isnan(self.lcs.cat['RA_1'][i]) |  np.isnan(self.lcs.cat['DEC_1'][i]):
                continue
            d, w = getlegacy(self.lcs.cat['RA_1'][i],self.lcs.cat['DEC_1'][i])
            ssfr = self.lcs.cat['logSFR'][i] - self.lcs.cat['logMstar'][i]

            s = "NSID {}, sSFR = {:.1f}, dSFR={:.1f}, BT={:.1f}".format(self.lcs.cat['NSAID'][i],ssfr,self.lcs_dsfr[i],self.lcs.cat[BTkey][i])
            s = '$ \Delta SFR={:.1f}, \ BT={:.1f}$'.format(self.lcs_dsfr[i],self.lcs.cat[BTkey][i])
            #except:
            #    s = "NSID {0}, dSFR={1:.1f}".format(self.lcs.cat['NSAID'][i],self.lcs_dsfr[i],)
            plt.xticks([])
            plt.yticks([])


            # add circle for apex
            if apexflag:
                plt.text(0,.9,'{},z={:.3f}'.format(i,self.lcs.cat['Z_1'][i]),fontsize=12,transform=plt.gca().transAxes,color='w')
                cir = Circle((30,30),radius=27/2,color='None',ec='0.5')
                plt.gca().add_patch(cir)
                
            if titleflag:
                plt.title(s,fontsize=8)
            nplot += 1
    def compare_BT(self,nbins=15,xmax=.3):
        
        flag1 = self.lcs.membflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)        
        coreBT = self.lcs.cat[BTkey][flag1]
        
        flag2 = self.lcs.infallflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        infallBT = self.lcs.cat[BTkey][flag2]
        
        flag3 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        gswBT = self.gsw.cat[BTkey][flag3]
        
        plt.figure()
        mybins = np.linspace(0,xmax,nbins)
        delta_bin = mybins[1]-mybins[0]
        mybins = mybins + 0.5*delta_bin
        xvars = [gswBT,coreBT,infallBT]
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        hatches = ['/','\\','|']
        for i in range(len(xvars)):
            plt.hist(xvars[i],bins=mybins,color=colors[i],density=True,\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=orders[i])#hatch=hatches[i])
            plt.hist(xvars[i],bins=mybins,color=colors[i],density=True,\
                     histtype='step',lw=lws[i],zorder=orders[i],label=labels[i])#hatch=hatches[i])
            plt.axvline(x=np.median(xvars[i]),color=colors[i],ls='--')
        plt.xlabel('$B/T$',fontsize=20)
        plt.ylabel('$Normalized \ Distribution$',fontsize=20)
        
        plt.legend()
        print('comparing BT: Core vs Infall')
        t = ks_2samp(coreBT,infallBT)
        print(t)
        print()
        print('comparing BT: Core vs Field')
        t = ks_2samp(coreBT,gswBT)
        print(t)
        print()        
        print('comparing BT: Field vs Infall')
        t = ks_2samp(gswBT,infallBT)
        print(t)
        print()
        print('number in core and infall = {}, {}'.format(sum(flag1),sum(flag2)))
    def compare_morph(self,nbins=10,xmax=.4,coreonly=False):
        '''what is this for?   '''
        plt.figure(figsize=(10,6))
        plt.subplots_adjust(wspace=.1,hspace=.4)
        # 2x2 figure
        nrow=2
        ncol=2
        if coreonly:
            lcsflag = self.lcs.membflag #| self.lcs.infallflag
            labels = ['Field','Core']                        
        else:
            lcsflag = self.lcs.membflag | self.lcs.infallflag
            labels = ['Field','LCS all']            
        field_cols=[BTkey,'ng']
        catalogs = [self.gsw.cat,self.lcs.cat,self.lcs.cat] 
        sampleflags = [self.gsw_mass_sfr_flag,self.lcs_mass_sfr_flag & self.lcs.membflag,self.lcs_mass_sfr_flag & self.lcs.infallflag]
        lowflags = [self.gsw_lowsfr_flag, self.lcs.lowsfr_flag,self.lcs.lowsfr_flag]

        #colors = ['0.5',darkblue,lightblue] # color for field and LCS
        #alphas = [.8,.5,.5]
        #lws = [2,2]
        #zorders = [2,2]
        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        zorders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        
        #for i,col in enumerate(field_cols):
        xlabels = [r'$\rm B/T$','Sersic n','g-i','prob Sc',r'$\rm \log_{10}(M_\star/M_\odot)$',r'$\rm \Delta \log_{10}(SFR)$']
        xlabels = [r'$\rm B/T$','Sersic n','prob Sc',r'$\rm \log_{10}(M_\star/M_\odot)$',r'$\rm \Delta \log_{10}(SFR)$']
        titles = [r'$\rm Normal \ SFR$',r'$\rm Suppressed \ SFR$']
        ylabels=[r'$\rm Norm. \ Distribution$',r'$\rm Norm. \ Distribution$']
        # BT

        allbins = [np.linspace(0,xmax,nbins),np.linspace(1,6,nbins),\
                   np.linspace(0.2,1.6,nbins),np.linspace(0,1,nbins),\
                   np.linspace(9.7,11,nbins),np.linspace(-1,1,nbins)]
        allbins = [np.linspace(0,xmax,nbins),np.linspace(1,6,nbins),\
                   np.linspace(0,1,nbins),\
                   np.linspace(9.7,11,nbins),np.linspace(-1,1,nbins)]
        plotcols = [0,1,2]
        plotcols = [0,3]        
        ncol=len(plotcols)
        colnumber = 0
        for col in plotcols:
            for i in range(len(catalogs)):
                if col == 0:
                    xvar = catalogs[i][BTkey]
                    plt.ylabel(ylabels[0])
                elif col == 1:
                    try:
                        # single component fit in GSWLC is ng
                        xvar = catalogs[i]['ng']
                    except KeyError:
                        xvar = catalogs[i]['ng_2']
                        #elif col == 2:
                #    xvar = catalogs[i]['gmag'] - catalogs[i]['imag']
                elif col == 2:
                    xvar = catalogs[i]['pSc'] #+ catalogs[i]['pSa']
                elif col == 3: # stellar mass
                    xvar = catalogs[i]['logMstar'] #+ catalogs[i]['pSa']
                elif col == 5: # delta SFR
                    x1 = catalogs[i]['logMstar']
                    y1 = catalogs[i]['logSFR']
                    #xvar = y1-get_BV_MS(x1)
                    xvar = y1-get_MS(x1,BTcut=self.cutBT)

                    
                # plot on top row normal SF galaxies
                plt.subplot(nrow,ncol,1+colnumber)            
                flag1 = sampleflags[i] & ~lowflags[i]
                print(labels[i],': number of galaxies in normal sf subsample = ',sum(flag1))
                plt.hist(xvar[flag1],color=colors[i],density=True,bins=allbins[col],\
                         histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i])
                plt.hist(xvar[flag1],color=colors[i],density=True,bins=allbins[col],\
                         histtype='step',lw=lws[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                
                if col == 0:
                    plt.ylabel(ylabels[0],fontsize=16)
                    plt.legend()
                    plt.axvline(x=0.3,ls='--',color='k')                    
                if colnumber == 1:
                    plt.text(-.1,1.05,titles[0],transform=plt.gca().transAxes,horizontalalignment='center')
                # plot on bottom row those with low delta SFR
                print(nrow,ncol,ncol+1+col)
                plt.subplot(nrow,ncol,ncol+1+colnumber)            
                flag2 = sampleflags[i] & lowflags[i]
                print(labels[i],': number of galaxies in suppressed sf subsample = ',sum(flag2))
                print()
                
                if col == 0:
                    plt.ylabel(ylabels[1],fontsize=16)
                    plt.axvline(x=0.3,ls='--',color='k')
                if colnumber == 1:
                    plt.text(-.1,1.05,titles[1],transform=plt.gca().transAxes,horizontalalignment='center')
                    
                plt.hist(xvar[flag2],color=colors[i],density=True,bins=allbins[col],\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                plt.hist(xvar[flag2],color=colors[i],density=True,bins=allbins[col],\
                         histtype='step',lw=lws[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])

                plt.xlabel(xlabels[col],fontsize=16)
                plt.yticks([],[])
            plt.subplot(nrow,ncol,1+colnumber)
            colnumber += 1

        # compare mass distributions
        # compare normal SF galaxies
        
        flag1 = sampleflags[0] & ~lowflags[0]
        flag2 = sampleflags[0] & lowflags[0]
        gswmass_normal = self.gsw.cat['logMstar'][flag1]
        gswmass_lowsfr = self.gsw.cat['logMstar'][flag2]      
        gswBT_normal = self.gsw.cat[BTkey][flag1]
        gswBT_lowsfr = self.gsw.cat[BTkey][flag2]      

        # core
        flag1 = sampleflags[1] & ~lowflags[1]
        flag2 = sampleflags[1] & lowflags[1]
        lcsmass_normal = self.lcs.cat['logMstar'][flag1]
        lcsmass_lowsfr = self.lcs.cat['logMstar'][flag2]
        lcsBT_normal = self.lcs.cat[BTkey][flag1]
        lcsBT_lowsfr = self.lcs.cat[BTkey][flag2]

        #infall
        flag1 = sampleflags[2] & ~lowflags[1]
        flag2 = sampleflags[2] & lowflags[1]
        infallmass_normal = self.lcs.cat['logMstar'][flag1]
        infallmass_lowsfr = self.lcs.cat['logMstar'][flag2]
        infallBT_normal = self.lcs.cat[BTkey][flag1]
        infallBT_lowsfr = self.lcs.cat[BTkey][flag2]


        
        print('######################################################')
        print('comparing mass distribution of normal SFR galaxies, field vs core')
        #print('######################################################')
        t = ks_2samp(gswmass_normal,lcsmass_normal)
        print(t)
        print('######################################################')
        print('comparing mass distribution of low SFR galaxies, field vs core')
        #print('######################################################')
        t = ks_2samp(gswmass_lowsfr,lcsmass_lowsfr)
        print(t)        
        print('######################################################')
        print('comparing mass distribution of low SFR galaxies, field vs infall')
        #print('######################################################')
        t = ks_2samp(gswmass_lowsfr,infallmass_lowsfr)
        print(t)
        print()

        print('######################################################')
        print('comparing mass distribution of normal vs low SFR galaxies, field')
        #print('######################################################')
        t = ks_2samp(gswmass_normal,gswmass_lowsfr)
        print(t)
        print('######################################################')
        print('comparing mass distribution of normal vs low SFR galaxies, core')
        #print('######################################################')
        t = ks_2samp(lcsmass_normal,lcsmass_lowsfr)
        print(t)
        print('######################################################')
        print('comparing mass distribution of normal vs low SFR galaxies, infall')
        #print('######################################################')
        t = ks_2samp(infallmass_normal,infallmass_lowsfr)
        print(t)

        print('######################################################')
        print('comparing BT distribution of normal vs low SFR galaxies, field')
        #print('######################################################')
        t = ks_2samp(gswBT_normal,gswBT_lowsfr)
        print(t)
        print('######################################################')
        print('comparing BT distribution of normal vs low SFR galaxies, core')
        #print('######################################################')
        t = ks_2samp(lcsBT_normal,lcsBT_lowsfr)
        print(t)
        print('######################################################')
        print('comparing BT distribution of normal vs low SFR galaxies, infall')
        #print('######################################################')
        t = ks_2samp(infallBT_normal,infallBT_lowsfr)
        print(t)

    def compare_morph_mmatch(self,nbins=10,xmax=.4,coreonly=False):
        '''compare distribution of BT for normal and suppressed galaxies   '''
        plt.figure(figsize=(10,9))
        plt.subplots_adjust(wspace=.25,hspace=.4)
        # 2x2 figure
        if coreonly:
            lcsflag = self.lcs.membflag #| self.lcs.infallflag
            labels = ['Field','Core']                        
        else:
            lcsflag = self.lcs.membflag | self.lcs.infallflag
            labels = ['Field','LCS all']



        ### MASS MATCHING OF FIELD SAMPLE
        # get mass-matched sample of field galaxies
        lcsflag = (self.lcs.membflag | self.lcs.infallflag) & self.lcs_mass_sfr_flag
        normalfield = self.gsw_mass_sfr_flag & ~self.gsw_lowsfr_flag
        lowfield = self.gsw_mass_sfr_flag & self.gsw_lowsfr_flag

        # cluster flags
        normalcluster = lcsflag & ~self.lcs_lowsfr_flag
        lowcluster = lcsflag & self.lcs_lowsfr_flag
        # get mass-matched samples of low and normal field galaxies
        keep_indices_normal = mass_match(self.lcs.cat['logMstar'][normalcluster],self.gsw.cat['logMstar'][normalfield],34278,nmatch=NMASSMATCH)

        keep_indices_low = mass_match(self.lcs.cat['logMstar'][lowcluster],self.gsw.cat['logMstar'][lowfield],34278,nmatch=NMASSMATCH)


        field_cols=[BTkey,'ng']
        

        xvars_normal = [self.gsw.cat[BTkey][normalfield][keep_indices_normal],\
                        self.lcs.cat[BTkey][self.lcs.membflag & self.lcs_mass_sfr_flag & ~self.lcs_lowsfr_flag],\
                        self.lcs.cat[BTkey][self.lcs.infallflag & self.lcs_mass_sfr_flag & ~self.lcs_lowsfr_flag],\
                        self.gsw.cat['logMstar'][normalfield][keep_indices_normal],\
                        self.lcs.cat['logMstar'][self.lcs.membflag & self.lcs_mass_sfr_flag & ~self.lcs_lowsfr_flag],\
                        self.lcs.cat['logMstar'][self.lcs.infallflag & self.lcs_mass_sfr_flag & ~self.lcs_lowsfr_flag]]

        xvars_low = [self.gsw.cat[BTkey][lowfield][keep_indices_low],\
                     self.lcs.cat[BTkey][self.lcs.membflag & self.lcs_mass_sfr_flag & self.lcs_lowsfr_flag],\
                     self.lcs.cat[BTkey][self.lcs.infallflag & self.lcs_mass_sfr_flag & self.lcs_lowsfr_flag],\
                     self.gsw.cat['logMstar'][lowfield][keep_indices_low],\
                     self.lcs.cat['logMstar'][self.lcs.membflag & self.lcs_mass_sfr_flag & self.lcs_lowsfr_flag],\
                     self.lcs.cat['logMstar'][self.lcs.infallflag & self.lcs_mass_sfr_flag & self.lcs_lowsfr_flag]]

        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        zorders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        
        #for i,col in enumerate(field_cols):
        xlabels = [r'$\rm B/T$',r'$\rm \log_{10}(M_\star/M_\odot)$','Sersic n','prob Sc',r'$\rm \Delta \log_{10}(SFR)$']
        titles = [r'$\rm Normal \ SFR$',r'$\rm Suppressed \ SFR$']
        ylabels=[r'$\rm Norm. \ Distribution$',r'$\rm Norm. \ Distribution$']
        # BT

        allbins = [np.linspace(0,xmax,nbins),np.linspace(9.7,11,nbins)]
                   
        # one row, 2 cols
        # loop over normal vs suppressed
        for j in range(2):
            col = j
            for k in range(2): # rows
                plt.subplot(2,2,j+1+2*k)
                # loop over field, core, infall
                for i in range(len(labels)):
                    # plot B/T distribution of normal galaxies
                    if j == 0:
                        xvar = xvars_normal[i+3*k]
                    else:
                        xvar = xvars_low[i+3*k]
                    plt.hist(xvar,color=colors[i],density=True,bins=allbins[k],\
                             histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i])
                    plt.hist(xvar,color=colors[i],density=True,bins=allbins[k],\
                             histtype='step',lw=lws[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                
                if j == 0:
                    plt.ylabel(ylabels[0],fontsize=22)
                    plt.legend()
                if k == 0:
                    plt.axvline(x=0.3,ls='--',color='k')                    

            
                plt.xlabel(xlabels[k],fontsize=22)
                #plt.yticks([],[])
                if k == 0:
                    if (j == 0):
                        plt.title("Normal SFR")
                    else:
                        plt.title("Suppressed SFR")

        print('COMPARING PROPERTIES OF NORMAL AND LOW GALAXIES')
        print('\t field, core, infall')
        print('\t BT')        
        print('\t field, core, infall')
        print('\t logMstar')        
        for i,j in zip(xvars_normal,xvars_low):
            print('KS results: ',ks_2samp(i,j))
            print('AD results: ',anderson_ksamp([i,j]))
            print()
            
               
    def BT_mstar_normal_low_sfr(self,nbins=10,xmax=.3,coreonly=False):
        ''' plot B/T vs Mstar for normal and low sfr galaxies   '''
        plt.figure(figsize=(10,6))
        plt.subplots_adjust(wspace=.25,hspace=.4)
        # 2x2 figure
        nrow=2
        ncol=2
        if coreonly:
            lcsflag = self.lcs.membflag #| self.lcs.infallflag
            labels = ['Field','LCS core']                        
        else:
            lcsflag = self.lcs.membflag | self.lcs.infallflag
            labels = ['Field','LCS all']            
        field_cols=[BTkey,'ng']
        catalogs = [self.gsw.cat,self.lcs.cat,self.lcs.cat] 
        sampleflags = [self.gsw_mass_sfr_flag,self.lcs_mass_sfr_flag & self.lcs.membflag,self.lcs_mass_sfr_flag & self.lcs.infallflag]
        lowflags = [self.gsw.lowsfr_flag, self.lcs.lowsfr_flag,self.lcs.lowsfr_flag]

        #colors = ['0.5',darkblue,lightblue] # color for field and LCS
        #alphas = [.8,.5,.5]
        #lws = [2,2]
        #zorders = [2,2]
        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        zorders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        
        #for i,col in enumerate(field_cols):
        xlabels = ['B/T','Sersic n','g-i','prob Sc','logMstar','$\Delta SFR$']
        xlabels = ['B/T','Sersic n','prob Sc','logMstar','$\Delta SFR$']        
        titles = ['Normal SFR','Low SFR']
        ylabels=['Norm. Counts','Norm. Counts']
        # BT

        plotcols = [0,3]
        ncol=len(plotcols)
        colnumber = 0
        for col in plotcols:
            for i in range(len(catalogs)):
                if col == 0:
                    xvar = catalogs[i][BTkey]
                    plt.ylabel(ylabels[0])
                elif col == 1:
                    try:
                        # single component fit in GSWLC is ng
                        xvar = catalogs[i]['ng']
                    except KeyError:
                        xvar = catalogs[i]['ng_2']
                        #elif col == 2:
                #    xvar = catalogs[i]['gmag'] - catalogs[i]['imag']
                elif col == 2:
                    xvar = catalogs[i]['pSc'] #+ catalogs[i]['pSa']
                elif col == 3: # stellar mass
                    xvar = catalogs[i]['logMstar'] #+ catalogs[i]['pSa']
                elif col == 5: # delta SFR
                    x1 = catalogs[i]['logMstar']
                    y1 = catalogs[i]['logSFR']
                    #xvar = y1-get_BV_MS(x1)
                    xvar = y1-get_MS(x1,BTcut=self.cutBT)

                    
                # plot on top row normal SF galaxies
                plt.subplot(nrow,ncol,1+colnumber)            
                flag1 = sampleflags[i] & ~lowflags[i]
                print(labels[i],': number of galaxies in subsample = ',sum(flag1))
                plt.hist(xvar[flag1],color=colors[i],density=True,bins=allbins[col],\
                         histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i])
                plt.hist(xvar[flag1],color=colors[i],density=True,bins=allbins[col],\
                         histtype='step',lw=lws[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                
                if col == 0:
                    plt.ylabel(ylabels[0],fontsize=16)
                    plt.legend()
                    plt.axvline(x=0.4,ls='--',color='k')                    
                if colnumber == 1:
                    plt.text(-.1,1.05,titles[0],transform=plt.gca().transAxes,horizontalalignment='center')
                # plot on bottom row those with low delta SFR
                print(nrow,ncol,ncol+1+col)
                plt.subplot(nrow,ncol,ncol+1+colnumber)            
                flag2 = sampleflags[i] & lowflags[i]
                if col == 0:
                    plt.ylabel(ylabels[1],fontsize=16)
                    plt.axvline(x=0.4,ls='--',color='k')
                if colnumber == 1:
                    plt.text(-.1,1.05,titles[1],transform=plt.gca().transAxes,horizontalalignment='center')
                    
                plt.hist(xvar[flag2],color=colors[i],density=True,bins=allbins[col],\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                plt.hist(xvar[flag2],color=colors[i],density=True,bins=allbins[col],\
                         histtype='step',lw=lws[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])

                plt.xlabel(xlabels[col],fontsize=16)
            plt.subplot(nrow,ncol,1+colnumber)
            colnumber += 1

        # compare mass distributions
        # compare normal SF galaxies
        
        flag1 = sampleflags[0] & ~lowflags[0]
        flag2 = sampleflags[0] & lowflags[0]
        gswmass_normal = self.gsw.cat['logMstar'][flag1]
        gswmass_lowsfr = self.gsw.cat['logMstar'][flag2]      

        flag1 = sampleflags[1] & ~lowflags[1]
        flag2 = sampleflags[1] & lowflags[1]
        lcsmass_normal = self.lcs.cat['logMstar'][flag1]
        lcsmass_lowsfr = self.lcs.cat['logMstar'][flag2]

        print('######################################################')
        print('comparing mass distribution of normal SFR galaxies')
        #print('######################################################')
        t = ks_2samp(gswmass_normal,lcsmass_normal)
        print(t)
        print('######################################################')
        print('comparing mass distribution of low SFR galaxies')
        #print('######################################################')
        t = ks_2samp(gswmass_lowsfr,lcsmass_lowsfr)
        print(t)
    def compare_field2others(self,nbins=10,xmax=.3,coreonly=False):
        '''compare properties of LCS normal, low and field low to normal field   '''

        ### properties to compare
        # BT
        # sersic n
        # pSc
        # B/A - b/c it seems like we have more edge on among low sfr
        #       want to make sure it's not just extinction!
        
        plt.figure(figsize=(14,6))
        plt.subplots_adjust(wspace=.25,hspace=.4)
        # 1x4 figure
        nrow=1
        ncol=4
        labels = ['LCS normal','LCS low', 'Field low']                        
        var_cols=[BTkey,'ng']
        catalogs = [self.gsw.cat,self.lcs.cat] 
        sampleflags = [self.gsw_mass_sfr_flag,self.lcs_mass_sfr_flag & lcsflag]
        lowflags = [self.gsw.lowsfr_flag, self.lcs.lowsfr_flag]

        lcsflag = lcs.membflag | lcs.infallflag
        catalogs = [self.lcs.cat[self.lcs_mass_sfr_flag & lcsflag & ~self.lcs.lowsfr_flag],\
                    self.lcs.cat[self.lcs_mass_sfr_flag & lcsflag & self.lcs.lowsfr_flag ],\
                    self.gsw.cat[self.gsw_mass_sfr_flag & self.gsw.lowsfr_flag]]
        # comparison catalog
        compcat = self.gsw.cat[self.gsw_mass_sfr_flag & ~self.gsw.lowsfr_flag]
        colors = ['b','r','0.5'] # color for field and LCS
        alphas = [.5,.5,.8]
        lws = [2,2,2]
        zorders = [2,2,2]
        #for i,col in enumerate(field_cols):
        xlabels = ['B/T','Sersic n','prob Sc','logMstar','B/A']
        keys = [BTkey,'ng','pSc','logMstar',]
        titles = ['Normal SFR','Low SFR']
        ylabels=['Norm. Counts','Norm. Counts']
        # BT

        allbins = [np.linspace(0,xmax,nbins),np.linspace(1,6,nbins),\
                   np.linspace(0.2,1.6,nbins),np.linspace(0,1,nbins),\
                   np.linspace(9.7,11,nbins),np.linspace(-1,1,nbins)]
        allbins = [np.linspace(0,xmax,nbins),np.linspace(1,6,nbins),\
                   np.linspace(0,1,nbins),\
                   np.linspace(9.7,11,nbins),np.linspace(-1,1,nbins)]
        for col in range(4):
            for i in range(len(catalogs)):
                if col == 0:
                    xvar = catalogs[i][BTkey]
                    plt.ylabel(ylabels[0])
                elif col == 1:
                    xvar = catalogs[i]['ng']
                #elif col == 2:
                #    xvar = catalogs[i]['gmag'] - catalogs[i]['imag']
                elif col == 2:
                    xvar = catalogs[i]['pSc'] #+ catalogs[i]['pSa']
                elif col == 3: # stellar mass
                    xvar = catalogs[i]['logMstar'] #+ catalogs[i]['pSa']
                elif col == 5: # delta SFR
                    x1 = catalogs[i]['logMstar']
                    y1 = catalogs[i]['logSFR']
                    #xvar = y1-get_BV_MS(x1)
                    xvar = y1-get_MS(x1,BTcut=self.cutBT)

                    
                # plot on top row normal SF galaxies
                plt.subplot(nrow,ncol,1+col)            
                flag1 = sampleflags[i] & ~lowflags[i]
                plt.hist(xvar[flag1],color=colors[i],density=True,bins=allbins[col],\
                         histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                if col == 0:
                    plt.ylabel(ylabels[0],fontsize=16)
                    plt.legend()
                if col == 2:
                    plt.text(-.1,1.05,titles[0],transform=plt.gca().transAxes,horizontalalignment='center')
                # plot on bottom row those with low delta SFR
                plt.subplot(nrow,ncol,ncol+1+col)            
                flag2 = sampleflags[i] & lowflags[i]
                if col == 0:
                    plt.ylabel(ylabels[1],fontsize=16)
                if col == 2:
                    plt.text(-.1,1.05,titles[1],transform=plt.gca().transAxes,horizontalalignment='center')
                    
                plt.hist(xvar[flag2],color=colors[i],density=True,bins=allbins[col],\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
                plt.xlabel(xlabels[col],fontsize=16)
            plt.subplot(nrow,ncol,1+col)
            

    def compare_BT_lowsfr_field_core(self,nbins=12,coreonly=False,infallonly=False,BTmax=1,nrow=1,plotsingle=True,show_xlabel=True,subplot_offset=0):
        ''' compare the B/T distribution of LCS low SFR galaxies with mass-matched sample drawn from low SFR field galaxies '''

        if coreonly:
            lcsflag = self.lcs.membflag #| self.lcs.infallflag
            labels = ['Field','Core']
        elif infallonly:
            lcsflag = self.lcs.infallflag #| self.lcs.infallflag
            labels = ['Field','Infall']
        else:
            lcsflag = self.lcs.membflag | self.lcs.infallflag
            labels = ['Field','LCS all']            
        

        # compare B/T distribution of LCS low SFR galaxies with
        # mass-matched sample drawn from low SFR field galaxies

        
        # select low SFR galaxies from field
        sampleflags = [self.gsw_mass_sfr_flag,self.lcs_mass_sfr_flag & lcsflag]
        lowflags = [self.gsw.lowsfr_flag, self.lcs.lowsfr_flag]
        

        lcsflag2 = sampleflags[1] & lowflags[1]
        lcsmass_lowsfr = self.lcs.cat['logMstar'][lcsflag2]
        lcs_lowsfr = self.lcs.cat[lcsflag2]        


        flag2 = sampleflags[0] & lowflags[0]
        gswmass_lowsfr = self.gsw.cat['logMstar'][flag2]      
        gswBT_lowsfr = self.gsw.cat[BTkey][flag2]

        flags2=[flag2,lcsflag2]
        # create a mass-matched sample of field galaxies with low SFR
        myrandom = np.random.random()
        keep_indices = mass_match(lcsmass_lowsfr,gswmass_lowsfr,nmatch=NMASSMATCH,seed=479)            

        gsw_lowsfr_mmatch = self.gsw.cat[flag2][keep_indices]
        
        # compare the B/T distributions
        # does low SFR sample have higher B/T than normal field galaxies
        if plotsingle:
            plt.figure(figsize=(12,3))
            plt.subplots_adjust(wspace=.01,bottom=.2)
        colors = ['0.5',darkblue,'k'] # color for field and LCS

        alphas = [.8,.5,1.]
        lws = [2,2,2]
        zorders = [2,2,2]
        histtypes = ['stepfilled','stepfilled','stepfilled']
        #for i,col in enumerate(field_cols):
        xlabels = [r'$\rm B/T$',r'$\rm S{e}rsic \ n$',r'$\rm prob \ Sc$', r'$g-i$','logMstar','$\Delta SFR$']
        #xlabels = ['B/T','Sersic n','prob Sc','logMstar','$\Delta SFR$']        
        
        labels = ['Field','LCS','Low SFR Field']
        if coreonly:
            labels = [r'$\rm Field$',r'$\rm Core$','Low SFR Field']
        elif infallonly:
            labels = [r'$\rm Field$',r'$\rm Infall$','Low SFR Field']
            colors = ['0.5',lightblue,'k'] # color for field and LCS            
        allbins = [np.linspace(0,BTmax,nbins),\
                   np.linspace(1,6,nbins),\
                   np.linspace(0,1,nbins),\
                   np.linspace(.5,2,nbins),\
                   np.linspace(9.7,11,nbins),np.linspace(-1,1,nbins)]
        
        xvars_gsw = [gsw_lowsfr_mmatch[BTkey],gsw_lowsfr_mmatch['ng'],gsw_lowsfr_mmatch['pSc'], gsw_lowsfr_mmatch['gmag']-gsw_lowsfr_mmatch['imag']]
        xvars_lcs = [lcs_lowsfr[BTkey],lcs_lowsfr['ng_1'],lcs_lowsfr['pSc'],lcs_lowsfr['gmag']-lcs_lowsfr['imag']]        
        ncols=4
        for col in range((ncols)):
            plt.subplot(nrow,ncols,col+1+subplot_offset)
            xvars = [xvars_gsw[col],xvars_lcs[col]]
            for i in range(len(xvars)):
                plt.hist(xvars[i],color=colors[i],density=True,bins=allbins[col],\
                         histtype=histtypes[i],lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
            if col == 3:
                plt.legend(loc='center right',fontsize=11)#,bbox_to_anchor=(-.02,.7))
                #plt.text(-.2,-.1,'Normalized Distribution',transform=plt.gca().transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=24)
                #plt.ylabel(r'$\rm Norm. \ Distribution$',fontsize=20)

            print()
            print()
            print('###################################################################')
            print('comparing LCS low SFR and mass-matched field low SFR:',xlabels[col])
            #print('LCS values : ',xvars[1])
            print()
            print('median LCS {:.2f} vs field {:.2f}'.format(np.median(xvars[1]),np.median(xvars[0])))
            print()
            t = ks_2samp(xvars[0],xvars[1])
            print('statistic={:.2f}, pvalue={:.2e}'.format(t[0],t[1]))
            print('')
            print('anderson-darling test')
            t = anderson_ksamp([xvars[0],xvars[1]])            
            print(t)
            if t[2] < .003:
                s = r'$\rm \bf pvalue={:.3f}$'.format(t[2])
            else:
                s = r'$\rm pvalue={:.3f}$'.format(t[2])
            plt.text(0.95,.9,s,fontsize=14,transform=plt.gca().transAxes,horizontalalignment='right')
            if show_xlabel:
                plt.xlabel(xlabels[col],fontsize=20)
                
                if col == 0:
                    if nrow == 1:
                        plt.ylabel(r'$\rm Normalized \ Distribution$',fontsize=20)
                    else:
                        ax = plt.gca()
                        plt.text(-.3,1,r'$\rm Normalized \ Distribution$',fontsize=20,transform=ax.transAxes,rotation=90,verticalalignment='center')
            else:
                plt.xticks([],[])
            plt.yticks([],[])
        
    def compare_BT_lowsfr_lcs_field_mmatch(self,nbins=12,coreonly=False,infallonly=False,BTmax=1,nrandom=100,nmatch=30,plotsingle=False):
        ''' compare the B/T distribution of LCS low SFR galaxies with mass-matched sample drawn from low SFR field galaxies '''

        if coreonly:
            lcsflag = self.lcs.membflag #| self.lcs.infallflag
            labels = ['Field','Core']
        elif infallonly:
            lcsflag = self.lcs.infallflag #| self.lcs.infallflag
            labels = ['Field','Infall']
        else:
            lcsflag = self.lcs.membflag | self.lcs.infallflag
            labels = ['Field','LCS all']            
        

        # compare B/T distribution of LCS low SFR galaxies with
        # mass-matched sample drawn from low SFR field galaxies

        
        # select low SFR galaxies from field
        sampleflags = [self.gsw_mass_sfr_flag,self.lcs_mass_sfr_flag & self.lcs.membflag,self.lcs_mass_sfr_flag & self.lcs.infallflag]
        lowflags = [self.gsw.lowsfr_flag, self.lcs.lowsfr_flag,self.lcs.lowsfr_flag]
        

        lcsflag2 = sampleflags[1] & lowflags[1]
        lcsmass_sample = self.lcs_mass_sfr_flag & (self.lcs.membflag | self.lcs.infallflag) & self.lcs.lowsfr_flag
        coremass_lowsfr = self.lcs.cat['logMstar'][lcsmass_sample]
        coreBT_lowsfr = self.lcs.cat[BTkey][lcsflag2]        


        flag2 = sampleflags[0] & lowflags[0]
        gswmass_lowsfr = self.gsw.cat['logMstar'][flag2]      
        gswBT_lowsfr = self.gsw.cat[BTkey][flag2]

        lcsflag2_infall = sampleflags[2] & lowflags[2]
        
        flags2=[flag2,lcsflag2,lcsflag2_infall]

        # repeat mass matching many times, but select one field galaxy for each core
        # create a mass-matched sample of field galaxies with normal SFR

        ##########################################################
        ### MAKE A LOOP TO RUN MASS MATCHING MANY TIMES
        ##########################################################        
        # store all the realization
        # store p values and distributions for all realizations

        all_keep_indices = []


        for i in range(nrandom):
            keep_indices = mass_match(coremass_lowsfr,gswmass_lowsfr,nmatch=nmatch)
            all_keep_indices.append(keep_indices)
        # compare the B/T distributions
        # does low SFR sample have higher B/T than normal field galaxies

        plt.figure(figsize=(10,4))
        plt.subplots_adjust(wspace=.3,bottom=.2)

        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        zorders = [1,3,2]
        lws = [3,4,3]
        alphas = [.4,0,.5]        
        
        
        histtypes = ['stepfilled','stepfilled','stepfilled']
        #for i,col in enumerate(field_cols):
        xlabels = ['B/T','Sersic n','g-i','prob Sc','logMstar','$\Delta SFR$']
        xlabels = ['B/T','Sersic n','prob Sc','logMstar','$\Delta SFR$']        
        
        labels = ['Field','LCS','Low SFR Field']
        if coreonly:
            labels = ['Field','Core','Low SFR Field']
        allbins = [np.linspace(0,BTmax,nbins),np.linspace(1,6,nbins),\
                   np.linspace(0,1,nbins),\
                   np.linspace(9.7,11,nbins),np.linspace(-1,1,nbins)]
        
        # BT
        ncols=3
        for col in range((ncols)):
            all_KS_pvalue = np.zeros(nrandom)
            all_AD_pvalue = np.zeros(nrandom)
            all_hist_yvalues = np.zeros((nrandom,len(allbins[col])-1))
            if col == 0:
                xvars = [self.gsw.cat[BTkey],self.lcs.cat[BTkey],self.lcs.cat[BTkey]]
            elif col == 1:
                xvars = [self.gsw.cat['ng'],self.lcs.cat['ng_2'],self.lcs.cat['ng_2']]
            #elif col == 2:
            #    xvars = [self.gsw.cat['gmag'] - self.gsw.cat['imag'],\
            #             self.lcs.cat['gmag']- self.lcs.cat['imag']]
            elif col == 2:
                xvars = [self.gsw.cat['pSc'],self.lcs.cat['pSc'],self.lcs.cat['pSc']] #+ catalogs[i]['pSa']
            plt.subplot(1,3,col+1)
            for i in range(len(xvars)):
                if i == 0:
                    for k in range(nrandom):
                        keep_indices = all_keep_indices[k]
                        xvar = xvars[i][flags2[i]][keep_indices]
                        t = plt.hist(xvar,color=colors[i],density=True,bins=allbins[col],\
                                 histtype='step',lw=lws[i],alpha=.05,zorder=zorders[i])#hatch=hatches[i])
                        # get median of hist and plot that
                        all_hist_yvalues[k] = t[0]
                        a = ks_2samp(xvars[0][flags2[0]][keep_indices],xvars[1][flags2[1]])
                        all_KS_pvalue[k] = a[1]
                        #a = anderson_ksamp(xvars[0][flags2[0]][keep_indices],xvars[1][flags2[1]])
                        #all_AD_pvalue[k] = a[1]
                        
                    # calc the median of all the distributions
                    med_hist = np.median(all_hist_yvalues,axis=0)
                    #t = plt.hist(med_hist,color=colors[i],density=True,bins=allbins[col],\
                    #             histtype=histtypes[i],lw=lws[i],alpha=.05,zorder=zorders[i],label=labels[i])#
                    x = (allbins[col][0:-1]+allbins[col][1:])/2.
                    y = med_hist
                    #print(x,y)
                    plt.plot(x,y,'ko',color=colors[i])#
                    percentiles = [16,50,84]
                    KSmin = np.percentile(all_KS_pvalue,percentiles[0])
                    KSmed = np.percentile(all_KS_pvalue,percentiles[1])
                    KSmax = np.percentile(all_KS_pvalue,percentiles[2])
                    print('#############')
                    print(xlabels[col])
                    print('KS test')
                    print('pvalue={:.2e} ({:.2e},{:.2e})'.format(KSmed,KSmin,KSmax))
                    print('')
                    #print('anderson-darling test')
                    #print('pvalue={:.2e} ({:.2e},{:.2e}_'.format(np.percentile(all_AD_pvalue,percentiles[1]),\
                    #                                             np.percentile(all_AD_pvalue,percentiles[0]),\
                    #                                             np.percentile(all_AD_pvalue,percentiles[2])))
                    
                else:
                    xvar = xvars[i][flags2[i]]
                    plt.hist(xvar,color=colors[i],density=True,bins=allbins[col],\
                         histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=zorders[i])#hatch=hatches[i])
                    plt.hist(xvar,color=colors[i],density=True,bins=allbins[col],\
                         histtype='step',lw=lws[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
            if col == 0:
                #plt.text(-.2,-.1,'Normalized Distribution',transform=plt.gca().transAxes,rotation=90,horizontalalignment='center',verticalalignment='center',fontsize=24)
                plt.ylabel('Norm. Distribution',fontsize=16)

            plt.xlabel(xlabels[col],fontsize=16)

        plt.legend()

        
    def compare_BT_lowsfr_field(self,nbins=12,coreonly=False,BTmax=0.4):
        ''' compare the B/T distribution of field galaxies with low and normal SFR '''
        # select low SFR galaxies from field
        
        sampleflags = [self.gsw_mass_sfr_flag]#,self.lcs_mass_sfr_flag & lcsflag]
        lowflags = [self.gsw.lowsfr_flag]#, self.lcs.lowsfr_flag]
        
        flag1 = sampleflags[0] & ~lowflags[0]
        flag2 = sampleflags[0] & lowflags[0]        
        gswmass_normal = self.gsw.cat['logMstar'][flag1]
        gswBT_normal = self.gsw.cat[BTkey][flag1]        
        
        gswmass_lowsfr = self.gsw.cat['logMstar'][flag2]      
        gswBT_lowsfr = self.gsw.cat[BTkey][flag2]
        
        # create a mass-matched sample of field galaxies with normal SFR
        keep_indices = mass_match(gswmass_lowsfr,gswmass_normal,34278,nmatch=NMASSMATCH)            

        massmatched_gswBT_normal = gswBT_normal[keep_indices]
        
        # compare the B/T distributions
        # does low SFR sample have higher B/T than normal field galaxies
        print('#######################################')
        print('comparing low SFR and mass-matched normal SFR')
        t = ks_2samp(gswBT_lowsfr,massmatched_gswBT_normal)
        print(t)
        print('#######################################')
        print('comparing normal SFR and normal SFR mass-matched to low SFR')
        t = ks_2samp(gswBT_normal,massmatched_gswBT_normal)
        print(t)

        plt.figure(figsize=(8,6))
        colors = ['k','0.5','c'] # color for field and LCS
        xvars = [gswBT_normal, massmatched_gswBT_normal,gswBT_lowsfr]
        alphas = [1.,.8,.5]
        lws = [2,2,2]
        zorders = [2,2,2]
        histtypes = ['step','stepfilled','stepfilled']
        #for i,col in enumerate(field_cols):
        xlabels = ['B/T','Sersic n','g-i','prob Sc','logMstar','$\Delta SFR$']
        labels = ['Normal Field','Normal Field, MM to Low SFR Field','Low SFR Field']
        # BT
        
        for i in range(len(xvars)):

            plt.hist(xvars[i],color=colors[i],density=True,bins=np.linspace(0,BTmax,nbins),\
                 histtype=histtypes[i],lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
        #plt.legend()
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks([],[])        
        plt.xlim(-.01,BTmax+.01)#,0,9])
        plt.xlabel(r'$\rm B/T$')
        plt.ylabel(r'$\rm Normalized \ Distribution$')

    def compare_BT_lowsfr_lcs_massmatch(self,nbins=12,infallonly=False,coreonly=False,lcsflag=None,BTmax=0.4):
        ''' compare the B/T distribution of field galaxies with low and normal SFR '''
        # select low SFR galaxies from field
        if coreonly:
            lcsflag = self.lcs.membflag #| self.lcs.infallflag
        elif infallonly:
            lcsflag = self.lcs.infallflag #| self.lcs.infallflag
        else:
            lcsflag = self.lcs.membflag | self.lcs.infallflag

        # compare B/T distribution of LCS low SFR galaxies with
        # mass-matched sample drawn from low SFR field galaxies

        
        # select low SFR galaxies from field
        sampleflags = [self.gsw_mass_sfr_flag,self.lcs_mass_sfr_flag & self.lcs.membflag, self.lcs_mass_sfr_flag & self.lcs.infallflag]
        lowflags = [self.gsw.lowsfr_flag, self.lcs.lowsfr_flag, self.lcs.lowsfr_flag]
        

        if coreonly:
            lcsflag2 = sampleflags[1] & lowflags[1]
        elif infallonly:
            lcsflag2 = sampleflags[2] & lowflags[2]
        else:
            lcsflag2 = (sampleflags[1] | sampleflags[2]) & lowflags[1]
        lcsmass_lowsfr = self.lcs.cat['logMstar'][lcsflag2]
        lcsBT_lowsfr = self.lcs.cat[BTkey][lcsflag2]        

        flag1 = sampleflags[0] & ~lowflags[0]
        gswmass_normal = self.gsw.cat['logMstar'][flag1]
        gswBT_normal = self.gsw.cat[BTkey][flag1]        
        flag2 = sampleflags[0] & lowflags[0]        
        gswmass_lowsfr = self.gsw.cat['logMstar'][flag2]      
        gswBT_lowsfr = self.gsw.cat[BTkey][flag2]
        
        # create a mass-matched sample of field galaxies with normal SFR
        keep_indices = mass_match(lcsmass_lowsfr,gswmass_normal,nmatch=NMASSMATCH)            

        massmatched_gswBT_normal = gswBT_normal[keep_indices]
        
        # compare the B/T distributions
        # does low SFR sample have higher B/T than normal field galaxies
        print('#######################################')
        print('comparing low SFR and mass-matched normal SFR')
        t = ks_2samp(lcsBT_lowsfr,massmatched_gswBT_normal)
        print(t)
        #print('#######################################')
        #print('comparing normal SFR and normal SFR mass-matched to low SFR')
        #t = ks_2samp(gswBT_normal,massmatched_gswBT_normal)
        #print(t)

        plt.figure(figsize=(8,6))
        colors = ['k','0.5',darkblue] # color for field and LCS
        if infallonly:
            colors = ['k','0.5',lightblue] # color for field and LCS
        xvars = [gswBT_normal, massmatched_gswBT_normal,lcsBT_lowsfr]
        alphas = [1.,.8,.5]
        lws = [2,2,2]
        zorders = [2,2,2]
        histtypes = ['step','stepfilled','stepfilled']
        #for i,col in enumerate(field_cols):
        xlabels = ['B/T','Sersic n','g-i','prob Sc','logMstar','$\Delta SFR$']
        labels = ['Normal Field','Normal Field, MM to low SFR Core','Low SFR Infall']
        if infallonly:
            labels = ['Normal Field','Normal Field, MM to low SFR Infall','Low SFR Infall']
        # BT
        
        for i in range(len(xvars)):

            plt.hist(xvars[i],color=colors[i],density=True,bins=np.linspace(0,BTmax,nbins),\
                 histtype=histtypes[i],lw=lws[i],alpha=alphas[i],zorder=zorders[i],label=labels[i])#hatch=hatches[i])
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axis([-.01,BTmax+.01,0,9.])        
        plt.xlabel('B/T')
        plt.ylabel('Normalized Distribution')

        
    def plot_dsfr_BT(self,nbins=15,xmax=.3,writefiles=False,nsersic_cut=10,ecut=1,BTline=None,mmatch=False):
        #nflag = (self.lcs.cat['ng_2'] < nsersic_cut)
        #sflag = (self.lcs.cat['p_el'] < ecut)
        flag1 =  self.lcs.membflag &  self.lcs_mass_sfr_flag
        coreBT = self.lcs.cat[BTkey][flag1]
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        #core_dsfr = y1-get_BV_MS(x1)
        core_dsfr = y1-get_MS(x1,BTcut=self.cutBT)
        
        flag2 = self.lcs.infallflag & self.lcs_mass_sfr_flag
        infallBT = self.lcs.cat[BTkey][flag2]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        #infall_dsfr = y2-get_BV_MS(x2)
        infall_dsfr = y2-get_MS(x2,BTcut=self.cutBT)        

        nflag = (self.gsw.cat['ng'] < nsersic_cut)

        if mmatch:
            ### MASS MATCHING OF FIELD SAMPLE
            # get mass-matched sample of field galaxies
            lcsflag = (self.lcs.membflag | self.lcs.infallflag) & self.lcs_mass_sfr_flag
            fieldflag = self.gsw_mass_sfr_flag 

            keep_indices = mass_match(self.lcs.cat['logMstar'][lcsflag],self.gsw.cat['logMstar'][fieldflag],34278,nmatch=NMASSMATCH)

            fieldBT = self.gsw.cat[BTkey][fieldflag][keep_indices]
            x3 = self.gsw.cat['logMstar'][fieldflag][keep_indices]
            y3 = self.gsw.cat['logSFR'][fieldflag][keep_indices]
        else:
            flag3 = self.gsw_mass_sfr_flag   #& self.gsw.field1
            fieldBT = self.gsw.cat[BTkey][flag3]
            x3 = self.gsw.cat['logMstar'][flag3]
            y3 = self.gsw.cat['logSFR'][flag3]
            
        #field_dsfr = y3-get_BV_MS(x3)
        field_dsfr = y3-get_MS(x3,BTcut=self.cutBT)


        
        plt.figure()
        plt.subplots_adjust(left=.2)
        mybins = np.linspace(0,xmax,nbins)        
        xvars = [fieldBT,coreBT,infallBT]
        yvars = [field_dsfr,core_dsfr,infall_dsfr]        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alpha_pts = [.4,.4,.4]
        alpha_lines = [.6,.6,.6]        
        markers = ['o','s','o']
        markersizes = [1,8,8]
        hatches = ['/','\\','|']
        allBT=[]
        alldsfr=[]
        for i in range(len(xvars)):
            if i > 0:
                plt.plot(xvars[i],yvars[i],'k.',color=colors[i],alpha=alpha_pts[i],zorder=orders[i],markersize=markersizes[i],\
                     marker=markers[i],label=labels[i])
            t = lcscommon.spearmanr(xvars[i],yvars[i])
            ybin,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='mean',bins=mybins)
            yerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='std',bins=mybins)
            nyerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='count',bins=mybins)            
            yerr = yerr/np.sqrt(nyerr)
            dbin = xbin[1]-xbin[0]
            if i == 0:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=4,label=labels[i],zorder=10)
            else:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=4,zorder=10)
            plt.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=colors[i],alpha=alpha_lines[i],zorder=8)            
            print('\n'+labels[i])
            print('r={:.4f}, pvalue={:.3e}'.format(t[0],t[1]))
            allBT.append(xvars[i].tolist())
            alldsfr.append(yvars[i].tolist())            
        plt.xlabel(r'$\rm B/T$',fontsize=22)
        plt.ylabel(r'$\rm \Delta \log_{10}SFR$',fontsize=22)

        #plt.axhline(y=MS_OFFSET,ls='--',color='k',label='$1.5\sigma_{MS}$')
        
        plt.axhline(y=-1*MS_OFFSET,ls='--',color='w',lw=4)
        plt.axhline(y=-1*MS_OFFSET,ls='--',color='b',lw=3,label=r'$\rm MS - 1.5\sigma$')
        plt.legend()
        if BTline is not None:
            plt.axvline(x=BTline,ls=':',color='k')
        print('\n Combined Samples: Spearman Rank')
        #t = lcscommon.spearmanr(allBT,alldsfr)
        #print(t)

        if writefiles:
            coreflag = (core_dsfr < -1*MS_OFFSET) & (coreBT > 0.3)
            outtab = Table([self.lcs.cat['NSAID'][flag1][coreflag],self.lcs.cat['RA_1'][flag1][coreflag],self.lcs.cat['DEC_1'][flag1][coreflag]],names=['NSAID','RA','DEC'])
            outtab.write('core-btgt03-dsfrlt045.fits',overwrite=True)
            # write out infall
            flag = (infall_dsfr < -1*MS_OFFSET) & (infallBT > 0.3)
            outtab = Table([self.lcs.cat['NSAID'][flag2][flag],self.lcs.cat['RA_1'][flag2][flag],self.lcs.cat['DEC_1'][flag2][flag]],names=['NSAID','RA','DEC'])
            outtab.write('infall-btgt03-dsfrlt045.fits',overwrite=True)
        return xvars,yvars
    def plot_dsfr_nsersic(self,nbins=15,xmax=6,writefiles=False,nsersic_cut=10,ecut=1,BTline=None,mmatch=False):
        #nflag = (self.lcs.cat['ng_2'] < nsersic_cut)
        #sflag = (self.lcs.cat['p_el'] < ecut)
        flag1 =  self.lcs.membflag &  self.lcs_mass_sfr_flag

        # use sersic index from simard+2011 single component fits
        
        coreBT = self.lcs.cat['ng_1'][flag1]
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        #core_dsfr = y1-get_BV_MS(x1)
        core_dsfr = y1-get_MS(x1,BTcut=self.cutBT)
        
        flag2 = self.lcs.infallflag & self.lcs_mass_sfr_flag
        infallBT = self.lcs.cat['ng_1'][flag2]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        #infall_dsfr = y2-get_BV_MS(x2)
        infall_dsfr = y2-get_MS(x2,BTcut=self.cutBT)        


        if mmatch:
            ### MASS MATCHING OF FIELD SAMPLE
            # get mass-matched sample of field galaxies
            lcsflag = (self.lcs.membflag | self.lcs.infallflag) & self.lcs_mass_sfr_flag
            fieldflag = self.gsw_mass_sfr_flag 

            keep_indices = mass_match(self.lcs.cat['logMstar'][lcsflag],self.gsw.cat['logMstar'][fieldflag],34278,nmatch=NMASSMATCH)

            fieldBT = self.gsw.cat['ng'][fieldflag][keep_indices]
            x3 = self.gsw.cat['logMstar'][fieldflag][keep_indices]
            y3 = self.gsw.cat['logSFR'][fieldflag][keep_indices]
        else:
            flag3 = self.gsw_mass_sfr_flag   #& self.gsw.field1
            fieldBT = self.gsw.cat['ng'][flag3]
            x3 = self.gsw.cat['logMstar'][flag3]
            y3 = self.gsw.cat['logSFR'][flag3]
            
        #field_dsfr = y3-get_BV_MS(x3)
        field_dsfr = y3-get_MS(x3,BTcut=self.cutBT)


        
        plt.figure()
        plt.subplots_adjust(left=.2)
        mybins = np.linspace(0,xmax,nbins)        
        xvars = [fieldBT,coreBT,infallBT]
        yvars = [field_dsfr,core_dsfr,infall_dsfr]        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alpha_pts = [.4,.4,.4]
        alpha_lines = [.6,.6,.6]        
        markers = ['o','s','o']
        markersizes = [1,8,8]
        hatches = ['/','\\','|']
        allBT=[]
        alldsfr=[]
        for i in range(len(xvars)):
            if i > 0:
                plt.plot(xvars[i],yvars[i],'k.',color=colors[i],alpha=alpha_pts[i],zorder=orders[i],markersize=markersizes[i],\
                     marker=markers[i],label=labels[i])
            t = lcscommon.spearmanr(xvars[i],yvars[i])
            ybin,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='mean',bins=mybins)
            yerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='std',bins=mybins)
            nyerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='count',bins=mybins)            
            yerr = yerr/np.sqrt(nyerr)
            dbin = xbin[1]-xbin[0]
            if i == 0:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=4,label=labels[i],zorder=10)
            else:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=4,zorder=10)
            plt.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=colors[i],alpha=alpha_lines[i],zorder=8)            
            print('\n'+labels[i])
            print('r={:.4f}, pvalue={:.3e}'.format(t[0],t[1]))
            allBT.append(xvars[i].tolist())
            alldsfr.append(yvars[i].tolist())            
        plt.xlabel(r'$\rm Sersic \ index$',fontsize=22)
        plt.ylabel(r'$\rm \Delta \log_{10}SFR$',fontsize=22)

        #plt.axhline(y=MS_OFFSET,ls='--',color='k',label='$1.5\sigma_{MS}$')
        
        plt.axhline(y=-1*MS_OFFSET,ls='--',color='w',lw=4)
        plt.axhline(y=-1*MS_OFFSET,ls='--',color='b',lw=3,label=r'$\rm MS - 1.5\sigma$')
        plt.legend()
        if BTline is not None:
            plt.axvline(x=BTline,ls=':',color='k')
        print('\n Combined Samples: Spearman Rank')
        #t = lcscommon.spearmanr(allBT,alldsfr)
        #print(t)

        if writefiles:
            coreflag = (core_dsfr < -1*MS_OFFSET) & (coreBT > 0.3)
            outtab = Table([self.lcs.cat['NSAID'][flag1][coreflag],self.lcs.cat['RA_1'][flag1][coreflag],self.lcs.cat['DEC_1'][flag1][coreflag]],names=['NSAID','RA','DEC'])
            outtab.write('core-btgt03-dsfrlt045.fits',overwrite=True)
            # write out infall
            flag = (infall_dsfr < -1*MS_OFFSET) & (infallBT > 0.3)
            outtab = Table([self.lcs.cat['NSAID'][flag2][flag],self.lcs.cat['RA_1'][flag2][flag],self.lcs.cat['DEC_1'][flag2][flag]],names=['NSAID','RA','DEC'])
            outtab.write('infall-btgt03-dsfrlt045.fits',overwrite=True)
        return xvars,yvars
    def plot_HIdef_BT(self,nbins=15,xmax=1,writefiles=False,nsersic_cut=10,ecut=1,BTline=None,mmatch=False):
        ''' plot HIdef vs BT for core, infall and field '''
        lcsFlag = self.lcs.cat['HIdef_flag'] & self.lcs_mass_sfr_flag & (self.lcs.membflag | self.lcs.infallflag)

        lcsCoreFlag = lcsFlag & self.lcs.membflag
        lcsInfallFlag = lcsFlag & self.lcs.infallflag        
        # select field galaxies with:
        # HI measurements
        # in sfr and stellar mass cuts

        fieldFlag = self.gsw.HIdef['HIdef_flag'] & self.gsw_mass_sfr_flag

        # get a mass-matched sample of field galaxies
        lcsmass = self.lcs.cat['logMstar'][lcsFlag]
        gswmass = self.gsw.cat['logMstar'][fieldFlag]        
        keep_indices = mass_match(lcsmass,gswmass,3199,nmatch=NMASSMATCH)        
        # plot HIdef vs dSFR

        HIdef_cats = [self.lcs.cat,self.lcs.cat,self.gsw.HIdef]
        HIdefKey = 'HIdef_Boselli'
        dsfr_vars = [self.lcs_dsfr,self.lcs_dsfr,self.gsw_dsfr]
        flags = [lcsCoreFlag,lcsInfallFlag,fieldFlag]


        

        coreBT = self.lcs.cat[BTkey][lcsCoreFlag]
        x1 = self.lcs.cat['logMstar'][lcsCoreFlag]
        y1 = self.lcs.cat['logSFR'][lcsCoreFlag]
        #core_dsfr = y1-get_BV_MS(x1)
        core_dsfr = y1-get_MS(x1,BTcut=self.cutBT)
        
        infallBT = self.lcs.cat[BTkey][lcsInfallFlag]
        x2 = self.lcs.cat['logMstar'][lcsInfallFlag]
        y2 = self.lcs.cat['logSFR'][lcsInfallFlag]
        #infall_dsfr = y2-get_BV_MS(x2)
        infall_dsfr = y2-get_MS(x2,BTcut=self.cutBT)        

        nflag = (self.gsw.cat['ng'] < nsersic_cut)

         
        fieldBT = self.gsw.cat[BTkey][fieldFlag][keep_indices]
        x3 = self.gsw.cat['logMstar'][fieldFlag][keep_indices]
        y3 = self.gsw.cat['logSFR'][fieldFlag][keep_indices]
            
        #field_dsfr = y3-get_BV_MS(x3)
        field_dsfr = y3-get_MS(x3,BTcut=self.cutBT)


        
        plt.figure()
        plt.subplots_adjust(left=.2)
        mybins = np.linspace(0,xmax,nbins)        
        xvars = [fieldBT,coreBT,infallBT]
        yvars = [field_dsfr,core_dsfr,infall_dsfr]        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alpha_pts = [.4,.4,.4]
        alpha_lines = [.6,.6,.6]        
        markers = ['o','s','o']
        markersizes = [1,8,8]
        hatches = ['/','\\','|']
        allBT=[]
        alldsfr=[]
        for i in range(len(xvars)):
            if i > 0:
                plt.plot(xvars[i],yvars[i],'k.',color=colors[i],alpha=alpha_pts[i],zorder=orders[i],markersize=markersizes[i],\
                     marker=markers[i],label=labels[i])
            t = lcscommon.spearmanr(xvars[i],yvars[i])
            ybin,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='mean',bins=mybins)
            yerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='std',bins=mybins)
            nyerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='count',bins=mybins)            
            yerr = yerr/np.sqrt(nyerr)
            dbin = xbin[1]-xbin[0]
            if i == 0:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=4,label=labels[i],zorder=10)
            else:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=4,zorder=10)
            plt.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=colors[i],alpha=alpha_lines[i],zorder=8)            
            print('\n'+labels[i])
            print('r={:.4f}, pvalue={:.3e}'.format(t[0],t[1]))
            allBT.append(xvars[i].tolist())
            alldsfr.append(yvars[i].tolist())            
        plt.xlabel(r'$\rm B/T$',fontsize=22)
        plt.ylabel(r'$\rm HI \ Def$',fontsize=22)

        #plt.axhline(y=MS_OFFSET,ls='--',color='k',label='$1.5\sigma_{MS}$')
        
        #plt.axhline(y=-1*MS_OFFSET,ls='--',color='w',lw=4)
        #plt.axhline(y=-1*MS_OFFSET,ls='--',color='b',lw=3,label=r'$\rm MS - 1.5\sigma$')
        plt.legend()
        if BTline is not None:
            plt.axvline(x=BTline,ls=':',color='k')
        print('\n Combined Samples: Spearman Rank')
        #t = lcscommon.spearmanr(allBT,alldsfr)
        #print(t)

        return xvars,yvars
    def plot_dsfr_mstar(self,nbins=15,xmax=.3,writefiles=False,nsersic_cut=10,ecut=1,BTline=None):
        nflag = (self.lcs.cat['ng_2'] < nsersic_cut)
        sflag = (self.lcs.cat['p_el'] < ecut)
        flag1 = sflag &  nflag & self.lcs.membflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)        
        coreBT = self.lcs.cat[BTkey][flag1]
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        #core_dsfr = y1-get_BV_MS(x1)
        core_dsfr = y1-get_MS(x1,BTcut=self.cutBT)
        
        flag2 = sflag & nflag & self.lcs.infallflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)
        infallBT = self.lcs.cat[BTkey][flag2]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        #infall_dsfr = y2-get_BV_MS(x2)
        infall_dsfr = y2-get_MS(x2,BTcut=self.cutBT)        

        nflag = (self.gsw.cat['ng'] < nsersic_cut)
        flag3 = nflag & (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)   #& self.gsw.field1
        fieldBT = self.gsw.cat[BTkey][flag3]
        x3 = self.gsw.cat['logMstar'][flag3]
        y3 = self.gsw.cat['logSFR'][flag3]
        #field_dsfr = y3-get_BV_MS(x3)
        field_dsfr = y3-get_MS(x3,BTcut=self.cutBT)
        
        plt.figure()
        mybins = np.linspace(9.7,11.25,nbins)        
        xvars = [x3,x1,x2]
        yvars = [field_dsfr,core_dsfr,infall_dsfr]        
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        alphas = [.5,.6,.6]
        markers = ['o','s','o']
        markersizes = [1,8,8]
        hatches = ['/','\\','|']
        allBT=[]
        alldsfr=[]
        for i in range(len(xvars)):
            if i > 0:
                plt.plot(xvars[i],yvars[i],'k.',color=colors[i],alpha=alphas[i],zorder=orders[i],markersize=markersizes[i],\
                     marker=markers[i],label=labels[i])
            t = lcscommon.spearmanr(xvars[i],yvars[i])
            ybin,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='mean',bins=mybins)
            yerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='std',bins=mybins)
            nyerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='count',bins=mybins)            
            yerr = yerr/np.sqrt(nyerr)
            dbin = xbin[1]-xbin[0]
            if i == 0:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=3,label=labels[i])
            else:
                plt.plot(xbin[:-1]+0.5*dbin,ybin,color=colors[i],lw=3)
            plt.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=colors[i],alpha=.4)            
            print('\n'+labels[i])
            print('r={:.4f}, pvalue={:.3e}'.format(t[0],t[1]))
            allBT.append(xvars[i].tolist())
            alldsfr.append(yvars[i].tolist())            
        plt.xlabel('$\log_{10}(M_\star)$',fontsize=20)
        plt.ylabel('$\Delta \log_{10}(SFR \ (M_\odot/yr))$',fontsize=20)

        plt.axhline(y=.45,ls='--',color='k',label='$1.5\sigma_{MS}$')
        plt.axhline(y=-.45,ls='--',color='k')
        plt.legend()
        if BTline is not None:
            plt.axvline(x=BTline,ls=':',color='k')
        print('\n Combined Samples: Spearman Rank')
        #t = lcscommon.spearmanr(allBT,alldsfr)
        #print(t)

        if writefiles:
            coreflag = (core_dsfr < -1*MS_OFFSET) & (coreBT > 0.3)
            outtab = Table([self.lcs.cat['NSAID'][flag1][coreflag],self.lcs.cat['RA_1'][flag1][coreflag],self.lcs.cat['DEC_1'][flag1][coreflag]],names=['NSAID','RA','DEC'])
            outtab.write('core-btgt03-dsfrlt045.fits',overwrite=True)
            # write out infall
            flag = (infall_dsfr < -1*MS_OFFSET) & (infallBT > 0.3)
            outtab = Table([self.lcs.cat['NSAID'][flag2][flag],self.lcs.cat['RA_1'][flag2][flag],self.lcs.cat['DEC_1'][flag2][flag]],names=['NSAID','RA','DEC'])
            outtab.write('infall-btgt03-dsfrlt045.fits',overwrite=True)


    def plot_BThist_env(self,nbins=10,xmax=.3,writefiles=False):
        ''' Plot normalized hist of BT for different environments '''
        flag1 = self.lcs.membflag &  self.lcs.sampleflag
        coreBT = self.lcs.cat[BTkey][flag1]
        
        flag2 = self.lcs.infallflag &  self.lcs.sampleflag
        infallBT = self.lcs.cat[BTkey][flag2]

        flag3 =  (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut) 
        fieldBT = self.gsw.cat[BTkey][flag3]
        
        plt.figure()
        mybins = np.linspace(0,xmax,nbins)        
        delta_bin = mybins[1]-mybins[0]
        mybins = mybins + 0.5*delta_bin

        xvars = [fieldBT,coreBT,infallBT]
        colors = ['.5',darkblue,lightblue]
        labels = ['Field','Core','Infall']
        orders = [1,3,2]
        lws = [3,4,3]
        markers = ['o','s','o']
        markersizes = [1,8,8]
        hatches = ['/','\\','|']

        alphas = [.4,0,.5]        
        hatches = ['/','\\','|']
        for i in range(len(xvars)):
            npoints = len(xvars[i])
            w = np.ones(npoints)/npoints
            #print(w)
            plt.hist(xvars[i],bins=mybins,color=colors[i],weights=w,\
                     histtype='stepfilled',lw=lws[i],alpha=alphas[i],zorder=orders[i])#hatch=hatches[i])
            plt.hist(xvars[i],bins=mybins,color=colors[i],weights=w,\
                     histtype='step',lw=lws[i],zorder=orders[i],label=labels[i])#hatch=hatches[i])
            
            
            #ybin,xbin = np.histogram(xvars[i],bins=mybins)
            #yerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='std',bins=mybins)
            #nyerr,xbin,binnumb = binned_statistic(xvars[i],yvars[i],statistic='count',bins=mybins)            
            #yerr = yerr/np.sqrt(nyerr)
            #dbin = xbin[1]-xbin[0]
            #plt.fill_between(xbin[:-1]+0.5*dbin,ybin+yerr,ybin-yerr,color=colors[i],alpha=.4)            
            #print('\n'+labels[i])
            #print('r={:.4f}, pvalue={:.3e}'.format(t[0],t[1]))
            #plt.plot(xbin,ybin/len(xvars[i]),'ko',color=colors[i],markersize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)        
        plt.xlabel('$B/T$',fontsize=20)
        plt.ylabel('$Fraction$',fontsize=20)
        plt.legend(fontsize=14)

    def get_legacy_images_BT(self,lcsflag=None,ncol=4,nrow=4,fignameroot='dsfr-bt-',figsize=(12,10)):
        '''
        PURPOSE:
        
        '''
        if lcsflag is None:
            lcsflag = self.lcs.cat['membflag']
        
        ids = np.arange(len(self.lcs.cat))[lcsflag]

        plt.figure(figsize=figsize)
        nfig=0
        nplot=0
        for i in ids:
            # open figure
            plt.subplot(nrow,ncol,np.remainder(nplot+1,nrow*ncol)+1)
            # get legacy image
            d, w = getlegacy(self.lcs.cat['RA_1'][i],self.lcs.cat['DEC_2'][i])
            plt.xticks([],[])
            plt.yticks([],[])
            plt.title("sSFR={0:.1f},dSFR={1:.1f},BT={2:.2f},n={3:.1f}".format(self.lcs.ssfr[i],self.lcs.dsfr[i],self.lcs.cat[BTkey][i],self.lcs.cat['SERSIC_N'][i]),fontsize=10)
            if np.remainder(nplot+1,nrow*ncol) == 0:
                # write out file and start a new one
                plt.savefig(fignameroot+str(nfig)+'.png')
                nfig+=1
                plt.figure(figsize=figsize)
            nplot += 1
        plt.savefig(fignameroot+str(nfig)+'.png')
    def core_getlegacy(self,figsize=(12,10),extraflag=None,dsfrcut=-0.45):
        ''' 
        PURPOSE:
        * get legacy images for core galaxies with low dsfr and high B/T  

        INPUT
        * extraflag = any cuts beyone membership, mass, ssfrcut, and dist from MS
        * dsfrcut = distance from MS, default = 1.5sigma = -0.45
        '''
        flag = self.lcs.membflag &  (self.lcs.cat['logMstar']> self.masscut)  \
            & (self.lcs.ssfr > self.ssfrcut) \
            & (self.lcs_dsfr < dsfrcut) #& (self.lcs.cat[BTkey] > 0.3)
        if extraflag is not None:
            flag = flag & extraflag
        print('getting legacy images for ',sum(flag),' galaxies')
        self.get_legacy_images_BT(flag,fignameroot='core-dsfr-bt-',figsize=figsize)
    def infall_getlegacy(self,figsize=(12,10),extraflag=None):
        '''get legacy images for core galaxies with low dsfr and high B/T  '''
        flag = self.lcs.infallflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut) &\
            (self.lcs_dsfr < -1*MS_OFFSET) #& (self.lcs.cat[BTkey] > 0.3)
        if extraflag is not None:
            flag = flag & extraflag
        print('getting legacy images for ',sum(flag),' galaxies')
        self.get_legacy_images_BT(flag,fignameroot='infall-dsfr-bt-',figsize=figsize)
    def gsw_nobt_getlegacy(self,figsize=(12,10),ncol=4,nrow=4,extraflag=None):
        '''get legacy images for core galaxies with low dsfr and high B/T  '''
        flag = self.gsw_mass_sfr_flag & ~(self.gsw.cat[BTkey] < 1.1)
        if extraflag is not None:
            flag = flag & extraflag
            
        print('getting legacy images for ',sum(flag),' galaxies')

        ids = np.arange(len(self.gsw.cat))[flag]
        fignameroot='gsw-no-bt-'        
        plt.figure(figsize=figsize)
        nfig=0
        nplot=0
        for i in ids:
            # open figure
            plt.subplot(nrow,ncol,np.remainder(nplot+1,nrow*ncol)+1)
            # get legacy image
            d, w = getlegacy(self.gsw.cat['RA'][i],self.gsw.cat['DEC'][i])
            plt.xticks([],[])
            plt.yticks([],[])
            #plt.title("sSFR={0:.1f},dSFR={1:.1f},BT={2:.2f},n={3:.1f}".format(self.gsw.ssfr[i],self.gsw.dsfr[i],self.lcs.cat[BTkey][i],self.lcs.cat['SERSIC_N'][i]),fontsize=10)
            if np.remainder(nplot+1,nrow*ncol) == 0:
                # write out file and start a new one
                plt.savefig(fignameroot+str(nfig)+'.png')
                nfig+=1
                plt.figure(figsize=figsize)
            nplot += 1


        plt.savefig(fignameroot+str(nfig)+'.png')
        
        
    def plot_dvdr_sfgals(self, figname1=None,figname2=None,coreflag=False):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)
        xmin,xmax,ymin,ymax = 0,3.5,0,3.5
        #if plotsingle:
        #    plt.figure(figsize=(8,6))
        #    ax=plt.gca()
        #    plt.subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
        #    plt.ylabel('$ \Delta v/\sigma $',fontsize=26)
        #    plt.xlabel('$ \Delta R/R_{200}  $',fontsize=26)
        #    plt.legend(loc='upper left',numpoints=1)

        # plot low sfr galaxies

        #lowsfrcoreflag = self.lowsfr_flag & self.membflag
        #lowsfrinfallflag = self.lowsfr_flag & self.infallflag
        if coreflag:
            lcsflag = self.lcs.membflag
        else:
            lcsflag = (self.lcs.membflag | self.lcs.infallflag)
        parentflag = self.lcs_mass_sfr_flag & lcsflag
        lowsfrflag = parentflag & self.lcs.lowsfr_flag #& self.infallflag
        normalsfrflag = parentflag & ~self.lcs.lowsfr_flag
        #####
        ## CHECK NUMBERS
        #####
        print('number in parent sample = ',sum(parentflag))        
        print('number in low SFR sample = ',sum(lowsfrflag))
        print('number in normal SFR sample = ',sum(normalsfrflag))        
        
        x=(self.lcs.cat['DR_R200'])
        y=abs(self.lcs.cat['DELTA_V'])
        x1 = x[normalsfrflag]
        y1 = y[normalsfrflag]        
        x2 = x[lowsfrflag]
        y2 = y[lowsfrflag]
        label='low SFR LCS'
        ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'Normal SFR LCS',label,'temp.pdf',ymin=-.02,ymax=3.02,xmin=0,xmax=3,nhistbin=10,ylabel='$\Delta v/\sigma$',xlabel='$\Delta R/R_{200}$',contourflag=False,alphagray=.7,hexbinflag=False,color2='k',color1='0.5',alpha1=1,ssfrlimit=-11.5,cumulativeFlag=True)

        
        
        ## DIVISION BETWEEN CORE/INFALL
        xl=np.arange(0,2,.1)
        ax1.plot(xl,-4./3.*xl+2,'k-',lw=3,color=darkblue)

        ## LABELS FOR CORE/INFALL
        props = dict(boxstyle='square', facecolor=darkblue, alpha=0.9)
        ax1.text(.03,.32,'CORE',transform=ax1.transAxes,fontsize=18,color='k',bbox=props)
        props = dict(boxstyle='square', facecolor=lightblue, alpha=0.6)        
        ax1.text(.6,.6,'INFALL',transform=ax1.transAxes,fontsize=18,color='k',bbox=props)        

        ## POINTS FOR LOW SFR CORE/INFALL
        ax1.plot(x2,y2,'ko',mec='k',markersize=10)                
        
        
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)            

    def plot_dvdr_sfgals_2panel(self, figname1=None,figname2=None,coreflag=False,HIflag=False):
        # log10(chabrier) = log10(Salpeter) - .25 (SFR estimate)
        # log10(chabrier) = log10(diet Salpeter) - 0.1 (Stellar mass estimates)
        xmin,xmax,ymin,ymax = 0,3.5,0,3.5
        #if plotsingle:
        #    plt.figure(figsize=(8,6))
        #    ax=plt.gca()
        #    plt.subplots_adjust(left=.1,bottom=.15,top=.9,right=.9)
        #    plt.ylabel('$ \Delta v/\sigma $',fontsize=26)
        #    plt.xlabel('$ \Delta R/R_{200}  $',fontsize=26)
        #    plt.legend(loc='upper left',numpoints=1)

        # plot low sfr galaxies

        #lowsfrcoreflag = self.lowsfr_flag & self.membflag
        #lowsfrinfallflag = self.lowsfr_flag & self.infallflag
        if coreflag:
            lcsflag = self.lcs.membflag
        else:
            lcsflag = (self.lcs.membflag | self.lcs.infallflag)
        parentflag = self.lcs_mass_sfr_flag & lcsflag
        lowsfrflag = parentflag & self.lcs.lowsfr_flag #& self.infallflag
        normalsfrflag = parentflag & ~self.lcs.lowsfr_flag
        #####
        ## CHECK NUMBERS
        #####
        print('number in parent sample = ',sum(parentflag))        
        print('number in low SFR sample = ',sum(lowsfrflag))
        print('number in normal SFR sample = ',sum(normalsfrflag))        
        
        x=(self.lcs.cat['DR_R200'])
        y=abs(self.lcs.cat['DV_SIGMA'])
        x1 = x[normalsfrflag]
        y1 = y[normalsfrflag]        
        x2 = x[lowsfrflag]
        y2 = y[lowsfrflag]
        print('just checking, number of low sfr sample = {}'.format(len(x2)))
        print('just checking, number of normal sfr sample = {}'.format(len(x1)))        
        label2=r'$\rm Low \ SFR$'

        ##  COMPARE POPULATIONS
        print("")
        print("delta R: normal vs suppressed")
        t = anderson_ksamp([x1,x2])
        print('Anderson-Darling: ',t)
        pvalue_x = t[2]
        print('pvalue = {:.3e}'.format(pvalue_x))
        print("")
        print("delta v/sigma: normal vs suppressed")
        t = anderson_ksamp([y1,y2])
        print('Anderson-Darling: ',t)
        pvalue_y = t[2]
        print('pvalue = {:.3e}'.format(pvalue_y))
        

        #ax1,ax2,ax3 = colormass(x1,y1,x2,y2,'Normal SFR LCS',label,'temp.pdf',ymin=-.02,ymax=3.02,xmin=0,xmax=3,nhistbin=10,ylabel='$\Delta v/\sigma$',xlabel='$\Delta R/R_{200}$',contourflag=False,alphagray=.7,hexbinflag=False,color2='k',color1='0.5',alpha1=1,ssfrlimit=-11.5,cumulativeFlag=True)

        ## USING SUBPLOT2GRID INSTEAD

        label1 = r'$\rm Normal \ SFR$'
        ymin=-.02
        ymax=3.02
        xmin=0
        xmax=3

        nhistbin=10,
        ylabel=r'$\rm \Delta v/\sigma$'
        xlabel=r'$\rm \Delta R/R_{200}$'
        color1='0.5'
        color2='k'        

        nplotx=4
        nploty=4
        plt.figure(figsize=(12,5))

        gs = gridspec.GridSpec(2,2)
        ax1 = plt.subplot(gs[:,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,1])
        plt.subplots_adjust(wspace=.25,hspace=.5)        
        #ax1 = plt.subplot2grid((nplotx,nploty),(0,0),rowspan=4,colspan=2)
        #ax2 = plt.subplot2grid((nplotx,nploty),(0,2),colspan=2,rowspan=2)
        #ax3 = plt.subplot2grid((nplotx,nploty),(2,2),colspan=2,rowspan=2)

        ax1.plot(x1,y1,'ko',markersize=5,label=label1,color=color1)
        ## POINTS FOR LOW SFR CORE/INFALL        
        ax1.plot(x2,y2,'ko',markersize=8,marker='v',label=label2,color=color2)
        ax1.axis([xmin,xmax,ymin,ymax])
        
        ax1.set_xlabel(xlabel,fontsize=20)
        ax1.set_ylabel(ylabel,fontsize=20)        
        
        ## DIVISION BETWEEN CORE/INFALL
        xline=np.linspace(0,3,50)
        yline=-4./3.*xline+2
        ax1.plot(xline,yline,'k-',lw=3,color=darkblue)

        # try shading the core region, rather than label
        ax1.fill_between(xline,yline,color=darkblue,alpha=.1,zorder=0)

        # try shading the core region, rather than label
        ax1.fill_between(xline,3*np.ones(len(xline)),y2=yline,color=lightblue,alpha=.1,zorder=0)
        
        ## LABELS FOR CORE/INFALL
        props = dict(boxstyle='square', facecolor=darkblue, alpha=0.9)
        #ax1.text(.03,.32,'CORE',transform=ax1.transAxes,fontsize=18,color='k',bbox=props)
        props = dict(boxstyle='square', facecolor=lightblue, alpha=0.6)        
        #ax1.text(.6,.6,'INFALL',transform=ax1.transAxes,fontsize=18,color='k',bbox=props)        



        
        ## ADD HISTOGRAMS ON THE RIGHT

        # HISTOGRAM OF DELTA V/SIGMA
        try:
            ax3.hist(x1,bins=len(x1),cumulative=True,normed=True,label=label1,histtype='step',color=color1,lw=2)
            ax3.hist(x2,bins=len(x2),cumulative=True,normed=True,label=label2,histtype='step',color=color2,lw=3)
            ax2.hist(y1,bins=len(x1),cumulative=True,normed=True,label=label1,histtype='step',color=color1,lw=2)
            ax2.hist(y2,bins=len(x2),cumulative=True,normed=True,label=label2,histtype='step',color=color2,lw=3)

        except AttributeError:
            ax3.hist(x1,bins=len(x1),cumulative=True,density=True,stacked=True,label=label1,histtype='step',color=color1,lw=2)
            ax3.hist(x2,bins=len(x2),cumulative=True,density=True,stacked=True,label=label2,histtype='step',color=color2,lw=3)
            ax2.hist(y1,bins=len(x1),cumulative=True,density=True,stacked=True,label=label1,histtype='step',color=color1,lw=2)
            ax2.hist(y2,bins=len(x2),cumulative=True,density=True,stacked=True,label=label2,histtype='step',color=color2,lw=3)
            

        ax3.set_xlabel(xlabel,fontsize=20)
        ax3.text(0,1.05,r'$\rm pvalue:\ low\ vs\ normal={:.2f}$'.format(pvalue_x),fontsize=14)        
        # HISTOGRAM OF DELTA R/R200


        ax2.set_xlabel(ylabel,fontsize=20)
        ax2.text(0,1.05,r'$\rm pvalue:\ low\ vs\ normal={:.2f}$'.format(pvalue_y),fontsize=14)

        ## ADD THE HI SOURCES
        if HIflag:
            lcsHIFlag = self.lcs.cat['HIdef_flag'] & self.lcs_mass_sfr_flag & (self.lcs.membflag | self.lcs.infallflag)
            ax1.plot(x[lcsHIFlag],y[lcsHIFlag],'bs',markerfacecolor='None',markersize=8,label=r'$\rm HI \ detect.$')
            try:
                ax3.hist(x[lcsHIFlag],bins=sum(lcsHIFlag),cumulative=True,normed=True,label='HI',histtype='step',color='b')
                ax2.hist(y[lcsHIFlag],bins=sum(lcsHIFlag),cumulative=True,normed=True,label='HI',histtype='step',color='b')
            except AttributeError:
                ax3.hist(x[lcsHIFlag],bins=sum(lcsHIFlag),cumulative=True,density=True,stacked=True,label='HI',histtype='step',color='b')
                ax2.hist(y[lcsHIFlag],bins=sum(lcsHIFlag),cumulative=True,density=True,stacked=True,label='HI',histtype='step',color='b')
                
            # fraction of normal SF galaxies with HI detections
            Nnormal = np.sum(normalsfrflag)
            NnormalHI = np.sum(normalsfrflag & lcsHIFlag)
            Nsuppressed = np.sum(lowsfrflag)
            NsuppressedHI = np.sum(lowsfrflag & lcsHIFlag)

            binom_err = binom_conf_interval(NnormalHI,Nnormal)#lower, upper
            frac =  NnormalHI/Nnormal
            print('fraction of normal galaxies with HI detections = {:.2f}-{:.2f}+{:.2f}'.format(frac,frac-binom_err[0],binom_err[1]-frac))
            print(binom_err)
            binom_err = binom_conf_interval(NsuppressedHI,Nsuppressed)#lower, upper
            frac =  NsuppressedHI/Nsuppressed
                  
            print('fraction of suppressed galaxies with HI detections = {:.2f}-{:.2f}+{:.2f}'.format(frac,frac-binom_err[0],binom_err[1]-frac))
            
            t = anderson_ksamp([y1,y[lcsHIFlag]])
            pvalue_y_HI = t[2]
            ax2.text(0,.9,r'$\rm pvalue:\ HI\ vs\ normal={:.2f}$'.format(pvalue_y_HI),fontsize=14)            
            # fraction of suppressed SF galaxies with HI detections
            print("DR of HI vs normal:")
            t = anderson_ksamp([x1,x[lcsHIFlag]])
            print('anderson Darling: ',t)
            print()
            print('KS: ',ks_2samp(x1,x[lcsHIFlag]))
            pvalue_x_HI = t[2]
            ax3.text(0,.9,r'$\rm pvalue:\ HI\ vs\ normal={:.3f}$'.format(pvalue_x_HI),fontsize=14)            
            # fraction of suppressed SF galaxies with HI detections

        ## ADD LEGENDS AFTER HI
        ax1.legend(loc='upper left',bbox_to_anchor=(-.02,1.12),ncol=3,handletextpad=.1,columnspacing=1.1)#,horizontalalignment='left')
        ax2.legend(loc='lower right')
        ax2.text(-.15,-.2,r'$\rm Cumulative Distribution $',fontsize=20,transform=ax2.transAxes,rotation=90,verticalalignment='center')
        #ax3.legend(loc='lower right')
        ax2.axis([-.05,3,-.05,1.2])
        ax3.axis([-.05,3,-.05,1.2])        
            
        if figname1 is not None:
            plt.savefig(figname1)
        if figname2 is not None:
            plt.savefig(figname2)            
    def plot_2d_image_phasespace(self,nbins=10):
        lcsHIFlag = self.lcs.cat['HIdef_flag'] & self.lcs_mass_sfr_flag & (self.lcs.membflag | self.lcs.infallflag)
        lcsflag = (self.lcs.membflag | self.lcs.infallflag)
        parentflag = self.lcs_mass_sfr_flag & lcsflag
        lowsfrflag = parentflag & self.lcs.lowsfr_flag #& self.infallflag
        normalsfrflag = parentflag & ~self.lcs.lowsfr_flag

        # define x,y data for phase space

        x=(self.lcs.cat['DR_R200'])
        y=abs(self.lcs.cat['DV_SIGMA'])
        
        # grid data
        mybins = np.linspace(0,2,nbins)

        lowbinned = binned_statistic_2d(x[lowsfrflag],y[lowsfrflag],np.ones(sum(lowsfrflag)),statistic='sum',bins=mybins)
        normalbinned = binned_statistic_2d(x[normalsfrflag],y[normalsfrflag],np.ones(sum(normalsfrflag)),statistic='sum',bins=mybins)

        totbinned = binned_statistic_2d(x[lcsflag],y[lcsflag],np.ones(sum(lcsflag)),statistic='sum',bins=mybins)
        HIbinned = binned_statistic_2d(x[lcsHIFlag],y[lcsHIFlag],np.ones(sum(lcsHIFlag)),statistic='sum',bins=mybins)
        # plot fraction with low SFR
        fracs = [lowbinned[0]/totbinned[0], HIbinned[0]/totbinned[0], HIbinned[0]/normalbinned[0]]
        titles = ['Fraction of galaxies with low SFR',\
                  'Fraction of galaxies with HI detections',\
                  'Fraction of normal SFR galaxies with HI detections']

        for i,f in enumerate(fracs):
            plt.figure()
            plt.imshow(f,origin='lower',cmap='Greys')
            cb = plt.colorbar()
            plt.title(titles[i])
        
    def plot_frac_suppressed(self,uvirflag=False,BTcut=None,plotsingle=True,massmatch=False):
        '''fraction of suppressed galaxies vs environment'''

        slimit = -1*MS_OFFSET
        flag1 = self.lcs.membflag &  (self.lcs_mass_sfr_flag)
        if BTcut is not None:
            flag1 = flag1 & (self.lcs.cat[BTkey] < BTcut)
        if uvirflag:
            flag1 = flag1 & (self.lcs_uvir_flag)

        coreBT = self.lcs.cat[BTkey][flag1]
        x1 = self.lcs.cat['logMstar'][flag1]
        y1 = self.lcs.cat['logSFR'][flag1]
        if BTcut is not None:
            core_dsfr = y1-get_MS_BTcut(x1,BTcut=self.cutBT)
        else:
            core_dsfr = y1-get_MS(x1,BTcut=self.cutBT)
        core_fsup = sum(core_dsfr < slimit)/len(core_dsfr)
        core_err = binom_conf_interval(sum(core_dsfr < slimit),len(core_dsfr))#lower, upper

        print('CORE')        
        print('frac suppressed = {:.3f}, {:.3f},{:.3f}'.format(core_fsup,core_err[0]-core_fsup,core_err[1]-core_fsup))
        
        flag2 = self.lcs.infallflag &  (self.lcs_mass_sfr_flag)
        if BTcut is not None:
            flag2 = flag2 & (self.lcs.cat[BTkey] < BTcut)
        
        if uvirflag:
            flag2 = flag2 & self.lcs_uvir_flag
        infallBT = self.lcs.cat[BTkey][flag2]
        x2 = self.lcs.cat['logMstar'][flag2]
        y2 = self.lcs.cat['logSFR'][flag2]
        #infall_dsfr = y2-get_BV_MS(x2)
        if BTcut is not None:
            infall_dsfr = y2-get_MS_BTcut(x2,BTcut=self.cutBT)
        else:
            infall_dsfr = y2-get_MS(x2,BTcut=self.cutBT)
        infall_fsup = sum(infall_dsfr < slimit)/len(infall_dsfr)
        infall_err = binom_conf_interval(sum(infall_dsfr < slimit),len(infall_dsfr))#lower, upper
        print('INFALL')        
        print(infall_fsup,infall_err-infall_fsup)
        
        flag3 = (self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)  #& self.gsw.field1
        if BTcut is not None:
            flag3 = flag3 & (self.gsw.cat[BTkey] < BTcut)

        fieldBT = self.gsw.cat[BTkey][flag3]

        if massmatch:
            if BTcut is not None:
                gsw_flag = self.gsw_mass_sfr_flag & (self.gsw.cat[BTkey] < BTcut)
            else:
                gsw_flag = self.gsw_mass_sfr_flag 
            keep_indices = mass_match(self.lcs.cat['logMstar'][self.lcs_mass_sfr_flag],\
                                      self.gsw.cat['logMstar'][gsw_flag],\
                                      nmatch=NMASSMATCH,seed=379)            
            x3 = self.gsw.cat['logMstar'][gsw_flag][keep_indices]
            y3 = self.gsw.cat['logSFR'][gsw_flag][keep_indices]
            fieldBT = self.gsw.cat[BTkey][gsw_flag][keep_indices]
        else:
            x3 = self.gsw.cat['logMstar'][flag3]
            y3 = self.gsw.cat['logSFR'][flag3]
            fieldBT = self.gsw.cat[BTkey][flag3]
        #field_dsfr = y3-get_BV_MS(x3)
        if BTcut is not None:
            field_dsfr = y3-get_MS_BTcut(x3,BTcut=self.cutBT)
        else:
            field_dsfr = y3-get_MS(x3,BTcut=self.cutBT)
        field_fsup = sum(field_dsfr < slimit)/len(field_dsfr)
        field_err = binom_conf_interval(sum(field_dsfr < slimit),len(field_dsfr))#lower, upper
        print('FIELD')
        print(field_fsup,field_err-field_fsup)
        print(np.array(field_err).shape)
        if plotsingle:
            plt.figure()
        xvars = [0,2,1]
        yvars = [field_fsup,core_fsup,infall_fsup]
        yerrs = [field_err,core_err,infall_err]                
        colors = ['.5',darkblue,lightblue]
        labels = ['$Field$','$Core$','$Infall$']
        orders = [1,3,2]
        lws = [3,4,3]
        alphas = [.8,.8,.8]
        markers = ['o','s','o']
        markersizes = 20*np.ones(3)
        for i in [0,2,1]: # change order so we go field, infall, core
            yerr = np.zeros((2,1))
            yerr[0] = yvars[i]-yerrs[i][0]
            yerr[1] = yerrs[i][1]-yvars[i]
            if plotsingle:
                markerfacecolor = colors[i]
                marker = 'o'
            else:
                markerfacecolor = colors[i]
                marker = '^'
            plt.errorbar(np.array(xvars[i]),np.array(yvars[i]),\
                         yerr=yerr,\
                         color=colors[i],alpha=alphas[i],zorder=orders[i],\
                         markersize=markersizes[i],\
                         fmt=marker,label=labels[i],markerfacecolor=markerfacecolor)
        if plotsingle:
            plt.xticks(np.arange(0,3),[r'$\rm Field$',r'$\rm Infall$',r'$\rm Core$'],fontsize=22)
            #plt.xlabel('$Environment$',fontsize=20)
            plt.ylabel(r'$\rm Suppressed \ Fraction $',fontsize=22)
        plt.xlim(-0.4,2.4)
        
    def compare_lcs_sfr(self,core=True,infall=True):
        '''hist of sfrs of core vs infall lcs galaxies'''
        plt.figure(figsize=(8,5))
        plt.subplots_adjust(bottom=.2,left=.05)
        mybins=np.linspace(-2.5,2,10)
        ssfr = self.lcs.cat['logSFR'] - self.lcs.cat['logMstar']
        flag1 = (self.lcs.cat['logSFR'] > -50) & (ssfr > self.ssfrcut) & (self.lcs.cat['logMstar'] > self.masscut)
        x1 = self.lcs.cat['logSFR'][self.lcs.cat['membflag'] & flag1 ]
        x2 = self.lcs.cat['logSFR'][~self.lcs.cat['membflag'] &( self.lcs.cat['DELTA_V'] < 3.) & flag1]
        print('KS test comparing SFRs:')
        t = ks_2samp(x1,x2)
        print(t)
        vars = [x1,x2]
        labels = ['Core','Infall']
        mycolors = ['r','b']
        myhatch=['/','\\']
        for i in [0,1]:
            #plt.subplot(1,2,i+1)
            if (i == 0) and not(core):
                continue

            if (i == 1) and not(infall):
                continue
            t = plt.hist(vars[i],label=labels[i],lw=2,bins=mybins,hatch=myhatch[i],color=mycolors[i],histtype='step')                
        plt.xlabel('$log_{10}(SFR/(M_\odot/yr))$',fontsize=24)
        #plt.axis([-2.5,2,0,50])
        plt.legend(loc='upper left')
        if not(core):
            outfile=homedir+'/research/LCS/plots/lcsinfall-sfrs'
        elif not(infall):
            outfile=homedir+'/research/LCS/plots/lcscore-sfrs'
        else:
            outfile=homedir+'/research/LCS/plots/lcscore-infall-sfrs'
        if self.cutBT:
            plt.savefig(outfile+'-BTcut.png')
            plt.savefig(outfile+'-BTcut.pdf')        
        else:
            plt.savefig(outfile+'.png')
            plt.savefig(outfile+'.pdf')
    def ks_stats(self,massmatch=False):
        '''
        GOAL:
        * write out tables to use for computing KS statistics for table in paper comparing 
          - LCS core/field vs GSWLC
          - LCS core vs field
          - B/T cut is set when calling program
          - with and without mass matching

        2022-06-24
        * adding sersic index so I can try cutting on that instead of 
        '''
        #flag3 = lcsflag &  (self.lcs.cat['logMstar']> self.masscut)  & (self.lcs.ssfr > self.ssfrcut)

        # store all KS test restults, for writing latex table
        all_results = []
        lcsmemb = self.lcs.membflag & self.lcs_mass_sfr_flag
        lcsinfall = self.lcs.infallflag & self.lcs_mass_sfr_flag

        mc = self.lcs.cat['logMstar'][lcsmemb]
        sfrc = self.lcs.cat['logSFR'][lcsmemb]
        #dsfrc = sfrc - get_BV_MS(mc)
        dsfrc = sfrc - get_MS(mc,BTcut=self.cutBT)
        zc = self.lcs.cat['Z_1'][lcsmemb]
        BTc = self.lcs.cat[BTkey][lcsmemb]
        
        mi = self.lcs.cat['logMstar'][lcsinfall]        
        sfri = self.lcs.cat['logSFR'][lcsinfall]
        #dsfri = sfri - get_BV_MS(mi)
        dsfri = sfri - get_MS(mi,BTcut=self.cutBT)
        zi = self.lcs.cat['Z_1'][lcsinfall]
        BTi = self.lcs.cat[BTkey][lcsinfall]
        
        field = self.gsw_mass_sfr_flag #(self.gsw.cat['logMstar'] > self.masscut) & (self.gsw.ssfr > self.ssfrcut)
        mf = self.gsw.cat['logMstar'][field]
        sfrf = self.gsw.cat['logSFR'][field]
        #dsfrf = sfrf - get_BV_MS(mf)
        dsfrf = sfrf - get_MS(mf,BTcut=self.cutBT)
        zf = self.gsw.cat['Z_1'][field]
        BTf = self.gsw.cat[BTkey][field]        

        # write out file to use in latex table program
        allnames = ['logmstar','logsfr','dlogsfr','z','BT']

        alldata = [[mc,sfrc,dsfrc,zc,BTc],\
                   [mi,sfri,dsfri,zi,BTi],\
                   [mf,sfrf,dsfrf,zf,BTf]]
        outnames = ['core','infall','field']
        for i in range(len(alldata)):
            
            stab = Table(alldata[i],names=allnames)
            if massmatch:
                basename = 'LCS-sfr-mstar-{}-mmatch'.format(outnames[i]) 
            else:
                basename = 'LCS-sfr-mstar-{}'.format(outnames[i]) 
            if args.cutBT:
                sfile = basename+'-BTcut.fits'
            else:
                sfile = basename+'.fits'.format(outnames[i])
            stab.write(sfile,format='fits',overwrite=True)
    def compare_HIdef(self,combineLCS=False):
        ''' 
        GOAL: compare HIdef for field and LCS core, infall

        '''

        # select LCS galaxies with:
        # HI measurements
        # in sfr and stellar mass cuts

        lcsFlag = self.lcs.cat['HIdef_flag'] & self.lcs_mass_sfr_flag & (self.lcs.membflag | self.lcs.infallflag)

        lcsCoreFlag = lcsFlag & self.lcs.membflag
        lcsInfallFlag = lcsFlag & self.lcs.infallflag        
        # select field galaxies with:
        # HI measurements
        # in sfr and stellar mass cuts

        fieldFlag = self.gsw.HIdef['HIdef_flag'] & self.gsw_mass_sfr_flag

        # get a mass-matched sample of field galaxies
        lcsmass = self.lcs.cat['logMstar'][lcsFlag]
        gswmass = self.gsw.cat['logMstar'][fieldFlag]        
        keep_indices = mass_match(lcsmass,gswmass,3199,nmatch=NMASSMATCH)        
        # plot HIdef vs dSFR

        HIdef_cats = [self.lcs.cat,self.lcs.cat,self.gsw.HIdef]
        HIdefKey = 'HIdef_Boselli'
        dsfr_vars = [self.lcs_dsfr,self.lcs_dsfr,self.gsw_dsfr]
        flags = [lcsCoreFlag,lcsInfallFlag,fieldFlag]
        labels = ['Core','Infall','Field']
        colors = [darkblue, lightblue,'0.5']
        markers = ['s','o','o']
        mecs = ['k','k','None']        
        alphas = [.8,.7,.4]
        msize = [8,8,4]
        fig, ax = plt.subplots(figsize=(8,6))
        #colors = ['.5',darkblue,lightblue]
        #labels = ['Field','Core','Infall']
        orders = [3,2,1]
        lws = [4,3,3]
        histalphas = [0,.5,.4]        
        
        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

        # make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)        
        for i in [2,1,0]:
            # plot LCS core        
            # plot LCS infall
            # plot field
            x = dsfr_vars[i][flags[i]]
            y = HIdef_cats[i][HIdefKey][flags[i]]
            if i == 2:
                x = dsfr_vars[i][flags[i]][keep_indices]
                y = HIdef_cats[i][HIdefKey][flags[i]][keep_indices]
                
            nbins=8
            ax.plot(x,y,'bo',color=colors[i],alpha=alphas[i],markersize=msize[i],label=labels[i],marker=markers[i],mec=mecs[i])

            # plot filled histogram, then same histogram with a heavier line
            try:
                ax_histx.hist(x,bins=nbins,cumulative=False,normed=True,histtype='stepfilled',color=colors[i],lw=lws[i],alpha=histalphas[i],zorder=orders[i])
                ax_histx.hist(x,bins=nbins,cumulative=False,normed=True,histtype='step',color=colors[i],lw=lws[i],zorder=orders[i])

            # plot y histograms
                ax_histy.hist(y,bins=nbins,cumulative=False,normed=True,histtype='stepfilled',orientation='horizontal',color=colors[i],lw=lws[i],alpha=histalphas[i],zorder=orders[i])
                ax_histy.hist(y,bins=nbins,cumulative=False,normed=True,histtype='step',orientation='horizontal',color=colors[i],lw=lws[i],zorder=orders[i])            
            except AttributeError:
                ax_histx.hist(x,bins=nbins,cumulative=False,density=True,stacked=True,histtype='stepfilled',color=colors[i],lw=lws[i],alpha=histalphas[i],zorder=orders[i])
                ax_histx.hist(x,bins=nbins,cumulative=False,density=True,stacked=True,histtype='step',color=colors[i],lw=lws[i],zorder=orders[i])

            # plot y histograms
                ax_histy.hist(y,bins=nbins,cumulative=False,density=True,stacked=True,histtype='stepfilled',orientation='horizontal',color=colors[i],lw=lws[i],alpha=histalphas[i],zorder=orders[i])
                ax_histy.hist(y,bins=nbins,cumulative=False,density=True,stacked=True,histtype='step',orientation='horizontal',color=colors[i],lw=lws[i],zorder=orders[i])            

        ax.set_xlabel(r'$\rm \Delta \log_{10} SFR$',fontsize=20)
        ax.set_ylabel(r'$\rm HI \ Deficiency $',fontsize=20)

        # plot line for suppressed galaxies
        ax.axvline(x=-1*MS_OFFSET,ls='--',color='b',label=r'$\rm MS-1.5\sigma $')                
        # add linear fit to dSFR vs HIdef for field galaxies
        x = dsfr_vars[2][flags[2]]
        y = HIdef_cats[2][HIdefKey][flags[2]]
        
        popt,V = np.polyfit(x,y,1,cov=True)
        xline = np.linspace(-.75,.75,100)
        yline = np.polyval(popt,xline)
        ax.plot(xline,yline,'k-',lw=2,label='Fit to Field')
        ax.legend()
        # print best-fit line slope and error
        print()
        print('best-field line for field = {:.3f}+/-{:.3f}'.format(popt[0],np.sqrt(V[0][0])))
        print()
        # test for correlation between HIdef and dSFR
        # field
        x = dsfr_vars[2][flags[2]]
        y = HIdef_cats[2][HIdefKey][flags[2]]
        print('Spearman Rank test between dSFR and HIdef for field:')
        t = lcscommon.spearmanr(x,y)
        print(t)
        print()
 
        x = dsfr_vars[0][flags[0]]
        y = HIdef_cats[0][HIdefKey][flags[0]]
        print('Spearman Rank test between dSFR and HIdef for core:')
        t = lcscommon.spearmanr(x,y)
        print(t)
        print()

        x = dsfr_vars[1][flags[1]]
        y = HIdef_cats[1][HIdefKey][flags[1]]
        print('Spearman Rank test between dSFR and HIdef for infall:')
        t = lcscommon.spearmanr(x,y)
        print(t)
        print()
        
        
        print('number is core={}, infall={},field={}'.format(sum(flags[0]),sum(flags[1]),sum(flags[2])))
        print()
        # ks test comparing LCS core and field
        x1 = dsfr_vars[0][flags[0]]
        x2 = dsfr_vars[2][flags[2]][keep_indices]
        print()
        print('comparing LCS core vs Field: dSFR')

        print(ks_2samp(x1,x2))

        t = anderson_ksamp([x1,x2])
        print('Anderson-Darling: ',t)
        
        
        y1 = HIdef_cats[0][HIdefKey][flags[0]]
        y2 = HIdef_cats[2][HIdefKey][flags[2]][keep_indices]
        print()
        print('comparing LCS core vs Field: HIdef')
        print(ks_2samp(y1,y2))        
        t = anderson_ksamp([y1,y2])
        print('Anderson-Darling: ',t)

        # ks test comparing LCS infall and field
        x1 = dsfr_vars[1][flags[1]]
        x2 = dsfr_vars[2][flags[2]][keep_indices]
        print()
        print('comparing LCS infall vs Field: dSFR')

        print(ks_2samp(x1,x2))

        t = anderson_ksamp([x1,x2])
        print('Anderson-Darling: ',t)
        
        
        y1 = HIdef_cats[1][HIdefKey][flags[1]]
        y2 = HIdef_cats[2][HIdefKey][flags[2]][keep_indices]
        print()
        print('comparing LCS infall vs Field: HIdef')
        print(ks_2samp(y1,y2))        
        t = anderson_ksamp([y1,y2])
        print('Anderson-Darling: ',t)


        
        # ks test comparing LCS core and infall
        x1 = dsfr_vars[1][flags[1]]
        x2 = dsfr_vars[0][flags[0]]
        print()
        print('comparing LCS infall vs core: dSFR')

        print(ks_2samp(x1,x2))

        t = anderson_ksamp([x1,x2])
        print('Anderson-Darling: ',t)
        
        
        y1 = HIdef_cats[1][HIdefKey][flags[1]]
        y2 = HIdef_cats[0][HIdefKey][flags[0]]
        print()
        print('comparing LCS core vs infall: HIdef')
        print(ks_2samp(y1,y2))        
        t = anderson_ksamp([y1,y2])
        print('Anderson-Darling: ',t)
        


        # repeat test, but use all LCS vs field
        x1 = dsfr_vars[0][lcsFlag]
        x2 = dsfr_vars[2][flags[2]][keep_indices]
        print()
        print('comparing all LCS vs Field: dSFR')
        print(ks_2samp(x1,x2))

        t = anderson_ksamp([x1,x2])
        print('Anderson-Darling: ',t)
        
        
        y1 = HIdef_cats[0][HIdefKey][lcsFlag]
        y2 = HIdef_cats[2][HIdefKey][flags[2]][keep_indices]
        print()
        print('comparing all LCS vs Field: HIdef')
        print(ks_2samp(y1,y2))        
        t = anderson_ksamp([y1,y2])
        print('Anderson-Darling: ',t)
        
        plt.savefig(plotdir+'/dsfr-HIdef.png')
        plt.savefig(plotdir+'/dsfr-HIdef.pdf')

    def frac_suppressed_Lx(self,coreflag=True):
        ''' plot fraction of suppressed galaxies vs Lx  '''

        # calculate the fraction of suppressed galaxies in the field
        gparentflag = self.gsw_mass_sfr_flag
        glowsfrflag = gparentflag & self.gsw.lowsfr_flag 
        gnormalsfrflag = gparentflag & ~self.gsw.lowsfr_flag 
        gnlow =  np.sum(glowsfrflag)
        gntot = np.sum(gnormalsfrflag)
        gbinom_err = binom_conf_interval(gnlow,gntot)#lower, upper
        gfrac = gnlow/gntot
        
        # loop through cluster names
        for i,cname in enumerate(lcscommon.clusternames):
            # keep sf galaxies that are in cluster
            # NOTE: somehow when I matched catalog again using topcat
            # the cluster names all became 12 characters long
            cname12 = cname+" "*(12-len(cname))
            cflag = self.lcs.cat['CLUSTER_1'] == cname12
            # calc fraction that are suppressed
            if coreflag:
                lcsflag = self.lcs.membflag
            else:
                lcsflag = (self.lcs.membflag | self.lcs.infallflag)
            parentflag = self.lcs_mass_sfr_flag & lcsflag & cflag
            lowsfrflag = parentflag & self.lcs.lowsfr_flag 
            #normalsfrflag = parentflag & ~self.lcs.lowsfr_flag 
            
            nlow = np.sum(lowsfrflag)
            ntot = np.sum(parentflag)
            print('cluster {}: nlow = {}, ntot = {}'.format(cname,nlow,ntot))
            if (ntot == 0):
                continue
            binom_err = binom_conf_interval(nlow,ntot)#lower, upper
            frac =  nlow/ntot
            yerr_low,yerr_high = frac-binom_err[0],binom_err[1]-frac
            yerr = np.zeros((2,1))
            yerr[0] = yerr_low
            yerr[1] = yerr_high
            #print('shape of yerr = {} ({})'.format(yerr.shape,yerr))
            plt.plot(lcscommon.clusterLx[cname],frac,'o',markersize=12,c=mycolors[i],label=cname)            
            plt.errorbar(np.array(lcscommon.clusterLx[cname]),np.array(frac),yerr=yerr,fmt='o',color=mycolors[i])

        plt.gca().set_xscale('log')
        plt.xlabel('$L_X \ (10^{44} \ erg/s) $',fontsize=20)
        plt.ylabel('$N_{Suppressed}/N_{SF} $',fontsize=20)


        # show field values
        x1,x2 = plt.xlim()
        xline = np.linspace(x1,x2,100)
        yline = np.ones(len(xline))
        plt.fill_between(xline,y1=gbinom_err[1]*yline,y2=gbinom_err[0]*yline,color='0.5',alpha=.5,label='Field')
        plt.legend()        
    def frac_HI_Lx(self,coreflag=True):
        ''' plot fraction of suppressed galaxies vs Lx  '''

        # field region
        # 140 < RA  < 230
        # 2 < DEC < 32
        # region covered by ALFALFA
        ramin=140.
        ramax=230.
        decmin=2.
        decmax=32
        
        # calculate the fraction of suppressed galaxies in the field
        ga100flag = (self.gsw.cat['RA'] > ramin) & (self.gsw.cat['RA'] < ramax) & \
            (self.gsw.cat['DEC'] > decmin) & (self.gsw.cat['DEC'] < decmax)
        
        gparentflag = self.gsw_mass_sfr_flag & ga100flag
        glowsfrflag = gparentflag & self.gsw.HIdef['HIdef_flag']
        gnormalsfrflag = gparentflag & ~self.gsw.lowsfr_flag 
        gnlow =  np.sum(glowsfrflag)
        gntot = np.sum(gnormalsfrflag)
        gbinom_err = binom_conf_interval(gnlow,gntot)#lower, upper
        gfrac = gnlow/gntot
        
        
        # loop through cluster names
        for i,cname in enumerate(lcscommon.clusternames):
            # skip clusters with partial alfalfa coverage
            if (cname.find('AWM4') > -1) | (cname.find('Hercules') > -1) | (cname.find('NGC6107') > -1):
                continue
            # keep sf galaxies that are in cluster
            cname12 = cname+" "*(12-len(cname))
            cflag = self.lcs.cat['CLUSTER_1'] == cname12
            
            #cflag = self.lcs.cat['CLUSTER_1'] == cname
            # calc fraction that are suppressed
            if coreflag:
                lcsflag = self.lcs.membflag
            else:
                lcsflag = (self.lcs.membflag | self.lcs.infallflag)
            parentflag = self.lcs_mass_sfr_flag & lcsflag & cflag
            lowsfrflag = parentflag & self.lcs.cat['HIdef_flag']
            
            nlow = np.sum(lowsfrflag)
            ntot = np.sum(parentflag)
            if (ntot == 0):
                continue
            binom_err = binom_conf_interval(nlow,ntot)#lower, upper
            frac =  nlow/ntot
            yerr_low,yerr_high = frac-binom_err[0],binom_err[1]-frac
            yerr = np.zeros((2,1))
            yerr[0] = yerr_low
            yerr[1] = yerr_high
            #print('shape of yerr = {} ({})'.format(yerr.shape,yerr))
            plt.plot(lcscommon.clusterLx[cname],frac,'o',markersize=12,c=mycolors[i],label=cname)            
            plt.errorbar(np.array(lcscommon.clusterLx[cname]),np.array(frac),yerr=yerr,fmt='o',color=mycolors[i])

        plt.gca().set_xscale('log')
        plt.xlabel('$L_X \ (10^{44} \ erg/s) $',fontsize=20)
        plt.ylabel('$N_{HI}/N_{SF} $',fontsize=20)


        # show field values
        x1,x2 = plt.xlim()
        xline = np.linspace(x1,x2,100)
        yline = np.ones(len(xline))
        plt.fill_between(xline,y1=gbinom_err[1]*yline,y2=gbinom_err[0]*yline,color='0.5',alpha=.5,label='Field')
        plt.legend()        
if __name__ == '__main__':
    ###########################
    ##### SET UP ARGPARSE
    ###########################

    parser = argparse.ArgumentParser(description ='Program to run analysis for LCS paper 2')
    parser.add_argument('--minmass', dest = 'minmass', default = 9.7, help = 'minimum stellar mass for sample.  default is log10(M*) > 9.7')
    parser.add_argument('--minssfr', dest = 'minssfr', default = -11.5, help = 'minimum sSFR for the sample.  default is sSFR > -11.5')    
    parser.add_argument('--cutBT', dest = 'cutBT', default = False, action='store_true', help = 'Set this to cut the sample by B/T < 0.3.')
    parser.add_argument('--BT', dest = 'BT', default = 0.3, help = 'B/T cut to use. Default is 0.3.')
    parser.add_argument('--cutN', dest = 'cutN', default = False, action='store_true', help = 'Set this to cut the sample by sersic n < 2.')
    parser.add_argument('--nsersic', dest = 'nsersic', default = 2.5, help = 'Sersic n cut to use. Default is 2.')
    parser.add_argument('--HIdef', dest = 'HIdef', default=False,action='store_true', help = 'Include HIdef information for GSWLC')    
    parser.add_argument('--ellip', dest = 'ellip', default = .75, help = 'ellipt cut to use.  Default is 1.1 (= no cut)')    
    #parser.add_argument('--cutBT', dest = 'diskonly', default = 1, help = 'True/False (enter 1 or 0). normalize by Simard+11 disk size rather than Re for single-component sersic fit.  Default is true.  ')    

    args = parser.parse_args()
    
    trimgswlc = True
    # 10 arcsec match b/w GSWLC-X2-NO-DR10-AGN-Simard2011-tab1 and Tempel_gals_below_13.fits in topcat, best,symmetric
    #gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-13-2020Nov11'
    
    # 10 arcsec match b/w GSWLC-X2-NO-DR10-AGN-Simard2011-tab1 and Tempel_gals_below_13_5.fits in topcat, best,symmetric
    #gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-13.5-2020Nov11'
    
    # 10 arcsec match b/w GSWLC-X2-NO-DR10-AGN-Simard2011-tab1 and Tempel_gals_below_12.cat in topcat, best,symmetric
    #gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-12.5-2020Nov11'
    gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-Tempel-13-2020Nov25'

    # matched to Simard table 3 so we could use the sersic index from the single component fit as
    # one measure of morphology
    gsw_basefile = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-tab3-Tempel-13-2021Jan07'        
    if trimgswlc:
        #g = gswlc_full('/home/rfinn/research/GSWLC/GSWLC-X2.dat')
        # 10 arcsec match b/s GSWLC-X2 and Tempel-12.5_v_2 in topcat, best, symmetric
        #g = gswlc_full('/home/rfinn/research/LCS/tables/GSWLC-Tempel-12.5-v2.fits')
        #g = gswlc_full(homedir+'/research/GSWLC/GSWLC-Tempel-12.5-v2-Simard2011-NSAv0-unwise.fits',cutBT=args.cutBT)
        
        
        g = gswlc_full(gsw_basefile+'.fits',cutBT=args.cutBT,HIdef=args.HIdef)                
        g.cut_redshift()
        g.save_trimmed_cat()
        gfull = g
    #g = gswlc('/home/rfinn/research/LCS/tables/GSWLC-X2-LCS-Zoverlap.fits')
    if args.cutBT:
        infile=gsw_basefile+'-LCS-Zoverlap-BTcut.fits'
    else:
        infile = gsw_basefile+'-LCS-Zoverlap.fits'
    if args.HIdef:
        HIdef_file = homedir+'/research/GSWLC/GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-tab3-Tempel-13-2021Jan07-HIdef-2021Aug28-trimmed.fits'
    else:
        HIdef_file = None
    g = gswlc(infile,HIdef_file=HIdef_file,args=args)

    #g.plot_ms()
    #g.plot_field1()
    lcsfile = homedir+'/research/LCS/tables/lcs-gswlc-x2-match.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS-KE-SFR-GSWLC-X2-NO-DR10-AGN-Simard2011-tab1.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS-GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-vizier-10arcsec.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS-GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-vizier-10arcsec.fits'


    # added HI def measurements from A100 catalog
    # this adds columns AGC, ..., logMH, logMH_err,
    # HIdef_Toribio, HIdef_Boselli, HIdef_Jones
    # HIdef_flag - this means it has HI, not that it is deficient!
    lcsfile = homedir+'/research/LCS/tables/LCS-GSWLC-X2-NO-DR10-AGN-Simard2011-tab1-vizier-10arcsec-Tempel-Simard-tab3-2021Apr21-HIdef-2021Aug28.fits'
    lcsfile = homedir+'/research/LCS/tables/LCS-GSWLC-NODR10AGN-Simard-tab1-tab3-A100-LCSsizes-fixednames-Tempel.fits'    
    lcs = lcsgsw(lcsfile,cutBT=args.cutBT,args=args)
    #lcs = lcsgsw('/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR_GSWLC2_X2.fits',cutBT=args.cutBT)    
    #lcs.compare_sfrs()

    b = comp_lcs_gsw(lcs,g,minmstar=float(args.minmass),minssfr=float(args.minssfr),cutBT=args.cutBT)
