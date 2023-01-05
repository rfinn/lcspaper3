#!/usr/bin/env python

'''
GOAL:
- test several models that describe how disks shrink and integrated SFR decreases as a result of outside-in quenching


USAGE
- from within ipython

%run ~/Dropbox/pythonCode/LCSsimulate-infall.py

t = run_sim(tmax=0,drdt_step=0.05,nrandom=1000)


NOTES:

Written by Rose A. Finn, 2/21/18
Updated 2019-2020 to incorporate total SFRs into the comparison.

Updating 2022 to incorporate delay time into modeling.

Loop over tau (exponential decay rate of SFRs in clusters) and 
tdelay (how long after infall for exponential decay to kick in).

Updating Jan-04-2023:
* this is a copy of ~/github/LCS/python/LCSsimulate_infall_sfrs_mp_v2.py
* updating this for paper3, which will combine the SFR and size measurements

'''



from astropy.io import fits,ascii
from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as tri
import argparse
from scipy.stats import ks_2samp, anderson_ksamp
# the incomplete gamma function, for integrating the sersic profile
from scipy.special import gammainc
from scipy.interpolate import griddata
#from astropy.table import Table
import os
<import LCScommon as lcommon
import multiprocessing as mp


homedir = os.getenv("HOME")
plotdir = homedir+'/research/LCS/plots/'


# import mass-matching function from lcs_paper2
import sys
sys.path.append(homedir+'/github/LCS/python/')
# just copying it over here for now...
#from lcs_paper2 import mass_match

infall_results = []
def collect_results_infall(result):

    global results
    infall_results.append(result)

massmatch_results = []
def collect_results_massmatch(result):

    global results
    massmatch_results.append(result)
###########################
##### SET UP ARGPARSE
###########################

parser = argparse.ArgumentParser(description ='Program to run simulation for LCS paper 2')
parser.add_argument('--BTcut', dest = 'BTcut', default = False, action='store_true',help = 'use sample with BTcut imposed')



parser.add_argument('--largeBT', dest = 'largeBT', default = False, action='store_true',help = 'use sample with BTcut > 0.4')


parser.add_argument('--plotonly', dest = 'plotonly', default = False, action='store_true',help = 'set this if just making plots from existing data and not running models')

parser.add_argument('--mintinfall', dest = 'mintinfall', default = False, action='store_true',help = 'set the min allowable tinfall to 1 Gyr, so that infall times range uniformly from 1Gy to tmax')

parser.add_argument('--model', dest = 'model', default = 1, help = 'infall model to use.  default is 1.  \n\tmodel 1 is shrinking 24um effective radius \n\tmodel 2 is truncatingthe 24um emission')
parser.add_argument('--sfrint', dest = 'sfrint', default = 1, help = 'method for integrating the SFR in model 2.  \n\tmethod 1 = integrate external sersic profile out to truncation radius.\n\tmethod 2 = integrate fitted sersic profile out to rmax.')
parser.add_argument('--pvalue', dest = 'pvalue', default = .005, help = 'pvalue threshold to use when plotting fraction of trials below this pvalue.  Default is 0.05 (2sigma).  For ref, 3sigma is .003.')
parser.add_argument('--tmax', dest = 'tmax', default = 3., help = 'maximum infall time.  default is 3 Gyr.  ')

parser.add_argument('--sampleks', dest = 'sampleks', default = False, action='store_true',help = 'run KS test to compare core/external size, SFR, Re24 and nsersic24.  default is False.')


args = parser.parse_args()
args.model = int(args.model)
args.sfrint = int(args.sfrint)
args.tmax = float(args.tmax)
args.pvalue = float(args.pvalue)

###########################
##### DEFINITIONS
###########################

mycolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mipspixelscale=2.45

###########################
##### PLOTTING LABELS
###########################
drdt_label1 = r'$\dot{R}_{24} \ (R_{24}/Gyr^{-1}) $'
drdt_label2 = r'$\dot{R}_{trunc} \ (R_24}/Gyr^{-1}) $'


def get_SFR_cut(logMstar,cutBT=False):
    ''' 
    get min allowable SFR as a function of stellar mass
        
    '''
    #return 0.53*logMstar-5.5
    # for no BT cut, e < 0.75
    #return 0.6*logMstar - 6.11 - 0.6

    # using full GSWLC, just cut to LCS redshift range
    if not(cutBT):
        return get_MS(logMstar) - 0.845
    else:
        #return get_MS(logMstar) - 0.947
        return get_MS(logMstar) - 0.947
def get_MS(logMstar, cutBT=False):
    # not BT cut
    # updating this after we expanded the mass range used for fitting MS
    if not(cutBT):
        #self.MS_std = 0.22
        #return 0.6*x-6.11

        # using full GSWLC, just cut to LCS redshift
        #return 0.592*logMstar - 6.18

        # using 2nd order polynomial fit
        return -0.1969*logMstar**2 + 4.4187*logMstar -24.607
    else:
        # you get the same thing from fitting the MS or from fitting peaks of gaussian
        #self.MS_std = 0.16
        #return 0.62*logMstar-6.35
        return -0.0935*logMstar**2 + 2.4325*logMstar -15.107


## infall rates
# uniform distribution between 0 and tmax Gyr
tmax = 2. # max infall time in Gyr


###########################
##### READ IN DATA FILE
##### WITH SIZE INFO
###########################

# updated input file to include SFR, n
#infile = homedir+'/research/LCS/tables/LCS-simulation-data.fits'


if args.BTcut:
    infile1 = homedir+'/research/LCS/tables/lcs-sfr-sim-BTcut.fits'
    infile2 = homedir+'/research/LCS/tables/gsw-sfr-sim-BTcut.fits'
else:
    infile1 = homedir+'/research/LCS/tables/lcs-sfr-sim.fits'
    infile2 = homedir+'/research/LCS/tables/gsw-sfr-sim.fits'

    
lcs = Table.read(infile1)
field = Table.read(infile2)


if args.largeBT:
    lcsflag = (lcs['BT'] > 0.4) & (lcs['BT'] < 0.8) 
    fieldflag = (field['BT'] > 0.4)& (field['BT'] < 0.8)
    lcs = lcs[lcsflag]
    field = field[fieldflag]

core_sfr = 10.**(lcs['logSFR'])
external_sfr = 10.**(field['logSFR'])

core_dsfr = lcs['logSFR'] - get_MS(lcs['logMstar'])
core_logmstar = (lcs['logMstar'])
external_logmstar = (field['logMstar'])


###########################
##### compare core/external
###########################
if args.sampleks:
    print('\ncore vs external: logMstar distribution')
    lcommon.ks(core_logmstar,external_logmstar,run_anderson=True)
    print()
    print('\ncore vs external: SFR distribution')
    lcommon.ks(core_sfr,external_sfr,run_anderson=True)

###########################
##### FUNCTIONS
###########################
def mass_match(input_mass,comp_mass,seed,nmatch=10,dm=.15,inputZ=None,compZ=None,dz=.0025):
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
            print('galaxies in slice < # requested',sum(flag),nmatch,input_mass[i],inputZ[i])
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
        return_index[int(i*nmatch):int((i+1)*nmatch)] = np.random.choice(comp_index[flag],nmatch,replace=True)

    return return_index

def grid_xyz(x,y,z,nbins=20,color=None):
    ''' bin non-equally spaced data for use with contour plot  '''     
    # https://matplotlib.org/3.3.2/gallery/images_contours_and_fields/irregulardatagrid.html
    # griddata
    xspan = np.linspace(int(min(x)),int(max(x)),nbins)
    yspan = np.linspace(int(min(y)),int(max(y)),nbins)   
    triang = tri.Triangulation(x,y)
    interpolator = tri.LinearTriInterpolator(triang,z)
    
    xgrid,ygrid = np.meshgrid(xspan,yspan)
    zgrid = interpolator(xgrid,ygrid)
    #t = griddata((xspan,yspan),z,method='linear')
    # plot contour levels
    #grid = griddata((x,y),z,(xgrid,ygrid),method='linear')
    return xgrid,ygrid,zgrid
def contour_xyz(x,y,z,ngrid=20,color=None):
    ''' bin non-equally spaced data for use with contour plot  ''' 
    # griddata
    xspan = np.arange(int(min(x)),int(max(x)),ngrid)
    yspan = np.arange(int(min(y)),int(max(y)),ngrid)    
    xgrid,ygrid = np.meshgrid(xspan,yspan)    
    grid = griddata((x,y),z,(xgrid,ygrid),method='linear')
    
    return grid


def model2_get_fitted_param(input,coeff):
    # here rtrunc is the truncation radius/Re
    return coeff[0]+coeff[1]*np.exp(coeff[2]*input)

def integrate_sersic(n,Re,Ie,rmax=6):
    bn = 1.999*n-0.327
    x = bn*(rmax/Re)**(1./n)    
    return Ie*Re**2*2*np.pi*n*np.exp(bn)/bn**(2*n)*gammainc(2*n, x)

def get_frac_flux_retained(n,ratio_before,ratio_after):
    # ratio_before = the initial value of R/Re
    # ratio_after = the final value of R/Re
    # n = sersic index of profile

    # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
    
    # calculate the loss in light
    bn = 1.999*n-0.327
    x_before = bn*(ratio_before)**(1./n)
    x_after = bn*(ratio_after)**(1./n)    
    frac_retained = gammainc(2*n,x_after)/gammainc(2*n,x_before) 
    return frac_retained

'''
I think the right way to do the integration (inspiration during my jog today) is to integrate
the sersic profile of the external profile(Re24, n24) from zero to inf.

for sim core galaxy, integrate sersic profile with new Re from zero to inf.  
Don't know what to do with sersic index.  As a first attempt, leave it 
the same as for the external galaxy.

gammainc = 1 when integrating to infinity

'''
def get_frac_flux_retained0(n,ratio_before,ratio_after,Iboost=1):
    '''
    calculate fraction of flux retained after shrinking Re of sersic profile

    PARAMS
    ------
    * ratio_before: the initial value of R24/Re_r
    * ratio_after: the final value of R24/Re_r
    * n: sersic index of profile

    # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)

    RETURNS
    -------
    * frac_retained: fraction of flux retained
    '''
    
    # calculate the loss in light
    bn = 1.999*n-0.327
    x_before = bn*(ratio_before)**(1./n)
    x_after = bn*(ratio_after)**(1./n)

    # everything is the same except for Re(24)

    ### check this!!!  gamma function might be different???
    frac_retained = Iboost*(ratio_after/ratio_before)**2
    return frac_retained



def get_frac_flux_retained_model2(n,Re,rtrunc=1,rmax=4,version=1,Iboost=1):
    '''
    return fraction of the flux retained by a truncated profile.

    this model integrates the after model to the truncation radius AND
    boosts the central intensity of the after model.
    The sersic index is unchanged.  

    PARAMS
    ------
    * n: sersic index of profile
    * Re: effective radius of sersic profile
    * rtrunc: truncation radius, in terms of Re; default=1
    * rmax: maximum extent of the disk, in terms of Re, for the purpose of integration; default=6
      - this is how far out the original profile is integrated to, rather than infinity
    * version:
      - 1 = integrate the truncated profile
      - 2 = integrate the sersic profile we would measure by fitting a sersic profile to the truncated profile
      - best to use option 1
    * Iboost: factor to boost central intensity by

    RETURNS
    -------
    * fraction of the flux retained
    
    '''
    if version == 1:
        # sersic index of profile
        # Re = effective radius of profile
        # n = sersic index of profile
        # rmax = multiple of Re to use a max radius of integration in the "before" integral
        # rtrunc = max radius to integrate to in "after" integral

        # ORIGINAL
        # L(<R) = Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
        # PROCESSED
        # L(<R) = boost*Ie Re^2 2 pi n e^{b_n}/b_n^2n incomplete_gamma(2n, x)
            
        # calculate the loss in light
        
        # this should simplify to the ratio of the incomplete gamma functions
        # ... I think ...

        # this is the same for both
        bn = 1.999*n-0.327
        
        x_after = bn*(rtrunc/Re)**(1./n)
        x_before = bn*(rmax)**(1./n)
        frac_retained = Iboost*gammainc(2*n,x_after)/gammainc(2*n,x_before)
        
    elif version == 2:
        # use fitted values of model to get integrated flux after
        # integral of input sersic profile with integral of fitted sersic profile
        Ie = 1
        sfr_before = integrate_sersic(n,Re,Ie,rmax=rmax)
        
        n2 = n*model2_get_fitted_param(rtrunc/Re,sersicN_fit)
        Re2 = Re*model2_get_fitted_param(rtrunc/Re,sersicRe_fit)
        Ie2 = Ie*model2_get_fitted_param(rtrunc/Re,sersicIe_fit)
        sfr_after = integrate_sersic(n2,Re2,Ie2,rmax=rmax)        
        
        frac_retained = Iboost*sfr_after/sfr_before

        
    return frac_retained


def get_whitaker_ms(logmstar,z):
    ''' get whitaker  '''
    pass


def get_sfr_mstar_at_infall(sfr0,mstar0,tinfall):
    ''' 
    get the sfr and stellar mass at the time of infall, 
    using a grid of models created by sfr-mstar-forward-model.py 

    INPUT:
    * sfr0 : array with redshift zero log10 SFRs of field galaxies
    * mstar0 : an array with z=0 log10 stellar mass values of field galaxies
    * tinfall : an array with the infall time for each field galaxy

    RETURNS:
    * sfr_infall : an array with the log10 sfr of each galaxy at the time of infall
    * mstar_infall : an array with the log10 stellar mass of each galaxy at the time of infall

    '''
    lookup_table = fits.getdata(homedir+'/research/LCS/sfr_modeling/forward_model_sfms.fits')    
    sfr_infall = np.zeros(len(sfr0))
    mstar_infall = np.zeros(len(sfr0))
    allindex = np.arange(len(lookup_table))
    for i in range(len(tinfall)):
        dsfr = np.abs(sfr0[i] - lookup_table['logSFR0'])
        dmstar = np.abs(mstar0[i] - lookup_table['logMstar0'])
        dtinfall = np.abs(tinfall[i] - lookup_table['lookbackt'])        
        
        distance = dsfr + dmstar + dtinfall

        # find entry that falls closest to input galaxy

        match_index = distance == np.min(distance)
        match_index = allindex[match_index]
        sfr_infall[i] = lookup_table['logSFR'][match_index]
        mstar_infall[i] = lookup_table['logMstar'][match_index]        
    
    
    # find closest match in using sfr0, mstar0, tinfall-tab['lookbackt']
    # returning infall times so I can use pool.apply_async
    # this is faster, but there is not guarantee that the first set of
    # infall times corresponds to the first set of data returned
    return [sfr_infall,mstar_infall,tinfall]

def get_fraction_mass_retained(t):
    ''' 
    use Poggianti+2013 relation to determine fraction of mass retained.  This applies to 
    stellar populations with ages greater than 1.9E6 yrs.  For younger pops, the fraction 
    retained is just one.

    INPUT:
    * time in yr since the birth of the stellar pop

    RETURNS:
    * fraction of the mass that is retained after mass loss when age of population is t_yr

    '''
    frac = np.ones(len(t))
    flag = t > 1.9e6
    frac[flag] = 1.749 - 0.124*np.log10(t[flag])
    return frac

def get_delta_mass(infall_sfr,infall_times,tau):
    '''
    compute the amount of mass gained by the galaxy since tinfall.
    this includes mass added from star formation, and mass lost.

    INPUT
    * infall_sfr : array of infall sfrs
    * infall_times : array containing time since infall for each galaxy in Gyr
    * tau : e-folding time associated with sfr decline in Gyr

    RETURNS
    * dMstar : mass created since t infall, including mass loss
    '''


    dMstar = np.zeros(len(infall_sfr))

    for i in range(len(infall_sfr)):
        t = np.linspace(.002,infall_times[i],1000)
        dt = t[1] - t[0]
        dMstar[i] = infall_sfr[i]*dt*1.e9*np.sum(np.exp(-1*(infall_times[i]-t)/tau)*get_fraction_mass_retained(t*1.e9))
    return dMstar
    
    pass
###############################
##### MAIN SIMULATION FUNCTION
###############################


def run_sim(taumax=6,nstep_tau=10,ndrawmass=60,tmax=7,ntdelay=14,model=1,plotsingle=True,plotflag=True,boostflag=False,debug=False):
    '''
    run simulations of disk shrinking

    PARAMS
    ------
    * taumax : max value for e-folding time associated with SFR decline, in Gyr
    * nstep_tau : number of steps to take between taumax and zero
    * nmassmatch : number of times to repeat the mass matching between sim-core and core, at each step of tau
    * tmax: maximum time that core galaxies have been in cluster, in Gyr; default = 2
    * nrandom : number of random iterations for each value of dr/dt
    * boostflag : set this to boost central intensity; can set this for both model 1 and 2
    * plotsingle : default is True;
      - use this to print a separate figure;
      - set to False if creating a multi-panel plot
    * plotflag : don't remember what this does


    RETURNS
    -------
    * best_drdt : best dr/dt value (basically meaningless)
      - b/c KS test is good at rejecting null hypothesis, but pvalue of 0.98 is not better than pvalue=0.97
    * best_sim_core : best distribution of core sizes? (basically meaningless)
    * ks_p_max : pvalue for best model (basically meaningless)
    * all_drdt : dr/dt for every model
    * all_p : p value for size comparison for every model
    * all_p_sfr : p value for SFR comparison for every model
    * all_boost : boost value for each model; this will be zeros if model != 3

    '''
    # pass in rmax
    #rmax = float(args.rmax)
    if debug:
        nstep_tau = 2
        nrandom = 2
        nmassmatch = 2

    # turns out that using lists is faster than initialized arrays - doing that instead
    all_p_dsfr = []
    all_p_sfr = []
    all_p_sfr_AD = []    
    all_tau = []
    all_boost = []
    all_tdelay  = []
    fquench_size = []
    fquench_sfr = []
    fquench = []

    #save the sfr and mstar values of the sim core
    all_simcore_sfr = []
    all_simcore_dsfr = []
    all_simcore_mstar = []    
    all_simcore_tau = []
    all_simcore_tdelay = []    
    tau_min = 0.5
    dtau = (taumax-tau_min)/nstep_tau
       

    ## PARALLELIZE GETTING INFALL TIMES AND PROGENITOR MASSES
    # GET SFR AND MSTAR AT t_infall

    # MAX TINFALL IS 7 GYR


    ## ASSUME INFALL TIMES ARE UNIFORM BETWEEN 0 AND TMAX
    i=78
    np.random.seed(5*i+29*i)
    infall_times = np.random.random(len(external_sfr))*tmax

    print("GETTING INFALL TIMES\n\n")
    # LOOP OVER T DELAY
    all_infall_times = []
    for tdelay in np.linspace(0,7,ntdelay):
        
        tactive = infall_times - tdelay
        ## MAKE SURE INFALL TIMES ARE NOT LESS THAN ZERO
        flag = tactive < 0
        if np.sum(flag) > 0:
            tactive[flag] = np.zeros(np.sum(flag),'d')
        print("tdelay = {:.2f}, tmax={:.2f}, max(tactive)={:.2f},tmax - max (tactive) = {:.2f}".format(tdelay,tmax,np.max(tactive),tmax-np.max(tactive)))
        #print()
        all_infall_times.append(tactive)

    print("GETTING SFR/Mstar AT INFALL TIME\n\n")        
    ## USE MULTIPROCESSING TO GET THE LIST OF SFR/MSTAR AT TIME OF INFALL GIVEN TDELAY
    #print('getting sfr/mstar at infall')
    infall_pool = mp.Pool(mp.cpu_count())
    myresults = [infall_pool.apply_async(get_sfr_mstar_at_infall,args=(np.log10(external_sfr),(external_logmstar),infall_times),callback=collect_results_infall) for infall_times in all_infall_times]
    
    infall_pool.close()
    infall_pool.join()
    infall_results = [r.get() for r in myresults]
        
    print("LOOPING OVER TDELAY AND TAU\n\n")        
    for j in range(len(infall_results)):
        logsfr_infall, logmstar_infall,tactive = infall_results[j]

                                                                        
        # select different values of tau
        for i in range(nstep_tau):

            tau = tau_min + i*dtau
            #print("tdelay = {:.2f}, tau = {:.2f}".format(j,tau))

            # UPDATING PROCEDURE TO ACCOUNT FOR EVOLUTION OF SFR AND STELLAR MASS
            # OF FIELD GALAXIES BETWEEN t_infall AND PRESENT.
            
            if boostflag:
                # model 3 involves boosting Ie in addition to truncating the disk,
                # so this requires another loop where Iboost/Ie0 ranges from 1 to 5
                # not sure if I can implement this as a third case
                # or I could just assign a random boost for each iteration
                # and increase nrandom when running model 3

                #
                # going with one boost factor per iteration for now
                # obviously, it's not realistic that ALL galaxies
                # would be boosted by the SAME factor
                # but this is an easy place to start
                
                boost = np.random.uniform(1,maxboost) # boost factor will range between 1 and maxboost
                
            else:
                boost = 1.0
            
            ########################################################
            # get predicted SFR of core galaxies by multiplying the
            # distribution of SFRs from the external samples by the
            # flux ratio you would expect from shrinking the
            # external sizes to the sim_core sizes
            # SFRs are logged, so add log of frac_retained 
            sim_core_sfr = boost*(10.**logsfr_infall)*np.exp(-1*tactive/tau)

            # calculate Mstar at z=0, give sfr decline and mass loss

            sim_core_mstar = 10.**(logmstar_infall) + get_delta_mass(10.**logsfr_infall,\
                                                                     tactive,tau)
            
                                  
            # CREATE A SIMULATED CORE SAMPLE THAT IS MASS-MATCHED TO THE CORE
            # repeat this 1000 times

            # try parallelizing the mass_match call
            #if nmassmatch < nproc:
            #    nproc = nmassmatch
            
            ## REMOVING MULTIPLE MASS MATCHING WITH FIELD
            q=9
            matched_indices = mass_match(core_logmstar,np.log10(sim_core_mstar),45*q+7*q,nmatch=ndrawmass,dm=.15)
            sim_core_mstar_matched = sim_core_mstar[matched_indices]
            sim_core_sfr_matched = sim_core_sfr[matched_indices]
            # SFR and Mstar are linear, so need to take the log
            # update for new SFR/passive cut
            quench_flag = np.log10(sim_core_sfr_matched) < get_SFR_cut(np.log10(sim_core_mstar_matched))

            ## CALCULATE KS AND ANDERSON-DARLING TESTS
            D1,p1 = ks_2samp(core_sfr,sim_core_sfr_matched[~quench_flag])
            D, crit_values, p3 = anderson_ksamp([core_sfr,sim_core_sfr_matched[~quench_flag]])

            ## SAVE OUTPUT
            fquench_sfr.append(sum(quench_flag)/len(quench_flag))
            all_p_sfr.append(p1)
            all_p_sfr_AD.append(p3)                                                    
            all_boost.append(boost)
            all_tau.append(tau)
            all_tdelay.append(tmax - np.max(tactive))
            print("tau={:.2f}, tmax={:.1f}, max(tactive)={:.1f}, tmax-max(tactive) = {:.1f}".format(tau,tmax,np.max(tactive),(tmax-np.max(tactive))))
            
    all_p_dsfr = np.zeros(len(all_p_sfr))
    newtab = Table([all_tau,all_tdelay,all_p_sfr,all_p_dsfr,fquench_sfr,all_p_sfr_AD],names=['tau','tdelay','p_sfr','p_dsfr','fquench','p_sfr_AD'])
    newtab_name = 'pvalues_tmax{:.0f}_ntdelay{:d}_ndrawmass{:d}.fits'.format(tmax,ntdelay,ndrawmass)
    newtab.write(newtab_name,format='fits',overwrite=True)
    
    
    return all_tau,all_tdelay,all_p_sfr,all_p_dsfr,fquench_sfr, all_p_sfr_AD

###########################
##### PLOT FUNCTIONS
###########################


def plot_hexbin(all_drdt,all_p,best_drdt,tmax,gridsize=10,plotsingle=True):
    if plotsingle:
        plt.figure()
    plt.subplots_adjust(bottom=.15,left=.12)
    myvmax = 1.*len(all_drdt)/(gridsize**2)*4
    #print 'myvmax = ',myvmax 
    plt.hexbin(all_drdt, all_p,gridsize=gridsize,cmap='gray_r',vmin=0,vmax=myvmax)
    if plotsingle:
        plt.colorbar(fraction=0.08)
    plt.xlabel(drdt_label1,fontsize=18)
    plt.ylabel(r'$p-value$',fontsize=18)
    #s = r'$t_{max} = %.1f \ Gyr, \ dr/dt = %.2f \ Gyr^{-1}, \ t_{quench} = %.1f \ Gyr$'%(tmax, best_drdt,1./abs(best_drdt))
    s = r'$t_{max} = %.1f \ Gyr$'%(tmax)
    #plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    plt.title(s,fontsize=18)
    output = 'sim_infall_tmax_%.1f.png'%(tmax)
    plt.savefig(output)

def plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,plotsingle=True):
    pvalue = args.pvalue
    if plotsingle:
        plt.figure()
    plt.subplots_adjust(bottom=.15,left=.12)
    mybins = np.linspace(min(all_drdt),max(all_drdt),100)
    t= np.histogram(all_drdt,bins=mybins)
    #print(t)
    ytot = t[0]
    xtot = t[1]
    flag = all_p < 0.05
    t = np.histogram(all_drdt[flag],bins=mybins)
    #print(t)
    y1 = t[0]/ytot
    flag = all_p_sfr < 0.05
    t = np.histogram(all_drdt[flag],bins=mybins)
    y2 = t[0]/ytot
    #plt.figure()

    # calculate the position of the bin centers
    xplt = 0.5*(xtot[0:-1]+xtot[1:])
    plt.plot(xplt,y1,'bo',color=mycolors[0],markersize=6,label='R24/Re')
    plt.plot(xplt,y2,'rs',color=mycolors[1],markersize=6,label='SFR')
    plt.legend()

    plt.xlabel(drdt_label1,fontsize=18)
    plt.ylabel(r'$Fraction(p<{:.3f})$'.format(pvalue),fontsize=18)
    #s = r'$t_{max} = %.1f \ Gyr, \ dr/dt = %.2f \ Gyr^{-1}, \ t_{quench} = %.1f \ Gyr$'%(tmax, best_drdt,1./abs(best_drdt))
    s = r'$t_{max} = %.1f \ Gyr$'%(tmax)
    #plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    plt.title(s,fontsize=18)
    output = 'frac_pvalue_infall_tmax_%.1f.png'%(tmax)
    plt.savefig(output)
    output = 'frac_pvalue_infall_tmax_%.1f.pdf'%(tmax)
    plt.savefig(output)

def plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax,plotsingle=True):
    if plotsingle:
        plt.figure(figsize=(8,6))
    plt.scatter(all_p,all_p_sfr,c=all_drdt,s=10,vmin=-1,vmax=0)
    plt.xlabel('$p-value \ size$',fontsize=18)
    plt.ylabel('$p-value \ SFR$',fontsize=18)

    plt.axhline(y=.05,ls='--')
    plt.axvline(x=.05,ls='--')
    plt.axis([-.09,1,-.09,1])
    ax = plt.gca()
    #ax.set_yscale('log')
    if plotsingle:
        plt.colorbar(label='$dr/dt$')        
        plt.savefig('pvalue-SFR-size-tmax'+str(tmax)+'Gyr-shrink0.png')
def plot_frac_below_pvalue_sfr(all_tau,all_p_sfr,tmax,nbins=100,plotsingle=True,pvalue=0.05,color=None,alpha=1,lw=1):
    #pvalue = args.pvalue
    if plotsingle:
        plt.figure()
        plt.subplots_adjust(bottom=.15,left=.12)
    mybins = np.linspace(min(all_tau),max(all_tau),nbins)
    t= np.histogram(all_tau,bins=mybins)
    # convert input to arrays

    all_tau = np.array(all_tau,'d')
    all_p_sfr = np.array(all_p_sfr,'d')

    #print(t)
    ytot = t[0]
    xtot = t[1]
    flag = all_p_sfr < pvalue
    t = np.histogram(all_tau[flag],bins=mybins)
    y2 = t[0]/ytot
    #plt.figure()

    # calculate the position of the bin centers
    xplt = 0.5*(xtot[0:-1]+xtot[1:])
    plotflag = ~np.isnan(y2)
    x = xplt[plotflag]
    y = y2[plotflag]
    if color is None:
        plt.plot(x,1-y,marker='s',markersize=6,label='tmax={:d}Gyr'.format(tmax),alpha=alpha,lw=lw)
    else:
        plt.plot(x,1-y,marker='s',markersize=6,label='tmax={:d}Gyr'.format(tmax),color=color,alpha=alpha,lw=lw)
        #plt.fill_between(
    print('pvalue = ',pvalue)    
    #s = r'$t_{max} = %.1f \ Gyr, \ dr/dt = %.2f \ Gyr^{-1}, \ t_{quench} = %.1f \ Gyr$'%(tmax, best_drdt,1./abs(best_drdt))
    s = r'$t_{max} = %.1f \ Gyr$'%(tmax)
    #plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    #plt.title(s,fontsize=18)
    plt.grid(b=True)
    if plotsingle:
        plt.legend()
        plt.xlabel(r'$\tau (Gyr)$',fontsize=18)
        #plt.ylabel(r'$Fraction(p<{:.3f})$'.format(pvalue),fontsize=18)
        plt.ylabel(r'$Fraction \ of \ Acceptable \ Models$'.format(pvalue),fontsize=18)
        output = 'frac_pvalue_sfr_infall_tmax_%.1f.png'%(tmax)
        plt.savefig(output)
        output = 'frac_pvalue_sfr_infall_tmax_%.1f.pdf'%(tmax)
        plt.savefig(output)
    return x,y

def plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax,plotsingle=True):
    if plotsingle:
        plt.figure(figsize=(8,6))
    plt.scatter(all_p,all_p_sfr,c=all_drdt,s=10,vmin=-1,vmax=0)
    plt.xlabel('$p-value \ size$',fontsize=18)
    plt.ylabel('$p-value \ SFR$',fontsize=18)

    plt.axhline(y=.05,ls='--')
    plt.axvline(x=.05,ls='--')
    plt.axis([-.09,1,-.09,1])
    ax = plt.gca()
    #ax.set_yscale('log')
    if plotsingle:
        plt.colorbar(label='$dr/dt$')        
        plt.savefig('pvalue-SFR-size-tmax'+str(tmax)+'Gyr-shrink0.png')

def plot_multiple_tmax(nrandom=100):
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=1,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax)    
    plt.subplot(2,2,2)
    run_sim(tmax=2,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,3)
    run_sim(tmax=3,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplot(2,2,4)
    run_sim(tmax=4,drdt_step=.05,nrandom=nrandom,plotsingle=False)
    plt.subplots_adjust(hspace=.5,bottom=.1)
    plt.savefig('sim_infall_multiple_tmax.pdf')
    plt.savefig('fig18.pdf')

def plot_multiple_tmax_wsfr(nrandom=100):
    plt.figure(figsize=(12,6))
    mytmax = [1,2,3,4]
    allax = []
    for i,tmax in enumerate(mytmax):
        plt.subplot(2,4,i+1)
        best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False)
        allax.append(plt.gca())
        plt.subplot(2,4,i+5)    
        #plot_sfr_size(all_p,all_p_sfr,all_drdt,tmax,plotsingle=False)
        plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,pvalue=0.05,plotsingle=False)        
        allax.append(plt.gca())


    plt.subplots_adjust(hspace=.5,wspace=.7,bottom=.1)
    cb = plt.colorbar(ax=allax,label='$dr/dt$')    
    plt.savefig('sim_infall_multiple_tmax_wsfr.pdf')
    plt.savefig('sim_infall_multiple_tmax_wsfr.png')    
    #plt.savefig('fig18.pdf')
def plot_multiple_tmax_wsfr2(nrandom=100):
    plt.figure(figsize=(10,8))
    mytmax = [1,2,3,4]
    allax = []
    for i,tmax in enumerate(mytmax):
        plt.subplot(2,2,i+1)
        if args.model < 3:
            best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False,plotflag=False)
        else:
            best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr,boost=run_sim(tmax=tmax,drdt_step=.05,nrandom=nrandom,plotsingle=False,plotflag=False)

        plot_frac_below_pvalue(all_drdt,all_p,all_p_sfr,tmax,nbins=100,plotsingle=False)        
        allax.append(plt.gca())


    plt.subplots_adjust(hspace=.5,wspace=.5,bottom=.1)
    #cb = plt.colorbar(ax=allax,label='$dr/dt$')    
    plt.savefig('sim_infall_multiple_tmax_wsfr.pdf')
    plt.savefig('sim_infall_multiple_tmax_wsfr.png')    
    #plt.savefig('fig18.pdf')

def plot_results(core,external,sim_core,best_drdt,tmax):
    plt.figure()
    mybins = np.arange(0,2,.2)
    plt.hist(core,bins=mybins,color='r',histtype='step',label='Core',lw='3',normed=True)
    plt.hist(external,bins=mybins,color='b',ls='-',lw=3,histtype='step',label='External',normed=True)
    plt.hist(sim_core,bins=mybins,color='k',hatch='//',histtype='step',label='Sim Core',normed=True)
    plt.subplots_adjust(bottom=.15)
    plt.xlabel('$R_{24}/R_d$', fontsize=22)
    plt.ylabel('$Frequency$',fontsize=22)
    s = '$dr/dt = %.2f /Gyr$'%(best_drdt)
    plt.text(0.02,.7,s,transform = plt.gca().transAxes)
    s = '$t_{quench} = %.1f  Gyr$'%(1./abs(best_drdt))
    plt.text(0.02,.65,s,transform = plt.gca().transAxes)
    s = '$t_{max} = %.1f  Gyr$'%(tmax)
    plt.text(0.02,.6,s,transform = plt.gca().transAxes)
    plt.legend(loc='upper left')


def plot_model3(all_drdt,all_p,all_p_sfr,boost,tmax=2):
    '''plot boost factor vs dr/dt, colored by pvalue'''
    plt.figure(figsize=(12,4))
    plt.subplots_adjust(wspace=.5)
    colors = [all_p,all_p_sfr]
    labels = ['size p value','sfr p value']
    titles = ['Size Constraints','SFR Constraints']
    v2 = .005
    allax = []
    for i in range(len(colors)):
        plt.subplot(1,2,i+1)
        plt.scatter(all_drdt,boost,c=colors[i],vmin=0,vmax=v2,s=15)
    
        plt.title(titles[i])
        plt.xlabel(drdt_label1,fontsize=16)
        plt.ylabel('I boost/I0',fontsize=16)
        allax.append(plt.gca())
    cb = plt.colorbar()
    cb.set_label('KS p value')
    plt.savefig(plotdir+'/model3-tmax'+str(tmax)+'-size-sfr-constraints.png')
    plt.savefig(plotdir+'/model3-tmax'+str(tmax)+'-size-sfr-constraints.pdf')

def plot_boost_3panel(all_drdt,all_p,all_p_sfr,boost,tmax=2,v2=.005,model=3):
    plt.figure(figsize=(14,4))
    plt.subplots_adjust(wspace=.01,bottom=.2)
    colors = [all_p,all_p_sfr,np.minimum(all_p,all_p_sfr)]
    labels = ['size p value','sfr p value','min p value']
    titles = ['Size Constraints','SFR Constraints','minimum(Size, SFR)']
    allax = []
    psize=30
    for i in range(len(colors)):
        plt.subplot(1,3,i+1)
        plt.scatter(all_drdt,boost,c=colors[i],vmin=0,vmax=v2,s=psize)
        plt.title(titles[i],fontsize=20)
        if i == 0:
            plt.ylabel('$I_{boost}/I_e$',fontsize=24)
        else:
            y1,y2 = plt.ylim()
            #t = plt.yticks()
            #print(t)
            plt.yticks([])
            plt.ylim(y1,y2)
        if model == 1:
            plt.xlabel(drdt_label1,fontsize=24)
        else:
            plt.xlabel(drdt_label2,fontsize=24)
        
        allax.append(plt.gca())
    cb = plt.colorbar(ax=allax,fraction=.08)
    cb.set_label('KS p value')

    plt.savefig(plotdir+'/model3-tmax'+str(tmax)+'-size-sfr-constraints-3panel.png')
    plt.savefig(plotdir+'/model'+str(model)+'-tmax'+str(tmax)+'-size-sfr-constraints-3panel.pdf')

def plot_drdt_boost_ellipse(all_drdt,all_p,all_p_sfr,boost,tmax=2,levels=None,model=3,figname=None,alpha=.5,nbins=20):
    '''plot error ellipses of drdt and boost'''
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=.15,bottom=.2)
    allz = [all_p,all_p_sfr]
    allcolors = [mycolors[0],'0.5']
    labels = ['size p value','sfr p value']
    titles = ['Size Constraints','SFR Constraints']
    if levels is None:
        levels = [.05,1]
    else:
        levels = levels
    allax = []
    psize=30
    for i in range(len(allz)):
        xgrid,ygrid,zgrid = grid_xyz(all_drdt,boost,allz[i],nbins=nbins)
        plt.contourf(xgrid,ygrid,zgrid,colors=allcolors[i],levels=levels,label=titles[i],alpha=alpha)
        if i == 0:
            zgrid0=zgrid

    # define region where both are above .05

    zcomb = np.minimum(zgrid0,zgrid)
    plt.contour(xgrid,ygrid,zcomb,linewidths=4,colors='k',levels=[.05,1])
    plt.ylabel('$SFR \ Boost \ Factor \ (I_{boost}/I_o)$',fontsize=20)
    if model == 1:
        plt.xlabel(drdt_label1,fontsize=20)
    else:
        plt.xlabel(drdt_label2,fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)    
    #plt.legend()
    #plt.xlim(-2,0)
    plt.text(.05,.9,'Model '+str(model),transform=plt.gca().transAxes,horizontalalignment='left',fontsize=20)
    if figname is not None:
        plt.savefig(plotdir+'/'+figname+'.png')
        plt.savefig(plotdir+'/'+figname+'.pdf')        


def plot_model1_3panel(all_drdt,all_p,all_p_sfr,tmax=2,v2=.005,model=1,vmin=-4):
    '''
    make a 1x3 plot showing
     (1) pvalue vs dr/dt for size
     (2) pvalue vs dr/dt for SFR
     (3) pvalue size vs pvalue SFR, color coded by dr/dt

    PARAMS
    ------
    * all_drdt : output from run_sum; disk-shrinking rate for each model
    * all_p : output from run_sum; KS pvalue for size comparision
    * all_p_sfr : output from run_sum; KS pvalue for SFR comparison
    * tmax : tmax of simulation, default is 2 Gyr
    * v2 : max value for colorbar; default is 0.005 for 2sigma
    * model : default is 1; could use this plot for models 1 and 2

    OUTPUT
    ------
    * save png and pdf plot in plotdir
    * title is: model3-tmax'+str(tmax)+'-size-sfr-constraints-3panel.pdf
    '''
    
    plt.figure(figsize=(14,4))
    plt.subplots_adjust(wspace=.5,bottom=.15)
    xvars = [all_drdt, all_drdt, all_p]
    yvars = [all_p, all_p_sfr, all_p_sfr]
    if model == 1:
        drdt_label = drdt_label1
    else:
        drdt_label = drdt_label2
    xlabels=[drdt_label,drdt_label,'pvalue Size']
    ylabels=['pvalue Size','pvalue SFR','pvalue SFR']    
    titles = ['Size Constraints','SFR Constraints','']
    allax = []
    for i in range(len(xvars)):
        plt.subplot(1,3,i+1)
        if i < 2:
            plt.scatter(xvars[i],yvars[i],s=10,alpha=.5)
            plt.title(titles[i])
        else:
            # plot pvalue vs pvalue, color coded by dr/dt
            plt.scatter(all_p,all_p_sfr,c=all_drdt,vmin=vmin,vmax=0,s=5)
            plt.title('Size \& SFR Constraints')

        plt.xlabel(xlabels[i],fontsize=20)
        plt.ylabel(ylabels[i],fontsize=20)        
        plt.ylim(-.02,1.02)
        allax.append(plt.gca())
    cb = plt.colorbar(ax=allax,fraction=.08)
    cb.set_label('dr/dt')
    plt.axhline(y=.05,ls='--')
    plt.axvline(x=.05,ls='--')
    ax = plt.gca()
    #plt.axis([-.01,.35,-.01,.2])
    xl = np.linspace(.05,1,100)
    y1 = np.ones(len(xl))
    y2 = .05*np.ones(len(xl))
    plt.fill_between(xl,y1=y1,y2=y2,alpha=.1)
    plt.savefig(plotdir+'/model'+str(model)+'-tmax'+str(tmax)+'-size-sfr-constraints-3panel.png')
    plt.savefig(plotdir+'/model'+str(model)+'-tmax'+str(tmax)+'-size-sfr-constraints-3panel.pdf')

def plot_quenched_fraction(all_drdt,all_boost, fquench_size,fquench_sfr,fquench,vmax=.5,model=1):
    plt.figure(figsize=(14,4))
    #plt.subplots_adjust(bottom=.15,left=.1)
    plt.subplots_adjust(wspace=.01,bottom=.2)    
    allax=[]
    colors = [fquench_size, fquench_sfr,fquench]
    # total quenching is the same as SFR quenching
    # can't have galaxies with zero size that still has SFR above detection limit
    # therefore, only need first two panels
    for i in range(len(colors)-1):
        plt.subplot(1,3,i+1)
        plt.scatter(all_drdt,all_boost,c=colors[i],vmin=0,vmax=vmax)
        allax.append(plt.gca())
        if model == 1:
            drdt_label = drdt_label1
        else:
            drdt_label = drdt_label2
        plt.xlabel(drdt_label,fontsize=24)
        
        if i == 0:
            plt.ylabel('$I_{boost}/I_e$',fontsize=24)
            plt.title('Frac with $R_{24} = 0$',fontsize=20)
        if i == 1:
            plt.yticks([])
            plt.title('Frac with  $SFR < Limit$',fontsize=20)
        if i == 2:
            plt.yticks([])
            plt.title('Combined Fractions',fontsize=20)
            
    plt.colorbar(ax=allax,label='Fraction',fraction=.08)

def compare_single(var,flag1,flag2,xlab):
        xmin=min(var)
        xmax=max(var)
        print('KS test comparing members and exterior')
        (D,p)=lcommon.ks(var[flag1],var[flag2])


        plt.xlabel(xlab,fontsize=18)

        plt.hist(var[flag1],bins=len(var[flag1]),cumulative=True,histtype='step',normed=True,label='Core',range=(xmin,xmax),color='k')
        plt.hist(var[flag2],bins=len(var[flag2]),cumulative=True,histtype='step',normed=True,label='Infall',range=(xmin,xmax),color='0.5')
        plt.legend(loc='upper left')        
        plt.ylim(-.05,1.05)
        ax=plt.gca()
        plt.text(.9,.25,'$D = %4.2f$'%(D),horizontalalignment='right',transform=ax.transAxes,fontsize=16)
        plt.text(.9,.1,'$p = %5.4f$'%(p),horizontalalignment='right',transform=ax.transAxes,fontsize=16)


        return D, p

def compare_cluster_exterior(sizes,coreflag,infallflag):
    plt.figure(figsize=(8,6))
    plt.subplots_adjust(bottom=.15,hspace=.4,top=.95)
    plt.subplot(2,2,1)
    compare_single(sizes['logMstar'],flag1=coreflag,flag2=infallflag,xlab='$ log_{10}(M_*/M_\odot) $')
    plt.legend(loc='upper left')
    plt.xticks(np.arange(9,12,.5))
    #plt.xlim(8.9,11.15)
    plt.subplot(2,2,2)
    compare_single(sizes['B_T_r'],flag1=coreflag,flag2=infallflag,xlab='$GIM2D \ B/T $')
    plt.xticks(np.arange(0,1.1,.2))
    plt.xlim(-.01,.3)
    plt.subplot(2,2,3)
    compare_single(sizes['ZDIST'],flag1=coreflag,flag2=infallflag,xlab='$ Redshift $')
    plt.xticks(np.arange(0.02,.055,.01))
    plt.xlim(.0146,.045)
    plt.subplot(2,2,4)
    compare_single(sizes['logSFR'],flag1=coreflag,flag2=infallflag,xlab='$ \log_{10}(SFR/M_\odot/yr)$')
    plt.text(-1.5,1,'$Cumulative \ Distribution$',fontsize=22,transform=plt.gca().transAxes,rotation=90,verticalalignment='center')


    
    
if __name__ == '__main__':

    # run program
    print('Welcome!')
    #if args.model == 3:
    #    best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr,all_boost = run_sim(tmax=args.tmax,drdt_step=0.05,nrandom=100,rmax=6)
    #else:
    #    best_drdt, best_sim_core,ks_p_max,all_drdt,all_p,all_p_sfr = run_sim(tmax=args.tmax,drdt_step=0.05,nrandom=100,rmax=6)
    # plot
    #plot_frac_below_pvalue(all_drdt,all_p,best_drdt,args.tmax,nbins=100,pvalue=0.05,plotsingle=True)


    
    # read in data file (should only do this once though, right?)
    #if not args.plotonly:
    #    lookup_table = fits.getdata(homedir+'/research/LCS/sfr_modeling/forward_model_sfms.fits')

    t_ndraw60 = run_sim(taumax=7,nstep_tau=14,tmax=7,ntdelay=14,ndrawmass=60,debug=False)
    pass
