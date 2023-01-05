#!/usr/bin/env python
from pylab import *
import os
from scipy.stats.stats import spearmanr
from scipy.stats import ks_2samp
from scipy.stats import scoreatpercentile
from scipy.stats.mstats import normaltest
#from anderson import *

pscale24=2.45#arcsec per pixel

pscalesdss=1.#arcsec per pixel
sdsspixelscale=0.396127#conversion for isophotal radii from pixels to arcseconds
mipspixelscale=pscale24
mipsconv_MJysr_to_uJy=141.086

mingalaxysize=2.*mipspixelscale

lcsramin=170.
lcsramax=250.
lcsdecmin=0.
lcsdecmax=38.

sbmin=13
sbmax=20

Mpcrad_kpcarcsec = 2. * pi/360./3600.*1000.
minsize_kpc=1.3*2 # one mips pixel at distance of hercules
clusternames=['MKW11', 'MKW8', 'AWM4', 'A2063', 'A2052', 'NGC6107', 'Coma', 'A1367', 'Hercules']
clusternamesbylx=['MKW11',  'NGC6107','MKW8', 'AWM4', 'Hercules','A1367','A2063', 'A2052', 'Coma'  ]
clusternamesbydistance=['A1367','MKW11', 'Coma','MKW8',  'NGC6107', 'AWM4','A2063', 'A2052', 'Hercules']

numberofstars={'MKW11':5, 'MKW8':2, 'AWM4':5, 'A2063':5, 'A2052':4, 'NGC6107':5, 'Coma':5, 'A1367':1, 'Hercules':1}

clusterRA={'MKW11':202.3800, 'MKW8':220.1796, 'AWM4':241.2375, 'A2063':230.7578, 'A2052':229.1896, 'NGC6107':244.333750, 'Coma':194.9531, 'A1367':176.1231, 'Hercules':241.3125,'MKW10':175.5449}

clusterDec={'MKW11':11.78861, 'MKW8':3.4530, 'AWM4':23.9206, 'A2063':8.6394, 'A2052':7.0003, 'NGC6107':34.901389, 'Coma':27.9807, 'A1367':19.8391, 'Hercules':17.7485,'MKW10':10.3059}

clustervel={'MKW11':6854., 'MKW8':8100., 'AWM4':9526., 'A2063':10481., 'A2052':10647., 'NGC6107':9197., 'Coma':6900., 'A1367':8400., 'Hercules':11100.,'MKW10':6158.}

clustersigma={'MKW11':361, 'MKW8':325., 'AWM4':500., 'A2063':660., 'A2052':562., 'NGC6107':500., 'Coma':1000., 'A1367':745., 'Hercules':689.}

clusterf80MJysr={'MKW11':4., 'MKW8':3.75, 'AWM4':3.5, 'A2063':4., 'A2052':4., 'NGC6107':3.25, 'Coma':2.25, 'A1367':3.5, 'Hercules':3.25}

clusterz={'MKW11':.022849,'MKW8':.027,'AWM4':.031755,'A2063':.034937,'A2052':.035491,'NGC6107':.030658,'Coma':.023,'A1367':.028,'Hercules':.037,'MKW10':.02054}

john_prefix={'MKW11':'mkw11','MKW8':'mkw8','AWM4':'awm4','A2063':'abell2063','A2052':'abell2052','NGC6107':'ngc6107','Coma':'coma','A1367':'abell1367','Hercules':'hercules'}

xraycontourlevels={'MKW11':[.85,1.69,2.54],'MKW8':[.49,.99,1.48,1.98],'AWM4':[.8,1.6,2.4],'NGC6107':[1.43,2.85,4.27],'A2052':[.9,1.8,2.7,3.6],'A2063':[.9,1.8,2.7,3.6],'Hercules':[.9,1.92,2.9,3.8],'A1367':[.6,1.17,1.76,2.35],'Coma':[.88,1.76,2.63,3.51]}#used contour option in ds9 to derive these

coma_badobjects=[142589,104115,104020,104022,142662,142763,142797,162768,162797]

spiral_nozoo={'MKW11':[70685, 143485, 143530,143570,169997,171141], \
              'MKW8':[15218,18127,145303,145586,165696], \
              'AWM4':[63611, 68238, 68271, 68272, 68283,  68287, 68288,  68338,68341,68387,68430,68435, 68436,68437, 68439,68432, 146715, 166624], \
              'NGC6107':[43707,  43712,  43787,  43857,  69538], \
              'A2052':[79550,79593, 79646,79665,79680, 79705, 145994, 166042], \
              'A2063':[72672, 72739, 72767, 72775,146137,166124], \
              'Hercules':[99840, 146607,146635], \
              'A1367':[140124,140145, 140176, 140194], \
              'Coma':[104125, 104232,142527,142561,142563,142568,142572, 142627,142642,142653,142656,142663,142666,142668,142669,142676,142682,142687,142689,142693,142695,142706,142710,142713,142715,142716,142723,142727,142737,142740,142745,142750,142755,142758,142765,142767,142769,142774,142779,142781,142793,142795,142801,142804,142806,142808,142809,142810,142815,142819,142825,142837,142847,142855,142873,142914,162740]}#used contour option in ds9 to derive these

zoo_overide_flag = [70637, 70658, 70676, 163589, 169994] # galaxies that are clearly spirals but have low probability of being a spiral (p_cs < 0.7) according to galaxy zoo

spiral_100_nozoo={'MKW11':[70685, 143485, 143530],'MKW8':[18127],'AWM4':[68283,  68288,  68338,  68341, 166624],'NGC6107':[43707,  43712,  43857],'A2052':[79646,145994,166042],'A2063':[166124],'Hercules':[99840, 146607],'A1367':[140124,140177],'Coma':[142572, 142668,142914,162740]}#100% sure these are spirals

elliptical_nozoo={'MKW11':[143436,143514,143529],\
                  'MKW8':[145280],\
                  'AWM4':[68279,68438,146626],\
                  'NGC6107':[146832, 146876, 146878,146880, 166860],
                  'A2052':[79600,  79610, 79705,79710,146012, 146037, 146041],\
                  'A2063':[72751],\
                  'Hercules':[146638,146664],\
                  'A1367':[113076,113458,  140164],\
                  'Coma':[103978,104022,104061,104115,142531,142552,142584,142585,142604,142605,142609,142611,142614,142615,142616,142622,142623,142628,142636,142637,142638,142647,142648,142649,142651,142658,142660,142661,142675,142677,142678,142681,142684,142690,142699,142705,142717,142721,142725,142729,142741,142743,142761,142787,142803,142813,142832,142852,142866,162659]}#used contour option in ds9 to derive these

irreg_nozoo={'MKW11':[143709, 171128],'MKW8':[18255,165628],'AWM4':[],'NGC6107':[],'A2052':[],'A2063':[],'Hercules':[146673, 146680, 166679],'A1367':[140170,  140183, 140184, 140186, 160496],'Coma':[142559,142560,142578,142590,142593,142613,142620,142631,142645,142652,142667,142673,142679,142697,142718,142733,142753,142762,142771,142786,142821,142823,142826,142831,142834,142849,162689]}#used contour option in ds9 to derive these

unsure_nozoo={'MKW11':[],'MKW8':[],'AWM4':[],'NGC6107':[],'A2052':[],'A2063':[72627],'Hercules':[146680, 166679],'A1367':[140170,140176,140194, 160496],'Coma':[]}

# galaxies to cut from sample, at least initially
# galaxies that have contamination be nearby neighbor.  see notes.
visual_cut={'MKW11':[70639,70694,143485,171004],'MKW8':[18111, 18171],'AWM4':[82134, 82188, 82209, 146626, 166655, 166699],'NGC6107':[43782, 43814, 69617, 69618],'A2052':[ 79388, 166086],'A2063':[72631, 72710, 72745, 72782, 146106, 146107, 146124, 146128, 146130,146135],'Hercules':[99056, 99644, 99822, 99859, 99872, 146607, 146659, 166638],'A1367':[113058, 113404,  140197],'Coma':[103612, 103628, 103648, 103784, 103831, 103833, 103844, 103924,  103933, 104001, 104004, 104035, 104126, 142655, 142840, 162793, 162831]}


#Group names
groupnames=['NRGb041','NRGb151','NRGb157','NRGb168','NRGb206','NRGb247','NRGb282','NRGb301','MKW8','NCG5846','NRGs076','NRGs272','NRGs385']
altgroupnames=['WBL226','MKW10','HCG59','WBL368','WBL404','MKW11test','Zw1400','WBL509','MKW8','NGC5846','WBL251','WBL477','NGC6107']
#location of Final images


# central biweight location as calculated from findbestbiweight code
# clusterbiweightcenter={'MKW11':6906,'MKW8':8098,'AWM4':9650,'A2063':10422,'A2052':10354.5,'NGC6107':9429,'Coma':6999,'A1367':6481,'Hercules':10957.5}

 #sbi values output from +/- 4000km/s and 1 degree velocity cut from findbestbiweight code
# clusterbiweightscale={'MKW11':392.37,'MKW8':491.32,'AWM4':476.67,'A2063':727.06,'A2052':626.32,'NGC6107':616.86,'Coma':937.03,'A1367':794.61,'Hercules':772.74}

# redid biweight calculations in Jan 2015 to use NSA as base catalog
# also implemented bootstrap resampling for errors
# central biweight location as calculated from LCSbiweight code
clusterbiweightcenter={'MKW11':6904,'MKW8':8039,'AWM4':9604,'A2063':10410,'A2052':10431,'NGC6107':9397,'Coma':7011,'A1367':6505,'Hercules':10917}

clusterbiweightcenter_errp={'MKW11':38,'MKW8':40,'AWM4':61,'A2063':72,'A2052':57,'NGC6107':57,'Coma':45,'A1367':55,'Hercules':50}

clusterbiweightcenter_errm={'MKW11':49,'MKW8':38,'AWM4':55,'A2063':74,'A2052':64,'NGC6107':53,'Coma':44,'A1367':54,'Hercules':53}

 #sbi values output from +/- 4000km/s and 1 degree velocity cut from findbestbiweight code
clusterbiweightscale={'MKW11':383,'MKW8':443,'AWM4':458,'A2063':862,'A2052':666,'NGC6107':578,'Coma':1054,'A1367':838,'Hercules':790}
clusterbiweightscale_errp={'MKW11':19,'MKW8':29,'AWM4':107,'A2063':42,'A2052':37,'NGC6107':47,'Coma':26,'A1367':31,'Hercules':29}
clusterbiweightscale_errm={'MKW11':27,'MKW8':31,'AWM4':95,'A2063':65,'A2052':45,'NGC6107':34,'Coma':29,'A1367':42,'Hercules':31}

# X-ray luminosity in 10^43 ergs/s
# from Bohringer et al 2000, and Mahdavi et Geller
#clusterLx={'MKW11':0.033,'MKW8':0.096,'AWM4':0.550,'A2063':1.940,'A2052':2.580,'NGC6107':0.33,'Coma':7.010,'A1367':1.510,'Hercules':0.980}
# from  http://bax.ast.obs-mip.fr/servlets/omp.servlet.ClusterQueryByName#
# Lx (10^44 ergs/s) in 0.1-2.4 keV band
clusterLx={'MKW11':0.073397, # Jones & Forman 1999
           'MKW8':0.096, # no Lx from bax
           'AWM4':0.51799, # Bohringer + 2000
           'A2063':2.196055, # Reiprich 2002
           'A2052':2.521777, # Reiprich 2002
           'NGC6107':0.331708, # Bohringer + 2000
           'Coma':7.766525, #http://cdsads.u-strasbg.fr/cgi-bin/nph-bib_query?bibcode=2002ApJ...567..716R&db_key=AST Reiprich
           'A1367':1.244663, # Reiprich 2002
           'Hercules':0.900308} # Reiprich 2002
clusterTx={'MKW11':0.96, #+/- 0.4, Osmond+ 2004
           'MKW8':3.29, # Cavagnolo +2009
           'AWM4':2.48, #+/- .06, Gasteldello + 2008
           'A2063':3.7, # no temp measurement at bax, see below
           'A2052':3.12, # -.05, +.06 Ikebe + 2002
           'NGC6107':-99.,  # no temp measurement in bax
           'Coma':8.25, #+/- 0.1, Arnaud 2001A&A
           'A1367':3.55, # +/- .05  Ikebe + 2002
           'Hercules':2.52} #+/- .12  Ikebe + 2002


# Tx, errdown, errup
clusterTx1={'MKW11':[0],'MKW8':[3.,.12,.12],'AWM4':[0],'A2063':[3.77,.06,.06],'A2052':[3.35,.02,.02],'NGC6107':[0],'Coma':[9.15,.17,.17],'A1367':[3.58,.06,.06],'Hercules':[0]}
# X-ray temp in keV; from Mittal et al 2011

clusterTx2={'MKW11':[0],'MKW8':[2.74,.03,.03],'AWM4':[0],'A2063':[3.70,.02,.02],'A2052':[2.98,.03,.03],'NGC6107':[0],'Coma':[7.31,.06,.06],'A1367':[2.56,.02,.02],'Hercules':[0]} #Frank+2013, ApJ, 764, 46 XMM-NEWTON observations

clusterLx2={'MKW11':0.033,'MKW8':(.692,.058),'AWM4':0.550,'A2063':(2.06,.027),'A2052':(2.18,.022),'NGC6107':0.083,'Coma':(11.1,.156),'A1367':(1.13,.009),'Hercules':0.980}

# list of L500 (1.e37 W), M500(1.e14 Msun) and R500 (Mpc) from Piffaretti+ 2011
clusterXray={'MKW11':[0.065077,	0.3805,	0.5078],'MKW8':[0.192567,0.7352,0.6316],'AWM4':[0.284521,0.9289,0.6815],'A2063':[1.138819,2.1598,0.9020],'A2052':[1.442058,2.4945,0.9465],'NGC6107':[0.168099,0.6744,0.6127],'Coma':[3.455556,4.2846,1.1378],'A1367':[1.104603,2.1398,0.9032],'Hercules':[0.508824,1.3202,0.7652]}


# these correpond to area w/uniform 24um coverage
# center x,y,dx,dy,rotation E of N, all in degrees
cluster24Box={'MKW11':array([202.36239,11.752736,1.3138054,3.046197,27.0001],'f'), 'MKW8':array([220.18764,3.4955922,1.3188409,3.040413,13.5],'f'), 'AWM4':array([241.21434,23.872723,1.3441978,3.0241238,10],'f'), 'A2063':array([230.77172,8.6817732,1.3126447,3.0415136,13.5001],'f'), 'A2052':array([229.19761,7.0403283,1.3194664,3.0412907,13.25],'f'), 'NGC6107':array([244.30039,34.934184,1.3199655,3.0435265,322],'f'), 'Coma':array([194.86318,27.865896,1.5391027,1.976467,29.5002],'f'), 'A1367':array([176.1019,19.799614,.51080152,.90025557,31.5],'f'), 'Hercules':array([241.3065,17.771646,.51029561,.93431905,19.5001],'f')}


#solar magnitude in SDSS filters
SolarMag={'u':6.39,'g':5.07,'r':4.62,'i':4.52,'z':4.48}

#cosmology
H0=70
OmegaL=0.7
OmegaM=0.3
h=H0/100.

# bell+2003 stellar mass coefficients for sdss filters
# diet Salpeter IMF - 30% lower than Salpeter IMF, less mass from lower-mass stars
# log10(chabrier) = log10(Salpeter) - .25 (used in SFR estimate)
# log10(chabrier) = log10(diet Salpeter) - 0.1 (used in Stellar mass estimates)
bellug={'g':[-.221,0.485],'r':[-.099,0.345],'i':[-.053,0.268],'z':[-.105,0.226]}
bellur={'g':[-.390,0.417],'r':[-.223,0.229],'i':[-.151,0.233],'z':[-.178,0.192]}
bellui={'g':[-.375,0.359],'r':[-.212,0.257],'i':[-.144,0.201],'z':[-.171,0.165]}
belluz={'g':[-.400,0.332],'r':[-.232,0.239],'i':[-.161,0.187],'z':[-.179,0.151]}
bellgr={'g':[-.499,1.519],'r':[-.306,1.097],'i':[-.222,0.864],'z':[-.223,0.689]}
bellgi={'g':[-.379,0.914],'r':[-.220,0.661],'i':[-.152,0.518],'z':[-.175,0.421]}
bellgz={'g':[-.367,0.698],'r':[-.215,0.508],'i':[-.153,0.402],'z':[-.171,0.322]}
bellri={'g':[-.106,1.982],'r':[-.022,1.431],'i':[0.006,1.114],'z':[-.952,0.923]}
bellrz={'g':[-.124,1.067],'r':[-.041,0.780],'i':[-.018,0.623],'z':[-.041,0.463]}

snr24cut=5.
deltaCutout=100.#width of cutouts in arcsec
ramin=170.#cuts for culling the ac
ramax=250.#cuts for culling the ac
decmin=0.
decmax=38.#cuts for culling the ac
zmin=0.01366#min z cut, z(coma)-3 sigma
zmax=0.04333#max z cut, z(A2052, which is  10900 km/s)+ 4*sigma
vmin=zmin*3.e5
vmax=zmax*3.e5
#cutoutpath='/home/rfinn/research/LocalClusters/cutouts/'
cutoutpath='/home/rfinn/research/LocalClusters/cutouts/'

Lsol=3.826e33#normalize by solar luminosity
bellconv=9.8e-11#converts Lir (in L_sun) to SFR/yr
bellconv=4.5e-44#Kenn 98 conversion from erg/s to SFR/yr, assumes salpeter IMF
catalog_radial_cut = 3. # mastertable radial cut in degrees
mypath=os.getcwd()
if mypath.find('Users') > -1:
    print("Running on Rose's mac pro")
    homedir='/Users/rfinn/'
elif mypath.find('home') > -1:
    print("Running on coma")
    homedir='/home/rfinn/'

mipsflux2umJyconv=141.086
nmgy_muJy_sqarc_conv=3.631/sdsspixelscale**2
MJy_muJy_sqarc_conv=141.09/mipspixelscale**2

def uJy2ABmag(f): 
    mag=23.9-2.5*np.log10(f)
    return mag
def ABmag2uJy(mag): # returns micro-Jy
    f=10.**((mag-23.9)/(-2.5))
    return f

sdss_sb_cut=.025*(sdsspixelscale**2)
sdss_sb_cut=ABmag2uJy(25.5)/nmgy_muJy_sqarc_conv
# use a lower limit for MIPS as well
mips_sb_cut=.1/2.5


def multiplotaxes(i):
    ax=gca()
    noylabel=[2,3,5,6,8,9]
    if i < 7:
        ax.set_xticklabels(([]))
    if i in noylabel:
        ax.set_yticklabels(([]))
def multiplotlabels(xl,yl):
    ax=gca()
    text(-.5,-.25,xl,fontsize=22,horizontalalignment='center',transform=ax.transAxes)
    text(-2.45,1.5,yl,fontsize=22,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')
def multiplotlabelsv2(xl,yl): # for figures with figsize=(6.5,4)
    ax=gca()
    text(-.5,-.3,xl,fontsize=14,horizontalalignment='center',transform=ax.transAxes)
    text(-2.35,1.6,yl,fontsize=14,verticalalignment='center',rotation=90,transform=ax.transAxes,family='serif')

def spearman_boot(x,y,N=5000,cont_int=68.):
    boot_rho=zeros(N,'f')
    boot_p=zeros(N,'f')
    for i in range(N):
        indices=randint(0,len(x)-1,len(x))
        xboot=x[indices]
        yboot=y[indices]
        boot_rho[i],boot_p[i]=spearmanr(xboot,yboot)
    return scoreatpercentile(boot_rho,per=50),scoreatpercentile(boot_p,per=50)#,boot_rho,boot_p 

def spearman(x,y):
    #rho,pvalue=spearmanr(x,y)
    rho,pvalue=spearman_boot(x,y)
    print('Spearman Rank Test:')
    print('rho = %6.2f'%(rho))
    print('p-vale = %6.5f (prob that samples are uncorrelated)'%(pvalue)) 
    return rho,pvalue

def spearman_with_errors(x,y,yerr,Nmc=1000,plotflag=False,verbose=False):
    ysim=np.zeros(Nmc,'f')
    rhosim=np.zeros(Nmc,'f')
    psim=np.zeros(Nmc,'f')

    for i in range(Nmc):
        ysim=np.random.normal(y,scale=yerr,size=len(y))
        rhosim[i],psim[i] = spearmanr(x,ysim)
    cave=np.mean(rhosim)
    cstd=np.std(rhosim)
    q1=50-34 # mean minus one std
    lower=np.percentile(rhosim,q1)
    q2=50+34 # mean minus one std
    upper=np.percentile(rhosim,q2)
    print('mean (median) = %5.2f (%5.2f), std = %5.2f'%(cave,np.median(rhosim),cstd))
    print('confidence interval from sorted list of MC fit values:')
    print('lower = %5.2f (%5.2f), upper = %5.2f (%5.2f)'%(lower,cave-cstd, upper,cave+cstd))
    k,pnorm=normaltest(rhosim)
    print('probability that distribution of slopes is normal = %5.2f'%(pnorm))
    if plotflag:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.hist(rhosim,bins=10,normed=True)
        plt.xlabel(r'$Spearman \ \rho $')
        plt.axvline(x=cave,ls='-',color='k')
        plt.axvline(x=lower,ls='--',color='k')
        plt.axvline(x=upper,ls='--',color='k')
        plt.subplot(1,2,2)
        plt.hist(np.log10(psim),bins=10,normed=True)
        plt.xlabel(r'$\log_{10}(p \ value)$')
    return rhosim,psim

def ks_boot(x,y,N=1,conf_int=68.):
    boot_p=zeros(N,'f')
    boot_D=zeros(N,'f')
    for i in range(N):
        xboot=x[randint(0,len(x)-1,len(x))]
        yboot=y[randint(0,len(y)-1,len(y))]
        boot_D[i],boot_p[i]=ks_2samp(xboot,yboot)
    return scoreatpercentile(boot_D,per=50),scoreatpercentile(boot_p,per=50) 
def ks(x,y,run_anderson=True,Nboot=1):
    '''
    GOAL:
    * compute KS test on two input arrays
    * print results
    * optional : run anderson-darling test

    INPUT:
    * x : one sample for comparison
    * y : second sample for comparison
    * run_anderson : set to true to run anderson-darling test
    * Nboot : default=1; number of bootstrap resamples to do for KS test;
              will calculate mean value of D and p-value

    OUTPUT:
    * D : KS D statistic
    * pvalue : KS p value

    '''
    if Nboot == 1:
        D,pvalue=ks_2samp(x,y)
    else:
        D,pvalue=ks_boot(x,y,N=Nboot)
    print('KS Test:')
    print('D = %6.2f'%(D))
    print('p-vale = %1.3e (prob that samples are from same distribution)'%(pvalue))
    if run_anderson:
        anderson(x,y)
    return D,pvalue

def anderson(x,y):
    t=anderson_ksamp([x,y])
    print('Anderson-Darling test Test:')
    print('D = %6.2f'%(t[0]))
    print('p-vale = %6.5f (prob that samples are from same distribution)'%(t[2])) 
    return t[0],t[2]

def findnearest(x1,y1,x2,y2,delta):#use where command
	matchflag=1
	nmatch=0
	d=sqrt((x1-x2)**2 + (y1-y2)**2)#x2 and y2 are arrays
	index=arange(len(d))
	t=index[d<delta]
	matches=t
	if len(matches) > 0:
		nmatch=len(matches)
		if nmatch > 1:
			imatch=index[(d == min(d[t]))]
		else:
			imatch=matches[0]			
	else:
		imatch = 0
		matchflag = 0

	return imatch, matchflag,nmatch

def drawbox(data,style):#feed in center x,y,dx,dy,rotation E of N
    #xcoords of unrotated box, going around CCW
    xl=array([data[0]-0.5*data[2],data[0]+0.5*data[2],data[0]+0.5*data[2],data[0]-0.5*data[2],data[0]-0.5*data[2]],'d')
    yl=array([data[1]-0.5*data[3],data[1]-0.5*data[3],data[1]+0.5*data[3],data[1]+0.5*data[3],data[1]-0.5*data[3] ],'d')

    xl=array([-0.5*data[2],+0.5*data[2],+0.5*data[2],-0.5*data[2],-0.5*data[2]],'d')
    yl=array([-0.5*data[3],-0.5*data[3],+0.5*data[3],+0.5*data[3],-0.5*data[3] ],'d')

    ang=data[4]*pi/180.*-1.#convert rotation to radians
    #rotate coordinates
    xp=cos(ang)*xl-sin(ang)*yl
    yp=sin(ang)*xl+cos(ang)*yl

    #put back on absolute scale
    xp=data[0]+xp
    yp=data[1]+yp
    #draw rotated box
    plot(xp,yp,style,zorder=20)

def transcoords(imge,coords):
    outcoords='junk.xy'
    s='rm '+outcoords
    os.system(s)
    iraf.imcoords.wcsctran(image=self.image,input=self.incoords,output=outcoords,inwcs='world',outwcs='logical',verbose='no')
    return outcoords
    
## convert SB in mag/sq arcsec to flux per pixel on mips
def convert_sb_to_fluxperpixel(sb):
    flux_zp_AB = 3631. # in Jy
    flux_zp_Vega = 7.17 # in Jy
    flux_zp=flux_zp_AB
        
    # conversion from image units of MJ/sr to micro-Jy (1 sq arcsec = 2.3504e-11 sr)
    conv_MJysr_uJy = 23.5045*(2.45**2)
    magzp=2.5*log10(flux_zp*1.e6/conv_MJysr_uJy)
    # m2 - m1 = 2.5 log10(f1/f2)
    flux_sb=10.**(-1.*sb/2.5)*flux_zp # flux (Jy) per sq arcsec

    # flux in micro-Jy
    flux_sb=flux_sb*1.e6
    # area of a pixel in sq arcsec
    parea = mipspixelscale**2

    # convert to uJy per sq pixel
    flux_sb=flux_sb*parea
    # convert to image units of MJy/sr
    flux_sb=flux_sb/conv_MJysr_uJy
    
    return flux_sb
def binxycolor(x,y,color,nbin=5,erry=False,use_median=False,equal_pop_bins=False):
    '''
    - bin x in nbin equally spaced bins
    - calculate the median y value in each bin
    - calculate the median color in each bin
    '''
    xbins = np.zeros(nbin,'f')
    ybins = np.zeros(nbin,'f')
    ybinerr = np.zeros(len(xbins),'f')
    colorbins = np.zeros(len(xbins),'f')
    if equal_pop_bins:
        sorted_indices = np.argsort(x)
        y = y[sorted_indices]
        x = x[sorted_indices]
        color = color[sorted_indices]
        n_per_bin = len(x)/nbin
        xbin_number = np.arange(len(x))/int(n_per_bin)
        #print xbin_number
        #print x
    else:
        xbin_number = np.array(((x-min(x))*nbin/(max(x)-min(x))),'i')
    for i in range(nbin):
        if use_median:
            xbins[i] = np.median(x[xbin_number == i])
            ybins[i] = np.median(y[xbin_number == i])
            colorbins[i] = np.median(color[xbin_number == i])
        else:
            xbins[i] = np.mean(x[xbin_number == i])
            ybins[i] = np.mean(y[xbin_number == i])
            colorbins[i] = np.mean(color[xbin_number == i])

        ybinerr[i] = np.std(y[xbin_number == i])/np.sqrt(sum(xbin_number == i))
                      
    if erry:
        return xbins,ybins,ybinerr,colorbins
    else:
        return xbins,ybins,colorbins
