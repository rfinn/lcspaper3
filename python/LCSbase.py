#!/usr/bin/env python

'''

GOAL:
- read catalog 


USAGE:
- in ipython


    
SIZE RATIO
- I set up self.size_ratio to be the size used in all plots.

REQUIRED MODULES:


**************************
written by Rose A. Finn
May 2018
**************************

'''

from astropy.io import fits
from astropy import constants as c 
from astropy import units as u
from astropy.table import Table, Column
import os
import sys
from astropy.cosmology import WMAP9 as cosmo
import numpy as np

# set environment variables to point to github repository
# call it GITHUB_PATH
# LCS_TABLE_PATH

# python magic to get environment variable
homedir = os.environ['HOME']

# assumes github folder exists off home directory
# and that LCS is a repository in github
lcspath = homedir+'/github/LCS/'
if not(os.path.exists(lcspath)):
    print(('could not find directory: ',lcspath))
    sys.exit()

Mpcrad_kpcarcsec = 2. * np.pi/360./3600.*1000.
minmass=9.7

mipspixelscale=2.45

minsize_kpc=1.3 # one mips pixel at distance of hercules

class galaxies:
    def __init__(self, lcspath):

        #self.jmass=fits.getdata(lcspath+'tables/LCS_Spirals_all_fsps_v2.4_miles_chab_charlot_sfhgrid01.fits')
        # use jmass.mstar_50 and jmass.mstar_err

        #self.agc=fits.getdata(lcspath+'tables/LCS_Spirals_AGC.fits')

        self.s=fits.getdata(lcspath+'tables/LCS_all_size.fits')

        self.gim2d=fits.getdata(lcspath+'tables/LCS_all.gim2d.tab1.fits')
        
        # dictionary to look up galaxies by NSAID
        self.nsadict=dict((a,b) for a,b in zip(self.s.NSAID,np.arange(len(self.s.NSAID))))
        self.NUVr=self.s.ABSMAG[:,1] - self.s.ABSMAG[:,4]
        self.upperlimit=self.s['RE_UPPERLIMIT'] # converts this to proper boolean array

        self.MAG24 = 2.5*np.log10(3631./(self.s.FLUX24*1.e-6))
        
        self.dL = cosmo.luminosity_distance(self.s.ZDIST)
        self.distmod_ZDIST = 5*np.log10(self.dL.value*1.e6)-5
        self.ABSMAG24 = self.MAG24 - self.distmod_ZDIST
        self.NUV24 = self.s.ABSMAG[:,1] - self.ABSMAG24
        if __name__ != '__main__':
            self.setup()
        self.logstellarmass = self.s.MSTAR_50
        self.clusterflag = (self.s.CLUSTER == b'Coma')| (self.s.CLUSTER == b'A2063')# | (self.s.CLUSTER == b'Hercules') | (self.s.CLUSTER == b'A1367')  # | (self.s.CLUSTER == b'A2052')
    def get_agn(self):
        self.AGNKAUFF=self.s['AGNKAUFF'] & (self.s.HAEW > 0.)
        self.AGNKEWLEY=self.s['AGNKEWLEY']& (self.s.HAEW > 0.)
        self.AGNSTASIN=self.s['AGNSTASIN']& (self.s.HAEW > 0.)
        self.AGNKAUFF= ((np.log10(self.s.O3FLUX/self.s.HBFLUX) > (.61/(np.log10(self.s.N2FLUX/self.s.HAFLUX)-.05)+1.3)) | (np.log10(self.s.N2FLUX/self.s.HAFLUX) > 0.)) #& (self.s.HAEW > 0.)
        # add calculations for selecting the sample
        self.wiseagn=(self.s.W1MAG_3 - self.s.W2MAG_3) > 0.8
        self.agnflag = self.AGNKAUFF | self.wiseagn

    def get_gim2d_flag(self):
        self.gim2dflag=self.gim2d.recno > 0. # get rid of nan's in Rd
    def get_galfit_flag(self):
        self.sb_obs=np.zeros(len(self.s.RA))
        flag= (~self.s['fcnumerical_error_flag24'])
        self.sb_obs[flag]=self.s.fcmag1[flag] + 2.5*np.log10(np.pi*((self.s.fcre1[flag]*mipspixelscale)**2)*self.s.fcaxisratio1[flag])

        self.nerrorflag=self.s['fcnumerical_error_flag24']
        self.badfits=np.zeros(len(self.s.RA),'bool')
        #badfits=array([166134, 166185, 103789, 104181],'i')'
        nearbystar=[142655, 143485, 99840, 80878] # bad NSA fit; 24um is ok
        #nearbygalaxy=[103927,143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        # checked after reworking galfit
        nearbygalaxy=[143485,146607, 166638,99877,103933,99056]#,140197] # either NSA or 24um fit is unreliable
        #badNSA=[166185,142655,99644,103825,145998]
        #badNSA = [
        badfits= nearbygalaxy#+nearbystar+nearbygalaxy
        badfits=np.array(badfits,'i')
        for gal in badfits:
            self.badfits[np.where(self.s.NSAID == gal)]  = 1

        self.galfitflag = (self.s.fcmag1 > .1)  & ~self.nerrorflag & (self.sb_obs < 20.) & (self.s.fcre1/self.s.fcre1err > .5)#20.)
        #override the galfit flag for the following galaxies
        self.galfit_override = [70588,70696,43791,69673,146875,82170, 82182, 82188, 82198, 99058, 99660, 99675, 146636, 146638, 146659, 113092, 113095, 72623,72631,72659, 72749, 72778, 79779, 146121, 146130, 166167, 79417, 79591, 79608, 79706, 80769, 80873, 146003, 166044,166083, 89101, 89108,103613,162792,162838, 89063]
        for id in self.galfit_override:
            try:
                self.galfitflag[self.nsadict[int(id)]] = True
            except KeyError:

                if self.prefix == 'no_coma':
                    print ('ids not relevant for nc')
                else:
                    sys.exit()
        #self.galfitflag = self.galfitflag 
        self.galfitflag[self.nsadict[79378]] = False
        self.galfitflag = self.galfitflag & ~self.badfits

    def get_size_flag(self):
        self.DA=np.zeros(len(self.s.SERSIC_TH50))
        #self.DA[self.membflag] = cosmo.angular_diameter_distance(self.s.CLUSTER_REDSHIFT[self.membflag]).value*Mpcrad_kpcarcsec)
        for i in range(len(self.DA)):
            if self.membflag[i]:
                self.DA[i] = cosmo.angular_diameter_distance(self.s.CLUSTER_REDSHIFT[i]).value*Mpcrad_kpcarcsec
            else:
                self.DA[i] = cosmo.angular_diameter_distance(self.s.ZDIST[i]).value*Mpcrad_kpcarcsec
        self.sizeflag=(self.s.SERSIC_TH50*self.DA > minsize_kpc) #& (self.s.SERSIC_TH50 < 20.)

    def select_sample(self):
        self.logstellarmass =  self.s.MSTAR_50 # self.logstellarmassTaylor # or
        self.massflag=self.s.MSTAR_50 > minmass
        self.Re24_kpc = self.s.fcre1*mipspixelscale*self.DA
        self.lirflag=(self.s.LIR_ZDIST > 5.2e8)

 
        self.sbflag=self.sb_obs < 20.

        self.sb_obs=np.zeros(len(self.s.RA))
        flag= (~self.s['fcnumerical_error_flag24'])
        self.sb_obs[flag]=self.s.fcmag1[flag] + 2.5*np.log10(np.pi*((self.s.fcre1[flag]*mipspixelscale)**2)*self.s.fcaxisratio1[flag])

        

        #self.agnkauff=self.s.AGNKAUFF > .1
        #self.agnkewley=self.s.AGNKEWLEY > .1
        #self.agnstasin=self.s.AGNSTASIN > .1
        self.dv = (self.s.ZDIST - self.s.CLUSTER_REDSHIFT)*3.e5/self.s.CLUSTER_SIGMA
        self.dvflag = abs(self.dv) < 3.

        self.sampleflag = self.galfitflag    & self.lirflag   & self.sizeflag & ~self.agnflag & self.sbflag & self.gim2dflag#& self.massflag#& self.gim2dflag#& self.blueflag2
        self.agnsampleflag = self.galfitflag    & self.lirflag   & self.sizeflag & self.agnflag & self.sbflag & self.gim2dflag#& self.massflag#& self.gim2dflag#& self.blueflag2
        self.sfsampleflag = self.sizeflag & self.lirflag & ~self.agnflag # & ~self.badfits
        self.irsampleflag = self.lirflag & self.sizeflag & ~self.agnflag
        self.HIflag = self.s.HIMASS > 0.

    def calculate_sizeratio(self):
        self.SIZE_RATIO_DISK = np.zeros(len(self.gim2dflag))
        print((len(self.s.fcre1),len(self.gim2dflag)))
        a =  self.s.fcre1[self.gim2dflag]*mipspixelscale # fcre1 = 24um half-light radius in mips pixels
        b = self.DA[self.gim2dflag]
        c = self.gim2d.Rd[self.gim2dflag] # gim2d half light radius for disk in kpc

        #calculate 24um size in kpc
        self.mipssize = self.s.fcre1*mipspixelscale * self.DA
        #calculate optical size in kpc
        self.optdisksize = self.gim2d.Rd

        # this is the size ratio we use in paper 1
        self.SIZE_RATIO_DISK[self.gim2dflag] =a*b/c
        self.SIZE_RATIO_DISK_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_DISK_ERR[self.gim2dflag] = self.s.fcre1err[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.gim2d.Rd[self.gim2dflag]

        # 24um size divided by the r-band half-light radius for entire galaxy (single component)
        self.SIZE_RATIO_gim2d = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_gim2d[self.gim2dflag] = self.s.fcre1[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.gim2d.Rhlr[self.gim2dflag]
        self.SIZE_RATIO_gim2d_ERR = np.zeros(len(self.gim2dflag))
        self.SIZE_RATIO_gim2d_ERR[self.gim2dflag] = self.s.fcre1err[self.gim2dflag]*mipspixelscale*self.DA[self.gim2dflag]/self.gim2d.Rhlr[self.gim2dflag]

        # 24um size divided by NSA r-band half-light radius for single component sersic fit
        self.SIZE_RATIO_NSA = self.s.fcre1*mipspixelscale/self.s.SERSIC_TH50
        self.SIZE_RATIO_NSA_ERR=self.s.fcre1err*mipspixelscale/self.s.SERSIC_TH50
        self.sizeratio = self.SIZE_RATIO_DISK
        self.sizeratioERR=self.SIZE_RATIO_DISK_ERR
        # size ratio corrected for inclination 
        self.size_ratio_corr=self.sizeratio*(self.s.faxisratio1/self.s.SERSIC_BA)
    def get_memb(self):
        self.dv = (self.s.ZDIST - self.s.CLUSTER_REDSHIFT)*3.e5/self.s.CLUSTER_SIGMA
        self.dvflag = abs(self.dv) < 3.

        self.membflag = abs(self.dv) < (-4./3.*self.s.DR_R200 + 2)

    def make_SFR(self):
        #use SF_ZCLUST for cluster members and SFR_ZDIST for others.
        #assume that these are Chabrier
        self.SFR_BEST = self.s.SFR_ZCLUST * np.array(self.membflag,'i') + np.array(~self.membflag,'i')*(self.s.SFR_ZDIST)

        #now scale SFRs down by 1.58 dex to convert from Salpeter IMF
        #(which Chary+Elbaz 2001 use) to Chabrier
        #this conversion comes from Salim+16.  
        self.SFR_BEST = self.SFR_BEST / 1.58
        self.LIR_BEST = self.s.LIR_ZCLUST * np.array(self.membflag,'i') + np.array(~self.membflag,'i')*(self.s.LIR_ZDIST)

    def get_UVIR_SFR(self):
        print('nothing happening here')
        # kennicutt and evans
        # calibrations of integrated SFRs
        # corrected FUV - LUV
        # eqn 12
        # log SFR(Msun/yr) = log Lx - log Cx
        # NUV - log Cx = 43.17
        # 24um - logCx = 42.69
        # Halpha - log Cx = 41.27


        # corrected NUV luminosity
        # L(NUV)corr = L(NUV)obs + 2.26L(25um) 

        # LCS_all_size has L_IR from Chary & Elbaz, and FLUX24
        # will need to calculate L24 from FLUX24
        # assuming L24 is close enough to L25 to be interchangeable
        #
        # The Kennicutt & Evans eqn needs L25 in units of ergs/s - nu L_nu
        #
        # the FLUX24 is probably in units of Jy or uJy.
        # multiply by frequency at 24um to convert to nu F_v
        wavelength = 24.e-6*u.m
        freq = c.c/wavelength
        flux = self.s.FLUX24*1.e-6*u.Jy
        self.nuFnu24 = flux*freq
        
        # multiply by 4 pi d_L**2 to get nu L_nu
        self.nuLnu24_ZDIST = self.nuFnu24 * 4 * np.pi * (cosmo.luminosity_distance(self.s.ZDIST))**2
        self.nuLnu24_ZCLUST = self.nuFnu24 * 4 * np.pi * (cosmo.luminosity_distance(self.s.CLUSTER_REDSHIFT))**2

        # NUV is 230 nm, according to Kennicutt & Evans
        wavelength_NUV = 230.e-9*u.m
        freq_NUV = c.c/wavelength_NUV
        
        # convert NSA NUV abs mag to nuLnu_NUV
        #flux_10pc = 10.**((22.5-self.s.ABSMAG[:,1])/2.5)
        # assume ABSMAG is in AB mag, with ZP = 3631 Jy
        # ******* NSA ABSMAG IS FOR H0=100 *********
        #flux_10pc = 3631.*10**(-1.*self.s.ABSMAG[:,1]/2.5)*u.Jy
        #dist = 10.*u.pc

        ## CALCULATING AGAIN USING THE CORRECT DISTANCE
        nuv_mag = 22.5 - 2.5*np.log10(self.s['NMGY'][:,1])
        fnu_nuv = 3631*10**(-1*nuv_mag/2.5)*u.Jy

        nuFnu_NUV = fnu_nuv*freq_NUV
        
        #self.nuLnu_NUV = flux_10pc*4*np.pi*dist**2*freq_NUV
        self.nuLnu_NUV_ZDIST = nuFnu_NUV * 4 * np.pi * (cosmo.luminosity_distance(self.s.ZDIST))**2
        self.nuLnu_NUV_ZCLUST = nuFnu_NUV * 4 * np.pi * (cosmo.luminosity_distance(self.s.CLUSTER_REDSHIFT))**2        
        
        self.nuLnu_NUV_cor_ZDIST = self.nuLnu_NUV_ZDIST.cgs + 2.26*self.nuLnu24_ZDIST.cgs
        self.nuLnu_NUV_cor_ZCLUST = self.nuLnu_NUV_ZCLUST.cgs + 2.26*self.nuLnu24_ZCLUST.cgs        

        
        #self.nuLnu_NUV = fnu_nuv*4*np.pi*(cosmo.luminosity_distance(self.s.ZDIST))**2*freq_NUV

        #self.nuLnu_NUV_cor = self.nuLnu_NUV.cgs + 2.26*self.nuLnu24_ZDIST.cgs
        
        self.logSFR_NUV = np.log10(self.nuLnu_NUV_cor_ZDIST.cgs.value) - 43.17
        # need relation for calculating SFR from UV only
        #
        # eqn 12
        # log SFR(Msun/yr) = log Lx - log Cx
        # NUV - log Cx = 43.17
        # 24um - logCx = 42.69
        # Halpha - log Cx = 41.27
        
        self.logSFR_NUV_KE = np.log10(self.nuLnu_NUV_ZDIST.cgs.value) - 43.17
        self.logSFR_IR_KE = np.log10(self.nuLnu24_ZDIST.cgs.value)-42.69
        self.logSFR_NUVIR_KE = np.log10(self.nuLnu_NUV_cor_ZDIST.cgs.value) - 43.17
        self.logSFR_NUVIR_KE_ZCLUST = np.log10(self.nuLnu_NUV_cor_ZCLUST.cgs.value) - 43.17        
        # repeating calculation using ZCLUSTER


        self.logSFR_NUV_ZCLUST = np.log10(self.nuLnu_NUV_cor_ZCLUST.cgs.value) - 43.17

        self.logSFR_NUV_BEST = self.logSFR_NUV_ZCLUST * np.array(self.membflag,'i') + np.array(~self.membflag,'i')*(self.logSFR_NUV_KE)
        self.logSFR_NUVIR_KE_BEST = self.logSFR_NUVIR_KE_ZCLUST * np.array(self.membflag,'i') + np.array(~self.membflag,'i')*(self.logSFR_NUVIR_KE)        
        self.SFR_NUV_BEST = 10**self.logSFR_NUV_BEST

    def update_table(self):
        
        # append Kennicutt & Evans SFRs at end of table
        t = Table(self.s)
        newcolumns = [self.membflag, self.lirflag, self.sampleflag,self.gim2dflag,self.sizeflag,self.sbflag,self.galfitflag,self.logSFR_NUV_KE, self.logSFR_IR_KE, self.logSFR_NUVIR_KE,self.logSFR_NUVIR_KE_BEST,self.sizeratio,self.sizeratioERR]
        newnames = ['membflag','lirflag','sampleflag','gim2dflag','sizeflag','sbflag','galfitflag2','logSFR_NUV_KE','logSFR_IR_KE','logSFR_NUVIR_KE','logSFR_NUVIR_KE_ZBEST','sizeratio','sizeratio_err']
        for i in range(len(newcolumns)):
            col = Column(newcolumns[i],name=newnames[i])
            t.add_column(col)
        print('updating table')
        t.write('/home/rfinn/research/LCS/tables/LCS_all_size_KE_SFR.fits',format='fits',overwrite=True)

    def fitms(self):
        flag = self.sampleflag & ~self.membflag & (self.logstellarmass > 9.5) & (self.logstellarmass < 10.5)
        c = np.polyfit(self.logstellarmass[flag],self.logSFR_NUV_BEST[flag],1)
        self.msline = c
        self.msx = np.linspace(8.75,11.25,10)
        self.msy = np.polyval(c,self.msx)

        # calculate distance of points from MS line

        # slope of line connecting point to MS is -1*1/slope of MS
        # so we can get an equation of the line connecting a point to MS using point-slope formula
        # then we need to know where they intersect
        # then calculate the distance b/w point and the point of intersection

        self.msperpdist = 1./np.sqrt(1+self.msline[0]**2) * (self.logSFR_NUV_BEST - np.polyval(self.msline,self.logstellarmass))
        self.msdist = self.logSFR_NUV_BEST - np.polyval(self.msline,self.logstellarmass)

    def calcssfr(self):
        self.logsSFR = self.logSFR_NUV_BEST - self.logstellarmass
        flag = self.sampleflag & ~self.membflag & (self.logstellarmass > 9.5) & (self.logstellarmass < 10.5)
        c = np.polyfit(self.logstellarmass[flag],self.logsSFR[flag],1)
        self.sSFRline = c
        self.sSFRx = np.linspace(8.75,11.25,10)
        self.sSFRy = np.polyval(c,self.sSFRx)
        self.sSFRdist = self.logsSFR - np.polyval(self.sSFRline,self.logstellarmass)
    def write_sizes_for_sim(self):
        # write out sizes.txt
        # size_ratio, size_err, core_fla
        t = Table([self.sizeratio,self.sizeratioERR, np.array(self.membflag,'i')])
        t.write('/home/rfinn/research/LCS/tables/sizes.txt',format='ascii',overwrite=True)
    def setup(self):
        self.get_agn()
        self.get_gim2d_flag()
        self.get_galfit_flag()
        self.get_memb()
        self.make_SFR()
        self.get_UVIR_SFR()

        self.get_size_flag()
        self.calculate_sizeratio()
        self.select_sample()
        self.fitms()
        self.calcssfr()
        self.update_table()        
        #self.write_sizes_for_sim()
def write_out_sizes():
    outfile = open(homedir+'/research/LCS/tables/sizes.txt','w')
    size = g.sizeratio[g.sampleflag]
    sizerr = g.sizeratioERR[g.sampleflag]
    myflag = g.membflag[g.sampleflag]
    bt = g.gim2d.B_T_r[g.sampleflag]
    ra = g.s.RA[g.sampleflag]
    dec = g.s.DEC[g.sampleflag]
    outfile.write('#R24/Rd size_err  core_flag   B/T RA DEC \n')
    for i in range(len(size)):
        outfile.write('%6.2f  %6.2f  %i   %.2f %10.9e %10.9e \n'%(size[i],sizerr[i],myflag[i],bt[i],ra[i],dec[i]))
    outfile.close()        

if __name__ == '__main__':
    g = galaxies(lcspath)
    g.get_agn()
    g.get_gim2d_flag()
    g.get_galfit_flag()
    g.get_memb()
    g.make_SFR()
    g.get_UVIR_SFR()
    #g.calcssfr()
    g.get_size_flag()
    g.calculate_sizeratio()
    g.select_sample()
    g.update_table()
