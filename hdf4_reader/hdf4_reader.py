import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import warnings
import pyhdf.SD as SD
from pyhdf.HDF import*
from pyhdf.VS import*
import h5py
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")        

molecule = ['Elect', 'H', 'H2', 'H+', 'H_2+', 'H_3+', 'He+', \
            'He', 'O+', 'O', 'C+', 'C', '13C+', '13C', \
            'OH+', 'OH', 'O2', 'CO+', 'CO', 'CH+', 'CH', \
            '13CO+', '13CO', '13CH+', '13CH', \
            'HCO+', 'H2O+', 'H2O', 'H13CO+', 'CH2+', \
            '13CH2+', 'H3O+', 'CH2', 'CH2+', \
            'SO2', 'SO+', 'SO', 'S+', 'S', 'OCS+', 'OCS', \
            'HS+', 'HS', 'HCS+', 'H2S+', 'CS+', 'CS', \
            '18O+', '18O', 'O18O', 'H318O+', \
            'H218O+', 'H218O', '18OH+', \
            '18OH', 'HC18O+', 'H13C18O+', \
            'C18O+', 'C18O', '13C18O+', '13C18O']


color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', 
                   'c', 'b', 'g', '#800080', '#FF0000', '#FA8072',
                   '#A0522D', '#COCOCO', '#D2B48C', '#008080', '#FF6347', 
                   '#40EOD0', '#EE82EE', '#F5DEB3', '#FFFFFF', '#FFFFOO',
                   '#9ACD32', '#580F41', '#7E1E9C', '#E50000', '#FF796C', 
                   '#A9561E', '#C5C9C7', '#D1B26F', '#029386', '#EF4026',
                   '#06C2AC', '#9AOEEA', '#FBDD7E',  '#FFFF14', '#BBF90F',
                   '#D2691E']

#heating and cooling rate reactions
hcreat = ['$H_2$ deexcitation', '$H_2$ photo-diss heating', '$H_2$ formation', \
          'OI(63 $\mu $m)', 'OI(44 $\mu $m)', 'OI(146 $\mu$ m)', 'cosmic ray', \
          'PE', '$^{12}CO$', '[CII](158 $\mu$ m)', '[CI](610 $\mu$ m)', \
          '[CI](230 $\mu$ m)', '[CI](370 $\mu$ m)', '[SiII](35 $\mu$ m)', \
          '$^{13}CO$ cooling', \
          'Lyman- alpha', '$H_2O$', 'gas-grain', 'OH', 'OI 6300', \
          '$H_2$ photo-diss cooling', 'C-ioniation heating', \
          'chemical heating']

debug = False
#=============================Functions================================#
def smooth(y, box_pts): # creating a smooth profile from diffusion rates
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def arrays(f):
 #   df = pd.DataFrame('depth', 'depth', 'abundace', 'column_density', 'diffusion_rates', 'gas_T', 'dust_T') 
    if not h5py.is_hdf5(f): #to check whether hdf5 or hdf4 format
        print ('reading hdf4 file', f)
        hdf = SD.SD(f)
        dataset = hdf.datasets()
        #print (dataset)
        sds = hdf.select('Depth::Temp::Abundances').get()
        #Av is 0 and pc is 1
        depths = sds[1, :]
        av = sds[0, :]
        tg = sds[2, :]
        td = sds[3, :]
        print ("maximum temperature = ", max(tg))
        abundances = sds[804:866, :]
        sds = hdf.select('Column-densities').get()
        CDD = sds[800:860, :] 
        HC_rates = hdf.select('Heating-Cooling-Rates').get()
        optical_depth = hdf.select('Optical-depths').get()
        rad_field = hdf.select('Rad-field::photorates').get()
        d = 'diffusion rates'
        if d in dataset:
            diff_rates = hdf.select('diffusion rates').get()
        else:
            diff_rates = np.zeros(shape = (400, len(depths)))
        rows = []
        for molecule_name, abund, depth in zip(molecule, abundances, depths):
            for it, ab in enumerate(abund):
                mol = molecule.index(molecule_name)
                rows.append({"it":it, "gas_temp": tg[it],  "dust_temp": td[it], "molecule":molecule_name, "depth":depths[it], 'Av':av[it], "abund":ab, \
                    "cdd":CDD[mol, it], "total_diff_rates": diff_rates[mol, it], "mol_diff_rates": diff_rates[mol+65, it], \
                            "therm_diff_rates": diff_rates[mol+130, it], "turb_diff_rates": diff_rates[mol+195, it], "V_turb": diff_rates[259, it], \
                            "V_ions": diff_rates[260, it],"V_neutral": diff_rates[261, it]})
                
        df = pd.DataFrame(rows)        
        row_rad_field = []
        for k in range(len(depths)):
            row_rad_field.append({'depth':depths[k],'H2dissrate':rad_field[1,k],'Cionisation':rad_field[2,k],'COdissrate':rad_field[3,k],\
                                  'CO12selfshield':rad_field[4,k],'CO13selfshield':rad_field[5,k],'H2formrate':rad_field[8,k],\
                                    'H2rovibrate':rad_field[16,k]})
        df_photo_rates = pd.DataFrame(row_rad_field) 
        row_hc_rates = []
        for k in range(len(depths)):
            row_hc_rates.append({'depth':depths[k],hcreat[0]:HC_rates[0,k],hcreat[1]:HC_rates[1,k],hcreat[2]:HC_rates[2,k] \
                                ,hcreat[3]:HC_rates[3,k],hcreat[4]:HC_rates[4,k],hcreat[5]:HC_rates[5,k],hcreat[6]:HC_rates[6,k] \
                                ,hcreat[7]:HC_rates[7,k],hcreat[8]:HC_rates[8,k],hcreat[9]:HC_rates[9,k],hcreat[10]:HC_rates[10,k], \
                                 hcreat[11]:HC_rates[11,k],hcreat[12]:HC_rates[12,k],hcreat[13]:HC_rates[13,k],hcreat[14]:HC_rates[14,k],\
                                 hcreat[15]:HC_rates[15,k],hcreat[16]:HC_rates[16,k] \
                                ,hcreat[17]:HC_rates[17,k],hcreat[18]:HC_rates[18,k],hcreat[19]:HC_rates[19,k],hcreat[20]:HC_rates[20,k],hcreat[21]:HC_rates[21,k] \
                                ,hcreat[22]:HC_rates[22,k] })
        df_hc_rates = pd.DataFrame(row_hc_rates) 
    else:
        print ('It is a hdf5 file. Please use arrays_h5 function to read') 
    return (df, df_hc_rates, df_photo_rates)

#if the results need some soomthening    
def check_diffusion_rates(dif_form_rates, dif_dest_rates):
    for mol in range(len(dif_dest_rates)):
        a = pd.Series(dif_form_rates[mol])
        a = a.interpolate()
        dif_form_rates[mol] = smooth(a,2)
    for mol in range(len(dif_dest_rates)):
        a = pd.Series(dif_dest_rates[mol])
        a = a.interpolate()
        dif_dest_rates[mol] = smooth(a,2)
    return (dif_form_rates, dif_dest_rates)

#to plot the abundace of one species
def plot_abundance_one(hdf, sp):
    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax. label_outer()
    ax. set_xlabel ('distance from the cloud surface(pc)')
    ax.set_ylabel( '$n_i$')
    ax.set_xlim(10**-5, 2.2)
    ax.set_title("$%s$"%(molecule[i]))
    ax.loglog(hdf[hdf.molecule == sp].depth,hdf[hdf.molecule == sp].abund linewidth = 2, \
            label = '$%s$'%(sp))
    ax.legend(loc = 'best')
    fig.tight_layout()
    return (fig)

#to plot the abundace of all species
def plot_abundance_one(hdf, sp):
    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax. label_outer()
    ax. set_xlabel ('distance from the cloud surface(pc)')
    ax.set_ylabel( '$n_i$')
    ax.set_xlim(10**-5, 2.2)
    ax.set_title("$%s$"%sp)
    ax.loglog(hdf[hdf.molecule == sp].depth,hdf[hdf.molecule == sp].abund, linewidth = 2, \
            label = '$%s$'%(sp))
    ax.legend(loc = 'best')
    fig.tight_layout()
    return (fig)

def plot_abundance_of_all(hdf):
    plt.rcParams['text.usetex'] = True
    for sp in hdf.molecule:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax. label_outer()
        ax. set_xlabel ('distance from the cloud surface(pc)')
        ax.set_ylabel( '$n_i$')
        ax.set_xlim(10**-5, 2.2)
        ax.set_title("$%s$"%sp)
        ax.loglog(hdf[hdf.molecule == sp].depth,hdf[hdf.molecule == sp].abund, linewidth = 2, \
                label = '$%s$'%(sp))
        ax.legend(loc = 'best')
        fig.tight_layout()
    return (fig)



