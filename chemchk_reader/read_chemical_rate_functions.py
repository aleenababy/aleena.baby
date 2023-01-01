
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib import rc
rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
from matplotlib.backends.backend_pdf import PdfPages
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex = True)
import pyhdf.SD as SD
from pyhdf.HDF import*
from pyhdf.VS import*
import seaborn as sns
sns.set_theme("paper")
sns.set_style("ticks")
sns.set_context("poster")

molecule = ['ELECTR', 'H', 'H2', 'H+', 'H2+', 'H3+', 'HE+', \
            'HE', 'O+', 'O', 'C+', 'C', '13C+', '13C', \
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
                   '#A0522D', '#D2B48C', '#008080', '#FF6347', 
                    '#EE82EE', '#F5DEB3', '#FFFFFF', 
                   '#9ACD32', '#580F41', '#7E1E9C', '#E50000', '#FF796C', 
                   '#A9561E', '#C5C9C7', '#D1B26F', '#029386', '#EF4026', 
                   '#06C2AC', '#FBDD7E',  '#FFFF14', '#BBF90F', 
                   '#D2691E']


sp_name = { 'ELECTR' :'e-', 'EL' :'e-', 'H':'H', 'H2': 'H$_2$', 'H+': 'H$^+$', 'H2+': 'H$_2^+$' , \
           'H2*': 'H$_2*$', 'H3+': 'H$_3^+$', 'HE+': 'He$^+$', 'HE': 'He', \
           'O+': 'O$^+$', 'O':'O', 'C+':'C$^+$', '13C+': '$^{13}$C$^+$' , 'OH+': 'OH$^+$'\
           , 'O2': 'O$_2$' , 'SO2': 'SO$_2$', 'CO+': 'CO$^+$', 'SO+': 'SO$^+$', 'CH+': 'CH$^+$', \
           'CH2': 'CH$_2$', 'HS+': 'HS$^+$', '13C': '$^{13}$C', '13CO': '$^{13}$CO', \
           '13CO+': '$^{13}$CO$^+$', '13C18O': '$^{13}$C$^{18}$O', '13C18O+': '$^{13}$C$^{18}$O$^+$', \
           'C18O': 'C$^{18}$O', '13CH': '$^{13}$CH' , '13CH+': '$^{13}$CH$^+$', 'H2O': 'H$_2$O', \
           'H2O+': 'H$_2$O$^+$', 'H13CO+': 'H$^{13}$CO$^+$', 'HCS+': 'HCS$^+$', 'CS+': '$CS^+$', \
           'CH2+': 'CH$_2^+$', '13CH2+': '$^{13}$CH$_2^+$', '13CH2': '$^{13}$CH$_2$', 'H3O+': 'H$_3$O$^+$', 'H3': '$H_3$', \
           'HCO+': 'HCO$^+$', 'S+': 'S$^+$' , 'CH3+': 'CH$_3^+$', 'O18O': 'O$^{18}$O', \
           '18O': '$^{18}O$' , '18OH': '$^{18}$OH', '18O+': '$^{18}$O$^+$', '18OH+': '$^{18}$OH$^+$', \
           'H13C18O+':'H$^{13}$C$^{18}$O$^+$', 'HC18O+':'HC$^{18}$O$^+$', 'H218O+':'H$_2^{18}$O$^+$', \
           'C18O+': 'C$^{18}$O$^+$', 'C':'C', 'OH':'OH', 'CO':'CO', 'CH':'CH', 'SO':'SO' \
           , 'S':'S', 'OCS':'OCS', 'HS':'HS', 'H2S+':'H$_2$S$^+$', 'CS':'CS', 'OCS+':'OCS$^+$' \
           , 'H318O+':'H$_3^{18}$O$^+$', 'H218O':'H$_2^{18}$O', 'PHOTON': '$\gamma_{FUV}$'  \
           , 'CRPHOT': 'CR$_{Phot}$' , '==> ': 'diffusion$', '--> ': '--> '}
 

debug = False
# =============================Functions================================#
def smooth(y, box_pts): # creating a smooth profile from diffusion rates
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode = 'same')
    return y_smooth
    
def check_diffusion_rates(dif_form_rates, dif_dest_rates):
    for mol in range(len(dif_dest_rates)):
        a = pd.Series(dif_form_rates[mol])
        a = a.interpolate()
        dif_form_rates[mol] = smooth(a, 2)
    for mol in range(len(dif_dest_rates)):
        a = pd.Series(dif_dest_rates[mol])
        a = a.interpolate()
        dif_dest_rates[mol] = smooth(a, 2)
    return (dif_form_rates, dif_dest_rates)


def chemical_rates(df, debug):
    delimiter = df[0][0]
    split = df[df[0] == delimiter].index.values
    if debug == True:
        print ('spliting into different positions')
    #created an array of position of different spatial grids
    formation = []
    destruction = []
    for i in range(len(split)-1): # loop over different spatial positions
        #'reactions':df[0][split[i]+1:split[i+1]]    'rates':df[1][split[i]+1:split[i+1]]
        species = []
        reactions = []
        rates = []
        pq = df[split[i]+1:split[i+1]]
        species_split = pq[pq[0].str.contains('REL. TO XNTOT :')].index.values
        
        for k in range(len(species_split)-1):
            line = df[0][species_split[k]]
            line = line.split()
            r = str(line[0])
            if r == 'H2':
                r = 'H_2'
            sp = df[species_split[k]:species_split[k+1]]
            form_split = sp[sp[0].str.contains('FORMATION REACTIONS')].index.values
            tot_f_split = sp[sp[0].str.contains('TOTAL FORMATION RATE')].index.values
            dest_split = sp[sp[0].str.contains('DESTRUCTION REACTIONS')].index.values
            tot_d_split = sp[sp[0].str.contains('TOTAL DESTRUCTION RATE')].index.values
            for p in form_split:
                start = p
                line = df[0][p]; line = line.split()
                end = start + int(line[0])+1
                tot = df[0][tot_f_split[0]]; tot = tot.split()
                to = tot[-1]
                for ik in range(start+ 1, end): #form_split[0]+1, form_split[0]+1+end
                    formation.append({'molecule':r, 'pos':i, 'i':ik, 'reac':df[0][ik], \
                                        'form_rate':df[1][ik], 'total_form_rate':to, \
                                        'per_form_rate':float(df[1][ik])/float(to)})
            for p in dest_split:
                start = p
                line = df[0][p]; line = line.split()
                end = start + int(line[0])+1
                tot = df[0][tot_d_split[0]]; tot = tot.split()
                to = float(tot[-1])
                for ik in range(start+ 1, end): #form_split[0]+1, form_split[0]+1+end
                    destruction.append({'molecule':r, 'pos':i, 'i':ik, 'reac':str(df[0][ik]), \
                                        'dest_rate':float(df[1][ik]), 'total_dest_rate':to, \
                                        'per_dest_rate':float(df[1][ik])/float(to)})
        
        chem_form = pd.DataFrame(formation)
        chem_dest = pd.DataFrame(destruction)
    if debug == True:
        print ('finished spliting into different species and reactions')    
    return (chem_form, chem_dest)

def check_chemchk_reactions(array):
    list_ = []
    for sp in array.molecule.unique():
        a = array[array.molecule == sp]
        for k in a.pos.unique():
            b = a[a.pos == k]
            count = b.reac.value_counts()
            if len(count[count > 1]) > 0:
                r = count[count > 1].index[0]
                if r in list_:
                    pass
                else:
                    list_.append(r)
    return (list_)
    
    
def chemchk_reader(f):
    tstart = datetime.now()
    check = os.path.exists(f)
    if os.path.exists(f):
        print ('reading the file', f)
        df = pd.read_fwf(f, header = None)
        debug = 0
        chem_form, chem_dest = chemical_rates(df, debug)
        
        list1 = check_chemchk_reactions(chem_form)
        list2 = check_chemchk_reactions(chem_dest)
        if len(list1) > 0:
            print ("repeated entires in the chemchk file formation")
            print ("Please check the file /rundir/pdrinpdata/chem_rates.dat")
            print ("The following reactions are repeated")
            print (list1)
        if len(list2) > 0:
            print ("repeated entires in the chemchk file destruction")
            print ("Please check the file /rundir/pdrinpdata/chem_rates.dat")
            print ("The following reactions are repeated")
            print (list2)                                
 
    else:
        print ('Unfortunately, I could not find this file at this location', f)

    tend = datetime.now()
    if debug == True: print('dataframe loaded in ', tend-tstart, 'seconds')
    return (chem_form, chem_dest)

def check_species(array, sp):
    if sp in array.molecule.unique():
        pass
    else:
        print ('formation/destruction of this species does not exist')
        print ('choose any of the following species(case sensitive):')
        print (array.molecule.unique())
        sp = input('which species you want to proceed with?')
    return (sp)


def check_reac(name):
    pq = name.split()
    for p in pq:
        pq[pq.index(p)] = sp_name[p]
    if len(pq) == 2:
        label_ = 'diffusion of %s'%(pq[1])
    else:
        if len(pq) == 3:
            label_ = r'%s + %s $\rightarrow$ %s'%(pq[0], pq[1], pq[3])
        if len(pq) == 4:
            label_ = r'%s + %s $\rightarrow$ %s '%(pq[0], pq[1], pq[3])
        if len(pq) == 5:
            label_ = r'%s + %s $\rightarrow$ %s + %s '%(pq[0], pq[1], pq[3], pq[4])
        if len(pq) == 6:
            label_ = r'%s + %s $\rightarrow$ %s + %s+ %s'%(pq[0], pq[1], pq[3], pq[4], pq[5])
    return(label_)
def check_for_av(hdf):
        
    if os.path.exists(hdf):
        print ('reading the file', hdf)
    else:
        print ('Not again! give me a proper file name')
        print ('I want to do some work')
        print ('Plotting with position index')
        
    return ()
# to plot the destruction reaction rates based on the position index.
# If combined with hdf file it will plot the 
# AV or pc on the x axis



def destruction_species(spe_name, chem_dest):
    fig, ax = plt.subplots(1, 1)
    for reac, table_reac in  chem_dest[chem_dest.molecule == "%s"%(spe_name)].groupby(["reac"]):
        ax.set_title('%s'%(sp_name[spe_name]), fontsize = 10)
        ax.set_ylabel('destruction rates ($cm^{-3}s{-1}$)')
        ax.set_xlabel('position index')
        if '==>' in reac or 'diffusion' in reac:
            ax.semilogy(table_reac.pos, table_reac.dest_rate, linewidth = 1.2, label = 'diffusion')
        else:
            ax.semilogy(table_reac.pos, table_reac.dest_rate, linewidth = 1.2, label = check_reac(reac))
    ax.legend( bbox_to_anchor = (1.15, 1), loc = 'upper left', fontsize = 6, borderaxespad = 0.)
    ax.label_outer()
    fig.tight_layout()
    return(fig)

def destruction_species_withdhf(spe_name, chem_dest, hdf):
    fig, ax = plt.subplots(1, 1)
    for reac, table_reac in  chem_dest[chem_dest.molecule == "%s"%(spe_name)].groupby(["reac"]):
        ax.set_title('%s'%(sp_name[spe_name]), fontsize = 10)
        ax.set_ylabel('destruction rates ($cm^{-3}s{-1}$)')
        ax.set_xlabel('distance from the cloud surface (pc)')
        p = hdf[hdf.molecule == spe_name].depth.reset_index()
        if '==>' in reac or 'diffusion' in reac:
            ax.semilogy(p['depth'][table_reac.pos], table_reac.dest_rate, linewidth = 1.2, label = 'diffusion')
        else:
            ax.semilogy(p['depth'][table_reac.pos], table_reac.dest_rate, linewidth = 1.2, label = check_reac(reac))
    ax.legend( bbox_to_anchor = (1.15, 1), loc = 'upper left', fontsize = 6, borderaxespad = 0.)
    ax.label_outer()
    fig.tight_layout()
    return(fig)

def formation_species(spe_name, chem_form):
    fig, ax = plt.subplots(1, 1)
    for reac, table_reac in  chem_form[chem_form.molecule == "%s"%(spe_name)].groupby(["reac"]):
        ax.set_title('%s'%(sp_name[spe_name]))
        ax.set_ylabel('formation rates ($cm^{-3}s{-1}$)')
        ax.set_xlabel('position index')
        if '==>' in reac or 'diffusion' in reac:
            ax.semilogy(table_reac.pos, table_reac.form_rate, linewidth = 1.2, label = 'diffusion')
        else:
            ax.semilogy(table_reac.pos, table_reac.form_rate, linewidth = 1.2, label = check_reac(reac))
    ax.legend( bbox_to_anchor = (1.15, 1), loc = 'upper left', fontsize = 6, borderaxespad = 0.)
    ax.label_outer()
    fig.tight_layout()
    return(fig)

def formation_species_withdhf(spe_name, chem_form, hdf):
    fig, ax = plt.subplots(1, 1)
    for reac, table_reac in  chem_form[chem_form.molecule == "%s"%(spe_name)].groupby(["reac"]):
        ax.set_title('%s'%(sp_name[spe_name]))
        ax.set_ylabel('formation rates ($cm^{-3}s{-1}$)')
        ax.set_xlabel('distance from the cloud surface (pc)')
        p = hdf[hdf.molecule == spe_name].depth.reset_index()
        if '==>' in reac or 'diffusion' in reac:
            ax.semilogy(p['depth'][table_reac.pos], table_reac.form_rate, linewidth = 1.2, label = 'diffusion')
        else:
            ax.semilogy(p['depth'][table_reac.pos], table_reac.form_rate, linewidth = 1.2, label = check_reac(reac))
    ax.legend( bbox_to_anchor = (1.15, 1), loc = 'upper left', fontsize = 6, borderaxespad = 0.)
    ax.label_outer()
    fig.tight_layout()
    return(fig)