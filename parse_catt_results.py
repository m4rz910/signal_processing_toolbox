# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 02:24:43 2020
@author: m4rz910
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = "black"

out_dir= os.path.join('./Output')

def generate_plot(file_name,dict_key):
    """
    E - energy-based 1/1-octave echograms E (red = energy = fuzzy and warm) 
    h -  B-format and Binaural Impulse Responses (IRs) (blue = pressure = clean and cold)
    """
    d = sio.loadmat(file_name)
    positionarray = np.arange(1,7)
    # octivebands = [125*2**i for i in range(0,8)]
    octivebands = [125,250,500,1000,2000,4000]
    fig, ax = plt.subplots(figsize=(10,5))
    red_series  = '{}_E'.format(dict_key)
    blue_series = '{}_h'.format(dict_key)
    for e, h, marker, receiver, color in zip(d[red_series].T, d[blue_series].T,
                            ['-','--',':','-.'], ['R1_','R2_','R3_','R4_'], ['r','b','g','y']): 
        # plt.plot(positionarray,e[:len(positionarray)],
        #          marker, label=receiver+red_series, c='tab:red')
        plt.plot(positionarray,h[:len(positionarray)],
                marker, label=receiver+blue_series,c = color,marker = "o",markersize = 8)
        plt.xticks(positionarray,octivebands)
        # ax.set_xticklabels(octivebands)
        
    plt.legend()
    if dict_key == "T30":
        ax.set_xlabel('Octave Bands [Hz]'); ax.set_ylabel('T-30 [s]')
        ax.set_ylim(0,1)
        print(dict_key,"axis for T30")
    elif dict_key == "EDT":
        ax.set_xlabel('Octave Bands [Hz]'); ax.set_ylabel('EDT [s]')
        ax.set_ylim(0,1.1)
        print(dict_key,"axis for EDT")      
    elif dict_key == "C80":
        ax.set_xlabel('Octave Bands [Hz]'); ax.set_ylabel('C-80 [ms]')
        print(dict_key,"axis for C80")
            
    # plt.xticks(ticks = [0,125,250,500,1000,2000,4000,8000,16000])
    fig.savefig(os.path.join(out_dir,'{}_diff2.png'.format(dict_key)), dpi=300, bbox_inches='tight')
    
if __name__ == '__main__':
    file_name = 'C:/Users/tzave/Downloads/RoseAuditorium_tzavelis_wei_diff2.MAT'
    for param in ['T30','C80','EDT']:
        generate_plot(file_name, param)
