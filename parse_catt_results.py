 
Table of Contents
1. Model Creation and Calibration	2
1.1. Computer-Aided Design (CAD) Process	2
1.2. CATT Acoustics Model	3
1.2.1. Room Geometry	3
1.2.2. Material Assignments	4
1.2.3. Source and Receiver Positions	5
1.2.4. Interactive RT Simulation and TUCT2 Settings	6
1.2.5. Simulation Results	6
2. Analysis of Acoustic Measurements	9
2.1. Analysis of C-80	9
2.2. Analysis of EDT and T-30	10
3. Model Calibration and Recommendations	12
Appendix I:	Provided Schematics	13
Appendix II:	CATT Model Schematics	14
Appendix III:	Data Analysis Script	15

 
	Model Creation and Calibration
Computer-Aided Design (CAD) Process 
2-D cross-sectional schematics (Appendix I: Provided Schematics) of the Cooper Union Rose Auditorium informed the creation of the 3D CAD model. There are three sections to the auditorium: stage, front seating area and rear seating area. Figure I 2 and Figure I 3 depict the 3D model created in OnShape. A separate CAD program—Rhino—was used to assign room surfaces to different ‘layers’ and create the final DXF file; in the future, it is recommended that a CAD program that has ‘layer’ functionality—like Rhino or AutoCAD—be used to create the model with layers from the outset. Building a model with layers saves a lot of time when assigning materials to surfaces in CATT Acoustics—discussed in the following section.
Figure 1 1: Rose Auditorium 3D CAD (Isometric View)
 

Figure 1 2: Rose Auditorium 3D CAD (Right View) (See Figure I 1)
 

 CATT Acoustics Model
 Room Geometry
CATT Acoustics has an import tool called ‘DXFTOGEO’ to generate the room geometry (GEO) file from a CAD model. With each plane already grouped by their material layer in the CAD program, adding absorption and scattering coefficients to each material was the only edit necessary in the GEO file. Figure 1 3 shows a color-coded visualization of the auditorium in CATT. Detailed schematics can be found in Appendix II:.
Figure 1 3: CATT Model Rear Perspective View 
 
  Material Assignments
Materials were assigned to each color-coded surface based on online images of the Rose Auditorium (Figure 1 4). 
Figure 1 4: Image of Rose Auditorium
 
The stage is primarily wood, while the stage sidewalls and top half of the front seating area walls were modeled with perforated gypsum boards. In the non-stage area, the floor area is entirely concrete. The ceiling is made with thin non-planar metal mesh— the irregular geometry made it difficult to find precise acoustic properties. Thus, ceiling modeling was simplified by using metal panels from the Vorlaender material database. Lastly, all chairs were modeled as a single entity, using the acoustic properties of concert chairs from the Vorlaender database. Table 1 1 summarizes the mapping of materials to room surfaces, while Table 1 2 below details what the acoustic properties of the materials.
Table 1 1: Room Surface Material Assignment
Material	Room Surface
Concrete	Floors (front and rear seating area)
Perforated Gypsum	Stage (walls and ceiling), 
Top half of the side walls (front seating area)
Medium Upholstered Concert Chairs, empty	Seating (front and rear seating area)
Wood, stage floor	Stage and Podium
Metal Panel Ceiling	Ceiling (front and rear seating area),
Bottom half of sidewalls (front seating area),
Sidewalls (back seating area)

Table 1 2: Room Material Absorption Coefficients¬
Material	125	250	500	1000	2000	4000
Concrete	1	3	5	2	2	2
Perforated Gypsum	45	55	60	90	86	75
Medium Upholstered Concert Chairs, empty	49	66	80	88	82	70
Wood, stage floor	10	7	6	6	6	6
Metal Panel Ceiling	59	80	82	65	27	23

 Source and Receiver Positions
A single source was placed in the middle of the stage to model a speaker. Three receivers (1, 4, 3) were placed in front of the source, while receiver 2 was placed close to the wall, parallel to receiver 1. Their arrangement is shown in Figure 1 5: Source and Receivers Positions. 
Figure 1 5: Source and Receivers Positions
 

 
 Interactive RT Simulation and TUCT2 Settings
The largest T-30 reverberation estimate was 0.406 seconds, informing the decision to use 2 seconds as impulse response length in the full simulation. The full simulation used 25,000 rays.
Table 1 3: Interactive RT Estimation [units: s]
Octave Band	125	250	500	1k	2k	4k	8k	16k
Eyring	0.304	0.203	0.178	0.175	0.285	0.330	0.338	0.246
Sabine	0.400	0.302	0.279	0.276	0.377	0.412	0.391	0.262
T-30   	0.383	0.256	0.240	0.242	0.365	0.406	0.386	0.261

 Simulation Results
T-30 is the time it takes for sound pressure to decrease by 30 dB. Figure 1 6 shows T-30 predicted by CATT for the octave bands been 125 – 4000 Hz. The reverberation time follows a trend consistent across all receivers: T-30 starts around 0.4-0.5s and drops to its lowest at 500 Hz, then rises at higher frequencies. This could indicate that frequencies around 500 Hz are being absorbed more by the room than other frequencies.
Figure 1 6 T30 of Impulse Response
 
 
C-80 measures how clear one would perceive sound emitted from a source. Figure 1 7: C80 of Impulse Response shows the C-80 index at each receiver. A general trend exists for all 4 receivers: the index rises at low frequencies, peaks at 500Hz, and drops at higher frequencies. It makes sense that frequencies with low reverberation (low T-30) would be perceived as clearer or crisper. 
Figure 1 7: C80 of Impulse Response
 
 
Early decay time (EDT) is yet another measure of reverberation time. EDT is defined as 6 times the duration needed for a sound to decay from 0dB to -10dB, therefore it includes information about the decay of early sound. Early sounds are the sound waves that arrive at the receiver first, and consist of direct sounds, some first and second-order reflections, and higher-order reflections with a short path length. Figure 1 8: EDT of Impulse Response shows that EDT at all receiver positions follows roughly an increasing trend. However, receiver 3 does not behave the same at high frequencies as the receivers 1, 2 and 4. 
Figure 1 8: EDT of Impulse Response
 
 
¬¬¬
	Analysis of Acoustic Measurements
Due to the COVID-19 pandemic, acoustic measurements collected from a previous year were analyzed. The purpose of analyzing real acoustic measurements is to use the results to tune the CATT model created in Part 1. The objective is to have our model’s T30, EDT, and C80 match the measurements as closely as possible.
To compute T30, EDT, and C80 for our measurements, the first step is to construct the impulse response using the source and receiver data. The source file had one chirp and but each recorder captured 25 of these chirps played in the auditorium. Each recorder was linearly averaged to obtain an average received chirp to compare to the source chirp. The transfer function was computed using Equation 2 1. Then, an IFFT of H_(R_i ) was computed to get h(t).
H_(R_i )=  Y_(avg_i)/X_source  (Equation 2 1)
 Analysis of C-80
Time windowing was performed before bandpass filtering to calculate C80 for each octave band. Equation 2 2 was used to calculate C80, and Figure 2 1 depicts results for each receiver, which were very different from those in our model. The clarity index is significantly lower in reality than our model predicts evidenced by the [1,9] range of Figure 2 1, while Figure 1 7 showed a range of [9-25]. Furthermore, the real results showed a convex trend compared to a concave one in the model. 
C_80=10*log_10⁡〖(∫_0^(80 ms)▒〖h^2 (t)dt〗)/(∫_80^∞▒〖h^2 (t)dt〗)〗  (Equation 2 2)
Figure 2 1: Measured C80 
Analysis of EDT and T-30
Due to an abundance of noise in the impulse response, EDT and T30 were calculated based on a modified impulse response, in which everything after t=0.5s was set to zero for h(t). This modification made the calculation of the early decay curve in the interval of interest more accurate, and therefore improved the analysis of EDT and T30. EDT was calculated as six times the duration it took for the impulse response to go from 0 dB to -10 dB, while T30 was two times the duration it took for it to go from -5 to -35 dB. EDT and T30 were calculated for each octave band and plotted in Figure 2 2 and Figure 2 3 respectively. The rough range of EDT in the model was [0.1-0.7], while in the measurements it was higher at [0.6-1.1]. Although both were concave, the measurements revealed that higher frequencies had lower EDT than lower frequencies- but the model predicted the opposite. The T30 measurements were also different in range and trend than the model. The model showed a relatively flat relationship for T30, while the measurements revealed a downwards slope for all the receivers.
Reverberation time is a measure of how fast sound wave energy decays in a room. The smaller the RT the faster the energy decays due to wall absorption and few reflections.  Since the measurements had a higher overall level of EDT and T30 then the model predicted, it could be that the model materials were too absorbent. Since there was more than one material used, a closer of EDT and T30 at each octave band is needed to diagnose which material to adjust. It is also possible that there are noise sources present in the Rose auditorium, such as HVAC equipment, that was not included in the model and could be contributing to the discrepancies.
Figure 2 2: Measured EDT 
Figure 2 3: Measured T30
 
 
	Model Calibration and Recommendations
This part is still in progress.
 
Provided Schematics
Figure I 1: Rose Auditorium Schematic Long-Side View
 

Figure I 2: Rose Auditorium Schematic Short-Side View
 

Figure I 3: Rose Auditorium Schematic Top View
 
CATT Model Schematics

Figure II 1: Model Schematic (Side View)
 
Figure II 2: Model Schematic (Top View)
 


Data Analysis Script

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
        ax.set_ylim(0,1)
        print(dict_key,"axis for EDT")      
    elif dict_key == "C80":
        ax.set_xlabel('Octave Bands [Hz]'); ax.set_ylabel('C-80 [ms]')
        print(dict_key,"axis for C80")
            
    # plt.xticks(ticks = [0,125,250,500,1000,2000,4000,8000,16000])
    fig.savefig(os.path.join(out_dir,'python_plots/{}.png'.format(dict_key)), dpi=300, bbox_inches='tight')
    
if __name__ == '__main__':
    file_name = os.path.join(out_dir,'Rose_project_test_4.mat')
    for param in ['T30','C80','EDT']:
        generate_plot(file_name, param)
 
Acoustic Toolbox

The Acoustic Toolbox developed throughout the class is available publically on Github: https://github.com/m4rz910/signal_processing_toolbox

The file most relevant to the final project is ‘final_part2.py’, along with the supporting ‘sstoolbox.py’ library. 
