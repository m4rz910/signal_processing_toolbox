from sstoolbox import SignalTB
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.io as sio
from scipy.signal import find_peaks

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.titleweight'] = 'bold'

m_dir = r"C:\Users\tzave\OneDrive - The Cooper Union for the Advancement of Science and Art\102-cooper\150-masters\soundandspace\ss_final_project\measurements"

plt.cla()


#SOURCE
source = sio.loadmat('./chirp.mat')
x_source = source['x'][:,0]
s_source = pd.Series(data=x_source,
                     index=np.linspace(start=0, stop=4,
                                       num=len(x_source),endpoint=True))
source = SignalTB(x=s_source,
                  fs=len(x_source)/4)


fig = plt.figure(figsize=(10,5))
for r_lab, color, line in zip(['R1','R2','R3','R4'],
                        ['r','b','g','y'],
                        ['-','--',':','-.']):
    #RECEIVER
    data, fs = sf.read(os.path.join(m_dir,'{}.wav'.format(r_lab)));
    N = len(data);
    L = N/fs;
    s = pd.Series(data=data,
                  index=np.linspace(start=0,stop=L,num=N,endpoint=True))
    receiver = SignalTB(x=s, fs=fs)
    #Linear average the receiver
    gxx_a, Y_avg = receiver.linear_a(n_intervals = 25)
    y_avg = SignalTB.my_ifft(X=Y_avg, N=int(N/25),
                             df=receiver.df, dt=receiver.dt)
    
    # #plot - Source and Receiver Time Series
    # plt.plot(source.x); plt.plot(y_avg); plt.grid()
    # plt.title('Padded Source and Linear Averaged Receiver')
    
    #Calculate H
    X_source = source.my_fft()
    #H = np.abs(Y_avg)/np.abs(X_source) #is transfer function complex or real? complex, right?
    H = Y_avg/X_source
    #plt.figure(); plt.plot(H); plt.title('H')
    
    #calculate h
    h = SignalTB.my_ifft(X=H, N=source.N, df=source.df, dt=source.dt) 
    h = pd.Series(data=np.abs(h),
                  index=h.index)
    #h.loc[0.5:] = 0.001  #setting everything after 0.5 seconds 'zero'
    
    # plt.figure();
    # plt.plot(20*np.log10(h/np.max(h)));
    # plt.title('h(t)_{}'.format(r_lab)); plt.ylabel('Amplitude (dB re: Max)')
    # #include overall EDC
    # edc = np.flip(np.cumsum(np.flip(h.values)**2 *receiver.dt)) # flip, cumulative sum and flip back
    # edc = pd.Series(data=edc, index=h.index) #schroeder
    # plt.plot(10*np.log10(edc/np.max(edc))) # Q3: is it right to use different ref here
    # #plt.xlim([0,1])
    

#     # ************ C80 **************
    #time window first
    peaks, _ = find_peaks(h,height=5)
    # fig = plt.figure(); plt.xlim([0,0.05]);
    # plt.plot(h); plt.plot(h.iloc[peaks], "x")
    h_windowed = h.copy()
    h_windowed = h_windowed.iloc[peaks[0]-50:] #50 samples back from first peak seems to work fine
    #plt.figure(); plt.plot(h_windowed); plt.xlim([0,0.5]);
    
    h_w_0_80 = h_windowed.copy(); h_w_80_inf = h_windowed.copy()
    h_w_0_80.iloc[int(80/(source.dt*1000)):] = 0 #make everything after 80 ms 'zero'
    h_w_80_inf.iloc[:int(80/(source.dt*1000))] =  0 #make everything before 80 ms 'zero'
    
    # Octive Band Filter
    cuts =  [125*2**i for i in range(0,6)]
    octives = []
    for i,dummy in enumerate(cuts):
        #h_w_0_80
        x = SignalTB.butter_bandpass_filter(data=h_w_0_80.values,
                                            lowcut=cuts[i]/np.sqrt(2), highcut=cuts[i]*np.sqrt(2),
                                            fs=fs, order=3)
        h_w_0_80_filt = pd.Series(data=x, index=h_w_0_80.index)
        
        #h_w_80_inf
        x = SignalTB.butter_bandpass_filter(data=h_w_80_inf.values,
                                            lowcut=cuts[i]/np.sqrt(2), highcut=cuts[i]*np.sqrt(2),
                                            fs=fs, order=3)
        h_w_80_inf_filt = pd.Series(data=x, index=h_w_80_inf.index)
        
        h_filt_80 = h_w_0_80_filt.apply(lambda x: x**2).sum()
        h_filt_inf = h_w_80_inf_filt.apply(lambda x: x**2).sum()
        
        octives.append((i, 10*np.log10(h_filt_80/h_filt_inf)))
    plt.plot(*zip(*octives),line, c=color,marker='o',markersize=8, label=r_lab);
plt.legend(); plt.grid(); plt.xlabel('Octive Band [Hz]'); plt.ylabel('C80');
plt.xticks(range(0,6),cuts)


#     # ************ EDT **************  
#     h.loc[0.5:] = 0.001  #setting everything after 0.5 seconds 'zero'
# # #    Octive Band Filter
#     cuts =  [125*2**i for i in range(0,6)]
#     octives = []
#     for i,dummy in enumerate(cuts):
#         x = SignalTB.butter_bandpass_filter(data=h.values,
#                                             lowcut=cuts[i]/np.sqrt(2), highcut=cuts[i]*np.sqrt(2),
#                                             fs=fs, order=3)        
#         h_filt = pd.Series(data=x, index=h.index)
        
#         edc_filt = np.flip(np.cumsum(np.flip(h_filt.values)**2 *receiver.dt)) # flip, cumulative sum and flip back
#         edc_filt = pd.Series(data=edc_filt, index=h_filt.index) #schroeder
        
#         # plt.figure()
#         # plt.plot(20*np.log10(h_filt/np.max(h_filt)));
#         # plt.plot(10*np.log10(edc_filt/np.max(edc_filt)));
        
#         edc_filt = 10*np.log10(edc_filt/np.max(edc_filt))
#         t_zero = edc_filt[round(edc_filt,1)==0].index[0]
#         t_30 = edc_filt[round(edc_filt,1)==-10].index[0]
#         octives.append((i, (t_30-t_zero)*6))
#     plt.plot(*zip(*octives),line, c=color,marker='o',markersize=8, label=r_lab);
# plt.legend(); plt.grid(); plt.xlabel('Octive Band [Hz]'); plt.ylabel('EDT');
# plt.xticks(range(0,6),cuts)

#     # ************ T30 ************** 
#     h.loc[0.5:] = 0.001  #setting everything after 0.5 seconds 'zero'
# # #    Octive Band Filter
#     cuts =  [125*2**i for i in range(0,6)]
#     octives = []
#     for i,dummy in enumerate(cuts):
#         x = SignalTB.butter_bandpass_filter(data=h.values,
#                                             lowcut=cuts[i]/np.sqrt(2), highcut=cuts[i]*np.sqrt(2),
#                                             fs=fs, order=3)        
#         h_filt = pd.Series(data=x, index=h.index)
        
#         edc_filt = np.flip(np.cumsum(np.flip(h_filt.values)**2 *receiver.dt)) # flip, cumulative sum and flip back
#         edc_filt = pd.Series(data=edc_filt, index=h_filt.index) #schroeder
        
#         # plt.figure()
#         # plt.plot(20*np.log10(h_filt/np.max(h_filt)));
#         # plt.plot(10*np.log10(edc_filt/np.max(edc_filt)));
        
#         edc_filt = 10*np.log10(edc_filt/np.max(edc_filt))
#         t_zero = edc_filt[round(edc_filt,1)==-5].index[0] #changed from edt
#         t_30 = edc_filt[round(edc_filt,1)==-35].index[0] #changed from edt
#         octives.append((i, (t_30-t_zero)*2)) #changed from edt
#     plt.plot(*zip(*octives),line, c=color,marker='o',markersize=8, label=r_lab);
# plt.legend(); plt.grid(); plt.xlabel('Octive Band [Hz]'); plt.ylabel('T30');
# plt.xticks(range(0,6),cuts)


