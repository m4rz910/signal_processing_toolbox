from sstoolbox import SignalTB
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.titleweight'] = 'bold'

m_dir = r"C:\Users\tzave\OneDrive - The Cooper Union for the Advancement of Science and Art\102-cooper\150-masters\soundandspace\ss_final_project\measurements"

plt.cla()

#Read file
data, fs = sf.read(os.path.join(m_dir,'R1.wav'));
N = len(data); print(N)
L = N/fs; print(L)
s = pd.Series(data=data,
              index=np.linspace(start=0,stop=L,num=N,endpoint=True))
s = SignalTB(x=s, fs=fs)
plt.plot(s.x); plt.grid()

#Linear Averaging
gxx_a, X_avg = s.linear_a(n_intervals = 25)
#gxx_a.plot()
x_avg = SignalTB.my_ifft(X=X_avg, N=int(N/25), df=s.df, dt=s.dt)
plt.plot(x_avg)

fig = plt.figure(figsize=(10,5))
plt.plot(x_avg);plt.grid()


# Octive Bands
lowcut  = 125
highcut = 4000
cuts =  [lowcut*2**i for i in range(0,6)]
print(cuts)

fig = plt.figure(figsize=(10,5))
octives = []
saved = []
for i,dummy in enumerate(cuts):
    x = SignalTB.butter_bandpass_filter(data=x_avg.values,
                                        lowcut=cuts[i]/np.sqrt(2), highcut=cuts[i]*np.sqrt(2),
                                        fs=fs, order=3)
    x_avg_filt = pd.Series(data=x, index=x_avg.index)
    s = SignalTB(x=x_avg_filt, fs=fs); saved.append(s)
    X = s.my_fft()
    octives.append((i, SignalTB.spl(np.abs(SignalTB.my_ifft(X=X, N=int(N/25), df=s.df, dt=s.dt)))))
plt.scatter(*zip(*octives));
plt.grid(); plt.xlabel('Octive Number'); plt.ylabel('SPL');
fig.savefig('./plots/hw6_octives.png', dpi=300, bbox_inches='tight')