




m_dir = r"C:\Users\tzave\OneDrive - The Cooper Union for the Advancement of Science and Art\102-cooper\150-masters\soundandspace\ss_final_project\measurements"

data, fs = sf.read(os.path.join(m_dir,'R1.wav'));
N = len(data); print(N)
L = N/fs; print(L)
s = pd.Series(data=data,
              index=np.linspace(start=0,stop=L,num=N,endpoint=True))
s = SignalTB(x=s, fs=fs)
fig = plt.figure(figsize=(10,5))
plt.plot(s.x); plt.grid()

gxx_a = s.linear_a(n_intervals = 25)
gxx_a.plot()