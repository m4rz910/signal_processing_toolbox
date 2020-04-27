clear all; clc; close all;

N = 4096;
T = 10;
x = randn(N,1);
t = linspace(0,T,N); %3

fs = 4096; %2
dt = 1/fs; 
df = fs/N;

figure(1)
plot(t,x) %5
title('time series')
xlabel('time')
ylabel('x')
%soundsc(x,fs)

figure(2)
f = (0:N-1)*df;
X = fft(x)*dt;
plot(f,abs(X))
title('magnitude')
xlabel('frequency')
ylabel('magnitude')

figure(3)
plot(f,angle(X)*360/(2*pi))
title('phase plot')
xlabel('frequency')
ylabel('phase [degrees]')

%excercise 3
X_m = ones(N,1);
x = ifft(X_m)/dt;
figure(5)
plot(t,x) %5
title('time series')
xlabel('time')
ylabel('x')
%soundsc(x,fs)


