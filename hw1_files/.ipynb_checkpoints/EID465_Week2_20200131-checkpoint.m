%% 0.4 Basics of DSP

clc;
clear all;
close all;

N = 10;
fs = 100;

dt = 1/fs;
t = (0:N-1)*dt;

x = randn(N,1);

figure(1)
plot(t,x);

X = fft(x)*dt;

df = fs/N;
f = (0:N-1)*df;

figure(2)
plot(f,abs(X));

figure(3)
plot(f,angle(X));

sum_check = sum(x.^2)*dt
sum_check2 = sum(abs(X).^2)*df

%%
clc;
clear all;
close all;

N = 4096;
fs = 4096;
dt = 1/fs;
t = (0:N-1)*dt;

x = randn(N,1);

figure(1)
plot(t,x);

X = fft(x);
df = fs/N;
f = (0:N-1)*df;

figure(2)
plot(f,abs(X));

figure(3)
plot(f,angle(X));

figure(4)
hist(x,1000)
%%
X = ones(N,1);
for ii = 2:N/2
    phase = rand()*2*pi;
    X(ii) = exp((1j)*phase);
    X(N-(ii-2)) = exp(-(1j)*phase);
end
x = ifft(X)/dt;
figure(5)
plot(t,x);

figure(6)
plot(f,abs(X));

figure(7)
plot(f,angle(X));
