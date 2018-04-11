%% inicjalizacja
clear ; close all; clc

%% zbior danych uczacych i testowych
%rand('state',dzien*rok);
%randn('state',dzien*rok);
%zbiór uczacy
x = [ones(200,1) rand(200,30)*2-1 ];
wp = [1 rand(1,10)*0.5+0.5, rand(1,10)*0.3, zeros(1,10)]'; %wektor prawdziwych wag
y = (wp'*x' + randn(1,200)*0.1)';
%zbiór testowy
xt = [ones(200,1) rand(200,30)*2-1];
yt = (wp'*xt' + randn(1,200)*0.1)';

%% selekcja istotnych atrybutów- funkcja stepwisefit
[~,~,~,inmodel]=stepwisefit(x(:,2:end),y);
x3=x(:,[true inmodel]); %usuni?cie nieistotnych atrybutów

w3 = (x3'*x3)^(-1)*x3'*y; %wagi atrybutów istotnych
y3=(w3'*x3')';
E3 = sum((y-y3).^2);

%dla zbioru testowego
[~,~,~,inmodel_t]=stepwisefit(xt(:,2:end),yt);
x3_t=xt(:,[true inmodel_t]); %usuni?cie nieistotnych atrybutów

w3_t = (x3_t'*x3_t)^(-1)*x3_t'*yt;
y3t=(w3_t'*x3_t')';
E3t = sum((yt-y3t).^2);


%% regresja grzbietowa
lambda=0:1000;
w4=ridge(y,x(:,2:end),lambda,0);

figure();
plot(lambda, w4); grid on;
xlabel('Parameter lambda [{\lambda}]');
ylabel('Wartosci macierzy wag [w4]');
title ('Wykres wartosci wag w zaleznosci od parametru {lambda}');

for i=1:length(lambda)
	y4(:,i)=(w4(:,i)'*x')';
    E4(i)=sum((y-y4(:,i)).^2);
end

figure();
plot(lambda, E4); grid on;
xlabel('Parametr lambda [{\lambda}]');
ylabel('Wartosci macierzy bledu [E4]');
title ('Wykres wartosci bledu E4 w zaleznosci od parametru {lambda} dla zbioru uczacego');


%dla zbioru testowego
w4t=ridge(yt,xt(:,2:end),lambda,0);
for i=1:length(lambda)
	y4t(:,i)=(w4t(:,i)'*xt')';
    E4t(i)=sum((yt-y4t(:,i)).^2);
end
figure();
plot(lambda, E4t); grid on;
xlabel('Parametr lambda [{\lambda}]');
ylabel('Wartosci macierzy bledu [E4t]');
title ('Wykres wartosci bledu E4t w zaleznosci od parametru {lambda} dla zbioru uczacego');


%% regresja lasso do regularyzacji
lambda=0:0.001:1;
[w5, FitInfo] = lasso(x(:,2:end),y,'Lambda',lambda); w5_0=FitInfo.Intercept;

figure();
plot(lambda, w5);

for i=1:length(lambda) 
    y5(:,i)=([w5_0(i); w5(:,i)]'*x')';
    E5(i)=sum((y-y5(:,i)).^2);
end
figure();
plot(lambda, E5);
xlabel('Parametr lambda [{\lambda}]');
ylabel('Wartosci macierzy bledu [E5]');
title ('Wykres wartosci bledu E5 w zaleznosci od parametru {lambda} dla zbioru uczacego');

%zbior testowy
for i=1:length(lambda) 
    y5t(:,i)=([w5_0(i); w5(:,i)]'*xt')';
    E5t(i)=sum((yt-y5t(:,i)).^2);
end
figure();
plot(lambda, E5t);
xlabel('Parametr lambda [{\lambda}]');
ylabel('Wartosci macierzy bledu [E5t]');
title ('Wykres wartosci bledu E5t w zaleznosci od parametru {lambda} dla zbioru uczacego');
