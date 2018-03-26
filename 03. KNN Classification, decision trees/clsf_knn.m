%% Uczenie maszynowe AiR, 2018
%%
%% Cwiczenie: K-najblizszych sasiadow
% Cel: Wykorzystanie wbudowanych funkcji do klasyfikacji danych przy pomocy
% algorytmu k-nn

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-nearest
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Prosze pobrac dane
clear all; close all; clc;
%Fisher's 1936 iris data
load fisheriris.mat
% Prosze zapoznac sie ze zbiorem danych -> zmienna Description w Workspace

%Prosze uzywac funkcji fitcknn(), funkcja knnclassify() zostanie w
%przyszlosci callkowicie wycofana

%Zadania

% Zaprojektuj klasyfikator typu k najblizszych sasiadow (k-NN) do
% rozpoznawania kwiatow irysa lub rodzaju arytmii.

% Zadanie 1

% Podziel zbior danych na uczacy i testowy. Losowo wybierz 5 danych do
%zbioru testowego
w = randi([1 150], 1, 5);
test = species(w,:);
test_meas = meas(w,:);
learn = species;
learn_meas = meas;
learn(w,:) = [];
learn_meas(w,:) = [];

% Zadanie 2
% Narysuj dane uczace oraz testowe
% wybrac 2 paramentry
%FIXME
figure(1);
hold on; grid on;
title('Learn and test data');
xlabel('Sepal length'); ylabel('Sepal width');
gscatter(test_meas(:,1), test_meas(:,2), test, 'rgb', 'o');
gscatter(learn_meas(:,1),learn_meas(:,2),learn, 'rgb', 'x');
legend('learn setosa','learn versicolor', 'learn virginica','test setosa',...
'test versicolor','test virginica','Location','best');
hold off;
% Zadanie 3
% Znajdz 5 punktow najblizszych punktowi badanemu (pierwszy ze zbioru testowego)
% Skorzystaj z funkcji knnsearch()

k = 5;
[eIdx,eD] = knnsearch(learn_meas(:,1:2),test_meas(:,1:2),'K',k,'distance','euclidean');
[mIdx,mD] = knnsearch(learn_meas(:,1:2),test_meas(:,1:2),'K',k,'distance','minkowski');

% Narysuj te punkty na wykresie

num = eIdx(:,1);
learn_meas(num,:);
figure(2)
gscatter(learn_meas(:,1),learn_meas(:,2),learn);
line(test_meas(:,1),test_meas(:,2),'Marker','x','Color','k',...
   'Markersize',10,'Linewidth',2,'Linestyle','none');
line(learn_meas(eIdx,1),learn_meas(eIdx,2),'Color',[.5 .5 .5],'Marker','o',...
   'Linestyle','none','Markersize',10);
line(learn_meas(mIdx,1),learn_meas(mIdx,2),'Color',[.5 .5 .5],'Marker','p',...
   'Linestyle','none','Markersize',10);
legend('setosa','versicolor','virginica','query point',...
'minkowski','chebychev','Location','best');


% Zadanie 4
%Ustal gatunki sasiadow. Skorzystaj z funckji tabulate()

for i=1:size(test)
    disp(['Results for ',num2str(i),' test']);
    tabulate(learn(eIdx(i,:)));
end

% Zadanie 5
% Wykorzystujac funkcj? fitcknn() stworz klasyfikator dla k=4

k=4;
Mdl = fitcknn(learn_meas, learn, 'NumNeighbors', k, 'Standardize', 1);

%Zadanie 6
%Sklasyfikuj dane ze zbioru testowego, funkcja predict()

predict(Mdl, test_meas);

%Zadanie 7
% Znajdz optymalna wartosc liczby najblizszych sasiadow k, np. for:
%Przydatne funckje: crossvalind(), fitcknn()
%Dokladnosc klasyfikatora: ACC

k = 150;
ACC = zeros(1,k);
for i=1:k
    knn = fitcknn(meas,species,'NumNeighbors',i);
    class = predict(knn, test_meas);
    pos_rec = cellfun(@strcmp, test, class);
    ACC(i) = sum(pos_rec)/size(pos_rec,1);
end
%Narysuj wykres zaleznosci dokladnosci klasyfikatora (ACC) od wartosci k.
 x = 1:k;
 plot(x, ACC, 'k'); grid on;
 xlabel('K value'); ylabel('Accuracy rate');
 title('Accuracy for different k parameter');

%Wybierz optymalna wartosc k
%FIXME

%Zadanie 8
% Przedstaw na wykresie granice klas
opt_k = k(ACC(k)==1);
%Stworz klasyfikator kNN dla 2 wybranych parametrow:
two_params = [1,2];
knn = fitcknn(learn_meas(:,two_params),learn,'NumNeighbors',opt_k);
X = learn_meas(:,two_params);
%Dane testowe- przestrzen (X- parametry)(odkomentuj:)
x1_range = min(X(:,1)):.01:max(X(:,1));
x2_range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1_range,x2_range);
XGrid = [xx1(:) xx2(:)];

% Sklasyfikuj dane ze zbioru testowego

p = predict(knn, XGrid);

% Narysuj wykres (gscatter())

figure(); grid on;
gscatter(XGrid(:, 1), XGrid(:, 2), p);
xlabel('Sepal length'); ylabel('Sepal width');
title('Limits of classes for every species');


% DODATKOWO
%Prosze zapoznac sie z parametrami funkcji fitcknn() : metryki
%odleglosci(distance metrics), wagi (Distance Weights) ect.





