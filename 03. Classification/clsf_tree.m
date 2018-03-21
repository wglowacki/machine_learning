%% Uczenie maszynowe AiR, 2018
%%
%% Cwiczenie: Drzewa decyzyjne
% Cel: Ilustracja roznych aspektow budowania drzew decyzyjnych i ich weryfikowania

% Prosze uzupelnic brakujace fragmenty zgodnie z instrukcja (FIXME)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decision Trees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Prosze pobrac dane
clear all; close all; clc
load fisheriris.mat

%Prosze zapoznac sie ze zmiennymi meas oraz species

% Zadanie 1
%Wyswietlic informacje statystyczne na temat probek danych takie jak:

% Srednia wartosc atrybutu
average_value = mean(meas);
% wartosc maksymalna atrybutu - wartosc minimalna atrybutu
max_value = max(meas);
% odchylenie standardowe
standard_deviation = std(meas);

%Zadanie 2
% Przy pomocy funkcji gscatter() prosze wyswietlic wczytane dane dla
% atrybutow:

% a) sepal length <->sepal width (kolumna 1 i 2)
figure();
subplot(1,2,1);
gscatter(meas(:,1),meas(:,2),species, 'rgb', 'x');
hold on; grid on; title('Sepal');
xlabel('Sepal length'); ylabel('Sepal width');

% b) petal length <->petal width (kolumna 3 i 4)
subplot(1,2,2);
gscatter(meas(:,3),meas(:,4),species, 'rgb', 'x');
hold on; grid on; title('Petal');
xlabel('Petal length'); ylabel('Petal width');

%Zadanie 3
% Utworz drzewo decyzyjne przy pomocy funkcji fitctree() lub
% ClassificationTree.fit()

sepal_dec_tree = fitctree([meas(:,1),meas(:,2)], species);
pet_dec_tree = fitctree([meas(:,3),meas(:,4)], species);

% Zapoznaj sie z powstalym obiektem

% Zadanie 4
% Zapoznaj sie z graficzna i regulowa reprezentacja drzewa (funkcja view()):

view(sepal_dec_tree, 'mode', 'graph');
view(pet_dec_tree, 'mode', 'graph');

% Zadanie 5
%Wyznacz klasy dla kazdego przykladu trenujacego (dane meas, funkcja predict()):

figure();
subplot(1,2,1);
sep_classes = predict(sepal_dec_tree, [meas(:,1),meas(:,2)]);
gscatter(meas(:,1), meas(:,2), sep_classes);
hold on; grid on; title('Sepal');
xlabel('Sepal length'); ylabel('Sepal width');
subplot(1,2,2);
pet_classes = predict(pet_dec_tree, [meas(:,3),meas(:,4)]);
gscatter(meas(:,3), meas(:,4), pet_classes);
hold on; grid on; title('Petal');
xlabel('Petal length'); ylabel('Petal width');

% Zadanie 6
% Wyznacz macierz bledow (ang. confusion matrix) przy pomocy funkcji
% confusion matrix - funkcja confusionmat()

%https://en.wikipedia.org/wiki/Confusion_matrix
sep_conf_matrix = confusionmat(species, sep_classes);
pet_conf_matrix = confusionmat(species, pet_classes);

% Zadanie 7
%Wyswietl macierz przy pomocy funkcji disp()

disp(sep_conf_matrix);
disp(pet_conf_matrix);

% Zadanie 8
% Wyznacz macierz bledow uzywajac funkcji plotconfusion() (wczesniej konwertuj species i wynik predykcji na wektory numeryczne:

[~, ~, a] = unique(species, 'stable');
[~, ~, b] = unique(sep_classes, 'stable');
[~, ~, c] = unique(pet_classes, 'stable');
a = a';
b = b';
c = c';
X = [];
for x = unique(a)
    X = [X; a == x];
end
sep_Y = [];
for y = unique(b)
    sep_Y = [sep_Y; b == y];
end
pet_Y = [];
for y = unique(c)
    pet_Y = [pet_Y; c == y];
end
figure()
plotconfusion(X,sep_Y);
figure()
plotconfusion(X, pet_Y);

% Zadanie 9
% Oblicz podstawowe miary na podstawie macierzy bledow:

rec = zeros(1, 3);
spec = zeros(1, 3);
prec = zeros(1, 3);
acc = zeros(1, 3);
%only for sepal
for x = 1:3
    TP = sep_conf_matrix(x, x);
    FP = sum(sep_conf_matrix(:, x)) - TP;
    FN = sum(sep_conf_matrix(x, :)) - TP;
    TN = sum(sum(sep_conf_matrix)) - TP - FP - FN;
    rec(x) = TP / (TP + FN);
    spec(x) = TN / (FP + TN);
    prec(x) = TP / (TP + FP);
    acc(x) = (TP + TN) / (TP + TN + FP + FN);
end

% czulosc:
rec = mean(rec);

% swoistosc:
spec = mean(spec);

% precyzja:
prec = mean(prec);

% dokladnosc:
acc = mean(acc);

% Zadanie 10
%% Funkcja fitctree() po ustawieniu parametru 'CrossVal' 'on' uzywa 10-krotnej walidacji krzyzowej (ang. 10-fold crossvalidation). 
%Utworz drzewo decyzyjne

val_tree = fitctree([meas(:,1),meas(:,2)], species, 'CrossVal', 'on');

% Odpowiedz na pytanie:
% Ile drzew zostalo wygenerowanych:
length(val_tree.Trained);

% Zadanie 11
%Wyswietl pierwsze drzewo:
view(val_tree.Trained{1}, 'mode', 'graph');

% Zadanie 12
%Ocen dzialanie modelu po wlaczeniu walidacji krzyzowej (ang. crossvalidation). 
%Odpowiedz:
cross_classes = kfoldPredict(val_tree);
confusion_matrix = confusionmat(species, cross_classes);
%Model dziala gorzej, co spowodowane jest roznym losowaniem i dzieleniem na
%baze testowa i uczaca. Wczesniej ustalane one byly statycznie, wybierany byl jedynie jeden przypadek.

% Zadanie 13
%Uzyj funkcji kfoldLoss() do oceny dzialania modelu.
fold_loss = kfoldLoss(val_tree);

% Zadanie 14
%W petli generuj drzewa dla parametru "minimalna liczba przykladow w lisciu" (MinLeaf) zmienianego od 2 do 100 (parametr m). 
%Kazde takie drzewo oceniaj w ACC:
num = 0;
opt_acc = 1;
ACC = [];
for m = 2:100
    fitted = fitctree([meas(:,1),meas(:,2)], species, 'CrossVal', 'on', 'MinLeafSize', m);
    act = kfoldLoss(fitted);
    ACC = [ACC, act];
    if act < opt_acc
        num = m;
        opt_acc = act;
    end
end
num; %optimal number


% Zadanie 15
%Sporzadz wykres bledu w zaleznosci od parametru m. Wyznacz optymalna wartosc m (najwieksza wartosc m, przy ktorej blad utrzymuje sie na niskim poziomie).

x = 2:100;
figure()
hold on; grid on;
plot(x, ACC, 'k');
xlabel('K value'); ylabel('Accuracy rate');
title('Error in function of minimal number of samples on leaf');

% Zadanie 16
% Pokaz optymalne drzewo w postaci graficznej:

optimal_tree = fitctree([meas(:,1),meas(:,2)], species, 'CrossVal', 'on', 'MinLeafSize', 4);
% The model contains 10 decision trees.  It is impossible to show the model
% in the form of one tree, as such a tree does not exist.  A particular
% tree may be selected from tree.Trained cell array.
disp(optimal_tree);

% Zadanie 17
% Porownaj blad osiagany przez drzewo optymalne z bledem osiaganym przez drzewo wygenerowane przy domyslnych ustawieniach parametrow:

% Matlab R2015a issues.  loss function does not cooperate.  Expecting worse
% results for 'optimal' model - effect of better generalization vs.
% overfitting, when no cross validation is performed.

tree_basic = fitctree([meas(:,3),meas(:,4)], species);
loss_basic = loss(tree_basic, [meas(:,3),meas(:,4)], species);
tree = fitctree([meas(:,3),meas(:,4)], species, 'CrossVal', 'on', 'MinLeafSize', 40);
loss = kfoldLoss(tree);
disp(loss_basic - loss)

%DODATKOWO:
%Prosze przeanalizowac:
%This example shows how to optimize hyperparameters automatically using fitctree. The example uses Fisher's iris data.
%??
% X = meas;
% Y = species;
% Mdl = fitctree(X,Y,'OptimizeHyperparameters','auto')

