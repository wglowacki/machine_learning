
%   TOPIC: Data analysis
%
% ------------------------------------------------------------------------

close all; clear all;
clearvars;

smarket = readtable('./data/smarket.csv');
smarket.Direction = categorical(smarket.Direction, {'Up','Down'});
summary(smarket);
smarket_sub = smarket(:, 1:8);
tab_smarket_sub = table2array(smarket_sub);
[Cor,p_vals] = corrcoef(tab_smarket_sub);
% smarket_cor_tbl - tabela zawierajaca "okrojone" dane (bez cechy `Direction`)
% Cor, p_val - macierze zawierajace odpowiednio wspólczynniki korelacji i p-wartoœci
Cor = array2table(Cor);
Cor.Properties.VariableNames = smarket_sub.Properties.VariableNames;
Cor.Properties.RowNames = smarket_sub.Properties.VariableNames;
p_vals = array2table(p_vals);
p_vals.Properties.VariableNames = smarket_sub.Properties.VariableNames;
p_vals.Properties.RowNames = smarket_sub.Properties.VariableNames;

