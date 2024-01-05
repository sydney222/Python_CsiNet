clc
clear
close all

envir = 'Indoor'; %'Indoor' or 'Outdoor'
encoded_dim = 32;  %compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
model_name = "model_CsiNet_"+envir+"_dim"+num2str(encoded_dim);

load("Sigmoid result\"+model_name+"_Quantization_Result.mat")
rho_all_sigmoid=rho_all;
NMSE_all_sigmoid=NMSE_all;
load("Tanh result\"+model_name+"_Quantization_Result.mat")
rho_all_tanh=rho_all;
NMSE_all_tanh=NMSE_all;

figure
tiledlayout(2,1)
nexttile
plot(nBitsVec,rho_all_sigmoid,'*-','LineWidth',1.5)
hold on
plot(nBitsVec,rho_all_tanh,'*-','LineWidth',1.5)
legend("Sigmoid","Tanh")
title("Correlation ("+envir+" Codeword-" + encoded_dim + ")")
xlabel("Number of Quantization Bits"); ylabel("\rho")
grid on
nexttile
plot(nBitsVec,NMSE_all_sigmoid,'*-','LineWidth',1.5)
hold on
plot(nBitsVec,NMSE_all_tanh,'*-','LineWidth',1.5)
legend("Sigmoid","Tanh")
title("NMSE ("+envir+" Codeword-" + encoded_dim + ")")
xlabel("Number of Quantization Bits"); ylabel("NMSE (dB)")
grid on
savefig(gcf,"Compare result\"+envir+"_"+num2str(encoded_dim)+"_Quantized.fig")
saveas(gcf,"Compare result\"+envir+"_"+num2str(encoded_dim)+"_Quantized.jpg")