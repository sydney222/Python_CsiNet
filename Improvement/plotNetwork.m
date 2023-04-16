function plotNetwork(net,encNet,decNet)
%plotNetwork Plot autoencoder network
%   plotNetwork(NET,ENC,DEC) plots the full autoencoder network together
%   with encoder and decoder networks.
fig = figure;
t1 = tiledlayout(1,2,'TileSpacing','Compact');
t2 = tiledlayout(t1,1,1,'TileSpacing','Tight');
t3 = tiledlayout(t1,2,1,'TileSpacing','Tight');
t3.Layout.Tile = 2;
nexttile(t2)
plot(net)
title("Autoencoder")
nexttile(t3)
plot(encNet)
title("Encoder")
nexttile(t3)
plot(decNet)
title("Decoder")
pos = fig.Position;
pos(3) = pos(3) + 200;
pos(4) = pos(4) + 300;
pos(2) = pos(2) - 300;
fig.Position = pos;
end