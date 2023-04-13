function [encNet,decNet] = helperCSINetSplitEncoderDecoder(net,encEnd)
%helperCSINetSplitEncoderDecoder Split encoder and decoder networks
%   [ENC,DEC] = helperCSINetSplitEncoderDecoder(NET,SL) plits the
%   autoencoder neural network, NET, into an encoder neural network, ENC, 
%   and a decoder neural network, DEC. SL is the name of the last layer of
%   the encoder section in NET. This function also adds a regression layer
%   as the output layer of the ENC and an image input layer as the input
%   layer to the DEC.
%
%   See also CSICompressionAutoencoderExample, helperCSINetLayerGraph.

%   Copyright 2022 The MathWorks, Inc.

lg = layerGraph(net);

encEndIdx = find( {lg.Layers.Name} == encEnd );
decStart = lg.Layers(encEndIdx+1).Name;
decInputSize = lg.Layers(encEndIdx+1).InputSize;
lg = disconnectLayers(lg, encEnd, decStart);

% Find and remove encoder layers from decoder
decLayerNames = {lg.Layers(encEndIdx+1:end).Name};
lgEnc = removeLayers(lg, decLayerNames);

% Find and remove decoder layers from encoder
encLayerNames = {lg.Layers(1:encEndIdx).Name};
lgDec = removeLayers(lg, encLayerNames);

%%% Add output layer to encoder and input layer to decoder
lgEnc = addLayers(lgEnc, regressionLayer(Name="Enc_out"));
lgEnc = connectLayers(lgEnc, encEnd, "Enc_out");

lgDec = addLayers(lgDec, featureInputLayer(decInputSize, Name="Dec_in"));
lgDec = connectLayers(lgDec, "Dec_in", decStart);

encNet = assembleNetwork(lgEnc);
decNet = assembleNetwork(lgDec);
end