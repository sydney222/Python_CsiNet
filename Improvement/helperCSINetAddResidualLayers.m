function lgraph = helperCSINetAddResidualLayers(lgraph,llName)
%helperCSINetAddResidualLayers Append residual layers for autoencoder
%   OUTLG = helperCSINetAddResidualLayers(INLG,LL) appends residual layers to
%   input layer graph, INLG, and returns the OUTLG layer graph. LL is the
%   name of the last layer of INLG. 
%
%   See also CSICompressionAutoencoderExample, helperCSINetLayerGraph.

%   Copyright 2022 The MathWorks, Inc.

layers1 = [ ...
    convolution2dLayer([3 3],8,"Padding","same","Name","Res_Conv_1_1")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","BN_1_1")
    leakyReluLayer(0.3,"Name","leakyRelu_1_1")
    
    convolution2dLayer([3 3],16,"Padding","same","Name","Res_Conv_1_2")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","BN_1_2")
    leakyReluLayer(0.3,"Name","leakyRelu_1_2")
    
    convolution2dLayer([3 3],2,"Padding","same","Name","Res_Conv_1_3")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","BN_1_3")

    additionLayer(2,"Name","add_1")

    leakyReluLayer(0.3,"Name","leakyRelu_1_3")
    ];
lgraph = addLayers(lgraph,layers1);
lgraph = connectLayers(lgraph,llName,"Res_Conv_1_1");
lgraph = connectLayers(lgraph,llName,"add_1/in2");

layers2 = [ ...
    convolution2dLayer([3 3],8,"Padding","same","Name","Res_Conv_2_1")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","BN_2_1")
    leakyReluLayer(0.3,"Name","leakyRelu_2_1")
    
    convolution2dLayer([3 3],16,"Padding","same","Name","Res_Conv_2_2")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","BN_2_2")
    leakyReluLayer(0.3,"Name","leakyRelu_2_2")
    
    convolution2dLayer([3 3],2,"Padding","same","Name","Res_Conv_2_3")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","BN_2_3")

    additionLayer(2,"Name","add_2")

    leakyReluLayer(0.3,"Name","leakyRelu_2_3")
    ];
lgraph = addLayers(lgraph,layers2);
lgraph = connectLayers(lgraph,"leakyRelu_1_3","Res_Conv_2_1");
lgraph = connectLayers(lgraph,"leakyRelu_1_3","add_2/in2");
end
