% This MATLAB script trains CSINet model proposed in
% 'Chao-Kai Wen, Wan-Ting Shih, and Shi Jin, "Deep learning for massive MIMO CSI feedback,”
% IEEE Wireless Communications Letters, 2018. [Online]. Available: https://ieeexplore.ieee.org/document/8322184/.'
% using in MATLAB®.

% Set network parameters
maxDelay = 32;
nTx = 32;
numChannels = 2;
compressRate = 1/4; % 1/4 | 1/16 | 1/32 | 1/64
environment = "indoor"; % "indoor" | "outdoor"

% Create CSINet deep network
CSINet = createCSINet(maxDelay, nTx, numChannels, compressRate);

% Analyze CSINet architecture visually
analyzeNetwork(CSINet);

%% Data loading
% Load training data
load(fullfile("data","DATA_Htrain"+extractBefore(environment,"door")+".mat"));
sampleSize = length(HT);
xTrain = reshape(HT',maxDelay, nTx, numChannels, sampleSize);
xTrain = permute(xTrain, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

% Load validation data
load(fullfile("data","DATA_Hval"+extractBefore(environment,"door")+".mat"));
sampleSize = length(HT);
xVal = reshape(HT', maxDelay, nTx, numChannels, sampleSize);
xVal = permute(xVal, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

%% Set training parameters and train the network
options = trainingOptions("adam", ...
    InitialLearnRate=5e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=100, ...
    LearnRateDropFactor=exp(-0.1), ...
    Epsilon=1e-7, ...
    GradientDecayFactor=0.9, ...
    SquaredGradientDecayFactor=0.999, ...
    MaxEpochs=1500, ...
    MiniBatchSize=500, ...
    Shuffle="every-epoch", ...
    Verbose=true, ...
    VerboseFrequency=400, ...
    ValidationData={xVal, xVal}, ...
    ValidationFrequency=400, ...
    OutputNetwork="best-validation-loss", ...
    Plots="none");

% Train network using trainNetwork function
[CSINet, trainInfo] = trainNetwork(xTrain, xTrain, CSINet, options);

%% Test trained CSINet
% Load truncated channel coefficient matrices
load(fullfile("data","DATA_Htest"+extractBefore(environment,"door")+".mat"));

% Load untruncated channel coefficient matrices
load(fullfile("data","DATA_HtestF"+extractBefore(environment,"door")+"_all.mat"));
testSampleSize = length(HT);

%%
xTest = reshape(HT', maxDelay, nTx, numChannels, testSampleSize);
xTest = permute(xTest, [2, 1, 3, 4]);
xHat = predict(CSINet, xTest);

% Construct complex data from 2-channel input
xTestr = HT(:, 1:1024);
xTesti = HT(:, 1024 + 1:end);
xTestc = complex(xTestr - 0.5, xTesti - 0.5);

% Construct complex estimated data from 2-channel input
xHatc = complex(xHat(:, :, 1, :) - 0.5, xHat(:, :, 2, :) - 0.5);
xHatc = reshape(xHatc, nTx, maxDelay, testSampleSize);

% Apply fft to the estimated complex channel matrix to construct the
% frequency domain channel matrix
xHatFreq = fft(cat(2, xHatc, zeros(nTx, 256-maxDelay, testSampleSize)), [], 2);
xHatFreq = xHatFreq(:, 1:125, :);

% Calculate the cosine similarity of channel matrices in frequency-spatial
% domain
xtestFreq = reshape(HF_all.', 125, nTx, testSampleSize);
xtestFreq = permute(xtestFreq, [2, 1, 3]);
n1 = squeeze(sqrt(sum(conj(xtestFreq).*xtestFreq, 1)));
n2 = squeeze(sqrt(sum(conj(xHatFreq).*xHatFreq, 1)));
aa = squeeze(abs(sum(conj(xtestFreq).*xHatFreq, 1)));
rho = real(mean(aa./(n1.*n2), 'All'));
fprintf("\nAt compression rate 1/%d, rho is %f\n",1/compressRate, rho);

% Calculate MSE between test & predicted channel matrices in angular-delay domain
power = sum(abs(xTestc).^2, 2);
nmse = 10.*log10(squeeze(sum(abs(xTest - xHat).^2, [1,2,3]))./power);
meanMSE = real(mean(nmse));
fprintf("\nAt compression rate 1/%d, nmse is %f\n",1/compressRate, meanMSE);

%% Save trained network
savedNetFileName = "model_CsiNet_"+environment+"_dim"+num2str(maxDelay*nTx*numChannels*compressRate)+".mat";
save(savedNetFileName, "CSINet")

%% Local functions
function autoencoderLGraph = createCSINet(maxDelay, nTx, numChannels, compressRate)
% Helper function to create CSINet

inputSize = [maxDelay nTx numChannels];
numElements = prod(inputSize);
encodedDim = compressRate*numElements;

autoencoderLGraph = layerGraph([ ...
    % Encoder
    imageInputLayer(inputSize,"Name","Htrunc", ...
    "Normalization","none","Name","input_1")

    convolution2dLayer([3 3],2,"Padding","same","Name","conv2d")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99, ...
    "VarianceDecay",0.99,"Name","batch_normalization")
    leakyReluLayer(0.3,"Name","leaky_re_lu")

    functionLayer(@(x)permute(stripdims(x),[3,2,1,4]), ...
      "Formattable",true,"Acceleratable",true,"Name","Enc_Permute1")

    functionLayer(@(x)dlarray(reshape(x,numChannels*nTx*maxDelay,1,1,[]),'CSSB'), ...
      "Formattable",true,"Acceleratable",true,"Name","Enc_Reshape")

    fullyConnectedLayer(encodedDim,"Name","dense")

    % Decoder
    fullyConnectedLayer(numElements,"Name","dense_1")

    functionLayer(@(x)permute(stripdims(x),[3,2,1,4]),"Formattable",true, ...
    "Acceleratable",true,"Name","Dec_Permute1")

    functionLayer(@(x)dlarray(reshape(x,numChannels,nTx,maxDelay,[]),'CSSB'), ...
    "Formattable",true,"Acceleratable",true,"Name","Dec_Reshape")

    functionLayer(@(x)permute(x,[2,1,3,4]), ...
    "Formattable",true,"Acceleratable",true,"Name","Dec_Permute2")
    ]);

residualLayers1 = [ ...
    convolution2dLayer([3 3],8,"Padding","same","Name","conv2d_1")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_1")
    leakyReluLayer(0.3,"Name","leaky_re_lu_1")

    convolution2dLayer([3 3],16,"Padding","same","Name","conv2d_2")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_2")
    leakyReluLayer(0.3,"Name","leaky_re_lu_2")

    convolution2dLayer([3 3],2,"Padding","same","Name","Res_Conv_1_3")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_3")

    additionLayer(2,"Name","add")

    leakyReluLayer(0.3,"Name","leaky_re_lu_3")
    ];

autoencoderLGraph = addLayers(autoencoderLGraph,residualLayers1);
autoencoderLGraph = connectLayers(autoencoderLGraph,"Dec_Permute2","conv2d_1");
autoencoderLGraph = connectLayers(autoencoderLGraph,"Dec_Permute2","add/in2");

residualLayers2 = [ ...
    convolution2dLayer([3 3],8,"Padding","same","Name","conv2d_4")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_4")
    leakyReluLayer(0.3,"Name","leaky_re_lu_4")

    convolution2dLayer([3 3],16,"Padding","same","Name","conv2d_5")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_5")
    leakyReluLayer(0.3,"Name","leaky_re_lu_5")

    convolution2dLayer([3 3],2,"Padding","same","Name","conv2d_6")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_6")

    additionLayer(2,"Name","add_1")

    leakyReluLayer(0.3,"Name","leaky_re_lu_6")
    ];

autoencoderLGraph = addLayers(autoencoderLGraph,residualLayers2);
autoencoderLGraph = connectLayers(autoencoderLGraph,"leaky_re_lu_3","conv2d_4");
autoencoderLGraph = connectLayers(autoencoderLGraph,"leaky_re_lu_3","add_1/in2");


autoencoderLGraph = addLayers(autoencoderLGraph, ...
    [convolution2dLayer([3 3],2,"Padding","same","Name","conv2d_7") ...
    sigmoidLayer("Name","conv2d_7_sigmoid") ...
    regressionLayer("Name","RegressionLayer_conv2d_7")]);

autoencoderLGraph = ...
    connectLayers(autoencoderLGraph,"leaky_re_lu_6","conv2d_7");
end