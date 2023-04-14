import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from scipy.io import savemat
import numpy as np
import math
import time

class RefineNet(nn.Module):
    def __init__(self, img_channels=2):
        super(RefineNet, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(img_channels, 8, kernel_size=(3, 3), padding=(1,1)),
                                  nn.BatchNorm2d(8, eps=1e-03, momentum=0.99),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1,1)),
                                  nn.BatchNorm2d(16, eps=1e-03, momentum=0.99),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  nn.Conv2d(16, 2, kernel_size=(3, 3), padding=(1,1)),
                                  nn.BatchNorm2d(2, eps=1e-03, momentum=0.99))

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        ori_x = x.clone()

        # concatenate
        x = self.conv(x) + ori_x

        return self.leakyRelu(x)


class CsiNet(nn.Module):
    def __init__(self, img_height=32, img_width=32, img_channels=2, residual_num=2, encoded_dim=512):
        super(CsiNet, self).__init__()

        img_total = img_height * img_width * img_channels

        self.conv1 = nn.Sequential(nn.Conv2d(img_channels, 2, kernel_size=(3, 3), padding=(1,1)),
                                   nn.BatchNorm2d(2, eps=1e-03, momentum=0.99),
                                   nn.LeakyReLU(negative_slope=0.3))

        self.dense = nn.Sequential(nn.Linear(img_total, encoded_dim),
                                   nn.Tanh(),
                                   #nn.Sigmoid(),
                                   nn.Linear(encoded_dim, img_total))

        self.decoder = nn.ModuleList([RefineNet(img_channels)
                                      for _ in range(residual_num)])

        self.conv2 = nn.Sequential(nn.Conv2d(img_channels, 2, kernel_size=(3, 3), padding=(1,1)),
                                   nn.Sigmoid())

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Encoder, convolution & reshape
        x = self.conv1(x)
        x = x.contiguous().view(batch_size, channels * height * width)

        # Dense & reshape
        x = self.dense(x)
        x = x.contiguous().view(batch_size, channels, height, width)

        # Residual decoders
        for layer in self.decoder:
            x = layer(x)

        # Final convolution
        x = self.conv2(x)

        # x = self.Encoder(x)
        # x = self.Decoder(x)

        return x

if __name__ == '__main__':
    envir = 'outdoor'  # 'indoor' or 'outdoor'
    # image params
    img_height = 32
    img_width = 32
    img_channels = 2
    img_total = img_height * img_width * img_channels
    # network params
    residual_num = 2
    encoded_dim = 32  # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

    # Data loading
    if envir == 'indoor':
        mat = sio.loadmat('data/DATA_Htrainin.mat')
        x_train = mat['HT']  # array
        mat = sio.loadmat('data/DATA_Hvalin.mat')
        x_val = mat['HT']  # array
        mat = sio.loadmat('data/DATA_Htestin.mat')
        x_test = mat['HT']  # array

    elif envir == 'outdoor':
        mat = sio.loadmat('data/DATA_Htrainout.mat')
        x_train = mat['HT']  # array
        mat = sio.loadmat('data/DATA_Hvalout.mat')
        x_val = mat['HT']  # array
        mat = sio.loadmat('data/DATA_Htestout.mat')
        x_test = mat['HT']  # array

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train_length = len(x_train)
    x_val_length = len(x_val)
    x_test_length = len(x_test)

    x_train = np.reshape(x_train, (x_train_length, img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
    x_val = np.reshape(x_val, (x_val_length, img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (x_test_length, img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

    x_train = torch.tensor(x_train)
    x_val = torch.tensor(x_val)
    x_test = torch.tensor(x_test)

    # device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    x_train = x_train.to(device)
    x_val = x_val.to(device)
    x_test = x_test.to(device)

    # model
    model = CsiNet(img_height=img_height, img_width=img_width, img_channels=img_channels, residual_num=residual_num, encoded_dim=encoded_dim).to(device)
    #print(model)

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-07)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # loss function
    criterion = nn.MSELoss()

    n_epochs = 1000
    batch_size = 200
    total_batches = int(x_train_length/batch_size)

    tStart = time.time()

    train_loss_history = []
    val_loss_history = []

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(n_epochs):
            print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
            for i in range(total_batches):
                x_batch = x_train[i*batch_size:(i+1)*batch_size,]
                model.train()
                out = model(x_batch)

                loss = criterion(out, x_batch)
                train_loss_history.append(loss.item())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            # Validating
            with torch.no_grad():
                model.eval()

                out = model(x_val)
                loss = criterion(out, x_val)
                print("Loss: ",loss.item())
                val_loss_history.append(loss.item())

    tEnd = time.time()
    training_time = tEnd - tStart
    print("It cost %f sec for training." % training_time)

    # Testing data
    tStart = time.time()
    x_hat = model(x_test)
    tEnd = time.time()
    print("It cost %f sec." % ((tEnd - tStart) / x_test.shape[0]))
    x_test = x_test.to('cpu')
    x_hat=x_hat.to('cpu')

    # Calcaulating the NMSE and rho
    if envir == 'indoor':
        mat = sio.loadmat('data/DATA_HtestFin_all.mat')
        X_test = mat['HF_all']  # array

    elif envir == 'outdoor':
        mat = sio.loadmat('data/DATA_HtestFout_all.mat')
        X_test = mat['HF_all']  # array

    X_test = torch.tensor(X_test)
    X_test = torch.reshape(X_test, (len(X_test), img_height, 125))
    x_test_real = torch.reshape(x_test[:, 0, :, :], (len(x_test), -1))
    x_test_imag = torch.reshape(x_test[:, 1, :, :], (len(x_test), -1))
    x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
    x_hat_real = torch.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = torch.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    x_hat_F = torch.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
    X_hat = torch.fft.fft(torch.cat((x_hat_F, torch.zeros((len(x_hat_C), img_height, 257 - img_width))), axis=2), axis=2)
    X_hat = X_hat[:, :, 0:125]

    n1 = torch.sqrt(torch.sum(torch.conj(X_test) * X_test, axis=1))
    n2 = torch.sqrt(torch.sum(torch.conj(X_hat) * X_hat, axis=1))
    aa = abs(torch.sum(torch.conj(X_test) * X_hat, axis=1))
    rho = torch.mean(aa / (n1 * n2), axis=1)
    X_hat = torch.reshape(X_hat, (len(X_hat), -1))
    X_test = torch.reshape(X_test, (len(X_test), -1))
    power = torch.sum(abs(x_test_C) ** 2, axis=1)
    power_d = torch.sum(abs(X_hat) ** 2, axis=1)
    mse = torch.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)
    NMSE = 10 * math.log10(torch.mean(mse / power))
    Correlation = torch.mean(rho).item().real

    print("In " + envir + " environment")
    print("When dimension is", encoded_dim)
    print("NMSE is ", NMSE)
    print("Correlation is ", Correlation)

    file = 'CsiNet_' + (envir) + '_dim' + str(encoded_dim) + time.strftime('_%m_%d_%H_%M')
    outfile = "result/result_%s.mat" % file
    savemat(outfile, {'train_loss_history': train_loss_history,
                      'val_loss_history': val_loss_history,
                      'training_time': training_time,
                      'NMSE': NMSE,
                      'Correlation': Correlation})

    outfile = "model/model_%s.pt" % file
    torch.save(model, outfile)

    # Export the model
    model.eval()
    outfile = "model/model_%s.mat" % file
    savemat(outfile, {'Enc_Conv_Weight': model.conv1[0].weight.detach().to('cpu').numpy(),
                      'Enc_Conv_Bias': model.conv1[0].bias.detach().to('cpu').numpy(),
                      'Enc_BN_Weight': model.conv1[1].weight.detach().to('cpu').numpy(),
                      'Enc_BN_Bias': model.conv1[1].bias.detach().to('cpu').numpy(),
                      'Enc_BN_Mean': model.conv1[1].running_mean.detach().to('cpu').numpy(),
                      'Enc_BN_Var': model.conv1[1].running_var.detach().to('cpu').numpy(),
                      'Enc_FC_Weight': model.dense[0].weight.detach().to('cpu').numpy(),
                      'Enc_FC_Bias': model.dense[0].bias.detach().to('cpu').numpy(),
                      'Dec_FC_Weight': model.dense[2].weight.detach().to('cpu').numpy(),
                      'Dec_FC_Bias': model.dense[2].bias.detach().to('cpu').numpy(),
                      'Res_Conv_1_1_Weight': model.decoder[0].conv[0].weight.detach().to('cpu').numpy(),
                      'Res_Conv_1_1_Bias': model.decoder[0].conv[0].bias.detach().to('cpu').numpy(),
                      'BN_1_1_Weight': model.decoder[0].conv[1].weight.detach().to('cpu').numpy(),
                      'BN_1_1_Bias': model.decoder[0].conv[1].bias.detach().to('cpu').numpy(),
                      'BN_1_1_Mean': model.decoder[0].conv[1].running_mean.detach().to('cpu').numpy(),
                      'BN_1_1_Var': model.decoder[0].conv[1].running_var.detach().to('cpu').numpy(),
                      'Res_Conv_1_2_Weight': model.decoder[0].conv[3].weight.detach().to('cpu').numpy(),
                      'Res_Conv_1_2_Bias': model.decoder[0].conv[3].bias.detach().to('cpu').numpy(),
                      'BN_1_2_Weight': model.decoder[0].conv[4].weight.detach().to('cpu').numpy(),
                      'BN_1_2_Bias': model.decoder[0].conv[4].bias.detach().to('cpu').numpy(),
                      'BN_1_2_Mean': model.decoder[0].conv[4].running_mean.detach().to('cpu').numpy(),
                      'BN_1_2_Var': model.decoder[0].conv[4].running_var.detach().to('cpu').numpy(),
                      'Res_Conv_1_3_Weight': model.decoder[0].conv[6].weight.detach().to('cpu').numpy(),
                      'Res_Conv_1_3_Bias': model.decoder[0].conv[6].bias.detach().to('cpu').numpy(),
                      'BN_1_3_Weight': model.decoder[0].conv[7].weight.detach().to('cpu').numpy(),
                      'BN_1_3_Bias': model.decoder[0].conv[7].bias.detach().to('cpu').numpy(),
                      'BN_1_3_Mean': model.decoder[0].conv[7].running_mean.detach().to('cpu').numpy(),
                      'BN_1_3_Var': model.decoder[0].conv[7].running_var.detach().to('cpu').numpy(),
                      'Res_Conv_2_1_Weight': model.decoder[1].conv[0].weight.detach().to('cpu').numpy(),
                      'Res_Conv_2_1_Bias': model.decoder[1].conv[0].bias.detach().to('cpu').numpy(),
                      'BN_2_1_Weight': model.decoder[1].conv[1].weight.detach().to('cpu').numpy(),
                      'BN_2_1_Bias': model.decoder[1].conv[1].bias.detach().to('cpu').numpy(),
                      'BN_2_1_Mean': model.decoder[1].conv[1].running_mean.detach().to('cpu').numpy(),
                      'BN_2_1_Var': model.decoder[1].conv[1].running_var.detach().to('cpu').numpy(),
                      'Res_Conv_2_2_Weight': model.decoder[1].conv[3].weight.detach().to('cpu').numpy(),
                      'Res_Conv_2_2_Bias': model.decoder[1].conv[3].bias.detach().to('cpu').numpy(),
                      'BN_2_2_Weight': model.decoder[1].conv[4].weight.detach().to('cpu').numpy(),
                      'BN_2_2_Bias': model.decoder[1].conv[4].bias.detach().to('cpu').numpy(),
                      'BN_2_2_Mean': model.decoder[1].conv[4].running_mean.detach().to('cpu').numpy(),
                      'BN_2_2_Var': model.decoder[1].conv[4].running_var.detach().to('cpu').numpy(),
                      'Res_Conv_2_3_Weight': model.decoder[1].conv[6].weight.detach().to('cpu').numpy(),
                      'Res_Conv_2_3_Bias': model.decoder[1].conv[6].bias.detach().to('cpu').numpy(),
                      'BN_2_3_Weight': model.decoder[1].conv[7].weight.detach().to('cpu').numpy(),
                      'BN_2_3_Bias': model.decoder[1].conv[7].bias.detach().to('cpu').numpy(),
                      'BN_2_3_Mean': model.decoder[1].conv[7].running_mean.detach().to('cpu').numpy(),
                      'BN_2_3_Var': model.decoder[1].conv[7].running_var.detach().to('cpu').numpy(),
                      'Dec_Conv_Weight': model.conv2[0].weight.detach().to('cpu').numpy(),
                      'Dec_Conv_Bias': model.conv2[0].bias.detach().to('cpu').numpy()})