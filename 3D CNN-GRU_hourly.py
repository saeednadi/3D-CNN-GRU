from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd 
import os
from numpy import array
from keras.layers.convolutional import Conv1D,MaxPooling1D,AveragePooling1D,Conv2D,AveragePooling2D,Conv3D,AveragePooling3D
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Activation,LSTM,LeakyReLU,GRU,Dropout,Input,Flatten
from keras.optimizers import Adam,Adagrad
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from matplotlib import pyplot
from keras import regularizers
from keras.callbacks import EarlyStopping
import keras
from keras.models import Model
import keras.backend as K
from datetime import datetime
from keras.regularizers import l2
os.chdir('D:\\codes thesis\\final\\CNN-GRU')
#for normalization
def normal_vector(X):
    data = np.reshape(X, (-1, X.shape[2]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data_ = data.reshape(X.shape[0], X.shape[1], X.shape[2],X.shape[3],X.shape[4])
    X = data_
    return X

def normal_vector_(auxiliary_input_array):
    data = np.reshape(auxiliary_input_array, (-1, auxiliary_input_array.shape[2]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data_ = data.reshape(auxiliary_input_array.shape[0], auxiliary_input_array.shape[1],
                         auxiliary_input_array.shape[2])
    auxiliary_input_array = data_
    return auxiliary_input_array

def D_normal_vector(X):
    data = np.reshape(X, (-1, X.shape[2]))
    data = scaler.inverse_transform(data)
    data_ = data.reshape(X.shape[0], X.shape[1], X.shape[2],X.shape[3],X.shape[4])
    X = data_
    return X
#==============================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
datasets =pd.read_csv('hourly_datasets.csv' ,header=None)
data_weather =pd.read_csv('weather_data.csv',header=None)
data_weather=data_weather.values
values=datasets.values
values = values.astype('float32')
#==============================================================================
A=19536
i=0
X=[]
y=[]
for i in range(0,19536):
    X=values[[i,A+i,(A*2)+i,(A*3)+i,(A*4)+i,(A*5)+i,(A*6)+i,
             (A*7)+i,(A*8)+i,(A*9)+i,(A*10)+i,(A*11)+i,(A*12)+i],:]
    i += 1
    y=np.append(y,X)
data=np.reshape(y,(values.shape[0],values.shape[1]))   
data_=np.reshape(data,(19536,13,4))
#==============================================================================
def split_sequences(sequences, time_lag, predict_time):
    end_ix = time_lag
    out_end_ix = end_ix + predict_time
    seq_x, seq_y = sequences[0:end_ix, :, :], sequences[end_ix:out_end_ix, :, 0]
    X = np.transpose(seq_x, (1, 2, 0))
    y = np.transpose(seq_y, (1, 0))
    X = X[np.newaxis, :, :, :]
    y = y[np.newaxis, :, :]    
    for i in range(1, len(sequences)):
        end_ix = i + time_lag
        out_end_ix = end_ix + predict_time
        if out_end_ix > len(sequences):	
            break
        seq_x, seq_y = sequences[i:end_ix, :, :], sequences[end_ix:out_end_ix, :, 0]
        seq_x_r = np.transpose(seq_x, (1, 2, 0))
        seq_y_r = np.transpose(seq_y, (1, 0))
        seq_x_r = seq_x_r[np.newaxis, :, :, :]
        seq_y_r = seq_y_r[np.newaxis, :, :]
        X = np.concatenate((X, seq_x_r), axis=0)
        y = np.concatenate((y, seq_y_r), axis=0)
    return X, y
#==============================================================================
time_lag=8
n_features=4
predict_time =1
output_station =13
X, y = split_sequences(data_,time_lag,predict_time)
X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1,X.shape[3])
X = normal_vector(X)
#y= normal_vector_(y)
Y= y[:,:,-1:]
n = Y.shape[1] * Y.shape[2]
y_ = Y.reshape((Y.shape[0], n))
n_split = 2*(365 * 24)
train_x, train_y = X[:n_split, :,:,:,:], y_[:n_split,:]
train_y = scaler.fit_transform(train_y)
test_x, test_y = X[n_split:, :,:,:,:], y_[n_split:,:]
auxiliary_input_array=data_weather
auxiliary_input_array = np.reshape(data_weather,(19536,13,7))
auxiliary_input_array = normal_vector_(auxiliary_input_array)
auxiliary_input_train=auxiliary_input_array[:n_split,:,:]
auxiliary_input_test=auxiliary_input_array[n_split:,:,:]
auxiliary_input_train=auxiliary_input_train[:len(train_x),:,:]
auxiliary_input_test=auxiliary_input_test[:len(test_x):,:,:]
#======================================================================================
print("Build Model ... ")
main_input =Input(shape=(train_x.shape[1], train_x.shape[2],
                 train_x.shape[3],train_x.shape[4]),name='main_input',dtype='float32')
auxiliary_input = Input(shape=(auxiliary_input_array.shape[1],auxiliary_input_array.shape[2]), name='aux_input')
Conv3D_layer1=Conv3D(64, kernel_size=(4,1,8),padding='same')(main_input)
Activation_layer=Activation('selu')(Conv3D_layer1)
reshape_1=keras.layers.Reshape((K.int_shape(Activation_layer)[1],
K.int_shape(Activation_layer)[2]*K.int_shape(Activation_layer)[3]*K.int_shape(Activation_layer)[4]))(Activation_layer)
gru_layer1=GRU(64, activation='tanh',return_sequences=True,
               dropout=0.02, recurrent_dropout=0.3)(reshape_1)
gru_layer2=GRU(64, activation='tanh',return_sequences=False,
               dropout=0.02, recurrent_dropout=0.3)(gru_layer1)
Dense_1=Dense(13,activation='linear')(gru_layer2)
reshape_3=keras.layers.Reshape((K.int_shape(Dense_1)[1],1))(Dense_1)
concatinate_layer=keras.layers.concatenate([reshape_3, auxiliary_input])
localy_conected=keras.layers.LocallyConnected1D(1, kernel_size=1, padding='valid')(concatinate_layer)
fllaten=Flatten()(localy_conected)
Dense_2 = Dense(600, activation='linear')(fllaten)
main_output = Dense(output_station,activation='linear', name='main_output')(Dense_2)
model=Model(inputs=[main_input, auxiliary_input] ,outputs=[main_output])
model.compile(loss='mae',optimizer='Adagrad')
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=35, verbose=1)
history=model.fit(x=[train_x, auxiliary_input_train],y=train_y,validation_split= 0.3,
                  epochs=800,batch_size=32,callbacks=[early_stop],verbose= 1,shuffle=False)    
pyplot.plot(history.history['loss'],'b', label='training history')
pyplot.plot(history.history['val_loss'],'r',label='testing history')
pyplot.title("Train and Test Loss function")
pyplot.legend()
pyplot.show()
yhat = model.predict([test_x,auxiliary_input_test])
y_pred = scaler.inverse_transform(yhat)
y_true = test_y
#=============================================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
for i in range(0,13):
    print('station:',i)
    MAPE=mean_absolute_percentage_error(y_true[:,i], y_pred[:,i])
    print('MAPE:' ,round(MAPE,2))
    mae=mean_absolute_error(y_true[:,i], y_pred[:,i])
    print('MAE:',round(mae,2))
    rmse = sqrt(mean_squared_error(y_true[:,i], y_pred[:,i]))
    print('RMSE: ', round(rmse,2))
    r2=r2_score(y_true[:,i], y_pred[:,i])
    r=np.corrcoef(y_true[:,i], y_pred[:,i])
    r2=r2_score(y_true[:,i], y_pred[:,i])
    print('R-squared:',round(r2,2))
    print('r coef:',round(r[1,0],2))
    print('*******')
print(' ')    
print('total accuracy af all stations: ')
MAPE=mean_absolute_percentage_error(y_true, y_pred)
print('MAPE:' ,round(MAPE,2) )
mae=mean_absolute_error(y_true, y_pred)
print('MAE:',round(mae,2))
rmse = sqrt(mean_squared_error(y_true, y_pred))
print('RMSE: ', round(rmse,2))
r2=r2_score(y_true, y_pred)
print('R-squared:',round(r2,4))
R=np.corrcoef(y_true[:,0], y_pred[:,0])+np.corrcoef(y_true[:,1], y_pred[:,1])+np.corrcoef(y_true[:,2], y_pred[:,2])+np.corrcoef(y_true[:,3], y_pred[:,3])+np.corrcoef(y_true[:,4], y_pred[:,4])+np.corrcoef(y_true[:,5], y_pred[:,5])+np.corrcoef(y_true[:,6], y_pred[:,6])+np.corrcoef(y_true[:,7], y_pred[:,7])+np.corrcoef(y_true[:,8], y_pred[:,8])+np.corrcoef(y_true[:,9], y_pred[:,9])
R=R/10
print('R ceef:',round(R[1,0],3))
#=======================================================================================
fig, axs = pyplot.subplots(5, 3,figsize=(11,7),sharex=True, sharey=False,gridspec_kw={'hspace': 0.3})
axs[0,0].plot(y_true[750:1000,0],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.5,
          label = 'Observed PM$_{2.5}$')
axs[0,0].plot(y_pred[750:1000,0],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[0,0].set_title('Rey', loc='left',fontsize=8,y=0.78,x=0.9)
axs[0,0].tick_params(labelsize=9)
axs[0,0].set_title('The Prediction of PM$_{2.5}$')

axs[0,1].plot(y_true[750:1000,1],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[0,1].plot(y_pred[750:1000,1],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[0,1].set_title('sh_mantaghe4', loc='left',fontsize=8,y=0.78,x=0.7)
axs[0,1].tick_params(labelsize=9)
axs[0,1].set_title('The Prediction of PM$_{2.5}$')

axs[0,2].plot(y_true[750:1000,2],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[0,2].plot(y_pred[750:1000,2],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[0,2].set_title('Pirouzi', loc='left',fontsize=8,y=0.78,x=0.85)
axs[0,2].tick_params(labelsize=9)
axs[0,2].set_title('The Prediction of PM$_{2.5}$')

axs[1,0].plot(y_true[750:1000,3],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[1,0].plot(y_pred[750:1000,3],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[1,0].set_title('Ponak', loc='left',fontsize=8,y=0.78,x=0.85)
axs[1,0].tick_params(labelsize=9)

axs[1,1].plot(y_true[250:550,4],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[1,1].plot(y_pred[250:550,4],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[1,1].set_title('Golbarg', loc='left',fontsize=8,y=0.78,x=0.82)
axs[1,1].tick_params(labelsize=9)

axs[1,2].plot(y_true[750:1000,5],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[1,2].plot(y_pred[750:1000,5],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[1,2].set_title('Setad bohran', loc='left',fontsize=8,y=0.78,x=0.72)
axs[1,2].tick_params(labelsize=9)

axs[2,0].plot(y_true[750:1000,6],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[2,0].plot(y_pred[750:1000,6],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[2,0].set_title('sh_mantaghe21', loc='left',fontsize=8,y=0.78,x=0.68)
axs[2,0].tick_params(labelsize=9)

axs[2,1].plot(y_true[750:1000,8],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[2,1].plot(y_pred[750:1000,8],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[2,1].set_title('Masudiye', loc='left',fontsize=8,y=0.78,x=0.8)
axs[2,1].tick_params(labelsize=9)

axs[2,2].plot(y_true[750:1000,7],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[2,2].plot(y_pred[750:1000,7],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[2,2].set_title('Shadabad', loc='left',fontsize=8,y=0.78,x=0.8)
axs[2,2].tick_params(labelsize=9)

axs[3,0].plot(y_true[750:1000,9],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[3,0].plot(y_pred[750:1000,9],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[3,0].set_title('Sharif', loc='left',fontsize=8,y=0.78,x=0.82)
axs[3,0].tick_params(labelsize=9)

axs[3,1].plot(y_true[250:550,10],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[3,1].plot(y_pred[250:550,10],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[3,1].set_title('Modares', loc='left',fontsize=8,y=0.78,x=0.81)
axs[3,1].tick_params(labelsize=9)

axs[3,2].plot(y_true[750:1000,11],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[3,2].plot(y_pred[750:1000,11],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[3,2].set_title('Sadr', loc='left',fontsize=8,y=0.78,x=0.86)
axs[3,2].tick_params(labelsize=9)

axs[4,0].plot(y_true[750:1000,12],linewidth=0.7,marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=3,linestyle='--',c='r',markeredgewidth=0.6,
          label = 'Observed PM$_{2.5}$')
axs[4,0].plot(y_pred[750:1000,12],c='black',linewidth=1,label = 'Predicted PM$_{2.5}$')
axs[4,0].set_title('Aghdasiye', loc='left',fontsize=8,y=0.78,x=0.79)
axs[4,0].tick_params(labelsize=9)
fig.delaxes(axs[4,1])
fig.delaxes(axs[4,2])
for ax in axs.flat:
    ax.set(xlabel='Time interval (hours)', ylabel='PM$_{2.5}$ $(μg/{m}^3)$')
#pyplot.xlabel('Time interval (hours)',fontsize=13,y=2.5,x=-0.7)
#pyplot.ylabel('PM$_{2.5}$ concentration',fontsize=13,y=2,x=-20)
pyplot.xlim([0,250])
pyplot.legend(bbox_to_anchor=(1.2, 0.9),prop={'size': 15})
pyplot.subplots_adjust(top=0.95,left = 0.05,right = 0.98)
pyplot.tight_layout()
pyplot.savefig("picturenew.jpg",dpi=500)

#===============================================================================
from matplotlib import pyplot as plt
fig, axes= plt.subplots(figsize=(5,4))
plt.plot(y_pred[1000:1500,5],linewidth=1, marker='o', markeredgecolor='r',
          markerfacecolor='none',markersize=5,linestyle='--',c='r',markeredgewidth=0.6)
plt.plot(y_true[1000:1500,5],c='black',linewidth=0.8)
plt.xlabel('Time interval(hour)',fontsize=15)
plt.ylabel('PM2.5 concentration',fontsize=15)
axes.tick_params(direction='out', length=6, width=2, colors='k',labelsize=17,
               grid_color='k', grid_alpha=0.5)
plt.legend(["Observed PM$_{2.5}$ $(μg/{m}^3)$", "Predicted PM$_{2.5}$ $(μg/{m}^3)$"],
            loc='best',fontsize='x-large')
plt.axis([0, 500, 0, 150])
plt.tick_params(labelsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("pic2.png",dpi=100)
#============================================
#scatter plot
from scipy.stats import gaussian_kde,linregress
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
y_true=y_true.flatten() 
y_pred=y_pred.flatten() 
fig, ax = plt.subplots(figsize=(7, 5))
slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
line = slope*y_true+intercept
plt.plot(y_true, line, 'black', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
xy = np.vstack([y_true, y_pred])
z = gaussian_kde(xy)(xy)
plt.scatter(y_true, y_pred,edgecolors='none',c= z,cmap='jet',s=10)
plt.text(100,20,'R-squared = %0.2f' % r2,fontsize=14)
line = mlines.Line2D([0, 1], [0, 1], color='r',linestyle='--')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.colorbar()
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.xlabel('Observed PM$_{2.5}$ $(μg/{m}^3)$',fontsize=15)
plt.ylabel('Predicted PM$_{2.5}$ $(μg/{m}^3)$',fontsize=15)
plt.legend(fontsize=14,loc='lower right')
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig("scatter.png",dpi=300) 
#=============================================================================
