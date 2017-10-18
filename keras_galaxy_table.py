import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import keras.utils as utils
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import plot_model

# Parameters ======================================
layer_nodes=[100]
act_func='relu'

# preprocessing data ===============================
# fix seed to make this code repeatable
np.random.seed(10)

# load mnist data
#(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()

label_data = np.loadtxt("./real_01/graph_label.txt", unpack=False)
x_train_image = np.array([]).reshape(0, 150)
y_train_label = np.array(label_data[:800])
x_test_image = np.array([]).reshape(0, 150)
y_test_label = np.array(label_data[800:1001])


for i in range(800):
    data = np.loadtxt("./real_01/graph" + str(i) + ".txt", unpack=False)
    x_train_image = np.vstack((x_train_image, np.array(data)))  

#print(x_train_image[-1])
    

for i in range(800,1000):
    data = np.loadtxt("./real_01/graph" + str(i) + ".txt", unpack=False)
    x_test_image = np.vstack((x_test_image, np.array(data))) 


# reshape to rows and normalize to 0~1
x_train=x_train_image.reshape(-1,22500).astype('float32')/9  # (-1, 784) 打-1表示column reshape成一條784，-1fit剩下的
x_test=x_test_image.reshape(-1,22500).astype('float32')/9

# change label to binary representation
y_train=utils.to_categorical(y_train_label)
y_test=utils.to_categorical(y_test_label)

# build model =======================================
model=Sequential()
# construct hidden layer
for n,nodes in enumerate(layer_nodes):
  if n==0:
    model.add(Dense(units=nodes,input_dim=22500,kernel_initializer='normal',activation=act_func))
  else:
    model.add(Dense(units=nodes,kernel_initializer='normal',activation=act_func))
# output layer
model.add(Dense(units=2,kernel_initializer='normal',activation='softmax'))
print('* Model Summary -------------')
print(model.summary())
#plot_model(model, to_file='model.png')  # 必須安裝pydot且import環境變數

# Train model ======================================
model.compile(loss='categorical_crossentropy',
  optimizer='adam', metrics=['accuracy'])

# https://keras.io/models/model/#fit
# batch_size: Integer or None. Number of samples per gradient update. If unspecified, it will default to 32.
# epochs: Integer, the number of times to iterate over the training data arrays.
# verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch.
train_history=model.fit(x=x_train,y=y_train,validation_split=0.2, 
                        epochs=10, batch_size=200,verbose=2)  # 一個batch選200個出來train，epoch代表完整train一輪

# save train result, to load it, use: model = load_model('model.h5')
model.save('model.h5')

# plot train results ===============================
def show_train_history(train_history,train_item,valid_item):
    plt.plot(train_history.history[train_item])
    plt.plot(train_history.history[valid_item])
    plt.title('Train History')
    plt.ylabel(train_item)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

# prediction ========================================
prediction=model.predict_classes(x_test)
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10)         
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
plot_images_labels_prediction(x_test_image,y_test_label,
                              prediction,idx=340)

# confusion matrix =============================

datatab=pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])
print(datatab)

df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
print(df[:2])
print(df[(df.label==5)&(df.predict==3)])

plot_images_labels_prediction(x_test_image,y_test_label
                              ,prediction,idx=340,num=1)
plot_images_labels_prediction(x_test_image,y_test_label
                              ,prediction,idx=1289,num=1)