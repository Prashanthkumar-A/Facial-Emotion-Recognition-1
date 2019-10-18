#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the required libraries
import pandas as pd
import numpy as np
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
import sys, os
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils


# In[2]:



# reading csv to dataframe
df=pd.read_csv('fer2013.csv')

# print(df.info())
# print(df["Usage"].value_counts())
# print(df.head())



x_train,y_train,x_test,y_test=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            x_train.append(np.array(val,'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            x_test.append(np.array(val,'float32'))
            y_test.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")


num_features = 64
num_labels = 7
batch_size = 32
epochs = 10
width, height = 48, 48


x_train = np.array(x_train,'float32')
y_train = np.array(y_train,'float32')
x_test = np.array(x_test,'float32')
y_test= np.array(y_test,'float32')

y_train=np_utils.to_categorical(y_train, num_classes=num_labels)
y_test=np_utils.to_categorical(y_test, num_classes=num_labels)


#normalizing data between oand 1
x_train -= np.mean(x_train, axis=0)
x_train /= np.std(x_train, axis=0)

x_test -= np.mean(x_test, axis=0)
x_test /= np.std(x_test, axis=0)

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)

x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)


# # Creating the CNN Model

# In[3]:


#1st convolution layer
model = Sequential()
l2_regularization=0.01
regularization = l2(l2_regularization)

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',kernel_regularizer=regularization, input_shape=(x_train.shape[1:]),padding='same'))
model.add(Conv2D(64,kernel_size= (5,5), activation='relu',kernel_regularizer=regularization,padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (5,5), activation='relu',kernel_regularizer=regularization,padding='same'))
model.add(Conv2D(64, (5,5), activation='relu',kernel_regularizer=regularization,padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer=regularization,padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer=regularization,padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Activation('softmax'))

# model.summary()


# In[4]:




#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(1e-4,beta_1=0.95),
              metrics=['accuracy'])


# In[5]:


#saving the model at each epoch
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

patience=50
base_path='models/'
# callbacks
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


# # Training the Model

# In[6]:



#Training the model
model.fit(x_train,y_train,verbose=1,batch_size=64,epochs=60,callbacks=callbacks,
          validation_data=(x_test,y_test))




#print('saving whole model')
#Saving the  model to  use it later on
#fer_json = model.to_json()
#with open("model.json", "w") as json_file:
 #   json_file.write(fer_json)
#model.save_weights("weights.h5")


# In[ ]:




