import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

train_loc=r'E:\KaaM\CastingProduct_inspection\casting_data\train'

data={'image':[],
      'label':[]}

for i in os.listdir(train_loc):
    #print(i)
    path=os.path.join(train_loc, i)
    #print(path)
    
    for j in os.listdir(path):
        data['image'].append(os.path.join(path, j))
        data['label'].append(i)

df=pd.DataFrame(data, index=None)

#print(df.head())

count=df['label'].value_counts().reset_index()
#print(count)

sns.countplot(df['label'])

#print(df.isna().sum())

from sklearn.model_selection import train_test_split

train,test=train_test_split(df, test_size=0.2, random_state=42)
new_train,val=train_test_split(train, test_size=0.2, random_state=42)


# print(f'train shape: {new_train.shape}')
# print(f'test shape: {test.shape}')
# print(f'validation shape: {val.shape}')

from keras.preprocessing.image import ImageDataGenerator

train_gen=ImageDataGenerator('''rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, brightness_range=None, 
    shear_range=0.3, zoom_range=0.1''',rescale=1/255)
test_gen=ImageDataGenerator(rescale=1/255)
val_gen=ImageDataGenerator(rescale=1/255)


train_data=train_gen.flow_from_dataframe(new_train, x_col='image', y_col='label',
                                         batch_size=32, class_mode='categorical',
                                         target_size=(300,300))

test_data=test_gen.flow_from_dataframe(test, x_col='image', y_col='label',
                                         batch_size=32, class_mode='categorical',
                                         target_size=(300,300), shuffle=(False))

val_data=val_gen.flow_from_dataframe(val, x_col='image', y_col='label',
                                         batch_size=32, class_mode='categorical',
                                         target_size=(300,300), shuffle=(False))

from keras.models import Sequential
from keras .layers import Dropout, MaxPool2D, Conv2D, Flatten, Dense

model=Sequential()
model.add(Conv2D(32,(3,3), padding='same', activation='relu', input_shape=(300,300,3)))
model.add(MaxPool2D((2,2), strides=2))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3), padding='same', activation='relu', input_shape=(300,300,3)))
model.add(MaxPool2D((2,2), strides=2))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3), padding='same', activation='relu', input_shape=(300,300,3)))
model.add(MaxPool2D((2,2), strides=2))
model.add(Dropout(0.4))

model.add(Conv2D(128,(3,3), padding='same', activation='relu', input_shape=(300,300,3)))
model.add(MaxPool2D((2,2), strides=2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(2, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint=ModelCheckpoint(r'E:\KaaM\CastingProduct_inspection\models\model_1.h5',
                           monitor='val_loss',
                           mode='min',
                           save_best_only=True,
                           verbose=1)

earlystop=EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=0,
                        verbose=1,
                        restore_best_weights=True)

callbacks=[checkpoint,earlystop]


history=model.fit(train_data, validation_data=test_data, epochs=50, callbacks=callbacks, batch_size=32)


plt.plot(history.history['val_accuracy'], label='val_accuracy', color='lightpink')
plt.plot(history.history['accuracy'], label='accuracy', color='k')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


plt.plot(history.history['val_loss'], label='val_loss', color='red')
plt.plot(history.history['loss'], label='loss', color='darkgrey')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()


from keras.preprocessing import image

from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

model1=load_model(r'E:\KaaM\CastingProduct_inspection\models\model_98.h5')

predictions=model1.predict(test_data)
preds=np.argmax(predictions, axis=1)

print(preds)
#print(preds)

labels = train_data.class_indices
print(labels)

labels = dict((v,k) for k,v in labels.items())
labels

preds = [labels[k] for k in preds]
preds

print(test.shape)
  
matrix = confusion_matrix(test.label , preds)
print(matrix)

sns.heatmap(matrix, annot=True)


sns.heatmap(matrix, annot=True)

print(classification_report(test.label, preds))


classes = list(train_data.class_indices.keys())
print(classes)

test_image = image.load_img(r'E:\KaaM\CastingProduct_inspection\cast_ok_0_989.jpeg', target_size = (300,300))
plt.axis("off")
plt.imshow(test_image)
plt.show()
 
test_image = image.img_to_array(test_image)/255
test_image = np.expand_dims(test_image, axis=0)

predicted_array = model1.predict(test_image)
predicted_value = labels[np.argmax(predicted_array)]
print(predicted_value)
predicted_accuracy = round(np.max(predicted_array) * 100, 2)
print(predicted_accuracy)





