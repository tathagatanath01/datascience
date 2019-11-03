from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

df = pd.read_csv("./miml_dataset/miml_labels_2.csv")
df["labels"]=df["labels"].apply(lambda x:x.split(","))



datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train,test = train_test_split(df,test_size=0.02,random_state=0)

from sklearn.model_selection import train_test_split
train_v, validation = train_test_split(train,test_size=0.022,random_state=0)


train_generator=datagen.flow_from_dataframe(
dataframe=train_v,
directory="./miml_dataset/images",
x_col="Filenames",
y_col="labels",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
classes=["desert", "mountains", "sea", "sunset", "trees"],
target_size=(100,100))

valid_generator=test_datagen.flow_from_dataframe(
dataframe=validation,
directory="./miml_dataset/images",
x_col="Filenames",
y_col="labels",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
classes=["desert", "mountains", "sea", "sunset", "trees"],
target_size=(100,100))

test_generator=test_datagen.flow_from_dataframe(
dataframe=test,
directory="./miml_dataset/images",
x_col="Filenames",
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(100,100))

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(train_generator,
                         steps_per_epoch = 1916,
                         epochs = 1,
                         validation_data = valid_generator,
                         validation_steps = 44)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


test_generator.reset()
pred=classifier.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

pred_bool = (pred >0.5)

predictions=[]
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
for row in pred_bool:
    l=[]
    for index,cls in enumerate(row):
        if cls:
            l.append(labels[index])
    predictions.append(",".join(l))
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)