# import the needed dependencies
import glob
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

image_names_p1 = glob.glob('data/HAM10000_images_part_1/*.jpg')
train_image_names = [image[image.rindex('/')+1:-4] for image in image_names_p1]

image_names_p2 = glob.glob('data/HAM10000_images_part_2/*.jpg')
testing_image_names = [image[image.rindex('/')+1:-4] for image in image_names_p2]

val_num = int(len(testing_image_names)*0.5)

val_image_names = testing_image_names[0:val_num]
test_image_names = testing_image_names[val_num::]

FilePath = "data/HAM10000_metadata.csv"
df = pd.read_csv(FilePath)

train_df = df.loc[df['image_id'].isin(train_image_names)]
train_df['folder'] =  'data/HAM10000_images_part_1/' + train_df['image_id'] + '.jpg'

val_df = df.loc[df['image_id'].isin(val_image_names)]
val_df['folder'] =  'data/HAM10000_images_part_2/' + val_df['image_id'] + '.jpg'

test_df = df.loc[df['image_id'].isin(test_image_names)]
test_df['folder'] =  'data/HAM10000_images_part_2/' + test_df['image_id'] + '.jpg'


def make_model(learning_rate, droprate, input_shape, inner_layer):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_shape, input_shape, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(input_shape, input_shape, 3))

    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    dense = keras.layers.Dense(inner_layer, activation='relu')(vectors)
    dropout = keras.layers.Dropout(droprate)(dense)
    outputs = keras.layers.Dense(7, activation="linear")(dropout)

    model = keras.Model(inputs, outputs)
    
    learning_rate = learning_rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

full_train = pd.concat([train_df, val_df])

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_full_generator = train_datagen.flow_from_dataframe(
    full_train,
    x_col='folder',
    y_col='dx',
    target_size=(150, 150),
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='folder',
    y_col='dx',
    target_size=(150, 150),
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'skin-lession-class_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
)

# Set up the early stopping callback with a patience of 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

learning_rate = 0.001
size = 25

model = make_model(learning_rate=learning_rate,
                   droprate=0.0, input_shape=150,
                   inner_layer=size)

history = model.fit(train_full_generator, epochs=30,
                    validation_data=test_generator, batch_size=32,
                   callbacks=[early_stopping, checkpoint])
