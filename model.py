from keras.callbacks import CSVLogger
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

TRAIN_PATH = './X_Full'
MASK_PATH = './Y_Full'

# DATA AUGMENTATION STEP
train_dataGen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

train_image_generator = train_dataGen.flow_from_directory(TRAIN_PATH, batch_size=1, subset="training", class_mode=None,
                                                          target_size=(512, 512), color_mode='grayscale')
train_mask_generator = train_dataGen.flow_from_directory(MASK_PATH, batch_size=1, subset="training", class_mode=None,
                                                         target_size=(512, 512), color_mode='grayscale')
val_image_generator = train_dataGen.flow_from_directory(TRAIN_PATH, batch_size=1, subset="validation", class_mode=None,
                                                        target_size=(512, 512), color_mode='grayscale')
val_mask_generator = train_dataGen.flow_from_directory(MASK_PATH, batch_size=1, subset="validation", class_mode=None,
                                                       target_size=(512, 512), color_mode='grayscale')

batch_size = 4


def my_gen(train_gen, mask_gen):
    while True:
        try:
            if (next(train_gen) is not None and next(mask_gen) is not None):
                data = next(train_gen)
                labels = next(mask_gen)
            yield data, labels
        except:
            pass


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# U-Net
s = Lambda(lambda x: x / 255)(inputs)

c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(s)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
c2 = BatchNormalization()(c2)
c2 = Activation('relu')(c2)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
c3 = BatchNormalization()(c3)
c3 = Activation('relu')(c3)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
c4 = BatchNormalization()(c4)
c4 = Activation('relu')(c4)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
c5 = BatchNormalization()(c5)
c5 = Activation('relu')(c5)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

image_number = 13368
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # , metrics=[]) #mean_iou
model.summary()
earlyStopper = EarlyStopping(patience=5, verbose=1)
csv_logger = CSVLogger('./trainingLog.csv', append=True, separator=';')

checkpoints = ModelCheckpoint(filepath='drive/Segmentation/model-1.h5', monitor='val_acc', verbose=1,
                              save_best_only=False, save_weights_only=False,
                              mode='auto', period=2)
results = model.fit_generator(my_gen(train_image_generator, train_mask_generator),
                              validation_data=my_gen(val_image_generator, val_mask_generator),
                              validation_steps=(image_number // batch_size),
                              steps_per_epoch=(image_number // batch_size),
                              epochs=100, use_multiprocessing=True, workers=8,
                              callbacks=[csv_logger, checkpoints], shuffle=True)
