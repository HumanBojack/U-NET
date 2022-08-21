import tensorflow as tf

# Input image dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Set the layers
## input
input = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
normalized_input = tf.keras.layers.Lambda(lambda x: x / 255)(input) # Change this by dividing after import maybe?


## Contraction
### C1 => 128x128x16
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(normalized_input)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

### C2 => 64x64x32
p1 = tf.keras.layers.MaxPool2D((2, 2), strides=2)(c1)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

### C3 => 32x32x64
p2 = tf.keras.layers.MaxPool2D((2, 2), strides=2)(c2)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

### C4 => 16x16x128
p3 = tf.keras.layers.MaxPool2D((2, 2), strides=2)(c3)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

### C5 => 8x8x256
p4 = tf.keras.layers.MaxPool2D((2, 2), strides=2)(c4)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


## Expansion
### U6 => U6 + C4 (16x16x256)
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
### C6 => 16x16x128
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

### U7 => U7 + C3 (32x32x128)
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
### C7 => 32x32x64
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

### U8 => U8 + C2 (64x64x64)
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
### C8 => 64x64x32
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

### U9 => U9 + C1 (128x128x32)
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=2, padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
### C9 => 128x128x16
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


## output
output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)