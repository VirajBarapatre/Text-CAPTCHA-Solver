# model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,
    Reshape, Activation, Add, LeakyReLU, GlobalAveragePooling2D, SpatialDropout2D
)

def residual_block(x, filters, kernel_size=3):
    """
    Create a residual block with given filters and kernel size
    """
    shortcut = x
    
    # First convolution layer
    x = Conv2D(
        filters, 
        kernel_size, 
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Second convolution layer
    x = Conv2D(
        filters, 
        kernel_size, 
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(
            filters, 
            1, 
            padding='same',
            kernel_initializer='he_normal'
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add skip connection
    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def build_model(input_shape=(80, 200, 1), num_chars=6, num_classes=36):
    """
    Build and return the model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)
    
    # First block
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    
    # Second block
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.3)(x)
    
    # Third block
    x = residual_block(x, 256)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.3)(x)
    
    # Fourth block
    x = residual_block(x, 512)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    x = Dense(num_chars * num_classes)(x)
    x = Reshape((num_chars, num_classes))(x)
    outputs = Activation('softmax')(x)
    
    # Create and return model
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # Test model building
    model = build_model()
    model.summary()