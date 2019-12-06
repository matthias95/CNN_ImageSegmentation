# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def getBuildInResNet50(trainable=False, pretrained=True):
    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=('imagenet' if pretrained else None)) # 'imagenet'
    
    for layer in resnet50.layers:
        layer.trainable = trainable

    layers = [ resnet50.get_layer('conv2_block3_out'), resnet50.get_layer('conv3_block4_out'), resnet50.get_layer('conv4_block6_out'), resnet50.get_layer('conv5_block3_out')]
    return resnet50, layers
  

def residualBlock(inputs, filters, stride=1):
    with tf.name_scope('residualBlock') as scope:
        convs = tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), strides=(stride,stride), padding='SAME'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='SAME'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1,1), strides=(1,1), padding='SAME'),
          tf.keras.layers.BatchNormalization()
        ])


        if (stride != 1) or (inputs.shape[-1] != (filters * 4)):
            skip = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1,1), strides=(stride,stride), padding='SAME')(inputs)
            skip = tf.keras.layers.BatchNormalization()(skip)
            return tf.keras.layers.ReLU()(convs(inputs) + skip)

        return tf.keras.layers.ReLU()(convs(inputs) + inputs)

def residualBlockV2(inputs, filters, stride=1):

    x = inputs
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), strides=(stride,stride), padding='SAME')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1,1), strides=(1,1), padding='SAME')(x)



    if (stride != 1) or (inputs.shape[-1] != (filters * 4)):
        return x + tf.keras.layers.Conv2D(filters=filters * 4, kernel_size=(1,1), strides=(stride,stride), padding='SAME')(inputs)

    return x + inputs


def getResNet50(residualBlock=residualBlock):
    inputs = tf.keras.Input((None, None, 3))

    x = inputs 

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='SAME')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = residualBlock(x, 64)
    x = residualBlock(x, 64)
    x = residualBlock(x, 64)

    x = residualBlock(x, 128, stride=2)
    x = residualBlock(x, 128)
    x = residualBlock(x, 128)
    x = residualBlock(x, 128)

    x = residualBlock(x, 256, stride=2)
    x = residualBlock(x, 256)
    x = residualBlock(x, 256)
    x = residualBlock(x, 256)
    x = residualBlock(x, 256)
    x = residualBlock(x, 256)

    x = residualBlock(x, 512, stride=2)
    x = residualBlock(x, 512)
    x = residualBlock(x, 512)
    
    resnet50 = tf.keras.Model(inputs=inputs, outputs=x)
    
    print('Using: ' + residualBlock.__name__)
    
    if residualBlock.__name__ == 'residualBlock':
        layers = [resnet50.get_layer('re_lu_9'), resnet50.get_layer('re_lu_21'), resnet50.get_layer('re_lu_39'), resnet50.get_layer('re_lu_48')]
    elif residualBlock.__name__ == 'residualBlockV2':
        layers = [resnet50.get_layer('re_lu_10'), resnet50.get_layer('re_lu_22'), resnet50.get_layer('re_lu_40'), resnet50.get_layer('re_lu_48')]
        
    else:
        print('Invalid residualBlock')
        
    return resnet50, layers

def oneByOneConvAndResize(inputs, size):
    diemnsionalityReduction = tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], padding='SAME', activation=None)
    return  tf.image.resize(diemnsionalityReduction(inputs), size, method=tf.image.ResizeMethod.BILINEAR)

def getSegmentationModel(resnet50, layers):
    shape = tf.shape(resnet50.input)

    addedLayers = tf.keras.layers.Add()([oneByOneConvAndResize(layer.output, [shape[1], shape[2]]) for layer in layers])
    addedLayers = tf.reshape(addedLayers, [shape[0],shape[1], shape[2]])
    output = tf.keras.activations.sigmoid(addedLayers)

    segmentationModel = tf.keras.Model(inputs=resnet50.input, outputs={'logits': addedLayers, 'sigmoid': output})
    return segmentationModel

class SegmentationModel(tf.Module):
    def __init__(self, model):
        super(SegmentationModel, self).__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec([None,None,3], tf.float32)])
    def __call__(self, x):
        x = x + np.array([[[-103.939, -116.779, -123.68 ]]], dtype=np.float32)
        x = tf.expand_dims(x,0)
        return self.model(x, training=False)['sigmoid'][0]
