print('Importing Keras')
from keras.models import Model
from keras.layers import concatenate, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

print('Importing TensorFlow')
import tensorflow as tf

print('Importing helper classes')
from yolo_utils import BatchGenerator, parse_annotation, normalize_image, _interval_overlap
from yolo_models import  FullYolo, TinyYolo, custom_loss
import numpy as np
import os
import cv2



input_size = 416
max_box_per_image = 10
anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
labels = ["stop", "pedestrianCrossing", "signalAhead", "keepRight", "speedLimit35", "speedLimit25"]

nb_box = len(anchors)//2
nb_class = len(labels)
class_wt = np.ones(nb_class, dtype='float32')
grid_w = 13
grid_h = 13


#training parameters
train_times = 4
valid_times = 1
nb_epochs = 10000
learning_rate = 1e-4
batch_size = 2
warmup_epochs = 3
object_scale = 5.0
no_object_scale = 1.0
coord_scale = 1.0
class_scale = 1.0
saved_weights_name = 'saved_weights.h5'
warmup_batches = 0 #defined later
debug = False

#data locations
path_images_training = '../LISA/training/images/'
path_annotations_training = '../LISA/training/annotations/'
path_images_valid = '../LISA/validation/images/'
path_annotations_valid = '../LISA/validation/annotations/'


def train(model, train_imgs, valid_imgs): 
    #configure training parameters for Keras
    generator_config = {
        'IMAGE_H'         : input_size, 
        'IMAGE_W'         : input_size,
        'GRID_H'          : grid_h,  
        'GRID_W'          : grid_w,
        'BOX'             : nb_box,
        'LABELS'          : labels,
        'CLASS'           : len(labels),
        'ANCHORS'         : anchors,
        'BATCH_SIZE'      : batch_size,
        'TRUE_BOX_BUFFER' : max_box_per_image,
    }    

    train_generator = BatchGenerator(train_imgs, 
                                     generator_config, 
                                     norm=normalize_image)
    valid_generator = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=normalize_image,
                                     jitter=False)   
                                 
    warmup_batches= warmup_epochs * (train_times*len(train_generator) + valid_times*len(valid_generator))

    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=custom_loss, optimizer=optimizer)


    #export weights every time loss has improved    
    checkpoint = ModelCheckpoint(saved_weights_name, 
                                 monitor='loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)


    #start training process
    model.fit_generator(generator        = train_generator, 
                        steps_per_epoch  = len(train_generator) * train_times, 
                        epochs           = warmup_epochs + nb_epochs, 
                        verbose          = 2 if debug else 1,
                        validation_data  = valid_generator,
                        validation_steps = len(valid_generator) * valid_times,
                        callbacks        = [checkpoint],
                        workers          = 3,
                        max_queue_size   = 8)      
    
    print('Finished training')


def _main_():
    print('Processing training data')
    # parse annotations of the training set
    train_imgs = parse_annotation(path_annotations_training, 
                                  path_images_training, 
                                  labels)

    print('Processing validation data')
    # parse annotations of the validation set
    valid_imgs = parse_annotation(path_annotations_valid, 
                                  path_images_valid, 
                                  labels)

    print('Creating model')
    model = TinyYolo()
    #model = FullYolo()
    model.summary()

    print('Starting training process')
    train(model, train_imgs, valid_imgs)


if __name__ == '__main__':
    _main_()
    
