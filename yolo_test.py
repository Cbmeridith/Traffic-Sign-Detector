import argparse
import os
import cv2
import numpy as np
from collections import namedtuple
from yolo_models import FullYolo, TinyYolo
from yolo_utils import normalize_image, sigmoid, softmax, BoundBox, bbox_iou, _interval_overlap
import datetime
from math import ceil
from configparser import ConfigParser

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#config for demo
class Config:
    def __init__(self, **fields):
        self.__dict__.update(fields)
        
config_app = Config(show_grid=False, enforce_threshold=True, do_suppression=True, do_detection=True)


#decode raw output of model
#source: https://github.com/experiencor/keras-yolo2
def decode_output(netout, anchors, nb_class, labels, conf_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > 0.05#conf_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    
                    boxes.append(box)

    # non-max suppression
    if(config_app.do_suppression):
        for c in range(nb_class):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in range(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                            boxes[index_j].classes[c] = 0
                        
    # remove the boxes which have low confidences
    boxes_final = []
    if config_app.enforce_threshold:
        for box in boxes:
            if(box.get_score() >= conf_threshold):
                boxes_final.append(box)
    else:
        boxes_final = boxes
    
    
    return boxes_final



#run an image through the ML model
def predict(model, image, input_size, max_boxes, anchors, labels):
        image_h, image_w, channels = image.shape
        image = cv2.resize(image, (input_size, input_size))
        image = normalize_image(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,max_boxes,4))

        netout = model.predict([input_image, dummy_array])[0]
        boxes  = decode_output(netout, anchors, len(labels), labels)

        return boxes


#display bounding boxes in feed
def draw_boxes(image, boxes, labels):
    #color each sign differently
    box_colors = {
        'stop': (255,0,0),
        'pedestrianCrossing': (255,255,0),
        'signalAhead': (0,255,0),
        'keepRight': (0,0,255),
        'speedLimit35': (255,0,255),
        'speedLimit25': (0,255,255)
    }
    
    image_h, image_w, _ = image.shape
    
    for box in boxes:
        label_num = box.get_label()
        label_str = labels[label_num]
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), box_colors[label_str], 3)
        cv2.putText(image, 
                    label_str + ' ' + str(box.get_score())[:4], 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    box_colors[label_str], 2)
        
    return image


#draw a grid to divide the feed into a width x height grid 
def draw_grid(image, grid_w, grid_h):
    image_h, image_w, _ = image.shape
    cell_w = int(ceil(image_w / grid_w))
    cell_h = int(ceil(image_h / grid_h))

    x = 0
    y = 0
    while y < image_h:
        while x < image_w:
            cv2.rectangle(image, (x, y), (x + cell_w, y + cell_h), (0,0,0), 1)
            x += cell_w
        y += cell_h
        x = 0

    return image
    

def _main_(args):
    window_title = 'Traffic Sign Detector'

    #read config file variables
    config_model = ConfigParser()
    config_model.read(args.config)

    print(config_model.get('Model', 'Type'))
    print(config_model.get('Model', 'Labels').split(','))

    model_type = config_model.get('Model', 'Type')
    weights = config_model.get('Model', 'Weights')
    input_size = int(config_model.get('Model', 'InputSize'))
    labels = config_model.get('Model', 'Labels').split(',')
    anchors = config_model.get('Model', 'Anchors').split(',')
    grid_w = int(config_model.get('Model', 'GridWidth'))
    grid_h = int(config_model.get('Model', 'GridHeight'))
    max_boxes = int(config_model.get('Model', 'MaxBoxesPerImage'))
    anchors = [float(anchor) for anchor in anchors]

    #create model
    model = None
    if model_type == 'Full':
        model = FullYolo()
    elif model_type == 'Tiny':
        model = TinyYolo()
    else:
        print('Model Type \"' + str(model_type) + '\" not valid. Must be Full or Tiny')

    #load trained weights into model
    model.load_weights(weights)

    
    #open new window with webcam feed
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_title)
    
    #process every camera frame through ML model
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        bounding_boxes = None
        if config_app.do_detection:
            bounding_boxes = predict(model, frame, input_size, max_boxes, anchors, labels)
            frame = draw_boxes(frame, bounding_boxes, labels)
            
        if(config_app.show_grid):
            frame = draw_grid(frame, grid_h, grid_w)
            
        #display frame with boxes drawn on it
        cv2.imshow(window_title, frame)
         
        key = cv2.waitKey(1)
        key_code = key % 256
        if key_code == 27: #esc
            break
        elif key_code == 96 or key_code == 48: #0 toggle detection
            print('Toggling detection: ' + str(not config_app.do_detection))
            config_app.do_detection = not config_app.do_detection
        elif key_code == 97 or key_code == 49: #1, toggle grid
            print('Toggling grid display: ' + str(not config_app.show_grid))
            config_app.show_grid = not config_app.show_grid
        elif key_code == 98 or key_code == 50: #2, toggle conf threshold
            print('Toggling minimum confidence threshold enforcement ' + str(not config_app.enforce_threshold))
            config_app.enforce_threshold = not config_app.enforce_threshold
        elif key_code == 99 or key_code == 51: #3, toggle non-max suppression
            print('Toggling non-max suppression ' + str(not config_app.do_suppression))
            config_app.do_suppression = not config_app.do_suppression
        elif key_code == 32: #space, save snapshot
            snap = 'snapshot-{date:%Y-%m-%d_%H-%M-%S}.png'.format( date=datetime.datetime.now() )
            cv2.imwrite(snap, frame)
            print('Saved snapshot!')
            
    #cleanup
    cam.release()
    cv2.destroyAllWindows()
    exit()


argparser = argparse.ArgumentParser()
argparser.add_argument('-cfg', '--config', required=True)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
