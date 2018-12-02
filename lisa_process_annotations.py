import os
import numpy as np
import cv2
from shutil import copyfile

classes = ['stop', 
           'pedestrianCrossing', 
           'signalAhead', 
           'keepRight', 
           'speedLimit35', 
           'speedLimit25']


path_frames_input_train = ['LISA/LISA_TS/vid0/frameAnnotations-vid_cmp2.avi_annotations/',
                           'LISA/LISA_TS/vid1/frameAnnotations-vid_cmp1.avi_annotations/',
                           'LISA/LISA_TS/vid1/frameAnnotations-vid_cmp1.avi_annotations/',
                           'LISA/LISA_TS/vid3/frameAnnotations-vid_cmp2.avi_annotations/',
                           'LISA/LISA_TS/vid4/frameAnnotations-vid_cmp2.avi_annotations/',
                           'LISA/LISA_TS/vid5/frameAnnotations-vid_cmp2.avi_annotations/',
                           'LISA/LISA_TS/vid6/frameAnnotations-MVI_0071.MOV_annotations/',
                           'LISA/LISA_TS/vid7/frameAnnotations-MVI_0119.MOV_annotations/',
                           'LISA/LISA_TS/vid8/frameAnnotations-MVI_0120.MOV_annotations/',
                           'LISA/LISA_TS/vid9/frameAnnotations-MVI_0121.MOV_annotations/',
                           'LISA/LISA_TS/vid10/frameAnnotations-MVI_0122.MOV_annotations/',
                           'LISA/LISA_TS/vid11/frameAnnotations-MVI_0123.MOV_annotations/']

path_frames_input_validation = ['LISA/LISA_TS_extension/training/2014-04-24_10-59/frameAnnotations-cam1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-04-24_10-59/frameAnnotations-cam1-2.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-04-24_11-43/frameAnnotations-cam1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_10-40/frameAnnotations-cam1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_11-00/frameAnnotations-cam1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_11-40/frameAnnotations-cam1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_16-29/1/frameAnnotations-1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_16-29/2/frameAnnotations-2.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_17-03/1/frameAnnotations-1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-05-01_17-03/2/frameAnnotations-2.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-07-07_17-06/frameAnnotations-1.avi_annotations/',
                                'LISA/LISA_TS_extension/training/2014-07-11_12-12/1/frameAnnotations-1.avi_annotations/']


path_frames_output_train = 'LISA/training/images/'
path_annotations_output_train = 'LISA/training/annotations/'

path_frames_output_validation = 'LISA/validation/images/'
path_annotations_output_validation = 'LISA/validation/annotations/'


class Annotation:
    def __init__(self, raw_ann):

        ann = raw_ann.split(';')
        
        self.filename = ann[0]
        self.class_name = ann[1]
        self.xmin = ann[2]
        self.ymin = ann[3]
        self.xmax = ann[4]
        self.ymax = ann[5]
        self.occluded = ann[6]
        self.on_another_road = ann[7]
        self.origin_filename = ann[8]
        self.frame_number = ann[9]
        #self.origin_track = ann[10]


#convert LISA CSV to XML
def generate_xml(path_video, path_xml, path_image_output):
    path_annotations = path_video + 'frameAnnotations.csv'
    file_csv = open(path_annotations, 'r')
    for line in file_csv:
        ann = Annotation(line)
        if(ann.class_name in classes):
            path_image = path_video + ann.filename
            #get image dimensions
            image = cv2.imread(path_image)
            image_height, image_width, image_depth = image.shape
        
            #convert annotations/image information to XML
            file_xml = open(path_xml + ann.filename + '.xml', 'w')
            file_xml.write('<annotation verified=\"yes\">\n')
            file_xml.write('\t<folder>' + ann.class_name + '</folder>\n')
            file_xml.write('\t<filename>' + ann.filename + '</filename>\n')
            file_xml.write('\t<path>Unspecified</path>\n')
            file_xml.write('\t<source>\n')
            file_xml.write('\t\t<database>' + 'LISA' + '</database>\n')
            file_xml.write('\t</source>\n')
            file_xml.write('\t<size>\n')
            file_xml.write('\t\t<width>' + str(image_width) + '</width>\n')
            file_xml.write('\t\t<height>' + str(image_height) + '</height>\n')
            file_xml.write('\t\t<depth>' + str(image_depth) + '</depth>\n')
            file_xml.write('\t</size>\n')
            file_xml.write('\t<segmented>0</segmented>\n')
            file_xml.write('\t<object>\n')
            file_xml.write('\t\t<name>' + ann.class_name + '</name>\n')
            file_xml.write('\t\t<pose>Unspecified</pose>\n')
            file_xml.write('\t\t<truncated>0</truncated>\n')
            file_xml.write('\t\t<difficult>0</difficult>\n')
            file_xml.write('\t\t<bndbox>\n')
            file_xml.write('\t\t\t<xmin>' + ann.xmin + '</xmin>\n')
            file_xml.write('\t\t\t<ymin>' + ann.ymin + '</ymin>\n')
            file_xml.write('\t\t\t<xmax>' + ann.xmax + '</xmax>\n')
            file_xml.write('\t\t\t<ymax>' + ann.ymax + '</ymax>\n')
            file_xml.write('\t\t</bndbox>\n')
            file_xml.write('\t</object>\n')
            file_xml.write('</annotation>')
            file_xml.close()
            
            #copy image to output folder
            copyfile(path_image, path_image_output + ann.filename) 
    file_csv.close()

#process training annotations
for path_video in path_frames_input_train:
    generate_xml(path_video, path_annotations_output_train, path_frames_output_train)

#process validation annotations
for path_video in path_frames_input_validation:
    generate_xml(path_video, path_annotations_output_validation, path_frames_output_validation)
