# The allowed time for parking vehicle is taken as 30 seconds here.
#Some modification still need to be made. In the case of one way traffic violation, The stopped vehicle in the right direction also detected as a rule violated vehicle. So the measurement of centroid value should be considered in the case of traffic violation(Since the centroid of the stopped vehicle does not chnage much).
#This is applicable for short duration videos as the given tolerance range is not suitable for long duration videos.
#Future scope: This method will be suitable for long duration videos if the tolerance range can change automatically.
               # More traffic violation can be detected using a a single system.
               # The involvement of centroid valuefor detecting one way traffic violation will make the system effective.
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('One_way_traffic', True, 'check one way traffic violation')
flags.DEFINE_boolean('Stop_checking', True, 'check the stopping of vehicle')
# Python code to find the maximum value
def maximum(k,max):
  for i in k:
      if int(i)>max:
          max=int(i)
  return max
#Python code to convert the coordinates into center and width coordinates
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes

# Python code to merge dict using update() method
def Merge(dict1, dict2):
    return(dict2.update(dict1))





def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    true_list=[]
    tot_count=0
    ref_list = []
    lst_count=0
    dict1={}
    Negative_route_id = []
    Positive_route_id = []




    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        total_number=0
        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car','truck','motorbike','bicycle','bus']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        repeat_time=0
        # update tracks
        total_number=0
        repeat_time = 0
        Dict={}

        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            rtol = 0.081
            atol = 1
            width=int(bbox[2])-int(bbox[0])
            height=int(bbox[3])-int(bbox[1])
            trackid = str(track.track_id)
            centroid = (int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2
            print("centroid{}".format(centroid))
            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),class_name, (int(bbox[0]), int(bbox[1]),int(bbox[2]),int(bbox[3]))))
            #To check if the vehicle is stopped or not
            b=str(track.track_id)
            if not Dict:
                    Dict = dict({b: [class_name,[int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])],width,height]})
            else:
                    Dict[b] = [class_name,[int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])],width,height]

        print(Dict)
        reqd_dict=Dict
        reqd_dict_sort = sorted(reqd_dict.items())
        sorted_dict = dict(reqd_dict_sort)
        if Dict!={}:
            try:
                dict1=dup_dict
            except:
                print("No dup dict")
            dup_dict=Dict
            Merge(dict1,dup_dict)
            sort = sorted(dup_dict.items())
            new_dict = dict(sort)
            ref_array=np.array([])
            if len(Dict)>0:
                lst_count=lst_count+1
            k=maximum(new_dict,0)
            for i in range(k):
                key_to_lookup = str(i + 1)
                if key_to_lookup in new_dict:
                    if len(ref_array) == 0:
                        ref_array = np.array(new_dict[key_to_lookup][1])
                    else:
                        ref_array = np.append(ref_array, new_dict[key_to_lookup][1])

                else:
                    random_array = np.random.randint(10, size=(4))
                    ref_array = np.append(ref_array, random_array)

            print("Reference_array{}".format(ref_array))
            for x in ref_array:
                total_number = total_number + 1
            count=total_number
            split_parts=int(count/4)
            a = np.split(ref_array, split_parts)

        try:
            if lst_count>1:
                for y in sorted_dict:
                    k=str(y)
                    m=int(y)-1
                    vehicle_name = sorted_dict[k][0]
                    trackid = int(k)
                    box0 = sorted_dict[k][1][0]
                    box1 = sorted_dict[k][1][1]
                    first_width=int(new_dict[k][2])
                    changed_width=int(sorted_dict[k][2])
                    first_height=int(new_dict[k][3])
                    changed_height=int(sorted_dict[k][3])
                    # Vehicle stop checking
                    second_array = np.array(sorted_dict[k][1])
                    print("a[m] {}".format(a[m]))
                    print("second_array{}".format(second_array))
                    compare = np.allclose(a[m], second_array, rtol, atol)
                    print(compare)
                    if compare == True:
                        true_count = 0
                        true_count = true_count + 1
                        if not true_list:
                            true_list.append(true_count)
                        else:
                            try:
                                if true_list[m]:
                                    true_list[m] = true_list[m] + true_count
                            except:
                                true_list.append(true_count)

                        if true_list[m] >(fps*30):
                            print("Track_id {} is stopped".format(int(k)))
                            cv2.putText(frame, vehicle_name + "  is stopped " , (int(box0), int(box1 - 30)), 0, 0.75, (255, 255, 255), 2)
                    print("true_list{}".format(true_list))
                    if FLAGS.one_way_traffic:
                            if frame_num%20==0:
                                  if changed_width>first_width or changed_height > first_height and true_list[m]>=frame_num :
                                      pass
                                  elif  changed_width>first_width or changed_height > first_height:
                                         print("The vehicle corresponding to the track_id is driving on the wrong way")
                                         Negative_route_id.append(int(k))
                                         if true_list[m]>60:
                                             Negative_route_id.remove(int(k))
                                         print("neg list{}".format(Negative_route_id))
                    for i in range(len(Negative_route_id)):
                        print("Track_id{} is violated".format(Negative_route_id[i]))
                        cv2.putText(frame, vehicle_name + "  is driving on the wrong way ",
                                    (int(box0), int(box1 - 40)), 0, 0.75,
                                    (255, 255, 255), 2)
        except:
            print("No object is tracked till now")
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if 0xFF == ord('q'): break

  

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
