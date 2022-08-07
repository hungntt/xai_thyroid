import base64
import json
import cv2
import tensorflow.compat.v1 as tf
import time
import warnings
import logging
from datetime import datetime
from RMQ import BasicRMQClient

now = datetime.now()
warnings.filterwarnings('ignore')

logname = '/logs/log-{}.log'.format(now.strftime("%Y-%m-%d"))
logging.basicConfig(filename=logname,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging.info('=' * 10 + ' LOG FILE FOR ' + '=' * 10)


def get_config(path_config):
    with open(path_config, 'r') as fin:
        config_model = json.load(fin)
    return config_model


# default_model = None
config_models = get_config('/config/model_config.json')
config_rmq = get_config('/config/server_config.json')

rmq_server = config_rmq['rmq_server']
rmq_port = config_rmq['rmq_port']
rmq_user = config_rmq['rmq_user']
rmq_password = config_rmq['rmq_password']

rmq_virtual_host = config_rmq['rmq_virtual_host']
rmq_source_queue = config_rmq['rmq_source_queue']
rmq_completed_exchange = config_rmq['rmq_completed_exchange']


def get_model(model_path):
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.GFile(model_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')

            img_input = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')

            sess = tf.Session(graph=graph)
    return sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes


def draw(image, boxs, color=(255, 0, 0), thickness=2, predict=False):
    for b in boxs:
        logging.info(b)
        start_point, end_point = (b[1], b[2]), (b[3], b[4])
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def detector(file_path, model_id, score_threshold=0.3, is_draw=False):
    img = cv2.imread(file_path, 1)  # 1 is color
    img_original = img.copy()
    h_img, w_img = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.reshape(1, img.shape[0], img.shape[1], 3)  # reshape of (1,w,h,3). Channels = 3
    boxs = []
    global default_model
    if model_id != default_model:
        for model in config_models:
            if model['model_id'] == model_id:
                logging.info('======== Switch mode {} ========'.format(model_id))
                global sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes
                sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
                        model['model_path'])
                feed_dict = {img_input: image}
                y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run(
                        [detection_boxes, detection_scores, num_detections, detection_classes],
                        feed_dict=feed_dict)
                default_model = model_id
                break
    else:
        feed_dict = {img_input: image}
        y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run(
                [detection_boxes, detection_scores, num_detections, detection_classes],
                feed_dict=feed_dict)

    for i in range(int(y_p_num_detections[0])):

        if y_p_scores[0][i] > score_threshold:
            logging.info(y_p_classes[0][i])
            logging.info(y_p_scores[0][i])
            logging.info(y_p_boxes[0][i])
            x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
            y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
            boxs.append((y_p_scores[0][i], x1, y1, x2, y2))
    if is_draw:
        img_draw = draw(image=img_original, boxs=boxs, predict=True)
        return boxs, img_draw
    else:
        return boxs, img_original


def callback_on_message(ch, method, properties, body):
    try:
        time_start = time.time()
        # byte array to bitmap
        str_json = body.decode('utf-8')
        data_total = json.loads(str_json.replace("'", '"'))
        data = data_total['message']
        file_path = data['image_path']
        model_id = config_models[0]['model_id']
        try:
            model_id = data['model_id']
        except:
            logging.error('Not selected model')
        try:
            content = data['content']
            image_64_decode = base64.b64decode(content)
            image_result = open('image.jpg', 'wb')  # create a writable image and write the decoding result
            image_result.write(image_64_decode)
            file_path = 'image.jpg'
        except:
            logging.error('Not encode to base64')
        boxs, image_draw = detector(file_path, model_id)
        logging.info(boxs)
        # Display the results
        data = {
            "bounding_boxes": [],
            "image_path": data['image_path'],
            "success": "true",
            'image_id': data['image_id']
        }
        id = 0
        for (conf, left, top, right, bottom) in boxs:
            object_detect = {
                "ObjectClassName": "thyroid_cancer",
                "confidence": conf,
                "coordinates": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom
                },
                "ObjectClassId": id
            }
            data['bounding_boxes'].append(object_detect)
            id += 1
        data_total['messageType'] = ['urn:message:{}'.format(rmq_completed_exchange)]
        data_total["destinationAddress"] = "rabbitmq://{}/DataClasses.Messages:FrameCompleted".format(rmq_server)
        data_total['message'] = data
        json_str = str(data_total)
        end_time = time.time()
        logging.info('=========> Time processing: {}s'.format(end_time - time_start))
        ch.basic_ack(delivery_tag=method.delivery_tag)
        rmq_client.publish_exchange(ch, rmq_completed_exchange, json_str)
    except:
        logging.error('========= Error =============')


sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
        config_models[0]['model_path'])
default_model = config_models[0]['model_id']
image = cv2.imread('data/samples/20653934_AI_14_NGUYEN_THI_HUONG_1970_20200529105316651.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = img.reshape(1, img.shape[0], img.shape[1], 3)
feed_dict = {img_input: image}
y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run(
        [detection_boxes, detection_scores, num_detections, detection_classes],
        feed_dict=feed_dict)
# Create RMQ client
rmq_client = BasicRMQClient(rmq_server, rmq_port, rmq_user, rmq_password, rmq_virtual_host)
# start processing messages from the rmq_source_queue
rmq_client.process(callback_on_message, rmq_source_queue)
