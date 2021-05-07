import os
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import cv2 as cv
import numpy as np

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import matplotlib.pyplot as plt
import json, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

WHITE = (255, 255, 255)
img, img0, outputs = None, None, None
img_1, file_1, curr_conf = None, None, 0.5

classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# determine the yolo output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers( )]

UPLOAD_FOLDER, YOLO_FOLDER, MASK_FOLDER, KEYPOINT_FOLDER = './uploads/', './yolo_found/', './mask_rcnn_found/', './keypoint_rcnn_found/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'], app.config['YOLO_FOLDER'], app.config['MASK_FOLDER'] = UPLOAD_FOLDER, YOLO_FOLDER, MASK_FOLDER
app.config["KEYPOINT_FOLDER"] = KEYPOINT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submit_form')
def submit_form():
    return render_template('submit_form.html')

@app.route('/submit_form', methods=['POST'])
def submit_form_post():
    text = request.form['text']
    if len(text) == 0:# it wil stay there till u won't enter anything
    	return redirect("http://127.0.0.1:5000/")
    else:
	    processed_text = text.upper()
	    caption = processed_text

	    if request.form.get('pic_valid')!=None:
	    	return 'test for pics, '+processed_text

	    return processed_text


@app.route('/pic_form')
def pic_form():
	return render_template('pic_upload.html')


@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)

		file = request.files['file']

		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)

		if file and allowed_file(file.filename):# http://127.0.0.1:5000/
			global file_1
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			file_1 = filename# passed to yolo/mask_rcc/keypoint_rcnn(according to what user selects)
			color_result = getDominantColor(image)
			flash('Image successfully uploaded and displayed below')
			return render_template('pic_upload.html', filename=filename, value=str(color_result))
		else:
			flash('Allowed image types are -> png, jpg, jpeg, gif')
			# return 
			redirect(request.url)


@app.route('/uploads/<filename>')
def display_image(filename):
	return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route('/pic_post_keypoint_rcnn', methods=['POST'])
def keypoint_rcnn_cnf_score():
	global curr_conf, file_1
	if request.method == 'POST':
		if request.form.get('new_cf') != None:

			curr_conf = int(request.form.get('new_cf'))/100
			image = cv2.imread("/home/arnav/Desktop/Flask_all/flask_cv/uploads/"+file_1)
			return keypoint_rcnn(curr_conf)

		file = request.files['file']
		if file :
			if allowed_file(file.filename):# http://127.0.0.1:5000/
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
				file_1 = filename 
				return keypoint_rcnn()
			else:
				flash('Allowed image types are -> png, jpg, jpeg, gif') 
				redirect(request.url)


@app.route('/pic_post_mask_rcnn', methods=['POST'])
def mask_rcnn_cnf_score():
	global curr_conf, file_1
	if request.method == 'POST':
		if request.form.get('new_cf') != None:

			curr_conf = int(request.form.get('new_cf'))/100
			image = cv2.imread("/home/arnav/Desktop/Flask_all/flask_cv/uploads/"+file_1)
			return mask_rcnn(curr_conf)

		file = request.files['file']
		if file :
			if allowed_file(file.filename):# http://127.0.0.1:5000/
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
				file_1 = filename 
				return mask_rcnn()
			else:
				flash('Allowed image types are -> png, jpg, jpeg, gif') 
				redirect(request.url)


@app.route('/pic_post_yolo', methods=['POST'])
def yolo_cnf_score():
	global curr_conf, file_1
	if request.method == 'POST':
		if request.form.get('new_cnf') != None:
			curr_conf = int(request.form.get('new_cnf'))/100

			image = cv2.imread("/home/arnav/Desktop/Flask_all/flask_cv/uploads/"+file_1)
			new_file_name, obj_now, cnf_scores, temp = load_img_yolo(image, file_1, curr_conf)
			
			if new_file_name == None:
				return 'No yolo pre-trained objects detected in the image'
			else:
				# flash('Some yolo pre-trained objects detected')
				return render_template('yolo.html', filename=new_file_name, obj=obj_now, cnf=cnf_scores, conf_now=curr_conf*100)

		file = request.files['file']
		if file :
			if allowed_file(file.filename):# http://127.0.0.1:5000/
				curr_conf = 0.5
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
				file_1 = filename
				new_file_name, obj_now, cnf_scores, temp = load_img_yolo(image, filename, curr_conf)
			
				if new_file_name == None:
					return 'No yolo pre-trained objects detected in the image'
				else:
					# flash('Some yolo pre-trained objects detected')
					return render_template('yolo.html', filename=new_file_name, obj=obj_now, cnf=cnf_scores, conf_now=curr_conf*100)	
			else:
				flash('Allowed image types are -> png, jpg, jpeg, gif')
				# return 
				redirect(request.url)


@app.route('/yolo')
def yolo():
	global curr_conf
	image = cv2.imread("/home/arnav/Desktop/Flask_all/flask_cv/uploads/"+file_1)
	new_file_name, obj_now, cnf_scores, temp = load_img_yolo(image, file_1, curr_conf)

	if new_file_name == None:
		return 'No yolo pre-trained objects detected in the image'
	else:
		# flash('Some yolo pre-trained objects detected')
		return render_template('yolo.html', filename=new_file_name, obj=obj_now, cnf=cnf_scores, conf_now=curr_conf*100)	 	

@app.route('/yolo_found/<filename>')
def yolo_display(filename):
	return send_from_directory(app.config["YOLO_FOLDER"], filename)


@app.route('/mask_rcnn')
def mask_rcnn(mask_cnf=0.5):
	img = cv2.imread("/home/arnav/Desktop/Flask_all/flask_cv/uploads/"+file_1)
	cfg = get_cfg()
	cfg.MODEL.DEVICE='cpu'
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = mask_cnf
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	predictor = DefaultPredictor(cfg)
	outputs = predictor(img)

	v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	cnf = [str(100*float(t)) for t in list(outputs['instances'].scores)]

	if len(outputs["instances"].pred_classes) == 0:
		return 'No mask_rcnn pre-trained objects detected in the image'
	else:
		plt.imsave('/home/arnav/Desktop/Flask_all/flask_cv/mask_rcnn_found/mask_rcnn_'+file_1, out.get_image())
		return render_template('mask_rcnn.html', filename='mask_rcnn_'+file_1, cnf=cnf, conf_now=mask_cnf*100)

@app.route('/mask_rcnn_found/<filename>')
def mask_rcnn_display(filename):
	return send_from_directory(app.config["MASK_FOLDER"], filename)


@app.route('/keypoint_rcnn')
def keypoint_rcnn(keypoint_cnf=0.5):
	img = cv2.imread("/home/arnav/Desktop/Flask_all/flask_cv/uploads/"+file_1)
	cfg = get_cfg()
	cfg.MODEL.DEVICE='cpu'
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = keypoint_cnf
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")
	predictor = DefaultPredictor(cfg)
	outputs = predictor(img)

	v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	cnf = [str(100*float(t)) for t in list(outputs['instances'].scores)]

	if len(outputs["instances"].pred_classes) == 0:
		return 'No pre-trained keypoints detected in the image'
	else:
		plt.imsave('/home/arnav/Desktop/Flask_all/flask_cv/keypoint_rcnn_found/keypoint_rcnn_'+file_1, out.get_image())
		return render_template('keypoint_rcnn.html', filename='keypoint_rcnn_'+file_1, cnf=cnf, conf_now=keypoint_cnf*100)

@app.route('/keypoint_rcnn_found/<filename>')
def keypoint_rcnn_display(filename):
	return send_from_directory(app.config["KEYPOINT_FOLDER"], filename)

def getDominantColor(image): 
	B, G, R = cv2.split(image) 
	B, G, R = np.sum(B),np.sum(G), np.sum(R) 
	color_sums = [B,G,R]
	color_values = {"0": "Blue","1":"Green", "2": "Red"} 
	return color_values[str(np.argmax(color_sums))]


def yolo_post_process(img, outputs, conf):
    H, W = img.shape[:2]

    boxes, confidences, classIDs = [], [], []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    # flash(len(indices),'yolov3 pre-trained objects found')
    obj_found = {}
    if len(indices) > 0:
      for i in indices.flatten():
          (x, y) = (boxes[i][0], boxes[i][1])
          (w, h) = (boxes[i][2], boxes[i][3])
          color = [int(c) for c in colors[classIDs[i]]]
          cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
          try:
          	obj_found[classes[classIDs[i]]].append(100*round(confidences[i], 2))
          except:
          	obj_found[classes[classIDs[i]]] = [100*round(confidences[i], 2)]
          text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
          cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return len(indices), obj_found


def load_img_yolo(img_file, name, conf_sco):
    global img, img0, outputs, ln, curr_conf

    # img0 = cv.imread(path)
    img = img_file.copy()
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)
    outputs = np.vstack(outputs)
    ind, obj = yolo_post_process(img, outputs, conf_sco)

    if ind > 0:
    	cv.imwrite('/home/arnav/Desktop/Flask_all/flask_cv/yolo_found/yolo_'+name, img)
    	return 'yolo_'+name, ', '.join(list(obj.keys())), ', '.join([str(obj[j]) for j in obj]), obj
    else:
    	# flash('Nothing found')
    	return None, None, None, None
	
if __name__ == "__main__":
	app.debug = True
	app.run()