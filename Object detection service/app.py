import cv2
import numpy as np
import json
import base64
from flask import Flask, request, jsonify
import object_detection as od
import gc
import os


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/', methods=['GET'])
def index():
    return jsonify({'Message': 'Please use /api/object_detection to access the api'})

@app.route('/api/object_detection', methods=['GET','POST'])
def object_detection():
    if request.method == "POST":
        labelsPath= "coco.names"
        cfgpath= "yolov3-tiny.cfg"
        wpath= "yolov3-tiny.weights"
        Lables= od.get_labels(labelsPath)
        CFG= od.get_config(cfgpath)
        Weights= od.get_weights(wpath)
        upload_dir = "file_directory"
        #creating a dictionary from json request body
        data = json.loads(request.json)  

        #opening the file to write to file system
        with open(upload_dir+"/"+data['id']+".jpg", "wb") as f:
            #writing decoding the image
            f.write(base64.b64decode(data['image']))
        #address of the imagefile to file system
        imagefile = str(upload_dir+"/"+data['id']+".jpg")
        
        img = cv2.imread(imagefile)
        #converting the image to numpy array
        npimg=np.array(img)
        image=npimg.copy()
        #converting the image to RGB
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # load the neural net.  Should be local to this method as its multi-threaded endpoint
        nets = od.load_model(CFG, Weights)
        predictions=od.do_prediction(image, nets, Lables) 
        #deleting the image from the server
        del(image)
        os.remove(imagefile)

        gc.collect()
        #formatting the response in json format.
        return jsonify(
            {
            "id": data['id'],
            "objects": predictions
            })
    if request.method == "GET":
        return jsonify({'Message': "Please send a POST request with body containing image uuid and image data in base64 encoded format"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)