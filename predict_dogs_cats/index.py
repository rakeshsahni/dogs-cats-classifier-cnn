from flask import Flask, url_for,render_template,request
import numpy as np
import cv2
from tensorflow.keras.models import load_model 
import os
app = Flask(__name__)

model = load_model('own_0-dogs_1-cats.h5')
def modify(img) :
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1,(128,128))
    # print(img1.shape)
    img1 = img1 / 255
    img1 = img1.reshape([1,128,128,3])
    return img1


@app.route("/",methods=['POST','GET'])
def Home() :
    res = ""
    if request.method == 'POST' : 
        upload_img = request.files['myfile']
        # print(upload_img)
        upload_img.save(os.path.join('static',upload_img.filename))
        if upload_img.filename != "" : 
            img = cv2.imread('static/'+upload_img.filename)
            img = modify(img)
            y_pred = model.predict(img)
            res = np.round(y_pred[0])
            # print(res)
            os.unlink(os.path.join('static',upload_img.filename))
        return render_template('index.html',dt = res)
    return render_template('index.html', dt = res)

if __name__ == "__main__" : 
    app.run(port=3000,debug=True)