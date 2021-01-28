from flask import Flask,request, render_template
from model.prediction import *

app = Flask(__name__,template_folder='templates')
@app.route('/')
def index():
   return render_template('index.html')


@app.route('/photoRecognize', methods=['POST'])
def photoRecognize():
    if request.method == 'POST':        
        data = request.files['image_data']
        if data == None:
            return 'no image received'
        else:
            # model.predict.predict returns a dictionary
            cv.dnn_registerLayer('Crop', CropLayer)
            net = cv.dnn.readNet("/app/model/deploy.prototxt", "/app/model/hed_pretrained_bsds.caffemodel")
            prediction = predict(data,net)
    else:
      return render_template('index.html')

    return jsonify(results=prediction)


if __name__ == '__main__':
   app.run(debug = True)