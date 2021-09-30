from flask import Flask
from flask import render_template
from gastric_cancer_pred import get_prediction
from flask import Flask,  render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def index():
    return render_template('index.html', title='This is an RNN based network to predict whether cancer exists')

@app.route('/upload2',methods=['GET'])
def upload_images():
    return render_template("upload.html")

@app.route('/getImg/',methods=['GET','POST'])
def getImg():	
    imgData = request.files["image"]
    path =basedir + "/static/upload/"    
    imgName = imgData.filename
    file_path = path + imgName
    imgData.save(file_path)
    url = file_path    
    return render_template("upload_ok.html", url=url)

@app.route('/prediction/',methods=['GET','POST'])
def prediction():
    url=request.args.get('url')    
    path=[url]
    result = get_prediction(path)
    return render_template('prediction.html',
                            pred = str(result) )

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080 )