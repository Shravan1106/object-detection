import os,flask
from flask import request, redirect, url_for,render_template,jsonify
from werkzeug.utils import secure_filename
import boto3
import csv
from imageai.Detection import ObjectDetection
import tensorflow as tf

app = flask.Flask(__name__)
app.config["DEBUG"] = True
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

output = {}
@app.route('/')
def fun():
    return render_template("index.html")  

@app.route('/getImageDetails', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        if 'data' not in request.files:
            return redirect(request.url)
        file = request.files['data']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join("detectedimages/" + filename))
        aws(filename)
        animalANDobjects(filename)
        return jsonify(output)


                                    # AWS IS USED FOR HUMAN DETECTION

def aws(filename):
    with open('credentials.csv','r') as inp:
        next(inp)
        reader = csv.reader(inp)
        for line in reader:
            aws_access_key_id = line[2]
            secret_access_key = line[3]
    image = 'detectedimages/'+filename
    client = boto3.client('rekognition',region_name='ap-south-1',aws_access_key_id = aws_access_key_id,aws_secret_access_key=secret_access_key)

    with open(image,'rb') as source_image:
        source_bytes = source_image.read()

    response = client.detect_labels(Image = {'Bytes':source_bytes}) 
    persons = 0
    for val in response['Labels']:
        if(val['Name'] == 'Person'):
            persons+=len(val['Instances'])
    output['persons'] = persons

                            # IMAGEAI IS USED FOR OBJECT AND ANIMAL DETECTION

def animalANDobjects(filename):
    tf.compat.v1.Session()
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "objdet.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image= "detectedimages/"+filename, output_image_path=os.path.join(execution_path , "imagenew.jpg"))

    animals = ['bird','cat','dog','horse','sheep','cow','elephant','bear', 'zebra','giraffe']
    nanimal = 0
    objects = 0
    for eachObject in detections:
        if eachObject["name"] in animals:
            nanimal += 1
        elif(eachObject["name"] != 'person'):
            objects += 1
    output['animals'] = nanimal
    output['objects'] = objects


if __name__ == "__main__":
    app.run(debug=True)