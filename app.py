import sys,os
from wasteDetection.pipeline.training_pipeline import TrainPipeline
from wasteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from wasteDetection.constant.application import APP_HOST, APP_PORT
from wasteDetection.exception import AppException
from wasteDetection.logger import logging


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"



@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(BASE_DIR, "yolov5/runs/train/yolov5s_results/weights/my_model.pt")
        source_path = os.path.join(BASE_DIR, "data/inputImage.jpg")
        detect_script = os.path.join(BASE_DIR, "yolov5/detect.py")

        # Run YOLO detection
        os.system(f"python {detect_script} --weights {weights_path} --img 416 --conf 0.5 --source {source_path}")

        # Find latest 'exp' folder dynamically
        runs_detect_dir = os.path.join(BASE_DIR, "yolov5/runs/detect")
        exp_dirs = sorted([d for d in os.listdir(runs_detect_dir) if d.startswith("exp")], reverse=True)
        latest_exp = os.path.join(runs_detect_dir, exp_dirs[0])

        result_img_path = os.path.join(latest_exp, "inputImage.jpg")

        opencodedbase64 = encodeImageIntoBase64(result_img_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        # Cleanup
        os.system("rm -rf yolov5/runs")

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)




@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)