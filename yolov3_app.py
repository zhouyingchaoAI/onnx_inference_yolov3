import io
import json

import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, jsonify, request

# FLASK_ENV=development FLASK_APP=yolov3_app.py flask run


app = Flask(__name__)
session = onnxruntime.InferenceSession("yolov3.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]


# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([transforms.Resize(255),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             [0.485, 0.456, 0.406],
#                                             [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((416, 416), Image.BICUBIC)
    image_data = np.array(img, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)

    input = {
                inname[0]: image_data,
                inname[1]: np.expand_dims(np.array(image_data.shape[2:], dtype=np.float32), 0)
    }
    prediction = session.run(outname, input)
    print(prediction)
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in prediction[2]:
        out_classes.append(idx_[1])
        out_scores.append(prediction[1][tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(prediction[0][idx_1])
    return out_boxes, out_scores, out_classes


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        out_boxes, out_scores, out_classes = get_prediction(image_bytes=img_bytes)
        out_boxes = np.array(out_boxes).tolist();
        out_scores = np.array(out_scores).tolist();
        out_classes = np.array(out_classes).tolist();
        print(out_boxes, out_scores, out_classes)
        return jsonify({'boxes':out_boxes, 'scores': out_scores, 'classes': out_classes})


if __name__ == '__main__':
    app.run()
