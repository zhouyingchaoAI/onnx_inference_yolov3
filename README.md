
# onnx_inference_yolov3
implement the yolov3.onnx inference with flask web service

## Prerequisites
- `python3`
- `PIL==5.4.1
- `onnxruntime-gpu==1.1.0`
- `flask==1.0.2
- `numpy==1.16.2

## 1.get yolov3.onnx
you can get the model [here](https://onnxzoo.blob.core.windows.net/models/opset_10/yolov3/yolov3.onnx)
## 2.start you inference service
```bashrc
$ cd onnx_inference_yolov3
$ FLASK_ENV=development FLASK_APP=yolov3_app.py flask run
```
## 3.use client.py to inference 
```bashrc
$ cd onnx_inference_yolov3
$ python3 client.py --file img.jpeg
```

