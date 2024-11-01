from ultralytics import YOLO

# Load a model
model = YOLO("yolov8-CBAM.yaml")  # build a new model from scratch

# Use the model
model.train(data="data.yaml", epochs=1)  # train the model
metrics = model.val()  # evaluate model performance on the validation set