import cv2
import numpy as np
import onnxruntime as ort

im_size = 64
names = ['rectangle', 'circle', 'triangle']


def preprocess_image(im):
    """Preprocess the image for model input."""
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (im_size, im_size))
    blob -= 0.5
    return blob


def predict(net, im):
    # Preprocess the image
    blob = preprocess_image(im)

    # Run the model
    inputs = {net.get_inputs()[0].name: blob}
    outputs = net.run(None, inputs)
    shape_output, parameter_output = outputs

    # Get the predicted shape and parameter
    shape_idx = np.argmax(shape_output)

    predicted_parameter = parameter_output[0]
    area = predicted_parameter[0]
    points = predicted_parameter[1:].reshape(-1, 2)
    return shape_idx, area, points


if __name__ == '__main__':
    # Read the image
    im = cv2.imread("./images/test/triangle/11.png")
    origin_im = cv2.imread("./images/test/triangle/11_origin.png")

    # Load the ONNX model
    net = ort.InferenceSession("./shape_model.onnx")

    shape_idx, area, points = predict(net, im)
    shape = names[shape_idx]

    print("Shape of the image: ", im.shape)
    print(f"Predicted shape: {shape}, Area: {area}, Points: {points}")
    origin_array = np.array(origin_im)
    area = np.sum(origin_array > 0) / (origin_array.size)
    area = np.array([area]).astype(np.float32)
    print(f"True area", area)

    # Plot the points
    points *= im_size
    for point in points:
        cv2.circle(im, tuple(point.astype(np.int32)), 4, (0, 255, 0), -1)
    cv2.imwrite("output.png", im)
