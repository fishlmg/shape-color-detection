import cv2
import numpy as np

im_size = 64
names = ['rectangle', 'circle', 'triangle']


def predict(net, im):
    # Preprocess the image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (im_size, im_size))
    blob -= 0.5
    blob /= 0.5

    # Forward pass
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    print(layer_names)
    outputs = net.forward(layer_names)
    parameter_output, shape_output = outputs

    # Get the predicted shape and parameter
    shape_idx = np.argmax(shape_output)

    predicted_parameter = parameter_output[0]
    area = predicted_parameter[0]
    points = predicted_parameter[1:].reshape(-1, 2)
    return shape_idx, area, points


if __name__ == '__main__':
    # Read the image
    im = cv2.imread("./images/test/rectangle/11.png")
    origin_im = cv2.imread("./images/test/rectangle/11_origin.png")

    # Load the model
    net = cv2.dnn.readNetFromONNX("./shape_model.onnx")

    shape_idx, area, points = predict(net, im)
    shape = names[shape_idx]

    print("Shape of the image: ", im.shape)
    print(f"Predicted shape: {shape}, Area: {area}, Points: {points}")
    print(f"True area", np.mean(origin_im) / 255)

    # # Plot the points
    # # origin = cv2.resize(origin, (im_size*10, im_size*10))
    # points *= im_size
    # for point in points:
    #     cv2.circle(im, tuple(point.astype(np.int32)), 4, (0, 255, 0), -1)
    # cv2.imwrite("output.png", im)
