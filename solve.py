import cv2
import numpy as np

# List of color ranges in HSV (this needs to be customized based on your specific colors)
color_ranges = [
    {"color": "red", "lower": (0, 100, 100), "upper": (10, 255, 255)},
    {"color": "green", "lower": (40, 100, 100), "upper": (70, 255, 255)},
    {"color": "blue", "lower": (100, 100, 100), "upper": (130, 255, 255)},
    {"color": "yellow", "lower": (20, 100, 100), "upper": (30, 255, 255)},
    {"color": "orange", "lower": (10, 100, 100), "upper": (20, 255, 255)},
    {"color": "purple", "lower": (130, 100, 100), "upper": (160, 255, 255)},
    {"color": "pink", "lower": (160, 100, 100), "upper": (180, 255, 255)},
    {"color": "cyan", "lower": (80, 100, 100), "upper": (100, 255, 255)},
    # {"color": "brown", "lower": (0, 100, 100), "upper": (20, 255, 255)},
    {"color": "black", "lower": (0, 0, 0), "upper": (180, 255, 30)},
    # {"color": "white", "lower": (0, 0, 200), "upper": (180, 20, 255)},
    # Add more color ranges as needed
]

names = ['rectangle', 'circle', 'triangle']

im_size = 64


# Function to extract and save shapes based on color
def extract_shapes(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    shape_count = 0
    shapes = []

    for color_range in color_ranges:
        # Create a mask for the current color
        mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Create a bounding box around each contour
            if cv2.contourArea(contour) < 100:
                continue

            # Create a blank binary image
            shape_img = np.zeros_like(img[:, :, 0])

            # Draw the contour on the binary image
            cv2.drawContours(shape_img, [contour], -1, (255), thickness=cv2.FILLED)

            # Crop the binary image to the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            shape_img = shape_img[y:y + h, x:x + w]

            # Save the extracted shape
            shape_count += 1
            shape_filename = f'outputs/shape_{shape_count}_{color_range["color"]}.png'
            cv2.imwrite(shape_filename, shape_img)
            shapes.append(shape_img)
            print(f'Saved {shape_filename}, w: {w}, h: {h}')
    return shapes


def predict(net, im):
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (im_size, im_size))
    blob -= 0.5
    blob /= 0.5

    # Forward pass
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)
    parameter_output, shape_output = outputs

    # Get the predicted shape and parameter
    shape_idx = np.argmax(shape_output)

    predicted_parameter = parameter_output[0]
    area = predicted_parameter[0]
    points = predicted_parameter[1:].reshape(-1, 2)
    return shape_idx, area, points


def reshape_image(im):
    h, w = im.shape
    max_dim = max(h, w)

    # Create a blank square image with the desired size
    square_image = np.zeros((im_size, im_size), dtype=np.uint8)

    # Calculate the scaling factor and resize the image
    scale = im_size / max_dim / 1.45
    new_w, new_h = int(w * scale), int(h * scale)
    resized_im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate the position to place the resized image on the blank square
    top_left_x = (im_size - new_w) // 2
    top_left_y = (im_size - new_h) // 2

    # Place the resized image on the blank square
    square_image[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_im

    return scale, square_image


if __name__ == '__main__':
    # Load the model
    network = cv2.dnn.readNetFromONNX("./shape_model.onnx")

    # Read the image
    image_path = "./data/d.png"
    image = cv2.imread(image_path)
    origin_image = image.copy()

    # Detect and extract shapes based on color
    extracted_images = extract_shapes(image)

    # Process each extracted shape
    total_area = {"triangle": 0, "rectangle": 0, "circle": 0}
    for shape_im in extracted_images:
        ratio, shape_im = reshape_image(shape_im)

        shape_idx, area, points = predict(network, shape_im)
        area /= ratio
        points *= im_size
        shape_name = names[shape_idx]
        total_area[shape_name] += area

        print(f"Predicted shape: {shape_name}, Area: {area}, Points: {points}")

        # Plot the points
        shape_im = cv2.cvtColor(shape_im, cv2.COLOR_GRAY2BGR)
        for point in points:
            cv2.circle(shape_im, tuple(point.astype(np.int32)), 2, (0, 255, 0), -1)

        filename = f'outputs/shape_{shape_name}_{area:.2f}.png'
        cv2.imwrite(filename, shape_im)
        print(f'Saved {filename}')

    print(f"Total area: {total_area}")
    # Calculate the fraction of each shape in the total area
    total_area_sum = sum(total_area.values())
    shape_fractions = {shape: area / total_area_sum for shape, area in total_area.items()}

    for shape, fraction in shape_fractions.items():
        print(f"{shape}: {fraction:.2f}%")
