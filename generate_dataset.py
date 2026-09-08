import os
import random

import cv2
import numpy as np

im_size = 64


class MaxTriesExceededError(Exception):
    pass

def verify_point(im, point, threshold=0.8):
    x, y = point[0], point[1]
    crop = im[max(0, y - 3):min(im_size, y + 4), max(0, x - 3):min(im_size, x + 4)]
    return np.mean(crop) > threshold


def random_point_in_mask(im, max_tries=1000):
    for _ in range(max_tries):
        point = np.random.randint(0, im.shape[0], 2)
        if verify_point(im, point):
            return tuple(point)
    raise MaxTriesExceededError("Could not find a valid point")


def random_point_in_line(p1, p2, mask, starting=0.1, ending=0.9, max_tries=1000):
    for _ in range(max_tries):
        p = p1 + np.random.uniform(starting, ending) * (p2 - p1)
        p = p.astype(np.int32)
        if verify_point(mask, p):
            return tuple(p)
    raise MaxTriesExceededError("Could not find a valid point")


def random_crop(im, p1, p2, covering='circle'):
    mask = im.copy()
    cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
    cv2.circle(mask, p2, 4, 0, -1)
    cv2.circle(mask, p2, 4, 0, -1)

    if covering == 'circle':
        p3 = random_point_in_mask(mask)
        # Draw a circle that passes through p1, p2, and p3
        center, radius = cv2.minEnclosingCircle(np.array([p1, p2, p3]))
        center = np.array(center, dtype=np.int32)
        radius = int(radius)
        cv2.circle(im, center, radius, 0, -1)

    elif covering == 'polygon':
        polygon_points = [p1, p2]
        for _ in range(random.randint(1, 2)):
            p = random_point_in_mask(mask)
            cv2.circle(mask, p, 5, 0, -1)
            polygon_points.append(p)
        centroid = np.mean(polygon_points, axis=0)

        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        polygon_points = sorted(polygon_points, key=angle_from_centroid)
        cv2.fillPoly(im, [np.array(polygon_points)], 0)
    return im


def create_rectangle(max_tries=1000):
    while max_tries >= 1:
        try:
            im = np.zeros((im_size, im_size), dtype=np.uint8)

            # Generate a random rectangle
            r = random.uniform(im_size / 3, im_size / 2)
            theta = random.uniform(np.pi / 6, np.pi / 4)
            hw, hh = int(r * np.cos(theta)), int(r * np.sin(theta))  # Half
            cx, cy = im_size // 2, im_size // 2
            points = np.array([
                [cx - hw, cy - hh],
                [cx + hw, cy - hh],
                [cx + hw, cy + hh],
                [cx - hw, cy + hh],
            ])

            cv2.fillPoly(im, [points], [255])
            origin = im.copy()

            iterations = random.randint(1, 2)
            while iterations >= 1:
                idx = random.randint(0, 3)
                mask = cv2.erode(im.copy(), np.ones((3, 3), np.uint8), iterations=3)
                p1 = random_point_in_line(points[idx], points[(idx + 1) % 4], mask=mask)
                cv2.circle(mask, p1, 4, 0, -1)
                cover_point = random.choice([True, False])
                if iterations == 1 and cover_point:
                    p2 = random_point_in_line(points[(idx + 1) % 4], points[(idx + 2) % 4], mask=mask)
                else:
                    p2 = random_point_in_line(points[idx], points[(idx + 1) % 4], mask=mask)
                random_crop(im, p1, p2, covering=random.choice(['circle', 'polygon']))
                iterations -= 1
            points = np.concatenate([np.array([[cx, cy]]), points[[0, 2]]], axis=0)
            points = points.astype(np.float32)
            points /= im_size
            return origin, im, points
        except MaxTriesExceededError:
            # print("In function create_rectangle: Could not find a valid point. Retrying...")
            max_tries -= 1


def create_triangle(max_tries=1000):
    while max_tries >= 1:
        try:
            im = np.zeros((im_size, im_size), dtype=np.uint8)
            # Generate a random triangle
            while im.mean() < 20:
                im = np.zeros((im_size, im_size), dtype=np.uint8)
                r = random.uniform(im_size / 3, im_size / 2)
                theta = np.random.uniform(0, 2 * np.pi, 3)
                cx, cy = im_size // 2, im_size // 2
                points = np.array([
                    [cx + r * np.cos(theta[0]), cy + r * np.sin(theta[0])],
                    [cx + r * np.cos(theta[1]), cy + r * np.sin(theta[1])],
                    [cx + r * np.cos(theta[2]), cy + r * np.sin(theta[2])],
                ]).astype(np.int32)
                cv2.fillPoly(im, [points], [255])
            origin = im.copy()

            iterations = random.randint(1, 2)
            while iterations >= 1:

                mask = cv2.erode(im.copy(), np.ones((3, 3), np.uint8))
                idx = random.randint(0, 2)
                p1 = random_point_in_line(points[idx], points[(idx + 1) % 3], mask=mask)
                cv2.circle(mask, p1, 4, 0, -1)

                cover_point = random.choice([True, False])
                if iterations == 1 and cover_point:
                    p2 = random_point_in_line(points[(idx + 1) % 3], points[(idx + 2) % 3], mask=mask)
                else:
                    p2 = random_point_in_line(points[idx], points[(idx + 1) % 3], mask=mask)

                random_crop(im, p1, p2, covering=random.choice(['circle', 'polygon']))
                iterations -= 1
            return origin, im, points.astype(np.float32)/im_size
        except MaxTriesExceededError:
            # print("In function create_triangle: Could not find a valid point. Retrying...")
            max_tries -= 1


def create_circle(max_tries=1000):
    while max_tries >= 1:
        try:
            im = np.zeros((im_size, im_size), dtype=np.uint8)

            # Generate a random circle
            r = random.uniform(im_size / 3, im_size / 2)
            cx, cy = im_size // 2, im_size // 2
            cv2.circle(im, (cx, cy), int(r), 255, -1)
            origin = im.copy()

            mask = cv2.erode(im.copy(), np.ones((3, 3), np.uint8))
            # cv2.imwrite("images/mask.png", im)

            iterations = random.randint(1, 2)
            while iterations >= 1:
                theta = np.random.uniform(0, 2 * np.pi)
                p1 = np.array([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.int32)

                cv2.circle(mask, p1, 4, 0, -1)

                theta = np.random.uniform(0, 2 * np.pi)
                p2 = np.array([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.int32)

                random_crop(im, p1, p2, covering=random.choice(['circle', 'polygon']))
                iterations -= 1
            points = np.array([[cx, cy], [cx, cy-r], [cx, cy+r]]).astype(np.int32)
            return origin, im, points.astype(np.float32)/im_size
        except MaxTriesExceededError:
            # print("In function create_circle: Could not find a valid point. Retrying...")
            max_tries -= 1


def create_dataset(num_images=1000, train_val_test_ratio=(0.7, 0.2, 0.1), root_dir="images"):
    # Create directories
    subsets = ['train', 'val', 'test']
    shapes = ['rectangle', 'circle', 'triangle']
    for subset in subsets:
        for shape in shapes:
            os.makedirs(f"{root_dir}/{subset}/{shape}", exist_ok=True)

    # Calculate the number of images for each subset
    num_train, num_val, num_test = [int(num_images * ratio) for ratio in train_val_test_ratio]

    # Create images for each subset
    for i, num_images in enumerate([num_train, num_val, num_test]):
        # Get the largest image index of previous generated images
        for j in range(num_images // 3):
            for shape, create_shape in zip(shapes, [create_rectangle, create_circle, create_triangle]):
                origin, im, points = create_shape()
                cv2.imwrite(f"{root_dir}/{subsets[i]}/{shape}/{j}.png", im)
                cv2.imwrite(f"{root_dir}/{subsets[i]}/{shape}/{j}_origin.png", origin)
                # Write the points to a file
                np.savetxt(f"{root_dir}/{subsets[i]}/{shape}/{j}.txt", points, fmt="%f")

    print("Dataset created successfully")
    print(f"Train images: {num_train}, Validation images: {num_val}, Test images: {num_test}")


if __name__ == '__main__':
    root_dir = "images"
    create_dataset(10000, (0.8, 0.15, 0.05), root_dir)
