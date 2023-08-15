import logging
import math

import cv2
import dlib
import torch
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from facePoints import facePoints
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

LOG = logging.getLogger(__name__)


def create_color_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    return bar


def is_black_white(image, threshold=192) -> bool:
    """
    Check if the image is black and whites
    :param image:
    :param threshold:
    :return:
    """

    # return prob_bt >= threshold
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # Check if all pixel values are the same
        unique_values = np.unique(image)
        if len(unique_values) == 1:
            return True

    return False


def resize(image, width: int = -1, height: int = -1):
    """
    Resize the image, -1 means auto, but the image won't be resized if both width and height are -1
    :param image:
    :param width: -1 means auto
    :param height: -1 means auto
    :return:
    """
    if width < 0 and height < 0:
        return image
    elif width < 0:
        ratio = height / image.shape[0]
        width = int(image.shape[1] * ratio)
    elif height < 0:
        ratio = width / image.shape[1]
        height = int(image.shape[0] * ratio)

    return cv2.resize(image, (width, height))


def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), biggest_only=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    flags = cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT if biggest_only else cv2.CASCADE_SCALE_IMAGE
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=flags
    )
    if len(faces) == 0:
        return []
    # note: shape of faces is (n, 4) where n is num of faces
    # Change the format of faces from (x, y, w, h) to (x, y, x+w, y+h)
    faces[:, 2:] += faces[:, :2]
    return faces


def mask_face(image, face):
    x1, y1, x2, y2 = face
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1: y2, x1: x2] = 255
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


def detect_skin_in_bw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    all_0 = np.isclose(skin, 0).all()
    return image if all_0 else skin, skin_mask


def load_model(model_type, state_dict):
    category_prefix = '_categories.'
    categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
    categories = [k[len(category_prefix):] for k in categories]

    model = model_type(categories)
    model.load_state_dict(state_dict)

    return model

def _load_image(image):
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def detect_landmarks_in_color(image):
    model_path = "src/stone/shape_predictor_68_face_landmarks.dat"
    frontal_face_detector = dlib.get_frontal_face_detector()
    face_landmark_detector = dlib.shape_predictor(model_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detect all faces in image
    faces = frontal_face_detector(gray_image, 0)
    if len(faces) == 0:
        return None
    faces = faces[0]
    face_rectangle = dlib.rectangle(int(faces.left()), int(faces.top()), int(faces.right()), int(faces.bottom()))
    detected_landmarks = face_landmark_detector(gray_image, face_rectangle)

    return detected_landmarks

def remove_shadows_highlights(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect shadows (you can adjust the threshold value)
    _, shadows_mask = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)

    # Apply thresholding to detect highlights (you can adjust the threshold value)
    _, highlights_mask = cv2.threshold(gray_image, 213, 255, cv2.THRESH_BINARY)

    # Combine the shadows and highlights masks
    extreme_mask = cv2.bitwise_or(shadows_mask, highlights_mask)

    # Create a full-white image with the same shape as the original
    white_image = np.ones_like(image) * 255

    # Mask out the shadows and highlights from the white image
    removed_extremes_image = cv2.bitwise_and(white_image, white_image, mask=extreme_mask)

    # Combine the original image with the white image where the extremes were removed
    result_image = cv2.bitwise_or(image, removed_extremes_image)

    skin_mask = cv2.bitwise_not(extreme_mask)

    return result_image, skin_mask


def detect_skin_in_color(image, model, landmarks):
   # load up model from https://github.com/WillBrennan/SkinDetector.git
    if model:
        device = 'cpu'
        # convert image to tensor
        fn_image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: _load_image(image)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        image = fn_image_transform(image)

        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            results = model(image)['out'] 
            results = torch.sigmoid(results)

            # results = skin area
            results = results > 0.5
            skin_mask = results.squeeze().cpu().numpy()
            skin_mask = skin_mask.astype(np.uint8) * 255
            
            image = image[0].cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image * (0.229, 0.224, 0.225)) + (0.485, 0.456, 0.406)
            image = (255 * image).astype(np.uint8)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            skin = cv2.bitwise_and(image, image, mask=skin_mask)

            # using landmarks to remove eyes, eyebrows, and mouth
            if landmarks is not None:
                eyebrows_points = list(range(17, 27))  # 17 to 26
                eyes_points = list(range(36, 48))  # 36 to 47
                mouth_points = list(range(48, 60))  # 48 to 60
            
                # Create a mask for the facial regions to remove
                mask = np.zeros(skin_mask.shape, dtype=np.uint8)

                # Draw filled polygons on the mask for each facial region
                cv2.fillPoly(mask, [np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eyebrows_points], dtype=np.int32)], 255)
                cv2.fillPoly(mask, [np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eyes_points], dtype=np.int32)], 255)
                cv2.fillPoly(mask, [np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in mouth_points], dtype=np.int32)], 255)

                # Invert the mask to keep everything except the facial regions
                mask_inverse = cv2.bitwise_not(mask)
                skin = cv2.bitwise_and(skin, skin, mask=mask_inverse)
                skin_mask = cv2.bitwise_and(skin_mask, mask_inverse)
            
            # removing any harsh shadows or highlights
            result_image, skin_mask = remove_shadows_highlights(skin)
            skin = cv2.bitwise_and(skin, skin, mask=skin_mask)


            # # use gaussian blur to further remove non skin areas
            # skin = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
            # nonzero_indices = np.nonzero(skin)
            # nonzero_values = skin[nonzero_indices]
            # print("HSV SKIN: ", max(nonzero_values))
            # skin_mask = cv2.inRange(skin, low_hsv, high_hsv)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            # skin_mask = cv2.GaussianBlur(skin_mask, ksize=(1, 1), sigmaX=0)

            # skin = cv2.bitwise_and(skin, skin, mask=skin_mask)
            # skin = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
            
            return skin, skin_mask
        
    else:   
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Defining skin Thresholds
        low_hsv = np.array([0, 48, 80], dtype=np.uint8)
        high_hsv = np.array([20, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(img, low_hsv, high_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, ksize=(3, 3), sigmaX=0)

        skin = cv2.bitwise_and(image, image, mask=skin_mask)

        all_0 = np.isclose(skin, 0).all()

        return image if all_0 else skin, skin_mask
        

def draw_rects(image, *rects, color=(255, 0, 0), thickness=2):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, thickness)
    return image

def euclidean_distance(colors):
    # input: array of dominant colors (BGR)
    b1, g1, r1 = colors[0]
    b2, g2, r2 = colors[1]
    return ((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2) ** 0.5

def blend_colors(color1, color2, proportion):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.strip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        return '#%02x%02x%02x' % rgb_color

    rgb_color1 = hex_to_rgb(color1)
    rgb_color2 = hex_to_rgb(color2)

    r = int(rgb_color1[0] * proportion + rgb_color2[0] * (1 - proportion))
    g = int(rgb_color1[1] * proportion + rgb_color2[1] * (1 - proportion))
    b = int(rgb_color1[2] * proportion + rgb_color2[2] * (1 - proportion))

    return rgb_to_hex((r, g, b))

def dominant_colors(image, to_bw, n_clusters=2):
    if to_bw:
        data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        data = image
    data = np.reshape(data, (-1, 3))
    data = data[np.all(data != 0, axis=1)]
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, colors = cv2.kmeans(data, n_clusters, None, criteria, 10, flags)
    labels, counts = np.unique(labels, return_counts=True)

    order = (-counts).argsort()
    colors = colors[order]
    counts = counts[order]

    props = counts / counts.sum()

    return colors, props


def blur(image, degree=25):
    """
    Blur the image
    :param image:
    :param degree: the degree of blur. The bigger, the more blur
    :return:
    """
    ksize = degree, degree
    return cv2.blur(image, ksize)

def group_skin_color(colors_array):
    """
    :param colors_array: array of dominant colors (BGR) in order of proportion
    :return: index of lighter group if dom colors are in separate groups, else False
    """
    def is_within_range(value, range_tuple):
        return range_tuple[0][0] <= value[0] <= range_tuple[1][0] and \
            range_tuple[0][1] <= value[1] <= range_tuple[1][1] and \
            range_tuple[0][2] <= value[2] <= range_tuple[1][2]

    color_groups = []
    # Define the RGB ranges for each group
    for color in colors_array:
        range_1 = ((208, 231, 243), (228, 237, 247))  # Colors 1-3 lightest
        range_2 = ((150, 189, 215), (186, 218, 234))  # Colors 4-6 middle
        range_3 = ((52, 65, 96), (86, 126, 160))      # Colors 7-10 darkest

        # Check if the RGB values fall within any of the defined ranges
        if is_within_range(color, range_1):
            color_groups.append(1)
        elif is_within_range(color, range_2):
            color_groups.append(2)
        elif is_within_range(color, range_3):
            color_groups.append(3)
        else:
            color_groups.append(0)

    if abs(np.diff(color_groups)) >= 1:
        return np.argmin(color_groups)
    return False

def swap_elements(array):
    if type(array) != list:
        temp = array[0].copy()
    else:
        temp = array[0]
    array[0] = array[1]
    array[1] = temp
    return array

def skin_tone(colors, props, skin_tone_palette, tone_labels, weight_factors=[1.0, 0.20]):
    lab_tones = [convert_color(sRGBColor.new_from_rgb_hex(rgb), LabColor) for rgb in skin_tone_palette]
    lab_colors = [convert_color(sRGBColor(rgb_r=r, rgb_g=g, rgb_b=b, is_upscaled=True), LabColor) for b, g, r in colors]

    if props[0] < 0.53: 
        # colors are balanced in proportion
        distances = [np.sum([delta_e_cie2000(c, label) * p for c, p in zip(lab_colors, props)]) for label in lab_tones]
    elif euclidean_distance(colors) < 100: 
        # colors are close together in distance but proportionally unbalanced
        distances = [np.sum([delta_e_cie2000(c, label) * p * w for c, p, w in zip(lab_colors, props, weight_factors)]) for label in lab_tones]
    else: 
        # colors are far apart in distance and proportionally unbalanced --> one is most likely shadow
        if group_skin_color(colors) == 1: 
            # if colors are in separate groups
            colors = swap_elements(colors)
            lab_colors = swap_elements(lab_colors)
        distances = [np.sum([delta_e_cie2000(c, label) * p * w for c, p, w in zip(lab_colors, props, weight_factors)]) for label in lab_tones]

    tone_id = np.argmin(distances)
    distance: float = distances[tone_id]
    tone_hex = skin_tone_palette[tone_id].upper()
    PERLA = tone_labels[tone_id]
    return colors, props, tone_id, tone_hex, PERLA, distance


def classify(image, is_bw, to_bw, skin_tone_palette, tone_labels, n_dominant_colors=2, verbose=False, report_image=None, use_face=True, model=False, distance=100):
    """
    Classify the skin tone of the image
    :param image:
    :param is_bw: whether the image is black and white
    :param to_bw: whether to convert the image to black and white
    :param skin_tone_palette:
    :param tone_labels:
    :param n_dominant_colors:
    :param verbose:
    :param report_image: the image to draw the report on
    :param use_face: whether to use face area for detection
    :return:
    """
    landmarks = detect_landmarks_in_color(image)
    detect_skin_fn = detect_skin_in_bw if is_bw else detect_skin_in_color
    skin, skin_mask = detect_skin_fn(image, model, landmarks)
    dmnt_colors, dmnt_props = dominant_colors(skin, to_bw, n_dominant_colors)
    # Generate readable strings
    hex_colors = ['#%02X%02X%02X' % tuple(np.around([r, g, b]).astype(int)) for b, g, r in dmnt_colors]
    prop_strs = ['%.2f' % p for p in dmnt_props]
    result = list(np.hstack(list(zip(hex_colors, prop_strs))))
    # Calculate skin tone
    dmnt_colors, dmnt_props, tone_id, tone_hex, PERLA, distance = skin_tone(dmnt_colors, dmnt_props, skin_tone_palette, tone_labels)
    accuracy = round(100 - distance, 2)
    result.extend([tone_hex, PERLA, accuracy])
    if not verbose:
        return result, None

    # 0. Create initial report image
    report_image = initial_report_image(image, report_image, skin_mask, use_face, to_bw)
    bar_width = 100

    # 1. Create color bar for dominant colors
    color_bars = create_dominant_color_bar(report_image, dmnt_colors, dmnt_props, bar_width)

    # 2. Create color bar for skin tone list
    palette_bars = create_tone_palette_bar(report_image, tone_id, skin_tone_palette, bar_width)

    # 3. Combine all bars and report image
    report_image = np.hstack([report_image, color_bars, palette_bars])
    msg_bar = create_message_bar(dmnt_colors, dmnt_props, tone_hex, distance, report_image.shape[1])
    report_image = np.vstack([report_image, msg_bar])
    return result, report_image


def initial_report_image(face_image, report_image, skin_mask, use_face, to_bw):
    report_image = face_image if report_image is None else report_image

    if to_bw:
        report_image = cv2.cvtColor(report_image, cv2.COLOR_BGR2GRAY)
    if use_face:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        skin_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        
    blurred_image = blur(report_image)
    non_skin_mask = cv2.bitwise_not(skin_mask)
    edges = cv2.Canny(skin_mask, 50, 150)
    report_image = cv2.bitwise_and(report_image, report_image, mask=skin_mask) + cv2.bitwise_and(blurred_image, blurred_image, mask=non_skin_mask)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(report_image, contours, -1, (255, 0, 0), 2)
    return report_image


def create_dominant_color_bar(report_image, dmnt_colors, dmnt_props, bar_width):
    color_bars = []
    total_height = 0
    for color, prop in zip(dmnt_colors, dmnt_props):
        bar_height = int(math.floor(report_image.shape[0] * prop))
        total_height += bar_height
        bar = create_color_bar(bar_height, bar_width, color)
        color_bars.append(bar)
    padding_height = report_image.shape[0] - total_height
    if padding_height > 0:
        padding = create_color_bar(padding_height, bar_width, (255, 255, 255))
        color_bars.append(padding)
    return np.vstack(color_bars)


def create_tone_palette_bar(report_image, tone_id, skin_tone_palette, bar_width):
    palette_bars = []
    tone_height = report_image.shape[0] // len(skin_tone_palette)
    tone_bgrs = []
    for tone in skin_tone_palette:
        hex_value = tone.lstrip('#')
        r, g, b = [int(hex_value[i:i + 2], 16) for i in (0, 2, 4)]
        tone_bgrs.append([b, g, r])
        bar = create_color_bar(tone_height, bar_width, [b, g, r])
        palette_bars.append(bar)
    padding_height = report_image.shape[0] - tone_height * len(skin_tone_palette)
    if padding_height > 0:
        padding = create_color_bar(padding_height, bar_width, (255, 255, 255))
        palette_bars.append(padding)
    bar = np.vstack(palette_bars)

    padding = 1
    start_point = (padding, tone_id * tone_height + padding)
    end_point = (bar_width - padding, (tone_id + 1) * tone_height)
    bar = cv2.rectangle(bar, start_point, end_point, (255, 0, 0), 2)
    return bar


def create_message_bar(dmnt_colors, dmnt_props, tone_hex, distance, bar_width):
    msg_bar = create_color_bar(height=50, width=bar_width, color=(243, 239, 214))
    b, g, r = np.around(dmnt_colors[0]).astype(int)
    dominant_color_hex = '#%02X%02X%02X' % (r, g, b)
    prop = f'{dmnt_props[0] * 100:.2f}%'

    font, font_scale, txt_colr, thickness, line_type = cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA
    x, y = 2, 15
    msg = f'- Dominant color: {dominant_color_hex}, proportion: {prop}'
    cv2.putText(msg_bar, msg, (x, y), font, font_scale, txt_colr, thickness, line_type)

    text_size, _ = cv2.getTextSize(msg, font, font_scale, thickness)
    line_height = text_size[1] + 10
    accuracy = round(100 - distance, 2)
    cv2.putText(msg_bar, f'- Skin tone: {tone_hex}, accuracy: {accuracy}', (x, y + line_height), font, font_scale,
                txt_colr, thickness, cv2.LINE_AA)

    return msg_bar


def process(image: np.ndarray, is_bw: bool, to_bw: bool, skin_tone_palette: list, tone_labels: list = None, new_width=-1, n_dominant_colors=2,
            scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), biggest_only=True,
            verbose=False, model=False, distance=100):
    image = resize(image, new_width)

    records, report_images = {}, {}
    # TODO: take out face coords 
    face_coords = detect_faces(image, scaleFactor, minNeighbors, minSize, biggest_only)
    n_faces = len(face_coords)

    if is_bw:
        records['NA'] = []
        report_images['NA'] = image
    else:
        record, report_image = classify(image, is_bw, to_bw, skin_tone_palette, tone_labels, n_dominant_colors, verbose=verbose, use_face=False, model=model, distance=distance)
        records['NA'] = record
        report_images['NA'] = report_image

    return records, report_images


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face_report_image(face, idx, image):
    if image is None:
        return None
    x1, y1, x2, y2 = face
    width = x2 - x1
    height = 20
    bar = np.ones((height, width, 3), dtype=np.uint8) * (255, 0, 0)
    report_image = image.copy()
    report_image[y2:y2 + height, x1:x2] = bar
    txt = f'Face {idx + 1}'
    text_color = (255, 255, 255)
    font_scale = 0.5
    thickness = 1
    text_size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = x1 + (width - text_size[0]) // 2
    text_y = y2 + 15
    cv2.putText(report_image, txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return report_image

def hex_to_bgr(hex_color):
   # Convert hex color to RGB format
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb_color[::-1]
