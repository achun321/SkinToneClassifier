import logging
import os
import shutil
import threading
from datetime import datetime
from functools import partial
from multiprocessing import freeze_support, cpu_count, Pool

import cv2
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import tkinter as tk
from tkinter import PhotoImage, Label
import os

from image import process, is_black_white, load_model, hex_to_bgr, euclidean_distance, blend_colors
from utils import build_arguments, alphabet_id, build_filenames, is_windows
from fcn import FCNResNet101


def patch_asscalar(a):
    return np.asarray(a).item()


setattr(np, "asscalar", patch_asscalar)

LOG = logging.getLogger(__name__)
lock = threading.Lock()


def process_image(filename, image_type_setting,
                  specified_palette, default_palette,
                  specified_tone_labels, default_tone_labels,
                  to_bw, new_width, n_dominant_colors,
                  scale, min_nbrs, min_size, verbose, model, distance):
    basename, extension = filename.stem, filename.suffix

    image: np.ndarray = cv2.imread(str(filename.resolve()), cv2.IMREAD_COLOR)
    if image is None:
        msg = f'{basename}.{extension} is not found or is not a valid image.'
        LOG.warning(msg)
        return {
            'basename': basename,
            'extension': extension,
            'message': msg,
        }
    is_bw = is_black_white(image)
    image_type = image_type_setting
    if image_type == 'auto':
        image_type = 'bw' if is_bw else 'color'
    if len(specified_palette) == 0:
        skin_tone_palette = default_palette['bw' if to_bw or is_bw else 'color']
    else:
        skin_tone_palette = specified_palette

    tone_labels = specified_tone_labels or default_tone_labels['bw' if to_bw or is_bw else 'color']
    if len(skin_tone_palette) != len(tone_labels):
        raise ValueError('argument -p/--palette and -l/--labels must have the same length.')

    try:
        records, report_images = process(image, is_bw, to_bw, skin_tone_palette, tone_labels,
                                         new_width=new_width, n_dominant_colors=n_dominant_colors,
                                         scaleFactor=scale, minNeighbors=min_nbrs, minSize=min_size, verbose=verbose, model=model, distance=distance)
        return {
            'basename': basename,
            'extension': extension,
            'image_type': image_type,
            'records': records,
            'report_images': report_images,
        }
    except Exception as e:
        msg = f'Error processing image {basename}: {str(e)}'
        LOG.error(msg)
        return {
            'basename': basename,
            'extension': extension,
            'message': msg,
        }


def main():
    # Setup logger
    now = datetime.now()
    os.makedirs('./log', exist_ok=True)

    logging.basicConfig(
        filename=now.strftime('./log/log-%y%m%d%H%M.log'),
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)4d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    args = build_arguments()

    filenames = build_filenames(args.images)
    is_single_file = len(filenames) == 1

    debug: bool = args.debug
    to_bw: bool = args.black_white
    graph: bool = args.graph
    group: bool = args.group

    default_tone_palette = {
        # Default scale
        # 'color': ["#373028", "#422811", "#513b2e", "#6f503c", "#81654f", "#9d7a54", "#bea07e", "#e5c8a6", "#e7c1b8", "#f3dad6",
        #           "#fbf2f3"],
        # Monk Scale for Skin Colors
        'color': ["#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96", "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"],
        # Refer to this paper:
        # Leigh, A., & Susilo, T. (2009). Is voting skin-deep? Estimating the effect of candidate ballot photographs on election outcomes.
        # Journal of Economic Psychology, 30(1), 61-70.
        'bw': ["#FFFFFF", "#F0F0F0", "#E0E0E0", "#D0D0D0", "#C0C0C0", "#B0B0B0", "#A0A0A0", "#909090", "#808080", "#707070", "#606060",
               "#505050", "#404040", "#303030", "#202020", "#101010", "#000000"]
    }

    specified_palette: list[str] = args.palette if args.palette is not None else []

    default_tone_labels = {
        'color': ['Monk' + alphabet_id(i) for i in range(len(default_tone_palette['color']))],
        'bw': ['B' + alphabet_id(i) for i in range(len(default_tone_palette['bw']))]
    }
    specified_tone_labels = args.labels

    for idx, ct in enumerate(specified_palette):
        if not ct.startswith('#') and len(ct.split(',')) == 3:
            r, g, b = ct.split(',')
            specified_palette[idx] = '#%02X%02X%02X' % (int(r), int(g), int(b))

    new_width = args.new_width
    n_dominant_colors = args.n_colors
    min_size = args.min_size[:2]
    scale = args.scale
    min_nbrs = args.min_nbrs
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    result_filename = os.path.join(output_dir, './result.csv')
    image_type_setting = args.image_type
    model = args.model
    distance = args.distance

    def write_to_csv(row: list):
        with lock:
            with open(result_filename, 'a', newline='', encoding='UTF8') as f:
                f.write(','.join(map(str, row)) + '\n')

    num_workers = cpu_count() if args.n_workers == 0 else args.n_workers

    pool = Pool(processes=num_workers)

    # Backup result.csv if exists
    if os.path.exists(result_filename):
        renamed_file = os.path.join(output_dir, './result_bak.csv')
        shutil.move(result_filename, renamed_file)
    header = 'file,image type,face id,' + ','.join(
        [f'dominant {i + 1},props {i + 1}' for i in range(n_dominant_colors)]) + ',skin tone,PERLA,accuracy(0-100)'
    write_to_csv(header.split(','))

    # Load checkpoint to model
    if model: 
        device = 'cpu'
        model = torch.load("pretrained/model_segmentation_skin_30.pth", map_location=torch.device(device))
        model = load_model(FCNResNet101, model)
        model.to(device).eval()
    else:
        model = False

    # Start
    partial_process_image = partial(process_image, image_type_setting=image_type_setting,
                                    specified_palette=specified_palette, default_palette=default_tone_palette,
                                    specified_tone_labels=specified_tone_labels, default_tone_labels=default_tone_labels,
                                    to_bw=to_bw, new_width=new_width, n_dominant_colors=n_dominant_colors,
                                    scale=scale, min_nbrs=min_nbrs, min_size=min_size, verbose=debug, model=model, distance=distance)

    with logging_redirect_tqdm():
        with tqdm(filenames, desc='Processing images', unit='images') as pbar:
            if graph:
                freq_dict = {}
            if group:
                group_dict = {}
            for result in pool.imap(partial_process_image, filenames):
                if 'message' in result:
                    write_to_csv([result['basename'], result['message']])
                    pbar.update()
                    continue

                basename = result['basename']
                extension = result['extension']
                image_type = result['image_type']
                records = result['records']
                report_images = result['report_images']

                pbar.set_description(f"Processing {basename}")
                n_faces = len(records)
                for face_id, record, in records.items():
                    if face_id == 'NA':
                        n_faces = 0  # Did not detect any faces
                    image_name = f'{basename}-{face_id}'
                    write_to_csv([image_name, image_type, face_id] + record)
                    pbar.set_postfix({
                        'Image Type': image_type,
                        '#Faces': n_faces,
                        'Face ID': face_id,
                        'Skin Tone': record[-3],
                        'Label': record[-2],
                        'Accuracy': record[-1]
                    })
 
                    if graph: 
                        if record[-3] not in freq_dict:
                            freq_dict[record[-3]] = [record[-2], 1]
                        else:
                            freq_dict[record[-3]][1] += 1
                    if group: 
                        # format: (skin_tone: [image_name, ...])  
                        if record[-3] not in group_dict:
                            group_dict[record[-3]] = [basename + extension]
                        else:
                            group_dict[record[-3]].append(basename + extension)
                if debug:
                    debug_dir = os.path.join(output_dir, f'./debug/{image_type}/faces_{n_faces}')
                    os.makedirs(debug_dir, exist_ok=True)
                    for face_id, report_image in report_images.items():
                        image_name = f'{basename}-{face_id}'
                        report_filename = os.path.join(debug_dir, f'{image_name}{extension}')
                        with lock:
                            cv2.imwrite(report_filename, report_image)
                        if is_single_file:
                            cv2.imshow(f'Skin Tone Classifier - {image_name}', report_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                pbar.update()

            if group:
                for hex_color, images in group_dict.items():
                    canvas_height = 220
                    canvas_width = 200 * len(images)
                    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

                    skin_tone_color = hex_to_bgr(hex_color)
                    cv2.rectangle(canvas, (0, 0), (canvas_width, 20), skin_tone_color, -1)
                    for i, image_path in enumerate(images):
                        image = cv2.imread("./test_images/" + image_path)
                        image = cv2.resize(image, (200, 200))  # Resize the image to fit the canvas
                        canvas[20:220, i * 200:(i + 1) * 200] = image

                    # Show the composite image in a pop-up window
                    cv2.imshow(f"Similar Skin Tone - {skin_tone_color}", canvas)
                    cv2.waitKey(0)  # Wait for a key press to close the window

            if graph:
                brightness_dict = {}
                for hex_color, data in freq_dict.items():
                    brightness = (int(hex_color[1:3], 16) + int(hex_color[3:5], 16) + int(hex_color[5:7], 16))/3
                    brightness_dict[hex_color] = (brightness, data[0], data[1])

                sorted_colors = sorted(brightness_dict.items(), key=lambda x: x[1][0]) # sort by brightness
                
                # format: (hex_color: (brightness, color_name, freq))
                color_names = [tuple[1][1] for tuple in sorted_colors]
                freqs = [tuple[1][2] for tuple in sorted_colors]
                colors = [tuple[0] for tuple in sorted_colors]

                # Plot the graph with colored bars
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_facecolor('#e6e6e6')
                bars = ax.bar(color_names, freqs, color=colors, zorder=3)

                # Set the font family
                font_family = 'Palatino'

                # Configure the font properties
                rcParams['font.family'] = font_family

                # Customize the appearance of the graph
                ax.set_xlabel('Color Names', fontsize=12, fontname=font_family)
                ax.set_ylabel('Frequency', fontsize=12, fontname=font_family)
                ax.set_title('Color Frequency by Brightness', fontsize=14, fontname=font_family)
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                # Add data labels to the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom', fontname=font_family)

                # Remove the spines
                ax.spines[:].set_visible(False)

                # Add white grid lines
                plt.grid(color='white', linestyle='-', linewidth=2, alpha=0.3, zorder=2)

                # Show the graph
                plt.tight_layout()
                plt.show()



if __name__ == '__main__':
    if is_windows():
        freeze_support()
    main()
