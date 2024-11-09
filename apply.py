import argparse
from ultralytics import YOLO
import os
import xml.etree.ElementTree as ET


def get_image_paths(input_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def save_results_to_xml(results, output_file):
    root = ET.Element("Results")

    for result in results:
        img_element = ET.SubElement(root, "Image", path=result['path'])

        for box in result['boxes']:
            box_element = ET.SubElement(img_element, "Box")
            box_element.set("x1", str(box[0]))  # 左上角 x 坐标
            box_element.set("y1", str(box[1]))  # 左上角 y 坐标
            box_element.set("x2", str(box[2]))  # 右下角 x 坐标
            box_element.set("y2", str(box[3]))  # 右下角 y 坐标

            # 如果需要，你还可以添加类别等其他信息
            # box_element.set("class", str(box[4]))  # 类别索引，如果有的话

    tree = ET.ElementTree(root)
    tree.write(output_file)


def main(input_dir, model_path, output_file):
    model = YOLO(model_path)

    image_paths = get_image_paths(input_dir)
    results = []

    for img_path in image_paths:
        result = model(img_path, save=True)

        if isinstance(result, list) and len(result) > 0:
            result_data = result[0]  # 获取第一张图像的结果
            boxes = result_data.boxes.xyxy.tolist()  # 获取框的坐标
        else:
            boxes = []

        results.append({
            'path': img_path,
            'boxes': boxes,
        })

    save_results_to_xml(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('-dir', '--input_dir', required=True, help='Input directory containing images')
    parser.add_argument('-model', '--model_path', required=True, help='Path to the trained model')
    parser.add_argument('-out', '--output_file', required=True, help='Output XML file path')

    args = parser.parse_args()
    main(args.input_dir, args.model_path, args.output_file)
