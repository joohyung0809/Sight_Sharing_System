import xml.etree.ElementTree as ET
import os
import shutil

def convert_to_yolo_format(xml_file, output_folder, class_index, images_output_folder):
    # XML 파일 파싱
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for image in root.findall('image'):
        image_name = image.get('name')
        # 지정된 output_folder에 TXT 파일 경로 설정
        txt_file_name = os.path.join(output_folder, image_name.replace('.jpg', '.txt'))
        yolo_segments = []

        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            if label == 'sidewalk':
                points = polygon.get('points').split(';')
                # YOLO 형식으로 변환
                yolo_points = []
                for point in points:
                    x, y = map(float, point.split(','))
                    # x와 y를 이미지 너비와 높이로 나누어 정규화
                    normalized_x = x / 1920.0
                    normalized_y = y / 1080.0
                    yolo_points.append((normalized_x, normalized_y))
                
                # YOLO 형식 문자열 생성
                yolo_string = f"{class_index} " + ' '.join(f"{x:.6f} {y:.6f}" for x, y in yolo_points)
                yolo_segments.append(yolo_string)

        # sidewalk가 있을 경우만 TXT 파일에 저장하고 이미지 복사
        if yolo_segments:
            # TXT 파일 저장
            with open(txt_file_name, 'w') as f:
                for segment in yolo_segments:
                    f.write(segment + '\n')
            # 이미지 복사
            image_path = os.path.join(os.path.dirname(xml_file), image_name)
            if os.path.exists(image_path):
                shutil.copy(image_path, images_output_folder)

def process_surface_folders(root_folder, output_folder, class_index, images_output_folder):
    # output_folder가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(images_output_folder, exist_ok=True)

    # root_folder 내 모든 폴더를 탐색
    for folder_name in os.listdir(root_folder):
        if folder_name.startswith("Surface_"):
            folder_path = os.path.join(root_folder, folder_name)
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xml'):
                    xml_file = os.path.join(folder_path, file_name)
                    convert_to_yolo_format(xml_file, output_folder, class_index, images_output_folder)

# 사용 예시
root_folder = '/root/hayong/YOLO_data/Surface_5'  # 탐색할 루트 폴더 경로로 변경
output_folder = '/root/hayong/YOLO_data/train/labels/'  # 저장할 TXT 파일의 경로를 설정
images_output_folder = '/root/hayong/YOLO_data/train/images/'  # 이미지가 복사될 경로
class_index = 0  # sidewalk에 대한 클래스 인덱스를 설정

process_surface_folders(root_folder, output_folder, class_index, images_output_folder)
