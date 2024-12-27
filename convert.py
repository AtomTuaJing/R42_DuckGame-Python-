import os
import xml.etree.ElementTree as ET

# Define paths
xml_folder = "C:/Users/usEr/Documents/Duck_Catching_Game/Python/data/labels/train"
output_folder = "C:/Users/usEr/Documents/Duck_Catching_Game/Python/data/labels"
classes_file = "classes.txt"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load class names into a list
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def convert_bbox(size, box):
    """Convert VOC bbox to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return x_center, y_center, width, height

# Process each XML file
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Prepare output file
    output_file = os.path.join(output_folder, os.path.splitext(xml_file)[0] + ".txt")
    with open(output_file, "w") as out_file:
        # Parse object data
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_names:
                continue  # Skip unknown classes
            class_id = class_names.index(class_name)

            bndbox = obj.find("bndbox")
            x_min = float(bndbox.find("xmin").text)
            y_min = float(bndbox.find("ymin").text)
            x_max = float(bndbox.find("xmax").text)
            y_max = float(bndbox.find("ymax").text)

            # Convert to YOLO format
            bbox = convert_bbox((width, height), (x_min, x_max, y_min, y_max))
            out_file.write(f"{class_id} {' '.join(map(str, bbox))}\n")

print("Conversion completed!")
