import sys
from pathlib import Path


current_dir = Path(__file__).parent.resolve()
project_root = current_dir / 'models_convert/original/ultralytics'


sys.path.append(str(current_dir))
sys.path.append(str(project_root))

from models_convert.original.ultralytics.ultralytics import YOLO

model_path = str(current_dir / "models_convert/original/yolo26s.pt")
model_path = str(current_dir / "models_convert/original/yolo26s-pose.pt")
video_path = str(current_dir.parent / "datasets/loco640.mp4")
# image_path = str(current_dir / "images/yolo26s-pose_det.jpg")


if __name__ == "__main__":
    model = YOLO(model_path)
    result = model(video_path, show=True, save=False)



