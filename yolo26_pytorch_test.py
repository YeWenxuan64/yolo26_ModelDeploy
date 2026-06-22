import sys
from pathlib import Path


current_dir = Path(__file__).parent.resolve()
project_root = current_dir / 'models_convert/original/ultralytics'

class temporary_sys_path:
    def __init__(self, new_path: str):
        self.new_path = str(new_path)

    def __enter__(self):
        sys.path.insert(0, self.new_path)
        return self

    def __exit__(self, etype, value, traceback):
        if self.new_path in sys.path:
            sys.path.remove(self.new_path)


sys.path.append(str(current_dir))
sys.path.append(str(project_root))

with temporary_sys_path(current_dir):
    from models_convert.original.ultralytics.ultralytics import YOLO

model_path = str(current_dir / "models_convert/original/yolo26s.pt")
video_path = str(current_dir.parent / "datasets/loco640.mp4")


if __name__ == "__main__":
    model = YOLO(model_path)
    result = model(video_path, show=True)



