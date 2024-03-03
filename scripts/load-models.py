import os

MODELS_DIR = "models"
WEIGHTS_CACHE_DIR = "_weights_cache"


class ModelLoader:
    def load_yolo_v8(self):
        import ultralytics

        model = ultralytics.YOLO("yolov8n.pt")
        model.export(format="onnx")

        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)

        os.rename("yolov8n.onnx", f"{MODELS_DIR}/yolo-v8.onnx")
        # onnx_model = YOLO('yolov8n.onnx')


def main():
    model_loader = ModelLoader()
    model_loader.load_yolo_v8()


if __name__ == "__main__":
    main()
