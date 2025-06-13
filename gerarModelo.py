from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

results = model.val()
metrics = results.results_dict

print("\n📊 Avaliação do Modelo:")
print(f"Precisão média:     {metrics.get('metrics/precision(B)', 0):.4f}")
print(f"Revocação média:    {metrics.get('metrics/recall(B)', 0):.4f}")
print(f"mAP@0.5:            {metrics.get('metrics/mAP50(B)', 0):.4f}")
print(f"mAP@0.5:0.95:       {metrics.get('metrics/mAP50-95(B)', 0):.4f}")

img_path = "imgTeste.jpeg"
if not os.path.exists(img_path):
    print("\n⚠️ Coloque uma imagem chamada 'teste.jpg' na raiz do projeto para ver a predição.")
else:
    pred = model.predict(img_path)[0]

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        class_id = int(cls.item())
        class_name = model.names[class_id]
        confidence = conf.item()

        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 10), f"{class_name} ({confidence:.2f})", fill="green")

    plt.imshow(img)
    plt.axis("off")
    plt.title("Predição com YOLOv8")
    plt.show()
