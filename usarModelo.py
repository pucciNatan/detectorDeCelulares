from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

modelo_path = "runs/detect/train/weights/best.pt"
model = YOLO(modelo_path)

img_path = "imgTeste.jpeg"

if not os.path.exists(img_path):
    print(f"⚠️ Imagem '{img_path}' não encontrada.")
else:
    results = model.predict(source=img_path, conf=0.01)[0]

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        class_id = int(cls.item())
        class_name = model.names[class_id]
        confidence = conf.item()

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{class_name} ({confidence:.2f})", fill="red")

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predição com modelo salvo: {os.path.basename(modelo_path)}")
    plt.show()
