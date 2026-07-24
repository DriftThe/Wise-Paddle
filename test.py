from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
import time
import contextlib
import cv2
import typing

pil_img_1 = Image.open("./inputs/input_1.png").convert("RGB")
pil_img_2 = Image.open("./inputs/input_2.png").convert("RGB")
pil_img_3 = Image.open("./inputs/input_3.png").convert("RGB")
model_path = "./model/PP-DocLayoutV3"
model = AutoModelForObjectDetection.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

images = [pil_img_1, pil_img_2, pil_img_3]
inputs = processor(images=images, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    st = time.perf_counter()
    outputs = model(**inputs)

# print(outputs)
target_sizes = torch.tensor([[img.height, img.width] for img in images])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)
et = time.perf_counter()
print(f"Total Process Time:{et - st}")

# 结果是一个列表，每个元素对应一张图的结果
# with open("outputs.txt", "w") as f:
#     with contextlib.redirect_stdout(f):
#         print(outputs)
#
# with open("resutls.txt", "w") as f:
#     with contextlib.redirect_stdout(f):
#         for i, result in enumerate(results):
#             print(f"图片 {i + 1}:")
#             for box in result["boxes"]:
#                 print(box)

for i, result in enumerate(results):
    frame = cv2.imread(f"./inputs/input_{i + 1}.png")
    for box in result["boxes"]:
        LST = box.tolist()
        cv2.rectangle(frame, (int(LST[0]), int(LST[1])), (int(LST[2]), int(LST[3])), (255, 0, 0), 2)
    cv2.imwrite(f"./outputs/output_{i}.png", frame)
