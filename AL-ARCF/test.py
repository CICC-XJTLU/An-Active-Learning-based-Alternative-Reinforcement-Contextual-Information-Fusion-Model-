import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import variable
import importlib.util

# 指定模块路径
module_path = r'F:\博士课题\CENet-update\networks\subnet\CEmodule.py'

# 加载模块
spec = importlib.util.spec_from_file_location("CEmodule", module_path)
CEmodule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CEmodule)

# 导入模型类
CE = CEmodule.CE

# 加载模型
model = CE()
model.load_state_dict(torch.load('F:\博士课题\CENet-update\saved_models\3_EM_ATT_forward-MOSEI.pth'))
model.eval()

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 打开视频文件
video_path = 'F:\博士课题\CENet-update\dataset\MOSEI\6.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 视频帧预处理函数
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).unsqueeze(0)
    return frame.to(device)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # 预处理帧
    processed_frame = preprocess_frame(frame)

    # 执行预测
    with torch.no_grad():
        outputs = model(processed_frame)
        _, predicted = torch.max(outputs.data, 1)

    # 在控制台输出预测结果（示例）
    print(f"Frame prediction: {predicted.item()}")

    # 可选：在帧上绘制预测结果或其他信息
    # draw_predictions(frame, prediction)

    # 显示视频帧（可选）
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
