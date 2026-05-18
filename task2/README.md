# HW2 Task2: VisDrone Detection + MOT + Line Crossing

## 1. 安装依赖
```bash
pip install -r requirements.txt
```

## 2. 转换 VisDrone 标注到 YOLO
```bash
python scripts/prepare_visdrone.py --root archive --include_test
```
转换后会生成：
- `.../labels/*.txt`（YOLO 格式）

## 3. 训练
默认脚本已设为 `yolov8m + 1280 + AdamW + AMP`：
```bash
python scripts/train_yolov8_1.py --data configs/visdrone.yaml --model yolov8m.pt --imgsz 1280 --epochs 140 --batch 6 --device 0
```
运行不同策略下重训练脚本类似

## 4. 模型评测
```bash
python scripts/evaluate.py
```
可进一步指定评测的模型权重等

## 5. 视频检测 + 跟踪 + 越线计数
```bash
python scripts/run_tasks_234.py
```
可指定使用的model路径，输出路径，输入的视频路径等

输出：
- `detect/crossing_events.csv`（跨线事件）
- `detect/track_log.csv`（逐帧轨迹日志）
- `tracking_count.mp4`
