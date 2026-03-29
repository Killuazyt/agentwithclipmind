# LEVIR-MCI 双时相变化理解纯视觉 Baseline（PyTorch）

本项目是一个可直接运行、可继续扩展的 LEVIR-MCI 双时相变化理解基线：
- 主任务：`change / no_change` 二分类
- 辅任务：`object / action / location` 多标签预测
- 第一版仅使用视觉分支（不含 CLIP、不含文本编码）

## 1. 项目结构

```text
Code/
├─ src/
│  ├─ __init__.py
│  ├─ configs.py
│  ├─ dataset.py
│  ├─ label_encoder.py
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ model.py
│  ├─ trainer.py
│  └─ utils.py
├─ outputs/
├─ train.py
├─ test.py
├─ requirements.txt
└─ README.md
```

## 2. 数据目录格式

默认配置按以下路径读取数据：
- 标签文件：`c:/Users/Killua/Desktop/clip+/LEVIR-MCI-dataset/label.json`
- 图像根目录：`c:/Users/Killua/Desktop/clip+/LEVIR-MCI-dataset/images`

图像目录组织：

```text
images/
├─ train/
│  ├─ A/   # T1
│  └─ B/   # T2
├─ val/
│  ├─ A/
│  └─ B/
└─ test/
   ├─ A/
   └─ B/
```

`label.json` 每条样本需包含字段：
- `filename`
- `split` (`train` / `val` / `test`)
- `changeflag` (0/1)
- `object_labels`
- `action_labels`
- `location_labels`

程序会按 `split` + `filename` 到 `{split}/A/{filename}` 与 `{split}/B/{filename}` 查找图像；若缺失会报出具体绝对路径。

## 3. 标签定义（固定顺序）

- `changeflag`：0/1
- `object_labels`：`["building", "road", "vegetation"]`
- `action_labels`：`["add", "remove", "replace", "rebuild"]`
- `location_labels`：
  `[
  "top_left", "top_right", "bottom_left", "bottom_right", "center",
  "top", "bottom", "left", "right", "corner"
  ]`

所有多标签均编码为 multi-hot。

## 4. 模型说明

- Backbone：共享权重 `ResNet18`
- 双时相编码：`f1 = Enc(T1)`，`f2 = Enc(T2)`
- 融合：`concat([f1, f2, abs(f2 - f1)])`
- 任务头（MLP）：
  - `change_head -> 1` logit
  - `object_head -> 3` logits
  - `action_head -> 4` logits
  - `location_head -> 10` logits

## 5. 损失与指标

联合损失：

```text
total_loss = change_loss + lambda_obj*object_loss + lambda_act*action_loss + lambda_loc*location_loss
```

默认：
- `lambda_obj=1.0`
- `lambda_act=1.0`
- `lambda_loc=0.5`

指标：
- change：`accuracy / precision / recall / f1`
- object/action/location：`micro precision / micro recall / micro f1`

## 6. 安装依赖

```bash
pip install -r requirements.txt
```

## 7. 训练命令

在 `c:/Users/Killua/Desktop/clip+/Code` 下执行：

```bash
python train.py
```

常用覆盖参数示例：

```bash
python train.py --batch-size 16 --epochs 30 --learning-rate 1e-4 --image-size 224 --device auto
```

调试标签编码打印（默认打印前 3 个训练样本）：

```bash
python train.py --debug-print-samples 5
```

## 8. 测试命令

```bash
python test.py --test-checkpoint c:/Users/Killua/Desktop/clip+/Code/outputs/checkpoints/best.pt
```

可选参数示例：

```bash
python test.py --test-checkpoint c:/Users/Killua/Desktop/clip+/Code/outputs/checkpoints/best.pt --test-split test --threshold 0.5 --pred-csv-name test_predictions.csv
```

## 9. 输出文件说明

默认输出目录：`c:/Users/Killua/Desktop/clip+/Code/outputs`

- `train.log`：训练日志
- `test.log`：测试日志
- `checkpoints/last.pt`：最后一轮 checkpoint
- `checkpoints/best.pt`：最佳 checkpoint（按 `save_metric`，默认 `change_f1`）
- `best_metrics.json`：最佳轮指标摘要
- `test_predictions.csv`：逐样本预测结果

CSV 字段：
- `filename`
- `change_prob`
- `change_pred`
- `object_pred`
- `action_pred`
- `location_pred`

## 10. 可扩展性与假设

- 当前实现为纯视觉 baseline，未引入 CLIP/文本分支。
- `model.py` 中视觉编码与融合模块已独立，后续可并入文本分支再融合。
- 预留了类别不平衡参数接口：
  - `change_loss_pos_weight`
  - `object_loss_weight`
  - `action_loss_weight`
  - `location_loss_weight`

---

给 Cursor：如果一次输出太长，请分多次输出，但每次都必须给完整文件内容，不能只说省略。
