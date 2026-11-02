"""YOLOv7 本機網站：圖片辨識 / 影片辨識 / 上傳 zip 訓練，可以切換模型

執行：.\venv\Scripts\python.exe app.py  → 瀏覽器開 http://127.0.0.1:7861
（介面和 yolov8-test 的網站相同，背後改成呼叫 yolov7 原本的程式）
"""
import csv
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
V7 = ROOT / 'yolov7'          # yolov7 原始碼；它的 import 都是從這個資料夾開始，所以要加進 sys.path
DATASETS = ROOT / 'datasets'
RUNS = V7 / 'runs'
sys.path.insert(0, str(V7))
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
BAR = re.compile(r'\s*\d+%\|.*')  # 進度條的「xx%|███ 7/8 …」部分

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402
from utils.plots import plot_one_box  # noqa: E402

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
HALF = DEVICE.type != 'cpu'   # GPU 上用半精度，速度較快
_models = {}   # 已載入的模型快取：路徑 → (model, names, stride)
_train = {'proc': None}  # 目前的訓練行程（一次只跑一個）
USAGE_LOG = ROOT / 'usage.log'
_log_lock = threading.Lock()


# ---------------- 使用紀錄 ----------------
def log_usage(request, action, model='', file='', result=''):
    """每次辨識 / 訓練記一行到 usage.log（逗號分隔，Excel 可以直接開）"""
    h = request.headers if request else {}
    # 從 ngrok 連進來時 client.host 一律是 127.0.0.1，真正的 IP 在 x-forwarded-for
    ip = (h.get('x-forwarded-for') or '').split(',')[0].strip() or (request.client.host if request else '')
    row = [time.strftime('%Y-%m-%d %H:%M:%S'), ip, 'ngrok' if 'x-forwarded-for' in h else '本機',
           h.get('user-agent', ''), action, model, Path(file).name if file else '', result]
    with _log_lock:
        new = not USAGE_LOG.exists()
        with open(USAGE_LOG, 'a', newline='', encoding='utf-8-sig' if new else 'utf-8') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['時間', 'IP', '來源', '瀏覽器', '動作', '模型', '檔案', '結果'])
            w.writerow(row)


# ---------------- 模型 ----------------
OFFICIAL_DESC = {'yolov7': '標準版', 'yolov7_training': '標準版，訓練用的版本', 'yolov7-tiny': '最小最快'}


def read_names(data_yaml):
    names = yaml.safe_load(Path(data_yaml).read_text(encoding='utf-8'))['names']
    return list(names.values()) if isinstance(names, dict) else list(names)


def trained_classes(run_dir):
    """從訓練時的 opt.yaml 找出資料集，讀出類別名稱"""
    try:
        opt = yaml.safe_load((run_dir / 'opt.yaml').read_text(encoding='utf-8'))
        names = read_names(V7 / opt['data'])  # 相對路徑是相對於 yolov7 資料夾
        return '、'.join(names) if len(names) <= 6 else f'{len(names)} 種（{"、".join(names[:3])}…）'
    except Exception:
        return '未知'


def list_models():
    """下拉選單的選項：(顯示文字, 模型路徑)。自己訓練的放前面（新的優先），官方模型放後面"""
    choices = []
    for p in sorted(RUNS.glob('train/*/weights/best.pt'), key=lambda p: p.stat().st_mtime, reverse=True):
        run = p.parents[1]
        when = time.strftime('%m/%d %H:%M', time.localtime(p.stat().st_mtime))
        choices.append((f'自己訓練：{run.name}（{when}）— 認得：{trained_classes(run)}', str(p.relative_to(V7))))
    for stem, desc in OFFICIAL_DESC.items():
        if (V7 / f'{stem}.pt').exists():
            choices.append((f'官方：{stem}（{desc}）— 認得 COCO 80 種常見物件', f'{stem}.pt'))
    return choices


def get_model(name):
    if name not in _models:
        model = attempt_load(str(V7 / name), map_location=DEVICE)
        if HALF:
            model.half()
        names = model.module.names if hasattr(model, 'module') else model.names
        _models[name] = (model, list(names), int(model.stride.max()))
    return _models[name]


def refresh_models():
    choices = list_models()
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def class_choices(model_name):
    """切換模型時，更新「只顯示這些物件」的選項"""
    names = get_model(model_name)[1] if model_name else []
    return gr.Dropdown(choices=names, value=[], multiselect=True)


# ---------------- 辨識共用 ----------------
COLORS = [[int(c) for c in np.random.default_rng(i).integers(0, 255, 3)] for i in range(1000)]  # 每個類別固定一個顏色


@torch.no_grad()
def predict(model_name, img0, conf, imgsz, iou, classes, max_det, augment, agnostic):
    """對一張 BGR 圖片做辨識，回傳 [(類別名稱, 分數, x1, y1, x2, y2), ...]（和 yolov7 detect.py 的流程相同）"""
    model, names, stride = get_model(model_name)
    imgsz = check_img_size(int(imgsz), s=stride)
    img = letterbox(img0, imgsz, stride=stride)[0]              # 等比例縮放 + 補灰邊
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR→RGB、HWC→CHW
    img = torch.from_numpy(img).to(DEVICE)
    img = (img.half() if HALF else img.float()) / 255.0
    pred = model(img[None], augment=augment)[0]
    ids = [names.index(c) for c in classes] or None
    det = non_max_suppression(pred, conf, iou, classes=ids, agnostic=agnostic)[0]
    det = det[:int(max_det)]                                    # NMS 已依分數排序，只留前 N 個
    det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape).round()  # 座標換回原圖大小
    return [(names[int(c)], float(s), *[int(v) for v in xyxy]) for *xyxy, s, c in det.tolist()]


def draw(img0, boxes, names, line_width, show_labels, show_conf):
    img = img0.copy()
    for name, score, *xyxy in boxes:
        label = (name if show_labels else '') + (f' {score:.2f}' if show_conf else '')
        plot_one_box(xyxy, img, color=COLORS[names.index(name)], label=label.strip() or None,
                     line_thickness=int(line_width) or None)
    return img


# ---------------- 圖片 ----------------
def detect_image(model_name, img, conf, imgsz, iou, classes, max_det, augment, agnostic,
                 line_width, show_labels, show_conf, request: gr.Request):
    if img is None:
        raise gr.Error('請先上傳圖片')
    img0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes = predict(model_name, img0, conf, imgsz, iou, classes, max_det, augment, agnostic)
    rows = [[b[0], round(b[1], 3), *b[2:]] for b in boxes]
    summary = (f'找到 {len(rows)} 個：' + '、'.join(f'{k} {v} 個' for k, v in Counter(x[0] for x in rows).most_common())
               if rows else '沒有找到任何物件（可以試著把「最低把握程度」調低）')
    out = draw(img0, boxes, get_model(model_name)[1], line_width, show_labels, show_conf)
    log_usage(request, '圖片辨識', model_name, result=f'{len(rows)} 個框')
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), summary, rows


# ---------------- 影片 ----------------
def detect_video(model_name, video, vid_stride, conf, imgsz, iou, classes, max_det, augment, agnostic,
                 line_width, show_labels, show_conf, request: gr.Request, progress=gr.Progress()):
    if not video:
        raise gr.Error('請先上傳影片')
    names = get_model(model_name)[1]
    stride = max(int(vid_stride), 1)
    cap = cv2.VideoCapture(video)
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30) / stride  # 跳幀時輸出影片的 fps 也要跟著降，播放速度才正確
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // stride or None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = Path(tempfile.mkdtemp()) / f'{Path(video).stem}_det.mp4'
    # 直接用 ffmpeg 寫 H.264，瀏覽器才能播放（OpenCV 的 mp4v 瀏覽器不支援）
    writer = imageio_ffmpeg.write_frames(str(out), (w, h), fps=fps, codec='libx264', pix_fmt_in='rgb24',
                                         macro_block_size=1, quality=7)
    writer.send(None)
    counts, n, i = Counter(), 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % stride:
            continue
        boxes = predict(model_name, frame, conf, imgsz, iou, classes, max_det, augment, agnostic)
        frame = draw(frame, boxes, names, line_width, show_labels, show_conf)
        writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
        counts.update({b[0] for b in boxes})
        n += 1
        if total:
            progress(min(n / total, 1), desc=f'{n}/{total} 幀')
    cap.release()
    writer.close()
    log_usage(request, '影片辨識', model_name, video, f'{n} 幀')
    summary = (f'影片共 {n} 個畫面（幀）。每種物件出現在幾個畫面裡：'
               + ('、'.join(f'{k} {v} 幀' for k, v in counts.most_common()) or '都沒有找到物件'))
    return str(out), summary


# ---------------- 訓練 ----------------
def find_split(root):
    """支援兩種結構：Roboflow 的 train/images、valid/images，以及 images/train、images/val"""
    train = val = None
    for d in sorted(root.rglob('*')):
        if not d.is_dir():
            continue
        n, p = d.name.lower(), d.parent.name.lower()
        if (n == 'images' and p in ('train', 'training')) or (p == 'images' and n.startswith('train')):
            train = train or d
        elif (n == 'images' and p in ('valid', 'val', 'validation')) or (p == 'images' and n.startswith('val')):
            val = val or d
    return train, val or train


def find_names(root):
    for f in root.rglob('*.yaml'):
        names = (yaml.safe_load(f.read_text(encoding='utf-8')) or {}).get('names')
        if names:
            return list(names.values()) if isinstance(names, dict) else list(names)
    for f in root.rglob('classes.txt'):
        return [x.strip() for x in f.read_text(encoding='utf-8').splitlines() if x.strip()]
    return None


def ascii_names(root):
    """把資料夾、檔名裡的中文、emoji 換成英數字。
    v7 用 cv2.imread 讀圖，Windows 上路徑有這些字元就會讀不到。圖片和標註用同一套規則換名，所以仍然對得起來"""
    renamed = 0
    for p in sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True):  # 由深到淺，先改裡面再改外面
        if not p.name.isascii():
            new = ''.join(c if c.isascii() else f'u{ord(c):x}' for c in p.stem).strip() or 'x'
            p.rename(p.with_name(new + p.suffix))
            renamed += 1
    return renamed


def prepare_dataset(zip_path, name):
    """解壓縮 zip → 找出 train/val 與類別名稱 → 產生 v7 格式的 data.yaml，回傳 (yaml 路徑, 說明文字)"""
    dest = DATASETS / name
    if dest.exists():
        raise gr.Error(f'datasets/{name} 已經存在，請換一個名稱')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    renamed = ascii_names(dest)
    train, val = find_split(dest)
    names = find_names(dest)
    if not train:
        raise gr.Error('zip 裡找不到 train 的 images 資料夾（需要 train/images 或 images/train）')
    if not names:
        raise gr.Error('zip 裡找不到類別名稱（需要含 names 的 yaml，或 classes.txt）')
    # v7 的 yaml 沒有 path 欄位，train/val 直接寫絕對路徑，另外要寫 nc（類別數）
    cfg = {'train': str(train), 'val': str(val), 'nc': len(names), 'names': names}
    yml = dest / 'data_local.yaml'
    yml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    n_train = sum(1 for _ in train.glob('*.*'))
    n_val = sum(1 for _ in val.glob('*.*'))
    info = (f'資料集：datasets/{name}\n訓練圖片 {n_train} 張（{train.relative_to(dest)}），'
            f'驗證圖片 {n_val} 張（{val.relative_to(dest)}）'
            + ('  ⚠ 沒有驗證集，用訓練集代替' if val == train else '') + f'\n類別：{names}\n'
            + (f'已把 {renamed} 個含中文或特殊符號的資料夾 / 檔名換成英數字（v7 讀不到這種路徑）\n' if renamed else ''))
    return yml, info


# 起始權重 → (模型結構 cfg, 預設超參數 hyp)
START = {'yolov7_training.pt': ('cfg/training/yolov7.yaml', 'data/hyp.scratch.custom.yaml'),
         'yolov7-tiny.pt': ('cfg/training/yolov7-tiny.yaml', 'data/hyp.scratch.tiny.yaml')}
# 專家設定：loss 與權重更新（預設值依起始模型的 hyp 檔而不同，切換起始模型時會自動帶入）
EXPERT_KEYS = ['box', 'cls', 'obj', 'cls_pw', 'obj_pw', 'fl_gamma', 'loss_ota', 'anchor_t', 'momentum', 'weight_decay']
HYP_KEYS = ['fliplr', 'flipud', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'scale', 'mosaic', 'lr0',
            *EXPERT_KEYS]  # 寫進 hyp.yaml 的參數，順序和介面上的 extra 一致


def expert_defaults(weights):
    """切換起始模型時，把專家設定換成該模型 hyp 檔的預設值"""
    hyp = yaml.safe_load((V7 / START[weights][1]).read_text(encoding='utf-8'))
    return [bool(hyp[k]) if k == 'loss_ota' else hyp[k] for k in EXPERT_KEYS]


def train(request: gr.Request, zip_file, data_yaml, name, weights, epochs, batch, imgsz, *extra):
    if _train['proc'] and _train['proc'].poll() is None:
        raise gr.Error('已經有訓練在進行中')
    name = re.sub(r'[^\w-]', '_', name.strip()) or 'exp'
    if (RUNS / 'train' / name).exists():
        raise gr.Error(f'runs/train/{name} 已經存在，請換一個名稱')

    if zip_file:
        yml, log = prepare_dataset(zip_file, name)
    elif data_yaml.strip():
        yml, log = Path(data_yaml.strip()), f'資料集：{data_yaml.strip()}\n'
        if not yml.is_absolute():
            yml = V7 / yml  # 相對路徑當作相對於 yolov7 資料夾（例如 data/coco128.yaml）
    else:
        raise gr.Error('請上傳 zip，或填寫現有的 data yaml 路徑')

    # v7 的資料變化、學習速度都寫在 hyp.yaml：讀出預設值，換成網頁上的設定，存成新檔
    cfg, hyp_file = START[weights]
    *hyp_vals, freeze, single_cls, cache = extra
    hyp = yaml.safe_load((V7 / hyp_file).read_text(encoding='utf-8'))
    hyp.update({k: int(v) if k == 'loss_ota' else float(v) for k, v in zip(HYP_KEYS, hyp_vals)})
    hyp_path = Path(tempfile.mkdtemp()) / 'hyp.yaml'   # train.py 會把它複製一份到 runs/train/<name>/hyp.yaml
    hyp_path.write_text(yaml.safe_dump(hyp, sort_keys=False), encoding='utf-8')

    cmd = [sys.executable, 'train.py', '--data', str(yml), '--weights', weights, '--cfg', cfg, '--hyp', str(hyp_path),
           '--epochs', str(int(epochs)), '--batch-size', str(int(batch)), '--img-size', str(int(imgsz)), str(int(imgsz)),
           '--workers', '0', '--name', name]   # Windows 上 workers 一定要 0
    if int(freeze):
        cmd += ['--freeze', str(int(freeze))]
    if single_cls:
        cmd.append('--single-cls')
    if cache:
        cmd.append('--cache-images')
    log += ('執行：' + ' '.join(cmd[1:]) + '\n調整過的超參數：'
            + ', '.join(f'{k}={hyp[k]}' for k in HYP_KEYS) + '\n\n')
    log_usage(request, '開始訓練', weights, zip_file or str(yml), f'名稱 {name}，{int(epochs)} 輪')
    yield log, None, gr.Dropdown()

    # text=True 會把進度條的 \r 也當成換行，所以可以逐行讀取
    proc = _train['proc'] = subprocess.Popen(cmd, cwd=V7, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding='utf-8', errors='replace',
                                             env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    lines, last = [], 0.0
    for line in proc.stdout:
        line = ANSI.sub('', line).rstrip()
        if not line:
            continue
        # 同一個 epoch 的進度條、驗證的進度條會一直更新，只保留最後一行
        if lines and (re.match(r'\s*\d+/\d+\s', line) and re.match(r'\s*\d+/\d+\s', lines[-1])
                      and line.split()[0] == lines[-1].split()[0]
                      or '%|' in line and BAR.sub('', line) == BAR.sub('', lines[-1])):
            lines[-1] = line
        else:
            lines.append(line)
        if time.time() - last > 0.5:
            last = time.time()
            yield log + '\n'.join(lines[-40:]), None, gr.Dropdown()
    proc.wait()

    result_png = RUNS / 'train' / name / 'results.png'
    status = '✅ 訓練完成' if proc.returncode == 0 else f'❌ 訓練失敗或已停止（exit {proc.returncode}）'
    log_usage(request, '訓練結束', weights, result=f'名稱 {name}：{status[2:]}')
    yield (log + '\n'.join(lines[-40:]) + f'\n\n{status}',
           str(result_png) if result_png.exists() else None, refresh_models())


def stop_train(request: gr.Request):
    p = _train['proc']
    if p and p.poll() is None:
        p.terminate()
        log_usage(request, '停止訓練')
        return '已要求停止訓練，幾秒後就會結束（已經存下的 last.pt 可以之後接續訓練）'
    return '目前沒有正在進行的訓練'


# ---------------- 介面 ----------------
# 「使用說明」分頁的內容（README.md 裡也有一份相同的說明）
GUIDE = """
## 這個網站能做什麼
1. **辨識**：上傳圖片或影片，讓模型把找到的東西框起來，並告訴你是什麼、有幾個。
2. **訓練**：上傳你自己標註好的圖片，教模型認得新的東西（例如安全帽的種類），訓練好的模型可以馬上拿來辨識。

## 先懂幾個名詞
| 名詞 | 意思 |
|---|---|
| 模型 | 已經學會認東西的「大腦」。官方模型認得人、車、狗等 80 種常見東西；自己訓練的只認得你教它的類別 |
| 框 | 模型找到一個東西，就在它周圍畫一個方框，旁邊寫上名稱和分數 |
| 把握程度（分數） | 0~1 之間，模型對這個框有多確定。0.9 = 很確定，0.3 = 不太確定 |
| 類別 | 模型認得的東西種類，例如 person、car、helmet |
| 幀 | 影片是由一張張圖片組成的，每一張叫一幀。一秒的影片通常有 30 幀 |
| 訓練輪數（epoch） | 把整份訓練資料從頭到尾看一遍叫一輪 |

## 辨識：怎麼用
1. 最上方「使用的模型」選一個模型（第一次用選「官方：yolov7」）。
2. 切到「圖片辨識」或「影片辨識」分頁，上傳檔案，按「開始辨識」。
3. 右邊會出現畫好框的結果，下面會寫找到了什麼、各幾個。圖片還會列出每個框的座標。

**結果不理想時，該調哪裡：**
| 遇到的狀況 | 調整方法 |
|---|---|
| 明明有東西，卻沒框到 | 「最低把握程度」調低（例如 0.15）；東西很小就把「辨識解析度」調到 960 或 1280 |
| 框到一堆不是的東西（誤判） | 「最低把握程度」調高（例如 0.5） |
| 同一個東西被框了好幾次 | 「重疊框合併門檻」調低（例如 0.3） |
| 擠在一起的人只框到一個 | 「重疊框合併門檻」調高（例如 0.6） |
| 只想看某幾種東西 | 「只顯示這些物件」選你要的類別 |
| 文字太多擋住畫面 | 「進階辨識設定」裡關掉名稱或分數 |
| 影片太長、跑很久 | 「每隔幾個畫面辨識一次」調大，例如 3 |

## 訓練：怎麼教模型認新東西
**1. 準備圖片**：收集你要辨識的場景照片，至少 100~200 張，越多越好。從影片擷取也可以。光線、角度、遠近要多樣，盡量接近實際要用的畫面。

**2. 標註**：用 [Roboflow](https://roboflow.com)（網頁版，免安裝）把每張圖裡的東西框起來，並選好類別名稱。
- 每個要辨識的東西都要框，漏框會讓模型以為那不是目標
- 類別名稱用英文，例如 `full_helmet`、`half_helmet`、`no_helmet`

**3. 匯出**：在 Roboflow 選 Export → 格式選 **YOLOv7** → 下載 zip。zip 裡會有 train、valid 資料夾和 data.yaml。

**4. 訓練**：到「訓練新模型」分頁，上傳 zip，取一個名稱（例如 helmet-v1），其他先用預設值，按「開始訓練」。
- 時間參考：這台電腦用 yolov7 訓練 128 張圖，每輪大約 2~3 分鐘。1000 張、50 輪大概要好幾個小時
- 訓練中可以按「停止訓練」，已經學到的會保留

**5. 看結果**：訓練完成後會顯示訓練曲線圖。
- 名稱有 loss 的曲線：**越來越低**代表模型在進步
- 名稱有 mAP 的曲線：**越來越高**代表越來越準（最高 1）。mAP@0.5 到 0.7 以上通常就堪用
- 曲線最後還在往好的方向走 → 再多訓練幾輪會更好

**6. 使用**：按最上方的「🔄 重新讀取模型清單」，選「自己訓練：helmet-v1」，就能用它辨識了。

**訓練效果不好時：**
| 狀況 | 怎麼辦 |
|---|---|
| 出現 out of memory 錯誤 | 「每次一起學幾張圖」調成 2，或起始模型改用 yolov7-tiny |
| mAP 很低 | 先檢查標註有沒有漏框、框錯；再增加圖片數量和訓練輪數 |
| 某一類特別差 | 那一類的圖片通常太少，多補一些 |
| 實際影片效果比訓練時差 | 訓練圖片和實際畫面差太多（角度、光線、距離），從實際影片擷取畫面來標註 |

## 注意事項
- 訓練時顯示卡會全力運作，請等訓練結束再辨識
- 一次只能進行一個訓練
- 每個參數的說明都寫在它的下方，「進階」區塊裡的設定通常不用改
"""

# 標記網頁語言為中文並禁止瀏覽器自動翻譯（Chrome 曾把「信心門檻」翻成「信心基金會」）
HEAD = ('<meta name="google" content="notranslate">'
        '<script>document.documentElement.lang="zh-Hant";'
        'document.documentElement.setAttribute("translate","no");</script>')

with gr.Blocks(title='YOLOv7 物件辨識') as demo:
    gr.Markdown('# YOLOv7 物件辨識\n'
                '選好模型後，上傳圖片或影片就能看到辨識結果。也可以上傳自己標註好的資料，訓練一個新的模型。'
                '**第一次使用？請先看最右邊的「📖 使用說明」分頁。**')
    with gr.Row():
        first = list_models()
        model_dd = gr.Dropdown(first, value=first[0][1] if first else None, label='使用的模型', scale=4,
                               info='官方模型只認得 COCO 資料集的 80 種常見物件（人、車、動物…）；自己訓練的模型只認得你標註的類別')
        refresh_btn = gr.Button('🔄 重新讀取模型清單', scale=1)

    # ---- 辨識參數（圖片、影片共用）----
    with gr.Row():
        conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label='最低把握程度',
                         info='模型會替每個框打 0~1 的分數，代表它有多確定。低於這個分數的框不顯示。'
                              '調高 → 框變少、誤判變少；調低 → 框變多、比較不會漏掉')
        iou = gr.Slider(0.1, 0.95, value=0.45, step=0.05, label='重疊框合併門檻',
                        info='兩個框重疊超過這個比例時，只留分數高的那個。同一個東西出現好幾個框 → 調低；'
                             '擠在一起的人被合併成一個 → 調高')
        imgsz = gr.Dropdown([320, 640, 960, 1280], value=640, label='辨識解析度',
                            info='圖片會先縮放到這個大小再辨識。物件很小（例如遠處的人頭）時調大，但速度會變慢')
    classes = gr.Dropdown(choices=get_model(first[0][1])[1] if first else [], value=[],
                          multiselect=True, label='只顯示這些物件',
                          info='留空 = 全部顯示。例如只選 person、motorcycle，其他東西就不會畫出來')
    with gr.Accordion('進階辨識設定（通常不用改）', open=False):
        with gr.Row():
            max_det = gr.Number(value=300, precision=0, minimum=1, maximum=300, label='一張圖最多幾個框',
                                info='v7 最多 300 個')
            augment = gr.Checkbox(value=False, label='加強辨識',
                                  info='把圖片翻轉、縮放後多辨識幾次再合併。比較準，但約慢 3 倍')
            agnostic = gr.Checkbox(value=False, label='重疊的框不分種類只留一個',
                                   info='同一個東西被判成兩種類別時（例如同一頂安全帽同時被判成全罩和 3/4 罩）打開')
        with gr.Row():
            line_width = gr.Number(value=0, precision=0, label='框線粗細',
                                   info='0 = 依照圖片大小自動決定')
            show_labels = gr.Checkbox(value=True, label='顯示物件名稱')
            show_conf = gr.Checkbox(value=True, label='顯示把握程度分數',
                                    info='物件很密集、文字擋住畫面時可以關掉')
    DET = [conf, imgsz, iou, classes, max_det, augment, agnostic, line_width, show_labels, show_conf]

    refresh_btn.click(refresh_models, outputs=model_dd)
    model_dd.change(class_choices, model_dd, classes)

    with gr.Tab('圖片辨識'):
        with gr.Row():
            img_in = gr.Image(type='numpy', label='上傳圖片')
            img_out = gr.Image(type='numpy', label='辨識結果')
        img_btn = gr.Button('開始辨識', variant='primary')
        img_sum = gr.Textbox(label='找到了什麼')
        img_tbl = gr.Dataframe(headers=['物件', '把握程度', '左上 x', '左上 y', '右下 x', '右下 y'],
                               label='每個框的詳細資料（座標單位：像素）')
        img_btn.click(detect_image, [model_dd, img_in, *DET], [img_out, img_sum, img_tbl])

    with gr.Tab('影片辨識'):
        with gr.Row():
            vid_in = gr.Video(label='上傳影片')
            vid_out = gr.Video(label='辨識結果')
        vid_stride = gr.Slider(1, 10, value=1, step=1, label='每隔幾個畫面辨識一次',
                               info='1 = 每個畫面都辨識。影片很長、只想快速看效果時調大（例如 3 = 只處理 1/3 的畫面，快 3 倍）')
        vid_btn = gr.Button('開始辨識', variant='primary')
        vid_sum = gr.Textbox(label='找到了什麼')
        vid_btn.click(detect_video, [model_dd, vid_in, vid_stride, *DET], [vid_out, vid_sum])

    with gr.Tab('訓練新模型'):
        gr.Markdown('**步驟**：上傳標註好的資料集壓縮檔 → 設定參數 → 按「開始訓練」。\n\n'
                    '- 壓縮檔格式：YOLO 格式（在 Roboflow 匯出時選 **YOLOv7**），裡面要有 train 和 valid 資料夾，以及寫有類別名稱的 data.yaml\n'
                    '- 訓練完成後，新模型會自動出現在最上方的「使用的模型」清單裡\n'
                    '- v7 會跑完全部的訓練輪數（沒有自動提早停止），而且比 v8 慢很多，輪數請斟酌\n'
                    '- 訓練時顯示卡會全力運作，請等訓練結束後再做辨識')
        with gr.Row():
            zip_in = gr.File(file_types=['.zip'], type='filepath', label='資料集壓縮檔（.zip）')
            with gr.Column():
                data_in = gr.Textbox(label='或：使用電腦裡現有的資料集設定檔',
                                     placeholder='例如 data/coco128.yaml（相對於 yolov7 資料夾；有上傳壓縮檔時不用填）')
                name_in = gr.Textbox(value='my-run', label='這次訓練的名稱',
                                     info='用來命名資料夾，例如 helmet-v1。不能和之前用過的名稱重複')
                w_in = gr.Dropdown([('yolov7（標準版，推薦）', 'yolov7_training.pt'),
                                    ('yolov7-tiny（最小最快，準確度較低，記憶體用得少）', 'yolov7-tiny.pt')],
                                   value='yolov7_training.pt', label='從哪個模型開始訓練',
                                   info='會在官方模型的基礎上，學習你的資料')
        with gr.Row():
            ep_in = gr.Number(value=50, precision=0, label='訓練輪數',
                              info='整份資料要從頭看幾遍。先用 50 試試，曲線還在進步就再加')
            bs_in = gr.Number(value=4, precision=0, label='每次一起學幾張圖',
                              info='越大越快，但越吃顯示卡記憶體。8GB 顯卡用 yolov7 時建議 4；tiny 可以用 16')
            sz_in = gr.Dropdown([640, 960, 1280], value=640, label='訓練解析度',
                                info='物件很小時用 960 或 1280，記憶體不夠就搭配調小上一格')

        with gr.Accordion('進階訓練設定：資料變化（通常不用改）', open=False):
            gr.Markdown('訓練時會把圖片隨機做一些變化，讓模型看過更多情況（不同光線、角度、遠近），比較不會只認得訓練用的那幾張。'
                        '**某種變化在真實畫面中不會出現，就把它關掉（設成 0）。**')
            with gr.Row():
                fliplr = gr.Slider(0, 1, value=0.5, step=0.05, label='左右翻轉的機率',
                                   info='左右對稱的東西（安全帽、人、車）保持 0.5；有方向性的（文字、左右箭頭）設 0')
                flipud = gr.Slider(0, 1, value=0.0, step=0.05, label='上下翻轉的機率',
                                   info='道路、監視器畫面不會上下顛倒，保持 0；空拍、顯微鏡畫面可以設 0.5')
                degrees = gr.Slider(0, 180, value=0, step=5, label='隨機旋轉的角度',
                                    info='監視器畫面通常是正的，保持 0；空拍或零件這類方向不固定的可以設 10~180')
            with gr.Row():
                hsv_h = gr.Slider(0, 0.1, value=0.015, step=0.005, label='顏色（色相）變化',
                                  info='顏色本身就是分類依據時（例如紅燈綠燈、不同顏色的帽子代表不同身分）設 0')
                hsv_s = gr.Slider(0, 1, value=0.7, step=0.05, label='鮮豔度變化', info='模擬陰天、晴天的色彩差異')
                hsv_v = gr.Slider(0, 1, value=0.4, step=0.05, label='亮度變化', info='模擬白天、傍晚、陰影')
            with gr.Row():
                scale = gr.Slider(0, 0.9, value=0.5, step=0.05, label='隨機放大縮小的幅度',
                                  info='物件遠近差很多時可以調到 0.7')
                mosaic = gr.Slider(0, 1, value=1.0, step=0.05, label='4 張拼 1 張的機率',
                                   info='讓一張圖裡有更多小物件和不同背景，對小物件很有幫助，建議保持 1')

        with gr.Accordion('進階訓練設定：學習方式（通常不用改）', open=False):
            with gr.Row():
                lr0 = gr.Number(value=0.01, label='學習速度',
                                info='每次調整模型的幅度。訓練曲線上下亂跳、不穩定時調低成 0.005')
                freeze = gr.Number(value=0, precision=0, label='凍結前幾層',
                                   info='0 = 全部都訓練。資料很少（100 張以下）時，yolov7 可以設 50（凍結 backbone），只微調後半段')
            with gr.Row():
                single_cls = gr.Checkbox(value=False, label='所有類別當成同一類',
                                         info='只想知道「有沒有」，不分種類時打開')
                cache = gr.Checkbox(value=False, label='先把圖片載入記憶體',
                                    info='資料集不大（幾千張以內）時打開，訓練會比較快')

        with gr.Accordion('專家設定：loss 與權重更新（不確定就不要改）', open=False):
            ed = dict(zip(EXPERT_KEYS, expert_defaults('yolov7_training.pt')))
            gr.Markdown('loss 是模型「錯了多少」的分數，訓練就是不斷讓 loss 變小。v7 的 loss 由三部分加總：'
                        '**box**（框的位置準不準）、**obj**（這裡有沒有東西）、**cls**（是哪一類）。'
                        '下面的權重決定三者各佔多少比重，調高某一項 = 讓模型更在意那方面的錯誤。\n\n'
                        '- 預設值來自起始模型的 hyp 檔（yolov7 和 tiny 不同），切換「從哪個模型開始訓練」時會自動換回該模型的預設值\n'
                        '- train.py 會再依「類別數、解析度」自動換算這三個權重，這裡填的是換算前的基準值\n'
                        '- 一次只改一項，和原本的結果比較 mAP，才知道有沒有幫助')
            with gr.Row():
                box_g = gr.Number(value=ed['box'], label='box：框位置的權重',
                                  info='框的位置、大小常常不準（框太大、偏一邊）時調高，例如 0.05 → 0.075')
                obj_g = gr.Number(value=ed['obj'], label='obj：有沒有東西的權重',
                                  info='常常漏掉東西、或把背景當成東西時調整')
                cls_g = gr.Number(value=ed['cls'], label='cls：分類的權重',
                                  info='框得到但常分錯類（例如全罩被判成半罩）時調高，例如 0.3 → 0.5')
            with gr.Row():
                cls_pw = gr.Number(value=ed['cls_pw'], label='cls_pw：分類的正樣本加權',
                                   info='大於 1 = 更重視「是這一類」的樣本，類別很不平均時可以試 1.5~2')
                obj_pw = gr.Number(value=ed['obj_pw'], label='obj_pw：有東西的正樣本加權',
                                   info='大於 1 = 更重視「這裡有東西」，漏框很嚴重時可以試 1.5~2')
                fl_gamma = gr.Number(value=ed['fl_gamma'], label='focal loss 強度（fl_gamma）',
                                     info='0 = 關閉。設 1.5 讓模型專注在難認的樣本，類別極度不平均時可以試；v7 通常保持 0')
            with gr.Row():
                loss_ota = gr.Checkbox(value=ed['loss_ota'], label='使用 OTA 標籤分配（loss_ota）',
                                       info='v7 的預設，比較準。關掉會改用舊的分配方式，訓練較快、記憶體用得較少')
                anchor_t = gr.Number(value=ed['anchor_t'], label='anchor 比例門檻（anchor_t）',
                                     info='物件寬高和 anchor 差幾倍以內才算配對成功。物件形狀很特別（很細長）時才需要調大')
            with gr.Row():
                momentum = gr.Number(value=ed['momentum'], label='動量（momentum）',
                                     info='每次調整時保留多少上一次的方向，讓訓練比較穩定。幾乎不用改')
                weight_decay = gr.Number(value=ed['weight_decay'], label='權重衰減（weight_decay）',
                                         info='讓模型參數不要變太大、避免只背下訓練資料（過擬合）。'
                                              '訓練集分數很高但驗證集很差時，可以試 0.001')
        expert = [box_g, cls_g, obj_g, cls_pw, obj_pw, fl_gamma, loss_ota, anchor_t, momentum, weight_decay]
        w_in.change(expert_defaults, w_in, expert)

        with gr.Row():
            train_btn = gr.Button('開始訓練', variant='primary')
            stop_btn = gr.Button('停止訓練', variant='stop')
        train_log = gr.Textbox(label='訓練過程', lines=20, max_lines=20, autoscroll=True)
        train_png = gr.Image(label='訓練曲線（loss 越來越低、mAP 越來越高，代表模型有在進步）')
        # 順序要和 HYP_KEYS 一致，最後接 freeze、single_cls、cache
        extra = [fliplr, flipud, hsv_h, hsv_s, hsv_v, degrees, scale, mosaic, lr0, *expert, freeze, single_cls, cache]
        train_btn.click(train, [zip_in, data_in, name_in, w_in, ep_in, bs_in, sz_in, *extra],
                        [train_log, train_png, model_dd])
        stop_btn.click(stop_train, outputs=train_log)

    with gr.Tab('📖 使用說明'):
        gr.Markdown(GUIDE)

if __name__ == '__main__':
    demo.queue().launch(server_name='127.0.0.1', server_port=7861, inbrowser=True, head=HEAD)
