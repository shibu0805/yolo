# ==============================================================
# detect.py － 物件偵測（推論）主程式
# 用訓練好的權重（.pt）對圖片、影片、資料夾、webcam 或串流做偵測，
# 並把畫好框的結果存到 runs/detect/exp*/。
# 範例：python detect.py --weights yolov7.pt --source inference/images/horses.jpg
# ==============================================================
# --- 標準函式庫與第三方套件 ---
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# --- YOLOv7 自己的模組 ---
# attempt_load：載入 .pt 權重成模型
from models.experimental import attempt_load
# LoadStreams：讀 webcam/串流；LoadImages：讀圖片/影片/資料夾
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# plot_one_box：在圖片上畫框和標籤
from utils.plots import plot_one_box
# select_device：選 GPU/CPU；TracedModel：把模型轉成 TorchScript 加速
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# 偵測主函式：讀入資料 → 模型推論 → NMS 過濾 → 畫框存檔
def detect(save_img=False):
    # 從命令列參數 opt 取出常用設定
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # 除非加了 --nosave，否則都會存結果圖（.txt 清單除外）
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # source 是數字（如 0）、.txt 清單或網路串流網址時，視為 webcam/串流模式
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 建立輸出資料夾，名稱自動遞增：exp、exp2、exp3...
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 初始化 log 與裝置（--device 0 用 GPU、cpu 用 CPU）
    # Initialize
    set_logging()
    device = select_device(opt.device)
    # GPU 上使用半精度 FP16，速度更快、更省記憶體
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 載入模型權重
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # stride：模型最大下採樣倍數（通常是 32），輸入尺寸必須是它的倍數
    stride = int(model.stride.max())  # model stride
    # 把 --img-size 調整成 stride 的倍數
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # 預設會 trace 模型（產生 traced_model.pt）加速推論；加 --no-trace 可關閉
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # 第二階段分類器（預設關閉，一般用不到）
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device, weights_only=False)['model']).to(device).eval()

    # 建立資料讀取器
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # 串流模式：檢查能不能開視窗顯示畫面
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # 圖片/影片模式
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 取得類別名稱，並替每個類別隨機產生一個框的顏色
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # GPU 預熱：先用一張全黑圖跑一次，讓後面的計時比較準
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # 主迴圈：逐張圖片／逐幀影片處理
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # numpy 圖片轉成 torch tensor，搬到 GPU
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # 像素值正規化到 0~1
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 單張圖補上 batch 維度：(3,H,W) → (1,3,H,W)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 輸入尺寸改變時再預熱幾次
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # 模型前向推論，輸出所有候選框
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # NMS（非極大值抑制）：用信心門檻 --conf-thres 和重疊門檻 --iou-thres 去掉重複的框
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 逐張處理偵測結果
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 設定結果圖與標註 .txt 的存檔路徑
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # gn：用來把座標正規化到 0~1 的比例（寬、高、寬、高）
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # 把框的座標從模型輸入尺寸換算回原圖尺寸
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 統計每個類別偵測到幾個，組成輸出字串，例如 "5 horses"
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 逐個框輸出結果
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # --save-txt：存成 YOLO 格式的 .txt（類別 x中心 y中心 寬 高）
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在圖上畫框，標籤格式為「類別 信心值」
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # 印出這張圖的偵測結果與推論、NMS 耗時
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # --view-img：開視窗即時顯示
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 存檔：圖片直接存成 jpg；影片則逐幀寫進 mp4
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    # 遇到新的影片時，建立新的 VideoWriter
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


# --- 命令列參數定義 ---（執行 python detect.py --help 可以查看全部參數）
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --weights：權重檔路徑，可以放多個做 ensemble
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    # --source：輸入來源，可以是圖片、影片、資料夾、0（webcam）或 rtsp/http 網址
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # --img-size：推論時的圖片尺寸
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # --conf-thres：信心門檻，低於這個值的框會丟掉（調高 → 框變少、誤判減少）
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # --iou-thres：NMS 重疊門檻
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # --device：0 = 第一張 GPU，cpu = 用 CPU，留空 = 自動選
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --view-img：即時顯示結果視窗
    parser.add_argument('--view-img', action='store_true', help='display results')
    # --save-txt：把偵測結果存成 .txt
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # --save-conf：.txt 裡多存信心值
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # --nosave：不存結果圖／影片
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # --classes：只偵測指定類別，例如 --classes 0 2（person、car）
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # --agnostic-nms：NMS 不分類別
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # --augment：推論時做資料增強（TTA），比較準但比較慢
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    # --project / --name：輸出到 project/name，預設 runs/detect/exp
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # --exist-ok：資料夾已存在時直接覆蓋，不自動遞增編號
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # --no-trace：不轉 TorchScript
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    # 關閉梯度計算後執行偵測
    with torch.no_grad():
        # --update：重新存檔權重（清掉 optimizer），一般用不到
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
