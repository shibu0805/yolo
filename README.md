# YOLOv7 測試專案（Windows 11 + RTX 5050）

> 給「會寫 Python、剛接觸 YOLO」的自己看的筆記。
> 第 3、4 節是觀念和程式導讀，第 8 節是訓練自己資料的完整流程。
> `yolov7\` 裡所有 .py 都已加上中文註解（只加 `#` 註解，程式邏輯沒有變動，已用 AST 比對驗證）。

---

## 安裝

### 需要的東西
| 項目 | 說明 |
|---|---|
| 作業系統 | Windows 10 / 11（測試環境是 Windows 11） |
| Python | 要有 `py` 指令：到 [python.org](https://www.python.org/downloads/) 安裝任何版本的 Python，安裝時勾選 **py launcher**。安裝程式會自己再裝 Python 3.10（yolov7 需要 3.10，太新的版本不行） |
| 顯示卡 | 建議 NVIDIA 顯卡，並把驅動程式更新到最新。 |
| 硬碟空間 | 約 6GB（套件約 5GB，模型約 160MB） |
| 網路 | 第一次安裝要下載約 3GB |

### 安裝步驟
1. **下載程式**：在 GitHub 頁面按綠色的 **Code** → **Download ZIP**，解壓縮到一個**路徑沒有中文、空白、emoji** 的資料夾，例如 `C:\yolo\yolov7-test`（有中文的路徑 OpenCV 會讀不到圖片）。
2. **安裝**：進到資料夾，**點兩下 `setup.bat`**。它會自動做三件事：
   - 建立 Python 3.10 的虛擬環境 `venv\`（所有套件都裝在這裡，不會影響電腦裡其他 Python）
   - 依照 `requirements.txt` 安裝固定版本的套件
   - 下載官方模型 `yolov7.pt`、`yolov7_training.pt`、`yolov7-tiny.pt` 到 `yolov7\`

   最後看到 `CUDA 可用: True` 代表顯示卡可以用；`False` 代表會用 CPU（請先更新 NVIDIA 驅動，再執行一次 `setup.bat`）。
3. **啟動網站**：**點兩下 `start.bat`**，瀏覽器會自動打開 http://127.0.0.1:7861 。黑色視窗不要關，關掉網站就停了。

> `setup.bat` 可以重複執行：已經裝好的會跳過，只補缺少的部分。安裝中斷、模型沒下載完，再點一次就好。

### 安裝遇到問題
| 狀況 | 怎麼辦 |
|---|---|
| `找不到 py 指令` | 重新安裝 Python，勾選 py launcher |
| `py install 3.10` 失敗 | 手動到 python.org 下載 **Python 3.10** 安裝，再執行 `setup.bat` |
| 下載模型失敗 | 網路問題，再執行一次；或手動從 https://github.com/WongKinYiu/yolov7/releases/tag/v0.1 下載三個 .pt 放進 `yolov7\` |
| `CUDA 可用: False` | 更新 NVIDIA 驅動程式；確認是 NVIDIA 顯卡（AMD、Intel 內顯只能用 CPU） |
| 想重裝 | 刪掉 `venv` 資料夾，再點 `setup.bat` |

### 手動安裝（不想用 setup.bat 的話）
```powershell
py -V:3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
# 再把三個官方 .pt 下載到 yolov7\（網址見上表）
.\venv\Scripts\python.exe app.py
```

### 版本
所有套件版本都固定在 `requirements.txt`（測試環境：Python 3.10.11、torch 2.11.0+cu128、numpy 1.23.5、gradio 6.30.0）。yolov7 原始碼是 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) commit `a207844`，加上中文註解和 `torch.load(weights_only=False)` 的修正（新版 torch 必須）。

### 授權
yolov7 原始碼是 GPL-3.0 授權（見 `yolov7\LICENSE.md`），所以這個專案也必須用 GPL-3.0 公開。

---

## 0. 快速開始（最常用的指令）

```powershell
cd yolov7-test                        # 你放這個專案的資料夾
.\venv\Scripts\Activate.ps1          # 啟用環境，提示字元前面會出現 (venv)
cd yolov7

# 偵測（圖片 / 影片 / 資料夾 / webcam 0）
python detect.py --weights yolov7.pt --conf 0.25 --source inference/images/horses.jpg

# 訓練
python train.py --workers 0 --device 0 --batch-size 4 --data data/mydata.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name my-run --hyp data/hyp.scratch.custom.yaml --epochs 100

# 評估
python test.py --weights runs/train/my-run/weights/best.pt --data data/mydata.yaml --img 640 --batch-size 8
```

### 本機網站（app.py）

```powershell
cd yolov7-test                        # 你放這個專案的資料夾
.\venv\Scripts\python.exe app.py     # 瀏覽器開 http://127.0.0.1:7861（會自動開啟）
```

介面和 `yolov8-test` 的網站相同（v8 用 7860、v7 用 7861，兩個可以同時開）：切換模型、圖片 / 影片辨識、上傳 zip 訓練，所有參數都能在網頁上調整。和 v8 版的差別：

- 辨識直接呼叫 yolov7 的 `attempt_load`、`letterbox`、`non_max_suppression`，和 `detect.py` 的流程相同
- 官方模型：`yolov7.pt`、`yolov7_training.pt`、`yolov7-tiny.pt`（tiny 是另外下載的，最小最快）
- 訓練：起始模型選 yolov7 時用 `cfg/training/yolov7.yaml` + `hyp.scratch.custom.yaml`，選 tiny 時用 `yolov7-tiny.yaml` + `hyp.scratch.tiny.yaml`；網頁上的資料變化和學習速度會寫成一份新的 hyp.yaml 傳給 train.py（會複製到 `runs\train\<name>\hyp.yaml`）
- 上傳的 zip 解壓縮到 `yolov7-test\datasets\<名稱>\`，自動產生 v7 格式的 `data_local.yaml`（絕對路徑 + `nc` + `names`）
- 「專家設定：loss 與權重更新」區塊可以調 hyp.yaml 的 box / obj / cls 權重、cls_pw / obj_pw、fl_gamma、loss_ota、anchor_t、momentum、weight_decay；預設值會跟著起始模型（yolov7 / tiny）的 hyp 檔切換。`iou_t` 在 v7 程式裡沒有作用（被註解掉），所以沒有放
- 沒有「幾輪沒進步就自動停止」（v7 不支援），每張圖最多 300 個框
- **使用紀錄**：每次圖片辨識、影片辨識、開始 / 結束 / 停止訓練，都會在 `usage.log` 記一行（時間、IP、本機或 ngrok、瀏覽器、動作、模型、檔名、結果），用記事本或 Excel 都能開。從 ngrok 來的人記的是對方真正的 IP（讀 `x-forwarded-for`）；只有 IP，不知道是誰
- 額外套件：`gradio==6.30.0`、`imageio-ffmpeg`（已確認 numpy 維持 1.23.5）

### 網站使用說明（給第一次使用的人）

> 網站上的「📖 使用說明」分頁是同一份內容。
#### 這個網站能做什麼
1. **辨識**：上傳圖片或影片，讓模型把找到的東西框起來，並告訴你是什麼、有幾個。
2. **訓練**：上傳你自己標註好的圖片，教模型認得新的東西（例如安全帽的種類），訓練好的模型可以馬上拿來辨識。

#### 先懂幾個名詞
| 名詞 | 意思 |
|---|---|
| 模型 | 已經學會認東西的「大腦」。官方模型認得人、車、狗等 80 種常見東西；自己訓練的只認得你教它的類別 |
| 框 | 模型找到一個東西，就在它周圍畫一個方框，旁邊寫上名稱和分數 |
| 把握程度（分數） | 0~1 之間，模型對這個框有多確定。0.9 = 很確定，0.3 = 不太確定 |
| 類別 | 模型認得的東西種類，例如 person、car、helmet |
| 幀 | 影片是由一張張圖片組成的，每一張叫一幀。一秒的影片通常有 30 幀 |
| 訓練輪數（epoch） | 把整份訓練資料從頭到尾看一遍叫一輪 |

#### 辨識：怎麼用
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

#### 訓練：怎麼教模型認新東西
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

#### 注意事項
- 網址：http://127.0.0.1:7861 。只有這台電腦能開，關掉黑色視窗（或按 Ctrl+C）網站就會停止
- 訓練時顯示卡會全力運作，請等訓練結束再辨識
- 一次只能進行一個訓練
- 每個參數的說明都寫在它的下方，「進階」區塊裡的設定通常不用改

---

## 1. 環境版本

| 項目 | 版本 |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| GPU | NVIDIA GeForce RTX 5050 Laptop 8GB（Blackwell 架構，sm_120） |
| Driver / CUDA | 617.42 / CUDA 13.4 |
| Python（venv） | 3.10.11（系統預設的 3.15 不使用） |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| numpy | 1.23.5 |
| YOLOv7 commit | a207844 |

---

## 2. 資料夾結構

```
yolov7-test\
├── README.md            ← 本檔
├── app.py               ← 本機網站（辨識 + 訓練）
├── requirements.txt     ← 套件清單（固定版本）
├── setup.bat / setup.ps1 ← 一鍵安裝（venv + 套件 + 官方模型）
├── start.bat            ← 點兩下啟動網站
├── datasets\            ← 從網站上傳的 zip 解壓縮到這裡
├── usage.log            ← 網站使用紀錄（第一次有人使用時自動產生）
├── venv\                ← Python 3.10 虛擬環境（所有套件都裝在這裡）
├── coco128\             ← 測試資料集（128 張圖），格式和自己的資料集一樣
│   ├── images\train2017\    ← 圖片
│   └── labels\train2017\    ← 每張圖對應一個 .txt 標註（另有 train2017.cache 快取）
├── videos\sample.mp4    ← 影片測試素材
├── train_log.txt        ← coco128 訓練的完整 log
└── yolov7\              ← 官方程式碼（已加中文註解、已修改 torch.load）
    ├── detect.py            ← 偵測（推論）
    ├── train.py             ← 訓練（一般模型）
    ├── train_aux.py         ← 訓練（P6 大模型 w6/e6/d6/e6e 專用）
    ├── test.py              ← 評估 mAP
    ├── export.py            ← 匯出 ONNX / TorchScript
    ├── hubconf.py           ← torch.hub 入口
    ├── yolov7.pt            ← 官方權重：直接拿來偵測 COCO 80 類
    ├── yolov7_training.pt   ← 官方權重：拿來當訓練起點（遷移學習）
    ├── yolov7-tiny.pt       ← 官方權重：tiny 小模型（偵測、訓練起點都用這個）
    ├── traced_model.pt      ← detect.py 自動產生的加速暫存檔，可以忽略
    ├── cfg\training\        ← 模型架構 yaml（yolov7-tiny / yolov7 / yolov7x / w6 / e6 …）
    ├── data\                ← 資料集 yaml（coco.yaml、coco128.yaml）與超參數 hyp.*.yaml
    ├── models\              ← 網路結構程式（yolo.py、common.py、experimental.py）
    ├── utils\               ← 工具（資料讀取、loss、NMS、畫圖、評估指標）
    └── runs\
        ├── detect\exp*\     ← 每次偵測的輸出
        ├── train\<name>\    ← 每次訓練的輸出
        └── test\<name>\     ← 每次評估的輸出
```

---

## 3. YOLO 觀念入門

### 3.1 物件偵測在做什麼
輸入一張圖，輸出好幾個「框」，每個框包含：**位置（x1, y1, x2, y2）＋類別＋信心值（0~1）**。
例如 `horses.jpg` → 5 個 horse 框。分類（classification）只回答「這張圖是什麼」，偵測還要回答「在哪裡、有幾個」。

### 3.2 YOLO 怎麼預測框（對照 `models/yolo.py`）
1. 圖片先用 **letterbox** 等比例縮放到 640×640，不足的地方補灰邊（`utils/datasets.py: letterbox`）。
2. 網路分成 **backbone**（擷取特徵）和 **head**（組合特徵、輸出預測），結構寫在 `cfg/training/yolov7.yaml`。
3. 最後在 **3 種尺度** 上預測：

   | 偵測層 | stride | 格點數（640 輸入時） | 負責 |
   |---|---|---|---|
   | P3 | 8 | 80×80 | 小物件 |
   | P4 | 16 | 40×40 | 中物件 |
   | P5 | 32 | 20×20 | 大物件 |

4. 每個格點有 **3 個 anchor**（預設的參考框尺寸，見 `yolov7.yaml` 的 `anchors:`）。每個 anchor 輸出 `4（框）+ 1（有沒有物件 obj）+ nc（各類別機率）` 個數字。
5. 所以 640 的圖一次會產生 (80²+40²+20²)×3 = **25200 個候選框**，大部分都是沒有用的。

> **stride** = 下採樣倍數，所以輸入尺寸必須是 32 的倍數（`check_img_size` 會自動調整）。
> **anchor** 不適合你的資料時（例如物件全部都非常細長），訓練開始會自動重新計算（`utils/autoanchor.py`，log 會顯示 `BPR`）。

### 3.3 從 25200 個框到最後結果：conf 與 NMS（對照 `utils/general.py: non_max_suppression`）
1. **信心門檻 `--conf-thres`**：丟掉 `obj × 類別機率 < 0.25` 的框。調高 → 框變少、誤判減少，但可能漏抓。
2. **NMS（非極大值抑制）**：同一個物件常會被好幾個框框到。保留分數最高的那個，刪掉和它 **IoU > `--iou-thres`（0.45）** 的其他框。
3. **IoU** = 兩框交集面積 ÷ 聯集面積，1 = 完全重疊，0 = 沒有重疊。

### 3.4 訓練的基本名詞

| 名詞 | 意思 |
|---|---|
| epoch | 整個訓練集看過一遍叫 1 個 epoch |
| batch size | 一次餵幾張圖給 GPU。越大越吃 GPU 記憶體 |
| iteration | 處理一個 batch = 1 次迭代。128 張圖、batch 8 → 每個 epoch 16 次 |
| loss | 模型預測和正確答案的差距，越小越好。YOLO 有三種（下表） |
| learning rate（lr） | 每次更新參數的步伐大小，由 `hyp` 的 `lr0`、`lrf` 控制 |
| warmup | 前 3 個 epoch 學習率從很小慢慢升高，避免一開始就訓練爆掉 |
| 遷移學習 | 從已經學過 COCO 的權重（`yolov7_training.pt`）開始訓練，只需要少量資料就能有不錯的效果 |
| 資料增強 | 訓練時隨機翻轉、縮放、改顏色、mosaic（4 張拼成 1 張），讓模型看過更多變化 |
| EMA | 權重的移動平均版本，比較穩定；存下來的 best.pt 就是 EMA 權重 |

三種 loss（訓練進度條上的 `box / obj / cls`，程式在 `utils/loss.py: ComputeLossOTA`）：

| loss | 意思 |
|---|---|
| box | 框的位置準不準（CIoU loss） |
| obj | 「這裡有沒有物件」判斷得對不對 |
| cls | 類別猜得對不對（只有 1 類時永遠是 0） |

### 3.5 評估指標（對照 `test.py`、`utils/metrics.py`）

| 指標 | 意思 |
|---|---|
| **P（Precision，精確率）** | 模型框出來的東西裡，有多少是對的。低 = 誤判多 |
| **R（Recall，召回率）** | 真正的物件裡，有多少被找到。低 = 漏抓多 |
| **mAP@.5** | IoU ≥ 0.5 就算框對，各類別 AP 的平均。最常拿來看的指標 |
| **mAP@.5:.95** | IoU 門檻從 0.5 到 0.95 的平均，要求框得非常準，數值通常比較低 |
| **best.pt** | `0.1×mAP@.5 + 0.9×mAP@.5:.95` 最高的那個 epoch |
| **last.pt** | 最後一個 epoch（中斷後用 `--resume` 接續訓練時使用） |

### 3.6 怎麼判斷訓練得好不好（看 `results.png`）
- **train loss 持續下降、mAP 持續上升** → 正常，還可以繼續訓練。
- **mAP 不再上升** → 差不多收斂了。
- **train loss 還在降，但 val loss 開始上升、mAP 下降** → **過擬合（overfitting）**：模型把訓練圖背起來了。解法：增加資料、加強資料增強、減少 epoch。
- **mAP 一直接近 0** → 通常是標註有問題（類別編號錯、座標沒有正規化、images/labels 對不起來）。先檢查 `train_batch0.jpg` 上的框對不對。

---

## 4. 程式架構導讀（建議閱讀順序）

### 4.1 偵測流程 `detect.py`
```
解析參數 → select_device → attempt_load(權重)            # models/experimental.py
        → TracedModel（加速）                            # utils/torch_utils.py
        → LoadImages / LoadStreams                       # utils/datasets.py（letterbox）
        → for 每張圖/每一幀:
              model(img) → non_max_suppression          # utils/general.py
              → scale_coords（換算回原圖座標）
              → plot_one_box（畫框）→ 存檔             # utils/plots.py
```

### 4.2 訓練流程 `train.py`
```
解析參數 → 讀 hyp.yaml、data.yaml
→ Model(cfg)  +  載入預訓練權重中形狀相符的部分            # models/yolo.py: Model / parse_model
→ 建立 optimizer（SGD）、學習率排程（cosine）、EMA
→ create_dataloader（mosaic 等資料增強）                   # utils/datasets.py: LoadImagesAndLabels
→ check_anchors                                           # utils/autoanchor.py
→ for epoch:
      for batch:
          前向 → ComputeLossOTA（box/obj/cls）→ 反向傳播 → 更新參數 → 更新 EMA   # utils/loss.py
      test.test() 算 mAP → 存 last.pt / best.pt            # test.py
→ plot_results 畫 results.png → strip_optimizer 縮小 .pt
```

### 4.3 建議閱讀順序
1. `detect.py`（最短，先看懂推論流程）
2. `models/yolo.py` 的 `Model`、`parse_model`、`IDetect.forward`（模型怎麼組起來、輸出怎麼解碼）
3. `cfg/training/yolov7.yaml` 對照 `models/common.py` 的 `Conv`、`SPPCSPC`、`RepConv`
4. `train.py`（主迴圈）→ `utils/datasets.py` 的 `__getitem__`（資料增強）→ `utils/loss.py` 的 `ComputeLossOTA`
5. `utils/general.py` 的 `non_max_suppression`、`test.py`

### 4.4 在自己的 Python 程式裡使用模型
已在本機測試可用（需要在 `yolov7\` 底下執行，或把它加進 `sys.path`）：

```python
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load('yolov7.pt', map_location=device).eval()   # 換成你的 best.pt 也可以
names = model.names

img0 = cv2.imread('inference/images/horses.jpg')                 # 原圖（BGR）
img = letterbox(img0, 640, stride=32)[0]                          # 縮放 + 補邊
img = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).to(device).float() / 255.0
img = img.unsqueeze(0)                                            # (1, 3, H, W)

with torch.no_grad():
    pred = model(img)[0]
det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

for *xyxy, conf, cls in det.tolist():
    print(names[int(cls)], round(conf, 2), [int(v) for v in xyxy])
```

---

## 5. 完整重現指令（PowerShell）

```powershell
# --- 下載權重 ---
cd yolov7
Invoke-WebRequest https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -OutFile yolov7.pt
Invoke-WebRequest https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt -OutFile yolov7_training.pt

# --- 預訓練模型偵測 ---
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
Invoke-WebRequest https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4 -OutFile ..\videos\sample.mp4
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source ../videos/sample.mp4

# --- coco128 ---
Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip -OutFile ..\coco128.zip
Expand-Archive ..\coco128.zip -DestinationPath ..
# 建立 data\coco128.yaml（train/val: ../coco128/images/train2017, nc: 80, names 複製自 data\coco.yaml）

# --- 訓練 ---
python train.py --workers 0 --device 0 --batch-size 8 --data data/coco128.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name test-run --hyp data/hyp.scratch.custom.yaml --epochs 5

# --- 用訓練結果偵測與評估 ---
python detect.py --weights runs/train/test-run/weights/best.pt --source inference/images/horses.jpg
python test.py --weights runs/train/test-run/weights/best.pt --data data/coco128.yaml --img 640 --batch-size 8 --name test-run --no-trace
```

---

## 6. 執行結果

| 步驟 | 結果 |
|---|---|
| 圖片偵測（yolov7.pt） | 5 horses，推論 43ms → `runs\detect\exp\horses.jpg` |
| 影片偵測（647 幀） | 以幀數計：person 231、car 125、bicycle 106（另有少量誤判：tennis racket 25、skateboard 12、chair 1）→ `runs\detect\exp2\sample.mp4`、`runs\detect\video\` |
| coco128 訓練 5 epochs | 12.3 分鐘；mAP@.5 0.821 → 0.824、mAP@.5:.95 0.613 → 0.621；產生 `weights\best.pt`、`results.png` → `runs\train\test-run\` |
| best.pt 偵測 | 5 horses，推論 28ms → `runs\detect\exp3\horses.jpg` |
| best.pt 評估（test.py） | P 0.804、R 0.739、mAP@.5 0.827、mAP@.5:.95 0.624，每張 39ms → `runs\test\test-run\` |

> coco128 本身就是 COCO 的子集，預訓練權重本來就會，所以 mAP 只小幅上升，這次只是確認流程能跑。訓練自己的資料時，變化會明顯很多。

---

## 7. 遇到的問題與解法

| 問題 | 原因 | 解法 |
|---|---|---|
| 系統 Python 是 3.15 | `numpy<1.24` 最高只支援 Python 3.11，3.15 沒有 wheel | 用 `py install 3.10` 另外安裝 3.10，再用 `py -V:3.10 -m venv venv` 建立 venv |
| RTX 5050 是 sm_120 | 舊版 torch（cu118/cu121）不支援 Blackwell 架構 | 安裝 cu128 版本的 torch（`torch.cuda.get_arch_list()` 要有 sm_120） |
| `pip install -r requirements.txt` 會換掉 numpy | torch 預設會裝 numpy 2.x | requirements 已限制 `<1.24`，會自動降到 1.23.5；保險起見再執行一次 `pip install "numpy<1.24"` |
| `weights_only` 錯誤（torch ≥ 2.6） | torch 2.6 起 `torch.load` 預設 `weights_only=True` | 每個 `torch.load(...)` 都加上 `weights_only=False`。參數裡有括號（例如 `torch.device('cpu')`），所以用會配對括號的腳本處理，不用簡單 regex |
| `torch.meshgrid` UserWarning | 新版 torch 的提醒 | 可忽略 |
| Windows 多行程 DataLoader | Windows 的 spawn 模式 | 訓練時一律加 `--workers 0` |
| batch 8 時 GPU 記憶體用到 8.69G（超過實體 8GB） | Windows 會把超出的部分借用系統記憶體，不會當掉但會變慢 | 正式訓練建議用 `--batch-size 4` |
| 訓練第一個 epoch 驗證很慢（每批約 80 秒） | 訓練初期候選框很多，NMS 比較慢 | 正常現象，之後就會變快 |

---

## 8. 訓練自己的資料（完整流程）

### 步驟 1：收集圖片
- 每個類別建議至少 **100~300 張**，越多越好；要包含不同角度、光線、背景、遠近。
- 可以放一些**完全沒有目標物的背景圖**（不用標註檔），能減少誤判。

### 步驟 2：標註
工具：**Roboflow**（網頁，可以直接匯出 YOLO 格式並自動切 train/val，最適合新手）、X-AnyLabeling、CVAT、LabelImg。
匯出時選 **YOLO（YOLOv5/v7 txt）格式**。每張圖對應一個同檔名的 `.txt`，每行代表一個物件：

```
類別編號  x中心  y中心  寬  高
0 0.512 0.433 0.200 0.350
1 0.150 0.700 0.080 0.120
```
- 類別編號從 **0** 開始，對應 yaml 裡 `names` 的順序。
- 座標都是**除以圖片寬高後的比例（0~1）**，不是像素。
- 框要**貼齊物件**，同一類別的標註標準要一致（例如有遮擋的要不要標）。

### 步驟 3：資料夾結構（建議放在 `yolov7-test\` 底下）
```
yolov7-test\mydata\
├── images\
│   ├── train\   a.jpg  b.jpg ...      ← 約 80~90%
│   └── val\     c.jpg ...             ← 約 10~20%（不要和 train 重複）
└── labels\
    ├── train\   a.txt  b.txt ...
    └── val\     c.txt ...
```
⚠ 程式是把路徑中的 `images` 換成 `labels` 來找標註（`utils/datasets.py: img2label_paths`），**資料夾名稱一定要叫 images 和 labels**，檔名也要一一對應。

### 步驟 4：建立 `yolov7\data\mydata.yaml`
```yaml
train: ../mydata/images/train     # 相對於 yolov7\ 資料夾的路徑
val: ../mydata/images/val
nc: 2                             # 類別數
names: ['cat', 'dog']             # 順序 = 類別編號 0, 1, ...
```
`cfg` 裡的 `nc: 80` **不用改**，程式會用 data yaml 的 nc 自動覆蓋。

### 步驟 5：開始訓練
```powershell
python train.py --workers 0 --device 0 --batch-size 4 --data data/mydata.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name my-run --hyp data/hyp.scratch.custom.yaml --epochs 100
```
訓練開始後先檢查：
1. log 中的 `train: Scanning ... N found, 0 missing, 0 empty, 0 corrupted`：**missing 太多代表標註沒有對到圖片**。
2. 打開 `runs\train\my-run\train_batch0.jpg`，確認**框的位置和類別都正確**（這是最重要的檢查）。
3. 可以另開一個終端機執行 `tensorboard --logdir runs/train`，用瀏覽器開 http://localhost:6006 看即時曲線。

### 步驟 6：參數說明

| 參數 | 意思 | 建議 |
|---|---|---|
| `--data` | 資料集 yaml | ⭐ 一定要改 |
| `--epochs` | 訓練輪數 | ⭐ 先跑 50~100 看曲線，還在上升就再加 |
| `--name` | 輸出到 `runs\train\名稱\` | ⭐ 每次取不同名字，方便比較 |
| `--batch-size` | 每批幾張圖 | 本機用 4（8 會超過 8GB）；出現 `CUDA out of memory` 就再調小 |
| `--weights` | 起始權重 | `yolov7_training.pt`（遷移學習，強烈建議）；填 `''` 就是從零開始 |
| `--cfg` | 模型大小 | `yolov7.yaml`（預設）；要快 → `yolov7-tiny.yaml` 搭配 `--hyp data/hyp.scratch.tiny.yaml`，起始權重 `yolov7-tiny.pt` 需另外下載 |
| `--img` | 訓練/驗證圖片尺寸 | 640 640；小物件多可以試 1280（記憶體用量約 4 倍） |
| `--hyp` | 超參數 yaml | `data/hyp.scratch.custom.yaml` |
| `--workers` | DataLoader 行程數 | **Windows 固定 0** |
| `--device` | 0 = GPU、cpu = CPU | 0 |
| `--resume` | 接續中斷的訓練 | `python train.py --resume runs/train/my-run/weights/last.pt` |
| `--freeze` | 凍結前 N 層不訓練 | 資料很少時可以試 `--freeze 50`（凍結 backbone） |
| `--single-cls` | 所有類別當成同一類 | 只關心「有沒有」時使用 |
| `--cache-images` | 圖片快取進 RAM | 資料集小時可以加速 |
| `--multi-scale` | 每個 batch 隨機縮放 ±50% | 物件大小差異大時可以試 |
| `--adam` | 改用 Adam optimizer | 一般用預設的 SGD 即可 |

### 步驟 7：超參數 `data/hyp.scratch.custom.yaml` 常調的項目
建議複製一份成 `hyp.mydata.yaml` 再修改。

| 項目 | 預設 | 意思 / 什麼時候調整 |
|---|---|---|
| `lr0` | 0.01 | 初始學習率。loss 爆掉或變 nan → 降到 0.005 |
| `lrf` | 0.1 | 最終學習率 = lr0 × lrf |
| `warmup_epochs` | 3.0 | 暖身的 epoch 數 |
| `box` / `obj` / `cls` | 0.05 / 0.7 / 0.3 | 三種 loss 的權重 |
| `anchor_t` | 4.0 | anchor 比例門檻 |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.7 / 0.4 | 色相 / 飽和度 / 亮度擾動。顏色是重要特徵時（例如紅綠燈）把 hsv_h 設 0 |
| `degrees` | 0.0 | 隨機旋轉角度。俯拍、零件等不分方向的物件可以設 10~180 |
| `translate` / `scale` | 0.2 / 0.5 | 平移 / 縮放幅度 |
| `fliplr` / `flipud` | 0.5 / 0.0 | 左右 / 上下翻轉機率。有方向性的物件（文字、左右箭頭）把 fliplr 設 0 |
| `mosaic` | 1.0 | 4 張拼 1 張的機率，對小物件很有幫助 |
| `mixup` | 0.0 | 兩張圖疊加的機率 |
| `loss_ota` | 1 | 1 = 使用 OTA 標籤分配（比較準）；0 = 比較快 |

### 步驟 8：看結果
`runs\train\my-run\`：

| 檔案 | 用途 |
|---|---|
| `weights\best.pt` | ⭐ 最佳模型，之後就用這個 |
| `weights\last.pt` | 最後一輪的模型（接續訓練用） |
| `results.png` | loss 與 mAP 曲線（判讀方式見 3.6） |
| `confusion_matrix.png` | 混淆矩陣：哪些類別容易搞混、哪些常被漏掉（被當成 background） |
| `PR_curve.png` / `F1_curve.png` | F1 曲線的最高點對應的信心值，就是最適合的 `--conf-thres` |
| `test_batch*_labels.jpg` vs `test_batch*_pred.jpg` | 驗證集的正確答案和模型預測，可以並排比較 |
| `hyp.yaml` / `opt.yaml` | 這次使用的所有參數（方便重現） |

### 步驟 9：使用模型
```powershell
python detect.py --weights runs/train/my-run/weights/best.pt --conf 0.25 --source 圖片或影片或資料夾路徑
python detect.py --weights runs/train/my-run/weights/best.pt --source 0                      # webcam
python detect.py --weights runs/train/my-run/weights/best.pt --source 路徑 --save-txt --save-conf   # 同時輸出 .txt 座標
python test.py   --weights runs/train/my-run/weights/best.pt --data data/mydata.yaml --img 640 --task val --verbose   # 逐類別 mAP
python export.py --weights runs/train/my-run/weights/best.pt --grid --end2end --simplify --img-size 640 640  # 匯出 ONNX（需要 pip install onnx onnxsim）
```

### 效果不好時的檢查順序
1. `train_batch0.jpg` 上的框對不對？→ 不對就是標註或類別編號有問題
2. 資料量夠不夠、val 和 train 是否太像或重複？
3. 換大一點的模型或圖片尺寸（`--img 1280`）
4. 增加 epochs、調整資料增強（見步驟 7）
5. 誤判多 → 加入背景圖、調高 `--conf-thres`；漏抓多 → 補充那類物件的資料

---

## 9. 常見錯誤

| 錯誤訊息 | 原因與解法 |
|---|---|
| `CUDA out of memory` | 調小 `--batch-size`（8 → 4 → 2）或 `--img` |
| `Label class X exceeds nc=Y` | 標註裡的類別編號 ≥ nc。檢查 yaml 的 nc 和標註 |
| `No labels in ... Can not train without labels` | 找不到標註：確認資料夾叫 `images` / `labels`，且檔名一一對應 |
| `non-normalized or out of bounds coordinate labels` | 標註座標不是 0~1 的比例。匯出時要選 YOLO 格式 |
| **改了標註但訓練結果沒變** | ⚠ 這版程式的標註快取**不會自動更新**（hash 檢查被註解掉了），改過標註要手動刪除 `labels\train.cache`、`labels\val.cache` |
| `_pickle.UnpicklingError: Weights only load failed` | torch ≥ 2.6 的 weights_only 問題，本專案已修正（第 7 節） |
| `AttributeError: module 'numpy' has no attribute 'int'` | numpy 版本太新：`pip install "numpy<1.24"` |
| 訓練卡住不動（Windows） | `--workers` 沒有設成 0 |
| `Dataset not found` | yaml 中的路徑錯誤。路徑是相對於 `yolov7\`（執行指令的位置） |
