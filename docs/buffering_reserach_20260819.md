# Joygen streaming research for input/output implementation 

> 測試規範：不論輸出端或輸入端測試，都要記錄每個 frame（或等效處理單位）的時間成本 log 表格，單位 ms，最終取平均。這樣才能量化「原始方法 vs 新方法」的延遲差異，也能找出真正的瓶頸在哪一段。下面各節提到「測試」的地方都適用這個規範，不重複寫。

## JoyGen 即時串流 Buffering — 可行方案研究

目的：回答「JoyGen 要支援即時串流，buffering 該用什麼單位（frame／bytes／text）、現有實作卡在哪裡、怎麼解」。以下內容基於直接讀 `JOY-MM/JoyGen` 原始碼得到的結論（`inference_audio2motion.py`、`inference_joygen.py` 已逐行讀過；`inference_edit_expression.py` 尚未逐行讀，標記為未確認），並比對同源模型家族 `yerfor/Real3DPortrait`、`yerfor/GeneFacePlusPlus`（JoyGen 的 audio2motion 直接沿用 Real3DPortrait 的 `audio2secc_vae`）——GeneFace++ 官方宣稱其 renderer 可即時推論，是「輸入端分塊可行」的間接佐證，但不是 JoyGen 這份 checkpoint 本身的證據，仍需本地驗證。

白板圖僅供參考（UDP bytes → 處理 → MP4/streaming → browser 的概念），下面把它對應到各節的具體解法。

### Summary

核心問題「JoyGen 的輸出與輸入 buffering 實作方式，包含文字和語音輸入，及 video streaming 輸出」，可行性結論：

- **輸出端（video streaming）：可行，風險最低**。現有程式碼已有逐 batch 的 frame generator，只需要接上即時編碼/傳輸，不需要動到模型本身（詳見第二節）。
- **輸入端（語音）：有條件可行**。能否用 sliding window 分塊餵給 audio2motion，取決於 VAE 模型是否依賴長距離上下文，目前沒有被驗證過，需要本地實測才能拍板（詳見第三、五節）。
- **輸入端（文字）：不直接支援**。JoyGen 只吃音訊特徵，文字需先經 streaming TTS 轉成音訊，格式須是 16kHz、單聲道、16-bit PCM，且切塊對齊視訊 fps（詳見第四節）。

三段管線目前靠寫檔＋CLI 參數交接，即使個別段落都改成 streaming，串接方式仍要重新設計，這件事排在較後面的優先順序（詳見第五節）。

---

## 一、JoyGen 現有輸入／輸出支援方式

JoyGen（含 audio2motion）原生只支援兩件事：

- **輸入**：`.wav / .mp3 / .mp4 / .avi` 音訊或影音檔案「路徑」，整檔讀入，沒有 streaming 介面。
- **輸出**：完整 MP4 檔案，所有畫面生成完才一次性用 ffmpeg 轉檔＋合併音軌。

文字輸入、逐 frame/逐 chunk 輸出，這兩者都不是原生支援的，需要額外轉換才能達成。轉換方式與需要注意的後果：

| 目標 | 需要的轉換 | 最終丟給 JoyGen 的格式 | 需注意的後果 |
|---|---|---|---|
| 文字輸入 | 文字 → streaming TTS → PCM 音訊 | **16kHz、單聲道、16-bit PCM**，切塊大小對齊視訊 fps | 多一段 TTS 延遲，需要用前述時間成本規範量測（見第四節） |
| video streaming 輸出 | 複製 `inference_joygen.py`，把逐 batch frame 接上常駐 ffmpeg pipe | rawvideo BGR frame → H.264 → MPEG-TS/UDP 或 fragmented MP4 | 需要額外的音畫同步機制；瀏覽器端還需要一個轉發/封裝層（不在 JoyGen 端範圍內） |

三段管線各自該用什麼 buffering 單位：

| 管線位置 | 資料型態 | 建議 buffering 單位 | 可行性 |
|---|---|---|---|
| Audio-to-Latent Mapper（audio2motion） | 音訊 → expression 係數 | bytes，固定長度 sliding window | 有條件可行，需本地驗證邊界失真 |
| Edit-Expression（3D 渲染） | expression 係數 → depth/pose 圖 | frame（逐張） | 未讀原始碼，無法確認 |
| Diffusion Decoder（joygen） | audio feature + latent → 畫面 | frame（batch=8 為單位） | 已確認可行，程式碼裡已有 generator |
| 最終輸出 | frame → 網路傳輸 | bytes（H.264/MP4 fragment 或 raw frame） | 目前不存在，需新建 |

以下各節分別說明每一段的細節、待解決事項，以及對應要新增的測試/腳本。

---

## 二、輸出端：frame-based buffering（優先做，風險最低）

### 現況

`inference_joygen.py` 的 `data_generator()` 已經是逐 batch（預設 8 frame）yield 的 generator，UNet decode 在 for-loop 裡逐批跑，`recon = vae.decode_latents(pred_latents)` 之後馬上就有畫面。模型與運算層對「frame 一算完就能立刻用」沒有阻礙，阻礙在後面的收尾邏輯。

### 問題

1. 迴圈算完的每個 frame 目前只是 append 進 `res_frame_list`，沒有立刻送出去。
2. frame 還要經過「貼回原圖」（`get_image(ori_img, res_crop_img, box)`）才是最終畫面，這一步也是逐張做的，同樣可以搬進迴圈裡即時做。
3. 最終輸出是等**全部** frame 都貼完、寫成 PNG 之後，才用一次性的 `os.system(ffmpeg image2 ...)` blocking 呼叫轉成 MP4，再合音軌——這是唯一真正卡住「即時」的地方，且事後還會 `shutil.rmtree()` 把 PNG 全刪掉。

### 解法

不改原檔，複製一份到 `ourproject/script/joygen_stream.py`，修改重點：

1. 把「decode → 貼回原圖」合併搬進 `data_generator` 的 for-loop 裡，frame 一產生就呼叫 `on_frame_ready(frame_bytes, frame_idx)` callback，而不是先塞進 list。
2. 用一個常駐的 ffmpeg subprocess 取代結尾的一次性 `os.system` 呼叫：
   ```python
   ffmpeg_proc = subprocess.Popen([
       "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
       "-s", f"{img_size}x{img_size}", "-r", str(fps), "-i", "-",
       "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
       "-f", "mpegts", "udp://<target_ip>:<port>"   # 或改成 fragmented mp4 走 websocket
   ], stdin=subprocess.PIPE)
   ```
   `on_frame_ready` 裡直接 `ffmpeg_proc.stdin.write(frame.tobytes())`，畫面產生後幾毫秒內就進到編碼器，不必等整支影片跑完。這條路徑對應白板上「bytes → UDP/MP4 → streaming」的分支。
3. 音訊要另外用 `-f s16le -i pipe:1` 或另一路 UDP 送，跟畫面對齊，這是新增的同步邏輯，ffmpeg 不會自動處理，列入第五節待解決事項。畫面（frame_index/fps）跟音訊（sample_count/sample_rate）要換算成統一的 PTS 才能對齊；因為畫面是整批（8 張）產生、節奏不穩定，需要額外指定 PTS 或做小型 jitter buffer 吸收落差。
4. PNG File Sink 保留當 debug 選項，預設關閉、只在 `--debug` flag 開啟時才寫檔；寫檔用獨立 thread/queue 做非同步處理，避免拖慢主要的 frame 產生迴圈，這樣可以跟即時串流同時跑，串流出問題時也能回頭翻 `output/frames/` 底下的圖對照。

> `inference_joygen.py` 的 `data_generator()` 已經是逐 batch（預設 8 frame）yield 的 generator，UNet decode 在 for-loop 裡逐批跑，`recon = vae.decode_latents(pred_latents)` 之後馬上就有畫面。模型與運算層對「frame 一算完就能立刻用」沒有阻礙，阻礙在後面的收尾邏輯。frame 是判斷「何時可以觸發送出」的單位，實際送出時會序列化成 bytes 才進得了 pipe／網路——後面第 2 點的 `frame.tobytes()` 就是這個轉換。

### 這個解法本身會產生什麼、需要什麼才能跑起來

要展示「frame-based 輸出即時串流」這件事本身，需要準備：

- 前一階段 `inference_edit_expression.py` 已經產生好的 `intermediate_dir`（逐 frame 的 pose/depth 檔案）——這是現有 pipeline 的既有依賴，不因為輸出端改了就消失。
- 一份 `audio_path`（可以是完整檔案，不需要是分塊音訊）——因為這個測試是要驗證「畫面產生後能不能立刻送出去」，不是驗證「音訊能不能分塊輸入」，兩者互不相依。
- 一個能接收 UDP/MPEG-TS 或 pipe 的接收端：本機先用 `ffplay udp://<ip>:<port>` 監聽（要先啟動 ffplay 再啟動推流端），不用先寫網頁或接瀏覽器就能即時看到畫面，用來驗證管線本身有沒有在動；跟第 4 點的 PNG debug sink 可以同時開，互不衝突。

### 本地測試該用哪種輸入、跟其他成員怎麼拆工

本地測試用輸入端維持現況即可：整段音訊檔，或已驗證可行的 0.5s 切片。inference_joygen.py 本來就是先把整段 audio_path 抽成 whisper feature、切成逐 frame 的 conditioning chunk，輸出端要驗證的是「frame 算完到送出去的延遲」，跟音訊是怎麼進來的無關，所以這樣測有 demo 價值，不用等輸入端 streaming 就位才開始。

> 輸入端之後改成 frame/bytes-based streaming，亦不會影響這裡輸出端的測試結果。

跟其他成員拆工的建議（對照你舉的例子，理解正確）：由於輸出端測試不依賴輸入端是否已經 streaming 化，**文字轉語音（TTS）這段可以交給負責 Moshi/文字流的成員獨立進行**，你這邊只需要提供輸入格式規格給他們（第一節表格裡的 16kHz、單聲道、16-bit PCM、切塊對齊 fps），自己專心把輸出端的 frame-based streaming 跑起來、量測時間成本，兩邊平行推進、之後再對接。

---

## 三、輸入端：bytes-based sliding window（次優先，風險中等，需要本地實測才能拍板）

### 現況

`inference_audio2motion.py` 的 `get_hubert()` / `get_mfcc()` 對整段音檔一次性抽特徵，`forward_audio2secc()` 也是整段一次 forward，沒有分塊邏輯。`save_wav16k()` 甚至是先整檔存到硬碟再讀回（`os.system(ffmpeg ...)`）。

### 問題

模型層級的風險:

1. **不知道 VAEModel 是否依賴長距離上下文**——如果是 attention-based 或雙向架構，直接切段餵可能在每段開頭/結尾出現表情不連續；如果主要靠局部音框，切段影響可能很小。現有程式碼沒有調用任何分段邏輯，代表這件事**沒有被驗證過**，不能假設可行、也不能假設不可行，必須本地測。
2. `save_wav16k()` 用「寫檔 → ffmpeg 轉檔 → 讀回」，就算輸入分塊了，這個 I/O pattern 本身也會拖慢延遲，需要換成純記憶體 resample。

> 系統整合層級的風險（TTS 是否支援 streaming、三段管線的檔案交接方式）另列在下方「前置條件」小節與第五節第 6 點。

### 解法

不改原檔，複製一份到 `ourproject/script/audio2motion_stream.py`，修改重點：

1. 把 `prepare_batch_from_inp()` 改成接受 in-memory numpy waveform（或持續被塞資料的 queue），`save_wav16k()` 的 ffmpeg 落地流程整段拿掉，改用 `torchaudio.transforms.Resample` 等效函式在記憶體裡做 resample。
2. 實作 sliding window 累積器：維持長度 `W`（例如 1–2 秒）的音訊 ring buffer，每收到新 chunk（例如 200–320ms，對齊 25fps 整數倍）就往尾端塞，buffer 滿了就對整個 window 跑一次 `forward_audio2secc()`，但只取這次 window 裡「新增音訊」對應的那段 exp 係數輸出，window 前段當上下文用完即丟。
3. 這個做法的正確性完全取決於第五節第一點的實測結果，**在測完之前不應套用到部署腳本 `setup.sh` 裡**。

### 若採用第四節方案 A（文字→TTS→音訊），這裡的解法需要先確認什麼

第四節建議文字走 TTS 轉音訊再進來，對 `audio2motion_stream.py` 而言，音訊來源是 mic 還是 TTS 輸出理論上沒有差別，都是 PCM bytes，本節解法不需要因此改變。但要先確認以下前置條件，才能放心讓 sliding window 解法生效：

1. **TTS 模組本身是否支援 streaming 輸出**——是逐 chunk 吐 PCM，還是也是整句合成完才給。如果 TTS 本身不支援 streaming，sliding window 設計會卡在 TTS 那端，不是 JoyGen 端能解決的問題。
2. **TTS 輸出格式是否直接就是 16kHz、單聲道、16-bit PCM**，如果不是，中間會多一段 resample，這段延遲要算進 latency budget。
3. **第五節第一點 VAEModel chunked inference 的實測結果**——不論音訊來源是誰，模型層級的限制都存在，這是共通的前置條件。

如果 1–3 都確認可行，才照本節解法實作；如果 TTS 端不支援 streaming（第 1 點不成立），代表輸入端從 TTS 那層就已經是整句節奏，此時本節的 sliding window 設計沒有意義，要退回「文字→完整合成語音→整段送進 JoyGen」，等於輸入端失去即時性，只剩輸出端還能做到 streaming——這個退回方案仍然有 demo 價值，只是不算真正的「即時」。

---

## 四、文字輸入：不建議讓 JoyGen 直接吃文字

### 現況

`save_wav16k()` 的 assert 只認 `.wav/.mp3/.mp4/.avi`，其餘格式直接失敗。JoyGen（含沿用的 Real3DPortrait audio2motion）整條模型鏈路的輸入張量都是從 hubert/whisper 音訊特徵抽出來的，沒有文字 embedding 的輸入分支。

### 兩個方案比較

| 方案 | 做法 | 評估 |
|---|---|---|
| A. 文字先轉語音，音訊照常進 JoyGen | 文字 token stream → streaming TTS → PCM bytes → 進第三節 sliding window buffer | **建議採用**。JoyGen 不用改，只是把「音訊來源」從 mic 換成 TTS 輸出流 |
| B. 訓練文字→表情係數的新模型，跳過音訊 | 需要重新訓練 text-to-expression 模型取代 audio2motion | 不建議：已經不是「串流化既有模型」，是重新做一個模型，超出 POC 範圍與時程 |

採用方案 A 後，測試時要額外記錄「加上 TTS 這一步」前後的每 frame（或等效單位）處理時間成本（ms，取平均，依開頭的測試規範），比較 original（純音訊直接輸入）vs new（文字→TTS→音訊）兩種方法的延遲差異，量化 TTS 到底增加了多少延遲，供整體 latency budget 討論用。

---

## 五、待解決事項與建議執行順序

建議先做風險低、能立刻拿到數據的輸出端，再做風險較高的輸入端：

1. **[先做] 輸出端 frame-based streaming 實作與時間量測**（對應第二節）：把 `joygen_stream.py` 跑起來，記錄逐 frame 時間成本，並跟原本「整段跑完才輸出」的 baseline 比較差異。這一步不依賴輸入端是否已經 streaming 化（見第二節「本地測試該用哪種輸入」）。
2. **[次做] 輸入端 audio2motion chunked inference 實測**（對應第三節）：驗證 VAEModel 對整段音訊 vs 切段音訊輸出的 exp 係數差異大小，決定 sliding window 方案能不能用、window 要多大、要不要 overlap。
3. **[待確認]** 讀 `inference_edit_expression.py` 原始碼，確認它能不能被改成逐 frame 即時輸出，還是天生要看到整段 exp 係數才能跑——目前完全未知，是這份研究最大的資訊缺口。
4. **[待設計]** frame 與 audio 之間的 timestamp 同步機制——輸出端一旦邊算邊送，畫面跟音訊各自走不同的 UDP/pipe，需要一致的時間基準，ffmpeg 不會自動處理。
5. **[待對齊]** 確認「UDP/MPEG-TS bytes → 瀏覽器可播放格式」中間要接什麼元件（WebRTC gateway？MSE + websocket relay？），屬於 architecture doc 的 Media Server 範疇，JoyGen 端修改到「產生可串流的 bytes」為止，要在 Task 3 跟其他成員對齊介面。
6. **[後續再處理]** 三支 script 之間目前靠檔案系統＋CLI 參數交接，就算個別都改成 streaming，彼此之間還是要重新設計成走記憶體物件或 local socket——等第 2、3 點答案出來，且確定值得做之後再動工，避免白工。

---

## 六、對應要新增的檔案（放在 `ourproject/script/`，不動原始碼）

這裡先列出對應關係，僅供快速對照。

- `ourproject/script/joygen_stream.py` — 複製自 `inference_joygen.py`，套用第二節解法。可以先做。
- `ourproject/script/audio2motion_stream.py` — 複製自 `inference_audio2motion.py`，套用第三節解法。等第五節第 2 點實測結果出來再動工。
- `ourproject/script/edit_expression_stream.py` — 待第五節第 3 點確認可行性後才知道怎麼改，目前先不建立空檔案。
