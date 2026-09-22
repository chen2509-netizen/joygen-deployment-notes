好的，我根據你提供的 `mem1_tts-streaming-spec_0911.md` 規格文件，幫你規劃 JoyGen input streaming 的實作方向。

## 核心要點

1. **對接第 1 層 TTS service HTTP `POST /synthesize`** — 直接拿 raw PCM bytes（octet-stream），不走 WebSocket / base64 那一層。
2. **支援 HTTP chunked transfer** — TTS 輸出是逐塊到達（10,240 bytes / 320ms 一塊），JoyGen 要能 on-the-fly 消化，不能等整個 response body 完整下載才開始處理。
3. **緩衝區 + jitter 容忍** — TTS 輸出節奏跟即時無關（視模型速度），JoyGen 要有緩衝區吸收輸入輸出的速度差。

## 實作建議

1. **`AudioRingBuffer` 類別**：實作一個 ring buffer，存 PCM int16 samples。
    - `put()`: 從 TTS `/synthesize` 拿到一塊寫入。支援 `requests.get().iter_content(chunk_size=10240)`，`put` 幾次都沒關係。
    - `read()`: `audio2motion` 要資料時用，一次吐固定 sample 數（例如 `sample_rate / fps`）。如果 buffer 不夠會 block 住等 `put`。
2. **`AudioFeeder` 類別**：專職播 TTS request + 餵 ring buffer。用獨立 thread 跑，免得 IO block 住主程式。
    - 進 `__init__()` 時 fire TTS request
    - 每拿到一塊 (`chunk_size=10240`) 就 `ringbuf.put()`
    - response 吃完時可以選擇 graceful stop 或是用 `time.sleep()` 模擬即時節奏繼續餵零值
3. **主程式改動**
    - `audio_feature` 改成初始化 `AudioFeeder` 和 `AudioRingBuffer`
    - data loading 那邊一律改讀 `ringbuf`，取代原本的 `whisper_chunks`

## 程式架構示意

```python
class AudioRingBuffer:
    def __init__(self, sample_rate: int, fps: int):
        self._chunk_size = sample_rate // fps  # 一幀的 sample 數
        self._buf = np.zeros(...)  # 實際存資料的 numpy array
        self._lock = threading.Lock()  # 讀寫同步用
    
    def put(self, samples: bytes):
        # 將 samples 寫進 self._buf；用 memoryview 或 struct.unpack 轉 int16
        # 如果這批 samples 會繞超過 buffer 尾，分兩段寫（ABC|DEF → DEFABC）
    
    def read(self, timeout=None) -> np.ndarray:  # 回傳 int16 nparray
        # 從 self._head 讀出 self._chunk_size 個 samples（一幀的量）
        # 如果 buffer 裡的資料不夠一幀，block 住等 put() 補進來
        # read 完後移動 self._head（如果繞到尾端，跳回頭）

class AudioFeeder(threading.Thread):
    def __init__(self, tts_endpoint: str, text: str, out_buffer: AudioRingBuffer):
        super().__init__(daemon=True)
        self._tts_endpoint = tts_endpoint
        self._text = text
        self._out_buffer = out_buffer
    
    def run(self):
        # 在這裡用 requests 打 TTS /synthesize endpoint
        # 用 response.iter_content(chunk_size=10240) iterate
        # 每拿到一塊 raw PCM bytes 就丟進 self._out_buffer.put(samples)
        # response body 吃完後 graceful stop（跳出 thread）

# 主程式
def main():
    # 其他初始化（VAE, UNet, ...）
    
    audio_buffer = AudioRingBuffer(sample_rate=16000, fps=args.fps)
    tts_feeder = AudioFeeder(tts_endpoint="http://foo:8001/synthesize", 
                             text=args.text,
                             out_buffer=audio_buffer)
    
    # data generator 那邊不變
    gen = data_generator(audio_buffer, latent_list, args.batch_size)
    
    # ...（跟原本一樣）
```

## 待釐清的點

1. `args.text` 怎麼來 — 是一開始全給嗎？有沒有可能邊生邊合成？
2. 要餵多久 — 一個 session 一句話？一個 session 一個 dialog 來回？要永遠餵嗎？
3. **「拿一幀的量」應該不是 `sample_rate / fps`** — 因為一幀 40ms，但 diffusion 是一次生一個 batch（8 幀 = 320ms）。`AudioRingBuffer.read()` 的 `self._chunk_size` 可能要改成 `(sample_rate / fps) * 8`。
4. **尾端不滿 320ms 怎麼辦** — 目前你說 `tts_service.py` 不會補零，會直接送長度不滿 10,240 bytes 的尾塊。JoyGen 端可以補零到 320ms，或是乾脆忽略（反正最後幾幀）。 

這個只是初步的架構 spec，實作上肯定還有細節要填，但大方向可以先朝這個目標去實驗。一邊做一邊修正設計是很正常的事 — 目前我們有最重要的起手式了。