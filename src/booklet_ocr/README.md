# Booklet OCR

批量读取歌词本或扫描图目录中的图片，一次性发送给 Gemini，输出原始响应、结构化 YAML 和合并后的 `.lrc`。

这个模块是独立于 `gemini_ocr` pipeline 和 `music_transcriber` 的子工具，适合处理没有音频、只有歌词本图片的场景。

## 运行方式

从 `src` 目录执行：

```bash
python -m booklet_ocr.main
```

仅查看将要处理的文件和输出命名，不实际请求：

```bash
python -m booklet_ocr.main --dry-run
```

如果要临时切换本地配置文件：

```bash
python -m booklet_ocr.main -c src/booklet_ocr/config.yaml
```

## 前置条件

- 已在项目根目录配置 `.env`
- `.env` 内可读取到 `GOOGLE_API_KEY`
- 主项目根目录存在 `ocr-cli-config.yaml`
- `src/booklet_ocr/config.yaml` 中的 `input_source_folder` 指向一个存在的图片目录

## 配置位置

子模块自己的配置在 `src/booklet_ocr/config.yaml`：

```yaml
model: gemini-3-flash-preview
prompt_file: src/booklet_ocr/prompts/gemini-booklet-ocr.md
input_source_folder: ../lyric-booklets/pending-OCR
supported_media_extensions:
  - .jpg
  - .jpeg
  - .png
output_root: outputs/booklet-ocr
warn_file_count_over: 20
small_image_media_resolution: MEDIA_RESOLUTION_HIGH
large_image_media_resolution: MEDIA_RESOLUTION_ULTRA_HIGH
pixel_threshold_for_ultra_high: 6000000
thinking_level: low
```

主项目的 `ocr-cli-config.yaml` 仍会被读取，但这里只复用：

- `.env` 加载入口
- `fee.exchange_rate`
- `model` 段中的模型元信息与计费配置

## 输入规则

- 输入目录只来自 `config.yaml` 中的 `input_source_folder`
- 支持绝对路径，也支持相对于项目根目录的路径
- 当前实现只枚举输入目录的直接子文件，不递归进入子目录
- 文件按文件名排序后一次性整体发送给 Gemini
- 如果文件数大于 `warn_file_count_over`，只打印告警，不自动拆批

## 动态清晰度

模块会按图片像素数自动决定每张图的上传清晰度：

- `< pixel_threshold_for_ultra_high`：使用 `small_image_media_resolution`
- `>= pixel_threshold_for_ultra_high`：使用 `large_image_media_resolution`

默认阈值是 `6000000` 像素，也就是约 6MP。

## 输出目录

输出写入主项目根目录下：

```text
outputs/booklet-ocr/<yyyymmdd>/
```

单次运行会生成一个时间戳前缀 `yyyymmdd-hhmmss`，并输出：

- `<timestamp>.response_raw.log`
- `<timestamp>.lyrics.yaml`
- `<timestamp>.lyrics.lrc`

其中：

- `response_raw.log` 会保留 Gemini 原始回答
- 若模型返回了 thought summary，也会一并写入 raw log
- `lyrics.yaml` 和 `lyrics.lrc` 从模型响应里的 fenced code block 提取

## Prompt 约定

默认 prompt 在 `src/booklet_ocr/prompts/gemini-booklet-ocr.md`。

当前要求模型：

- 输出两个 fenced code block，分别是 YAML 和 LRC
- 保留日文原文，不要罗马字化
- 如果歌名是用假名拼出的外来词/非日语单词，可额外返回 `title_actual_spelling`

这个 `title_actual_spelling` 目前不参与程序逻辑，只是预留给编辑器联想使用。

## 拆分合并后的 LRC

如果一次运行产出的是一个合并 `.lrc`，可以使用示例脚本按 `[ti:...]` 拆成单曲文件：

```bash
python src/booklet_ocr/split_combined_lrc.py.example outputs/booklet-ocr/20260408/20260408-153503.lyrics.lrc
```

也可以手动指定输出目录：

```bash
python src/booklet_ocr/split_combined_lrc.py.example outputs/booklet-ocr/20260408/20260408-153503.lyrics.lrc --output-dir outputs/booklet-ocr/20260408/custom-split
```

默认会输出到与输入 LRC 同级的：

```text
<timestamp>.split-lrc/
```

## 说明

- 本模块不依赖输入图片文件名具有歌曲语义，输出归档按日期和时间戳组织
- 当前不做自动分批、自动重试或超量兜底
- 歌曲名占位文件接口已预留，但尚未正式启用
