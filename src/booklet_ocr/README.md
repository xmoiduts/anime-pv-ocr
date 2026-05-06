# Booklet OCR

批量读取歌词本或扫描图目录中的图片，一次性发送给 Gemini，输出原始响应、结构化 YAML，并由 Python 自动转换出合并后的 `.lrc`。

这个模块是独立于 `gemini_ocr` pipeline 和 `music_transcriber` 的子工具，适合处理没有音频、只有歌词本图片的场景。

# 运行方式

## 推荐：一键工作流

入口: `run_ocr_workflow.bat` 可用来串起日常手动 OCR 流程：

```text
文件流转：

Downloads → pending-OCR → booklet_ocr.main → finishing-OCR / Done / 留在 pending-OCR
```

### 如何使用一键工作流
推荐把 `src/booklet_ocr/run_ocr_workflow.bat` 创建快捷方式到桌面。之后把要处理的歌词本图片下载到系统 Downloads 目录，双击快捷方式即可：

1. 自动：扫描 Downloads 中最近一段时间{可配置}内检测到的图片
2. 手动：展示文件名、大小和修改时间，让你确认全部接受或逐张选择
3. 自动：将选中的图片移动到 `input_source_folder` 指向的 `pending-OCR`
4. 自动：调用 `python -m booklet_ocr.main` 执行 Gemini OCR
5. 手动：OCR 成功后，询问是否把已处理图片移动到 `finishing-OCR`、`Done`，或留在 `pending-OCR`

也可以从命令行直接运行：

```bat
src\booklet_ocr\run_ocr_workflow.bat
```

工作流默认扫描最近 `default_lookback_minutes` 分钟内的 Downloads 图片。这里的“最近”使用文件的修改时间和创建/到达时间两者中较新的一个，因此旧图片新放入 Downloads 后也会被视为最新文件。临时调整时间窗：

```bat
src\booklet_ocr\run_ocr_workflow.bat --minutes 5
```

仅查看将会匹配哪些 Downloads 图片，不移动文件、不请求 大模型：

```bat
src\booklet_ocr\run_ocr_workflow.bat --dry-run
```

## 另一种方法：手动调用 OCR 入口

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
- 使用一键工作流时，`finishing_ocr_folder` 和 `done_folder` 指向的归档目录也应存在，或允许脚本自动创建
- `run_ocr_workflow.bat` 会按脚本自身位置定位 `ocr_workflow.py`，复制项目到新位置后不需要修改本机绝对路径

## 配置位置

子模块自己的配置在 `src/booklet_ocr/config.yaml`：

```yaml
model: gemini-3-flash-preview
prompt_file: src/booklet_ocr/prompts/gemini-booklet-ocr.md
input_source_folder: ../lyric-booklets/pending-OCR
finishing_ocr_folder: ../lyric-booklets/finishing-OCR
done_folder: ../lyric-booklets/Done
supported_media_extensions:
  - .jpg
  - .jpeg
  - .png
  - .webp
  - .bmp
  - .tif
  - .tiff
output_root: outputs/booklet-ocr
raw_log_suffix: response_raw.log
yaml_suffix: lyrics.yaml
lrc_suffix: lyrics.lrc
warn_file_count_over: 20
small_image_media_resolution: MEDIA_RESOLUTION_HIGH
large_image_media_resolution: MEDIA_RESOLUTION_ULTRA_HIGH
pixel_threshold_for_ultra_high: 600000
thinking_level: low
default_lookback_minutes: 120
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
- 一键工作流会先从系统 Downloads 目录筛选最近图片，再移动到 `input_source_folder`
- 如果最近 `default_lookback_minutes` 分钟内没有图片命中，工作流会检查 Downloads 中最新的文件；若该文件是支持的图片格式，则以它的检测时间为锚点向前回溯同样的分钟数并收集图片
- 如果 Downloads 仍没有命中，但 `pending-OCR` 已有图片，工作流只提示已有图片数量，不做预移动，直接调用 `booklet_ocr.main`
- 一键工作流支持的 Downloads 图片后缀固定为 `.jpg`、`.jpeg`、`.png`、`.webp`、`.bmp`、`.tif`、`.tiff`

## 动态清晰度

模块会按图片像素数自动决定每张图的上传清晰度：

- `< pixel_threshold_for_ultra_high`：使用 `small_image_media_resolution`
- `>= pixel_threshold_for_ultra_high`：使用 `large_image_media_resolution`

当前配置阈值是 `600000` 像素，也就是约 0.6MP；高于或等于该阈值的图片会使用 `MEDIA_RESOLUTION_ULTRA_HIGH`。

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
- `lyrics.yaml` 从模型响应里的 YAML fenced code block 提取
- `lyrics.lrc` 由 Python 从 `lyrics.yaml` 的 `songs[].title` 和 `songs[].lyrics` 自动生成

## Prompt 约定

默认 prompt 在 `src/booklet_ocr/prompts/gemini-booklet-ocr.md`。

当前要求模型：

- 只输出一个 YAML fenced code block，不输出 LRC
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
- 一键工作流只在 `booklet_ocr.main` 成功返回后询问如何处理图片；你可以归档到 `finishing-OCR` / `Done`，也可以选择留在 `pending-OCR`
- 如果 OCR 失败，图片会留在 `pending-OCR` 方便排查或重跑
