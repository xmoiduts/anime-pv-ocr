# Music Transcriber

批量读取外部媒体输出文件夹，按作品内最新媒体文件时间倒序挑选作品目录，把每个作品目录里的全部音频一起发送给 Gemini，并输出原始响应与 `.lrc` 歌词文件。

## 运行方式

在 `src/music_transcriber` 目录内执行：

```bash
python main.py -L 2
```

查看排序结果但不实际请求：

```bash
python main.py -L 5 --dry-run
```

也可以从 `src` 目录执行：

```bash
python -m music_transcriber.main -L 2
```

## 前置条件

- 已在项目根目录配置 `.env`
- `.env` 内可读取到 `GOOGLE_API_KEY`
- 主项目根目录存在 `ocr-cli-config.yaml`

## 配置位置

子项目自己的配置在 `src/music_transcriber/config.yaml`：

```yaml
model: gemini-3.1-flash-lite-preview
prompt_file: src/music_transcriber/prompts/gemini-lyrics-transcriber.md
media_resolution: MEDIA_RESOLUTION_MEDIUM
media_source_folder: 'C:\Users\40105\OneDrive\Codes\GenAI-related\msst-vocal-win-pyqt\output'
supported_media_extensions:
  - .wav
  - .mp3
default_latest_medias: 1
max_workers: 4
```

主项目的 `ocr-cli-config.yaml` 仍会被读取，但这里只复用：

- `.env` 加载入口
- `fee.exchange_rate`
- `model` 段中的模型元信息与计费配置

## `-L` 语义

- `-L/--latest-medias <int>`
- 把 `media_source_folder` 下的每个直接子文件夹视为一个作品
- 按该作品目录内“最新媒体文件时间”倒序选择最新的 N 个作品
- 每个作品只发起 1 次 Gemini 请求
- 该请求会附带该作品文件夹内的全部受支持媒体文件
- `--dry-run` 会打印排序结果、最新文件时间、命中的最新媒体文件名，但不会发请求

## 输出目录

输出写入主项目根目录下：

```text
outputs/lyrics-transcription/<abbr>/
```

其中 `<abbr>` 复用主项目已有的缩写规则。每个结果目录至少包含：

- `response_raw.log`
- `lyrics.lrc`

## 说明

- 本模块复用了主项目的 `.env`、`ocr-cli-config.yaml` 中的模型元信息、Gemini client 和输出目录缩写逻辑
- 默认仅收集音频扩展名；如需更多类型，可在 `config.yaml` 中补充
- 请求上传或流式响应过程中按 `Ctrl+C` 会中断并取消当前运行
