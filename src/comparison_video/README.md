# Comparison Video Generator

生成展示 `pipeline:all` 结果的对比视频，并复用源视频音轨。

## 功能特性

- 四分区布局：当前 OCR 帧、歌词面板、原视频、dig-hard strips、spotter grid
- Fill-forward 时间轴：OCR 帧与歌词都按最近有效帧延续
- 性能优化：`strip/spotter` 页面缓存、后台 prefetch、细粒度 trace
- 音视频输出：FFmpeg 管道编码视频并从原媒体混入音频（若存在）
- 实时预览：可选 OpenCV 预览窗口，支持键盘控制

## 使用方法

### 基本用法

```bash
# 从 src/ 目录运行
cd src
python -m comparison_video.main -i <media_substring>
```

### 启用预览

```bash
python -m comparison_video.main -i AYAKAKI --preview
```

### 启用性能 trace

```bash
python -m comparison_video.main -i AYAKAKI --trace-perf
```

### 指定 trace 文件

```bash
python -m comparison_video.main -i AYAKAKI --trace-perf --trace-file ../outputs/trace.json
```

## 预览控制

启用 `--preview` 后：

| 按键 | 功能 |
|------|------|
| `P` / `空格` | 开关预览窗口 |
| `Q` / `ESC` | 退出并保存（保存到当前帧） |

## 前置条件

1. 先运行 `pipeline:all`：
   ```bash
   python src/gemini_ocr.py -i <media> --pipeline all
   ```
2. 输出目录 `<outputs>/<media_hash>/spotter-results/` 下存在：
   - `*.yaml`（spotter）
   - `digger_results_*.yaml`
   - `ocr_results_*.yaml`
3. 系统可用 `ffmpeg`（`ffmpeg -version`）

## 配置项

在 `ocr-cli-config.yaml` 的 `task.make-comparison-videos` 中配置：

```yaml
make-comparison-videos:
  target_fps: 6.0
  stripping: 5
  spotter_grid:
    cols: 4
    rows: 3
  layout:
    output_width: 1920
    output_height: 1080
    left_ratio: 0.4
    left_top_ratio: 0.5
    right_top_ratio: 0.6
    right_bottom_left_ratio: 0.5
  effects:
    fade_opacity: 0.5
    spotter_color: [0, 255, 0]   # 绿色（BGR）
    digger_color: [255, 0, 0]    # 蓝色（BGR）
    highlight_color: [0, 0, 255] # 兼容旧字段，strip 实际用 digger_color
  prefetch:
    enabled: true
    depth: 2
  preview:
    enabled: false
    scale: 0.5
```

## 当前显示规则说明

### Spotter Grid

- 被 OCR 选中的格子（`source=spotter/digger`）保持正常亮度
- 未被选中的格子统一减淡
- `current frame` 仅决定“粗边框高亮”，不决定亮度

### Strip Grid

- active strip = `<= current_frame_id` 的最大 strip frame id
- active strip 若被 OCR 选中：蓝框；未选中：灰框
- 未被 OCR 选中的 strip 一律减淡，被选中的 strip 一律不减淡

## 性能 trace 事件

开启 `--trace-perf` 后，会输出 Chrome Trace JSON（可在 `chrome://tracing` 或 Perfetto 打开）。

重点关注：

- `pipeline.composite_frame`
- `renderer.spotter_grid.prefetch_page`
- `renderer.strip_grid.prefetch_page`
- `renderer.spotter_grid.prefetch_seq_*`
- `renderer.strip_grid.prefetch_seq_*`
- `renderer.strip_grid.prefetch_seek_only_*`

## 输出

- 默认输出：`outputs/<media_hash>/comparison_video.mp4`
- 编码：H.264 (`libx264`) + `yuv420p`
- 音频：自动尝试复用源媒体音轨（AAC 复用）

## 故障排除

### 输出无音频

- 确认源媒体本身有音轨（可用 `ffprobe -show_streams`）
- 确认 `ffmpeg` 可正常读取源媒体

### 生成很慢 / CPU 利用率不高

- 先开 `--trace-perf`，看 `prefetch_page` 与 `decode_frame`
- 适当调小分辨率（`layout.output_width/height`）
- 适当调整 `prefetch.depth`

### 预览无响应

- 确保在有 GUI 的环境运行
- Windows 环境下检查显示权限


