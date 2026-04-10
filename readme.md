Anime PV OCR tool
======

# What is this?
Convert PV (Promotional Video, i.e. music video for anime-style songs) into lyrics text.
## Why this OCR tool is needed for PV lyrics?
Lyrics in PVs are usually not available as a normal, copyable text stream. Instead, they are rendered as stylized on-screen text, often animated, positioned freely, and visually inconsistent from scene to scene. This tool reads them optically so the lyrics can be extracted as text.

# Design principles
- CLI first, yaml then, not considering GUI/web interface.
- Google Gemini-centered, no \[whisper, deepseek-ocr, paddle-ocr\] (yet).
- Working as a pipeline, no agent yet.

# Usage

## OCR a PV (Music Video with rasterized lyric artword)
The main OCR entry point is `main.py`.

### Before running it:
- set `GOOGLE_API_KEY` in `.env` file
- put the source video in `medias/`
- configure tasks/pipelines in `ocr-cli-config.yaml`

### Basic usage:

```bash
python main.py -i "<media name substring>"
```

Examples:

```bash
# quick API connectivity check
python main.py --hello

# input: -i <media name substring>
python main.py -i "hats a H"
  # If you have a media file named "Whats a Hero.mp4" in `medias/`, it will match, and same hereinafter.
  # If you don't pass a task or pipeline name, a default task: `spotter` will be run.

# pass a task name: -t <task name>
python main.py -t ocr-filtered -i "hats a H"

# passing a pipeline name: -p <pipeline name>
python main.py -p all -i "hats a H"
  # we have defined an example pipeline named "all", which serializes all OCR steps that we use:
  # spotter -> dig-hard-samples -> ocr-filtered

  # pipeline without hard-sample digging
  python main.py -p no-hard-samples -i "hats a H"

# limit the frame range used for grid generation
python main.py -i "hats a H" -r "1-300"

```

## tasks explained
- spotter: ...
- dig-hard-samples: ...
- ocr-filtered: ...

## pipelines explained
- all: spotter -> dig-hard-samples -> ocr-filtered
- no-hard-samples: spotter -> ocr-filtered


## Useful flags:
- `-t, --task`: run one task from `ocr-cli-config.yaml` (defaults to `spotter`)
- `-p, --pipeline`: run a named pipeline from `ocr-cli-config.yaml`
- `-i, --input-file`: substring used to match the target media/output folder
- `-r, --range`: frame range such as `1-300` or `255-`
- `--suffix`: pick a specific prior result set when multiple runs exist
- `--prompt-file`: override the prompt markdown file for the run
- `-m, --model`: override the Gemini model
- `-b, --base-url`: use a custom Gemini-compatible endpoint

Outputs are written under `outputs/` for the matched media item.

# Other sub-tools
The project utilizes Gemini API functionalities, some side tools are also co-located in the same project to reuse the existing API communication functionality:

## Music Transcriber
Given the music files only, transcribe the lyrics from autio without visual media.

Usage see as src/music_transcriber/README.md

## Comparison Video Generator
Visualizes the output of pipeline:all tasks.

Usage see as src/comparison_video/README.md

## Lyric Booklet OCR
Extract lyric text from lyric booklet images.

Usage see as src/booklet_ocr/README.md