You are recognizing lyrics text from booklet or scan images for one release.

All attached images belong to the same booklet batch. Read them together, preserve the printed language/script, and do not invent content that is not visible.

Goals:

1. Recover song titles and lyrics from the booklet pages.
2. Preserve line breaks when they are visually meaningful for lyrics.
3. Ignore decorative background text unless it is part of the actual title or lyrics.
4. Keep Japanese text in Japanese script; do not romanize it.
5. If a title or lyric segment is uncertain, keep the best guess but note uncertainty in YAML.
6. If a song title is written in kana but is clearly spelling a non-Japanese word or loanword, add a YAML field with the most likely intended real-word spelling. This field is only for editor autocomplete hints and does not need to be present when not applicable.

Output exactly one fenced YAML code block and nothing else.

The Python workflow will generate downstream LRC text from this YAML, so do not output an LRC block.

```yaml
songs:
  - title: ""
    title_actual_spelling: ""
    title_guess_confidence: high
    lyrics: |
      ...
```
