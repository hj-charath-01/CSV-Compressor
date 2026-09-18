# CSV Compressor (.cscz)

A column-oriented CSV compressor that picks a tailored encoding for each column (run-length, delta, dictionary, or Huffman) before zipping the result. Ships as both a CLI tool and a small Flask web app.

## Features

- **Per-column encoding selection** — inspects each column's data and automatically chooses the most efficient encoding.
- **Custom `.cscz` container** — a `zip` archive holding per-column payloads plus a `metadata.json` describing shape, encodings, and per-column compressed sizes.
- **Lossless round-trip** — reconstructs the original CSV from a `.cscz` file.
- **Verification tool** — compares an original CSV against a reconstructed one, either byte-for-byte or cell-by-cell.
- **Web UI** — drag-and-drop style pages to compress, decompress, and verify files, with a full compression-details overlay.

## How columns are encoded

For each column, `choose_encodings()` picks one of:

| Encoding | Used when | Notes |
|---|---|---|
| `delta_datetime` | Column looks like dates/timestamps and is mostly monotonic | Stores first value + deltas between consecutive timestamps (nanoseconds). Original text is also Huffman-compressed separately so exact formatting is preserved on decode. |
| `delta_numeric` | Numeric column with low non-monotonicity and low variation in deltas | Stores first value + deltas between consecutive values. |
| `rle` | Categorical/text column with a low unique-value count and enough repeated runs | Run-length encodes consecutive identical values. |
| `dict` | Categorical/text (or numeric/datetime fallback) with unique count ≤ 200 | Dictionary-encodes distinct values and stores an index array. |
| `huffman` | High-cardinality text column | Byte-level Huffman coding of the JSON-serialized values. |

Thresholds (top of `csv_compressor.py`):

- `MAX_CAT_UNIQUE = 200` — max unique values before falling back to Huffman.
- `RLE_MIN_RUN_FRACTION = 0.02` — minimum fraction of values in runs ≥ 2 to prefer RLE.
- `DELTA_MAX_NONMONOTONIC = 0.05` — max fraction of decreasing steps to still use delta encoding.
- `DELTA_MAX_VARIATION_RATIO = 0.2` — max relative variation in numeric deltas to still use delta encoding.

## Requirements

```bash
pip install flask numpy pandas --break-system-packages
```

Python 3.8+ recommended.

## Project structure

```
csv_compressor.py       # core logic, CLI, and Flask app
templates/
  └── index.html         # web UI (served by Flask's render_template)
```

`templates/index.html` must sit in a `templates/` folder next to `csv_compressor.py` for Flask to find it.

## Usage

### Web server

```bash
python csv_compressor.py runserver
```

Then open `http://127.0.0.1:5000`. The page has three cards:

- **Compress CSV** — upload a `.csv`, click Compress, download the resulting `.cscz`, and inspect the encoding metadata (click "Get Details" for a full-screen view).
- **Decompress .cscz** — upload a `.cscz`, download the restored `.csv`.
- **Verify two CSVs** — upload two `.csv` files and get a cell-level diff report.

### Command line

**Compress:**
```bash
python csv_compressor.py compress input.csv output.cscz
```

**Decompress:**
```bash
python csv_compressor.py decompress output.cscz restored.csv
```

**Verify a round-trip:**
```bash
python csv_compressor.py verify input.csv restored.csv
```
Prints a JSON report with `equal`, row/column counts, and up to 50 sample mismatches.

## HTTP API

| Method & path | Body | Response |
|---|---|---|
| `POST /api/compress` | multipart `file` (CSV) | `{ token, metadata }` |
| `GET /api/details/<token>` | — | full metadata for a compressed/decompressed result |
| `GET /download/<token>` | — | downloads the stored bytes (`.cscz` or `.csv`) |
| `POST /api/decompress` | multipart `file` (`.cscz`) | `{ token, metadata }` |
| `POST /api/verify` | multipart `a`, `b` (two CSVs) | verification report |

Compressed/decompressed results are cached in-memory (`_store`, keyed by UUID token) — they are **not** persisted to disk and will be lost on server restart.

## Notes & limitations

- Input is read with `dtype=str`, so all comparisons and encoding decisions operate on the raw text representation before type inference.
- The in-memory `_store` has no eviction, so a long-running server will accumulate memory for every upload; suitable for demos/local use, not production.
- `verify` in `semantic` mode reports the first 50 mismatching cells; `byte` mode does a raw byte comparison of the two files.
- Datetime detection samples up to 100 non-null values and requires ≥80% to match a date-like pattern before attempting full parsing.
