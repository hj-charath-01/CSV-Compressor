#!/usr/bin/env python3
"""
csv_compressor_web.py

Single-file Flask web app + CLI for compressing/decompressing/verifying CSV files.

Changes from prior versions:
 - DOES NOT store original.csv inside the archive (per your request).
 - CLI exposes only three subcommands: compress, decompress, verify (no extra flags).

Web UI:
 - Upload a CSV and click Compress -> returns a downloadable .cscz and a token to fetch details.
 - After compressing, click Get Details -> shows per-column encoding and compressed-bytes.
 - Upload a .cscz and click Decompress -> returns decompressed CSV.
 - Upload two CSVs and click Verify -> runs semantic verification and returns JSON report.

Run:
  - As server: python csv_compressor_web.py runserver
  - CLI: python csv_compressor_web.py compress input.csv output.cscz
         python csv_compressor_web.py decompress input.cscz output.csv
         python csv_compressor_web.py verify orig.csv recon.csv

"""
from __future__ import annotations

import argparse
import io
import json
import csv
import zipfile
import warnings
import os
import sys
import gzip
import bz2
import lzma
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import heapq
import time
import uuid
import tempfile

from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings / thresholds
# ---------------------------------------------------------------------------
MAX_CAT_UNIQUE = 200
RLE_MIN_RUN_FRACTION = 0.02
DELTA_MAX_NONMONOTONIC = 0.05
DELTA_MAX_VARIATION_RATIO = 0.2
PROGRESS_UPDATE_INTERVAL = 0.1

# ---------------------------------------------------------------------------
# Utility: small in-memory store for generated files (token -> payload)
# ---------------------------------------------------------------------------
_store: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Core helpers (RLE, delta, dict, huffman) - adapted from earlier script
# ---------------------------------------------------------------------------

def detect_datetime_series(s: pd.Series, sample_size=100) -> Tuple[pd.Series, bool]:
    vals = s.dropna().astype(str)
    if len(vals) == 0:
        return s, False
    sample = vals.sample(min(sample_size, len(vals)), random_state=0)
    date_like = (
        sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}") |
        sample.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
    )
    if date_like.mean() < 0.8:
        return s, False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            conv = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if conv.notna().sum() >= 0.9 * len(s):
            return conv, True
    except Exception:
        pass
    return s, False


def rle_encode(values: List[Any]) -> Tuple[List[Any], List[int]]:
    if len(values) == 0:
        return [], []
    vals = []
    lengths = []
    prev = values[0]
    count = 1
    for v in values[1:]:
        if pd.isna(v) and pd.isna(prev):
            count += 1
        elif v == prev:
            count += 1
        else:
            vals.append(prev)
            lengths.append(count)
            prev = v
            count = 1
    vals.append(prev)
    lengths.append(count)
    return vals, lengths


def rle_decode(values: List[Any], lengths: List[int]) -> List[Any]:
    out: List[Any] = []
    for v, L in zip(values, lengths):
        out.extend([v] * L)
    return out


def delta_encode_numeric_preserve(arr: np.ndarray) -> Dict[str, Any]:
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "first_index": None, "first": None, "deltas": []}
    arr = arr.astype(float)
    notnan_idx = np.where(~np.isnan(arr))[0]
    if len(notnan_idx) == 0:
        return {"n": n, "first_index": None, "first": None, "deltas": [None] * n}
    first_index = int(notnan_idx[0])
    first_value = float(arr[first_index])
    deltas: List[Optional[float]] = [None] * n
    for i in range(first_index + 1, n):
        a = arr[i - 1]
        b = arr[i]
        if np.isnan(a) or np.isnan(b):
            deltas[i] = None
        else:
            deltas[i] = float(b - a)
    return {"n": n, "first_index": first_index, "first": first_value, "deltas": deltas}


def delta_decode_numeric(payload: Dict[str, Any]) -> List[str]:
    n = int(payload.get("n", 0))
    if n == 0:
        return []
    first_index = payload.get("first_index")
    first = payload.get("first")
    deltas = payload.get("deltas", [None] * n)
    out: List[Optional[float]] = [None] * n
    if first_index is None:
        return [""] * n
    out[first_index] = float(first)
    for i in range(first_index + 1, n):
        d = deltas[i]
        if d is None:
            out[i] = None
        else:
            prev = out[i - 1]
            if prev is None:
                out[i] = None
            else:
                out[i] = prev + float(d)
    return [("" if v is None else str(v)) for v in out]


def delta_encode_datetime_preserve(conv_series: pd.Series) -> Dict[str, Any]:
    n = len(conv_series)
    if n == 0:
        return {"n": 0, "first_index": None, "first": None, "deltas": []}
    ts_values = conv_series.apply(lambda x: None if pd.isna(x) else int(pd.Timestamp(x).value)).tolist()
    first_index = None
    first_ns = None
    for idx, v in enumerate(ts_values):
        if v is not None:
            first_index = idx
            first_ns = int(v)
            break
    deltas: List[Optional[int]] = [None] * n
    if first_index is None:
        return {"n": n, "first_index": None, "first": None, "deltas": deltas}
    prev = first_ns
    for i in range(first_index + 1, n):
        cur = ts_values[i]
        if cur is None:
            deltas[i] = None
            prev = None
        else:
            if prev is None:
                deltas[i] = None
                prev = int(cur)
            else:
                deltas[i] = int(cur - prev)
                prev = int(cur)
    return {"n": n, "first_index": first_index, "first": first_ns, "deltas": deltas}


def delta_decode_datetime(payload: Dict[str, Any]) -> List[str]:
    n = int(payload.get("n", 0))
    if n == 0:
        return []
    first_index = payload.get("first_index")
    first = payload.get("first")
    deltas = payload.get("deltas", [None] * n)
    out_ns: List[Optional[int]] = [None] * n
    if first_index is None:
        return [""] * n
    out_ns[first_index] = int(first)
    prev = out_ns[first_index]
    for i in range(first_index + 1, n):
        d = deltas[i]
        if d is None:
            out_ns[i] = None
            prev = None
        else:
            if prev is None:
                out_ns[i] = None
                prev = None
            else:
                prev = int(prev + d)
                out_ns[i] = prev
    res: List[str] = []
    for v in out_ns:
        if v is None:
            res.append("")
        else:
            res.append(pd.to_datetime(int(v)).isoformat())
    return res


def dict_encode_series(s: pd.Series) -> Dict[str, Any]:
    vals = []
    for v in s.tolist():
        if pd.isna(v):
            vals.append('__nan__')
        elif v == '':
            vals.append('__empty__')
        else:
            vals.append(v)
    uniq = list(dict.fromkeys(vals))
    index_map = {v: i for i, v in enumerate(uniq)}
    indices = [index_map[v] for v in vals]
    return {'dict': uniq, 'indices': indices, 'n': len(vals)}


def dict_decode(payload: Dict[str, Any]) -> List[str]:
    dict_list = payload.get('dict', [])
    indices = payload.get('indices', [])
    out: List[str] = []
    for i in indices:
        v = dict_list[i]
        if v == '__nan__' or v is None:
            out.append('')
        elif v == '__empty__':
            out.append('')
        else:
            out.append(v)
    return out


# Huffman functions (kept for completeness from advanced version)
class HuffmanNode:
    def __init__(self, weight: int, symbol: Optional[int] = None, left=None, right=None):
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.weight < other.weight


def build_huffman_codes(data: bytes) -> Dict[int, str]:
    if len(data) == 0:
        return {}
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    pq = []
    for sym, w in freq.items():
        heapq.heappush(pq, (w, HuffmanNode(w, symbol=sym)))
    if len(pq) == 1:
        _, node = heapq.heappop(pq)
        return {node.symbol: '0'}
    while len(pq) > 1:
        w1, n1 = heapq.heappop(pq)
        w2, n2 = heapq.heappop(pq)
        merged = HuffmanNode(w1 + w2, symbol=None, left=n1, right=n2)
        heapq.heappush(pq, (merged.weight, merged))
    _, root = heapq.heappop(pq)
    codes: Dict[int, str] = {}

    def walk(node: HuffmanNode, prefix: str):
        if node.symbol is not None:
            codes[node.symbol] = prefix or '0'
            return
        walk(node.left, prefix + '0')
        walk(node.right, prefix + '1')

    walk(root, '')
    return codes


def huffman_encode_bytes(data: bytes) -> Dict[str, Any]:
    codes = build_huffman_codes(data)
    bit_chunks: List[str] = []
    for b in data:
        bit_chunks.append(codes[b])
    bitstr = ''.join(bit_chunks)
    bit_length = len(bitstr)
    pad = (8 - (bit_length % 8)) % 8
    if pad:
        bitstr += '0' * pad
    b_arr = bytearray()
    for i in range(0, len(bitstr), 8):
        byte = bitstr[i:i + 8]
        b_arr.append(int(byte, 2))
    codes_str = {str(k): v for k, v in codes.items()}
    return {'codes': codes_str, 'bit_length': bit_length, 'bin': bytes(b_arr)}


def huffman_decode_bytes(codes_str: Dict[str, str], bin_data: bytes, bit_length: int) -> bytes:
    if bit_length == 0:
        return b''
    codes = {v: int(k) for k, v in codes_str.items()}
    bits = []
    for b in bin_data:
        bits.append(format(b, '08b'))
    bitstr = ''.join(bits)[:bit_length]
    out = bytearray()
    cur = ''
    i = 0
    while i < len(bitstr):
        cur += bitstr[i]
        i += 1
        if cur in codes:
            out.append(codes[cur])
            cur = ''
    return bytes(out)

# ---------------------------------------------------------------------------
# Encoding decision
# ---------------------------------------------------------------------------

def choose_encodings(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    enc = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        col_meta: Dict[str, Any] = {'dtype': str(s.dtype)}
        conv, is_dt = detect_datetime_series(s)
        if is_dt:
            conv_ints = conv.dropna().view('int64').to_numpy()
            if len(conv_ints) <= 1:
                col_meta['encoding'] = 'delta_datetime'
            else:
                diffs = np.diff(conv_ints)
                nonmono = np.sum(diffs < 0)
                frac_nonmono = nonmono / max(1, len(diffs))
                if frac_nonmono <= DELTA_MAX_NONMONOTONIC:
                    col_meta['encoding'] = 'delta_datetime'
                else:
                    col_meta['encoding'] = 'raw'
            enc[col] = col_meta
            continue

        if pd.api.types.is_numeric_dtype(s):
            arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
            valid = ~np.isnan(arr)
            if valid.sum() <= 1:
                col_meta['encoding'] = 'delta_numeric'
            else:
                diffs = np.diff(arr[valid]) if valid.sum() > 1 else np.array([])
                if len(diffs) == 0:
                    frac_nonmono = 0.0
                    var_ratio = 0.0
                else:
                    nonmono = np.sum(diffs < 0)
                    frac_nonmono = nonmono / max(1, len(diffs))
                    median_delta = np.median(np.abs(diffs)) if len(diffs) > 0 else 0.0
                    if median_delta == 0:
                        var_ratio = 0.0
                    else:
                        var_ratio = np.median(np.abs(diffs - np.median(diffs))) / (median_delta + 1e-12)
                if frac_nonmono <= DELTA_MAX_NONMONOTONIC and var_ratio <= DELTA_MAX_VARIATION_RATIO:
                    col_meta['encoding'] = 'delta_numeric'
                else:
                    col_meta['encoding'] = 'raw'
            enc[col] = col_meta
            continue

        unique_vals = int(s.nunique(dropna=False))
        col_meta['unique_count'] = unique_vals
        if unique_vals <= MAX_CAT_UNIQUE:
            vals = s.astype(object).fillna('__nan__').tolist()
            total_in_runs = 0
            i = 0
            while i < len(vals):
                j = i + 1
                while j < len(vals) and vals[j] == vals[i]:
                    j += 1
                run_len = j - i
                if run_len >= 2:
                    total_in_runs += run_len
                i = j
            if total_in_runs >= RLE_MIN_RUN_FRACTION * n:
                col_meta['encoding'] = 'rle'
            else:
                col_meta['encoding'] = 'dict'
        else:
            col_meta['encoding'] = 'huffman'
        enc[col] = col_meta
    return enc

# ---------------------------------------------------------------------------
# Fallback compressors used previously (kept but optional)
# ---------------------------------------------------------------------------

def try_fallback_compressions_stream(orig_path: str, show_progress: bool = True) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        size = os.path.getsize(orig_path)
    except Exception:
        size = None

    candidates: List[Tuple[str, bytes]] = []

    # gzip
    try:
        buf = io.BytesIO()
        with (open(orig_path, 'rb')) as fin, gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as gz:
            read = 0
            while True:
                chunk = fin.read(65536)
                if not chunk:
                    break
                gz.write(chunk)
                read += len(chunk)
        gz_bytes = buf.getvalue()
        candidates.append(('gzip', gz_bytes))
    except Exception:
        pass

    # bz2
    try:
        compressor = bz2.BZ2Compressor(9)
        compressed_parts = []
        with open(orig_path, 'rb') as fin:
            while True:
                chunk = fin.read(65536)
                if not chunk:
                    break
                compressed_parts.append(compressor.compress(chunk))
            compressed_parts.append(compressor.flush())
        bz_bytes = b''.join(compressed_parts)
        candidates.append(('bz2', bz_bytes))
    except Exception:
        pass

    # lzma
    try:
        compressor = lzma.LZMACompressor(preset=9)
        compressed_parts = []
        with open(orig_path, 'rb') as fin:
            while True:
                chunk = fin.read(65536)
                if not chunk:
                    break
                compressed_parts.append(compressor.compress(chunk))
            compressed_parts.append(compressor.flush())
        lz_bytes = b''.join(compressed_parts)
        candidates.append(('lzma', lz_bytes))
    except Exception:
        pass

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: len(x[1]))
    return candidates[0][1], candidates[0][0]

# ---------------------------------------------------------------------------
# Compression (in-memory ZIP) - NOTE: does NOT store original.csv inside archive
# ---------------------------------------------------------------------------

def compress_csv_inmemory_bytes(input_csv_path: str, show_progress: bool = False) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress input CSV into an in-memory .cscz (zip) and return (zip_bytes, details_metadata)
    This function does NOT write original.csv into the archive.
    details_metadata is the metadata dict that would be written into metadata.json and also
    contains per-column compressed size estimates computed after creating the in-memory zip.
    """
    df = pd.read_csv(input_csv_path, dtype=str)
    nrows, ncols = df.shape
    enc = choose_encodings(df)

    in_mem = io.BytesIO()
    with zipfile.ZipFile(in_mem, 'w', compression=zipfile.ZIP_DEFLATED) as Z:
        metadata = {
            'original_shape': [nrows, ncols],
            'columns': {},
            'pandas_version': pd.__version__,
            'created': datetime.utcnow().isoformat() + 'Z',
            'original_file': os.path.basename(input_csv_path),
            'store_original': False,
            'fallback': None
        }
        for col in df.columns:
            s = df[col]
            col_info = enc[col].copy()
            enc_name = col_info['encoding']
            if enc_name == 'rle':
                vals, lengths = rle_encode(s.fillna('__nan__').tolist())
                payload = {'values': vals, 'lengths': lengths, 'n': len(s)}
                Z.writestr(f'cols/{col}.rle.json', json.dumps(payload, ensure_ascii=False))
                col_info['rle_file'] = f'cols/{col}.rle.json'
            elif enc_name == 'delta_numeric':
                arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
                payload = delta_encode_numeric_preserve(arr)
                Z.writestr(f'cols/{col}.delta.json', json.dumps(payload, ensure_ascii=False))
                col_info['delta_file'] = f'cols/{col}.delta.json'
            elif enc_name == 'delta_datetime':
                conv = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
                payload = delta_encode_datetime_preserve(conv)
                Z.writestr(f'cols/{col}.deltadt.json', json.dumps(payload, ensure_ascii=False))
                col_info['deltadt_file'] = f'cols/{col}.deltadt.json'
            elif enc_name == 'dict':
                payload = dict_encode_series(s)
                Z.writestr(f'cols/{col}.dict.json', json.dumps(payload, ensure_ascii=False))
                col_info['dict_file'] = f'cols/{col}.dict.json'
            elif enc_name == 'huffman':
                vals = []
                for v in s.tolist():
                    if pd.isna(v):
                        vals.append('__nan__')
                    elif v == '':
                        vals.append('__empty__')
                    else:
                        vals.append(v)
                json_bytes = json.dumps(vals, ensure_ascii=False).encode('utf-8')
                huff = huffman_encode_bytes(json_bytes)
                Z.writestr(f'cols/{col}.huff.bin', huff['bin'])
                meta = {'codes': huff['codes'], 'bit_length': huff['bit_length'], 'n': len(vals)}
                Z.writestr(f'cols/{col}.huff.json', json.dumps(meta, ensure_ascii=False))
                col_info['huff_file'] = f'cols/{col}.huff.json'
                col_info['huff_bin'] = f'cols/{col}.huff.bin'
            else:
                buf = io.StringIO()
                writer = csv.writer(buf)
                for v in s.tolist():
                    if pd.isna(v):
                        writer.writerow(['__nan__'])
                    elif v == '':
                        writer.writerow(['__empty__'])
                    else:
                        writer.writerow([v])
                Z.writestr(f'cols/{col}.raw.csv', buf.getvalue())
                col_info['raw_file'] = f'cols/{col}.raw.csv'
            metadata['columns'][col] = col_info
        Z.writestr('metadata.json', json.dumps(metadata, ensure_ascii=False, default=str))
    in_mem.seek(0)
    zip_bytes = in_mem.getvalue()

    # compute per-column compressed sizes
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as Z2:
        info_map = {info.filename: info for info in Z2.infolist()}
        per_column_sizes: Dict[str, int] = {}
        for col, info in metadata['columns'].items():
            enc_name = info['encoding']
            files = []
            if enc_name == 'rle' and 'rle_file' in info:
                files = [info['rle_file']]
            elif enc_name == 'delta_numeric' and 'delta_file' in info:
                files = [info['delta_file']]
            elif enc_name == 'delta_datetime' and 'deltadt_file' in info:
                files = [info['deltadt_file']]
            elif enc_name == 'dict' and 'dict_file' in info:
                files = [info['dict_file']]
            elif enc_name == 'huffman':
                files = [info['huff_file'], info['huff_bin']] if 'huff_file' in info and 'huff_bin' in info else []
            elif enc_name == 'raw' and 'raw_file' in info:
                files = [info['raw_file']]
            else:
                for fname in info_map:
                    if fname.startswith(f'cols/{col}.'):
                        files.append(fname)
            s = 0
            for fname in files:
                if fname in info_map:
                    s += info_map[fname].compress_size
            per_column_sizes[col] = s
        metadata['per_column_compressed_bytes'] = per_column_sizes
        metadata['zip_size'] = len(zip_bytes)
    return zip_bytes, metadata

# ---------------------------------------------------------------------------
# Decompress in-memory (reconstruct CSV bytes)
# ---------------------------------------------------------------------------

def decompress_cscz_bytes(cscz_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Decompress .cscz bytes and return (csv_bytes, metadata)
    If a fallback compressed original is present the function will attempt to restore it; otherwise
    it reconstructs from per-column encodings.
    """
    with zipfile.ZipFile(io.BytesIO(cscz_bytes), 'r') as Z:
        meta = json.loads(Z.read('metadata.json').decode('utf-8'))
        # try fallback inner
        fb = meta.get('fallback')
        if fb:
            inner = fb.get('inner_name')
            algo = fb.get('algo')
            if inner and inner in Z.namelist():
                inner_bytes = Z.read(inner)
                try:
                    if algo == 'gzip':
                        orig = gzip.decompress(inner_bytes)
                    elif algo == 'bz2':
                        orig = bz2.decompress(inner_bytes)
                    elif algo == 'lzma':
                        orig = lzma.decompress(inner_bytes)
                    else:
                        orig = None
                    if orig is not None:
                        return orig, {'restored_from': 'fallback', 'metadata': meta}
                except Exception:
                    pass
        # otherwise reconstruct
        cols = list(meta['columns'].keys())
        nrows = meta['original_shape'][0]
        reconstructed = {col: [''] * nrows for col in cols}
        for col in cols:
            info = meta['columns'][col]
            enc = info['encoding']
            if enc == 'rle':
                payload = json.loads(Z.read(info['rle_file']).decode('utf-8'))
                vals = payload['values']; lengths = payload['lengths']
                out = rle_decode(vals, lengths)
                reconstructed[col] = [("" if v == '__nan__' else v) for v in out[:nrows]]
            elif enc == 'delta_numeric':
                payload = json.loads(Z.read(info['delta_file']).decode('utf-8'))
                out = delta_decode_numeric(payload)
                reconstructed[col] = out[:nrows]
            elif enc == 'delta_datetime':
                payload = json.loads(Z.read(info['deltadt_file']).decode('utf-8'))
                out = delta_decode_datetime(payload)
                reconstructed[col] = out[:nrows]
            elif enc == 'dict':
                payload = json.loads(Z.read(info['dict_file']).decode('utf-8'))
                out = dict_decode(payload)
                reconstructed[col] = out[:nrows]
            elif enc == 'huffman':
                meta_json = json.loads(Z.read(info['huff_file']).decode('utf-8'))
                bin_data = Z.read(info['huff_bin'])
                decoded_bytes = huffman_decode_bytes(meta_json['codes'], bin_data, int(meta_json['bit_length']))
                vals = json.loads(decoded_bytes.decode('utf-8'))
                reconstructed[col] = [("" if v == '__nan__' or v == '__empty__' else v) for v in vals[:nrows]]
            else:
                raw_text = Z.read(info['raw_file']).decode('utf-8').splitlines()
                out = []
                for r in raw_text:
                    if r == '__nan__':
                        out.append('')
                    elif r == '__empty__':
                        out.append('')
                    else:
                        out.append(r)
                reconstructed[col] = out[:nrows]
        # build CSV bytes
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(cols)
        for i in range(nrows):
            row = [reconstructed[c][i] for c in cols]
            writer.writerow(row)
        csv_bytes = buf.getvalue().encode('utf-8')
        return csv_bytes, {'restored_from': 'reconstructed', 'metadata': meta}

# ---------------------------------------------------------------------------
# Verify helper (reads from given file paths)
# ---------------------------------------------------------------------------

def verify_csv_roundtrip(orig_path: str, recon_path: str, mode: str = 'semantic') -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if mode == 'byte':
        try:
            same = False
            if os.path.getsize(orig_path) == os.path.getsize(recon_path):
                with open(orig_path, 'rb') as fa, open(recon_path, 'rb') as fb:
                    while True:
                        a = fa.read(65536); b = fb.read(65536)
                        if not a and not b:
                            same = True; break
                        if a != b:
                            same = False; break
            report['equal'] = same
        except Exception as e:
            report['error'] = str(e)
        report['mode'] = 'byte'
        return report

    try:
        a_df = pd.read_csv(orig_path, dtype=str)
    except Exception as e:
        report['error'] = f"Failed to read original CSV: {e}"; return report
    try:
        b_df = pd.read_csv(recon_path, dtype=str)
    except Exception as e:
        report['error'] = f"Failed to read reconstructed CSV: {e}"; return report

    a_df = a_df.fillna('')
    b_df = b_df.fillna('')

    nrows_a, ncols_a = a_df.shape
    nrows_b, ncols_b = b_df.shape
    report['nrows_a'] = nrows_a; report['ncols_a'] = ncols_a
    report['nrows_b'] = nrows_b; report['ncols_b'] = ncols_b

    nrows = min(nrows_a, nrows_b)
    ncols = min(ncols_a, ncols_b)

    mismatches = []
    for i in range(nrows):
        for j in range(ncols):
            va = a_df.iat[i, j]
            vb = b_df.iat[i, j]
            if (va or '') != (vb or ''):
                mismatches.append({'row': i, 'col': a_df.columns[j] if j < ncols_a else j, 'a': va, 'b': vb})
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break
    report['equal'] = (len(mismatches) == 0 and nrows_a == nrows_b and ncols_a == ncols_b)
    report['mismatch_count_shown'] = len(mismatches)
    report['mismatches_sample'] = mismatches
    report['reason'] = 'content-mismatch' if not report['equal'] else 'equal'
    return report

# ---------------------------------------------------------------------------
# Flask web UI
# ---------------------------------------------------------------------------
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compress', methods=['POST'])
def api_compress():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'no file uploaded'}), 400
    # write to temp file and call compress_inmemory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as t:
        path = t.name
        f.save(path)
    try:
        zip_bytes, metadata = compress_csv_inmemory_bytes(path)
        token = str(uuid.uuid4())
        _store[token] = {'bytes': zip_bytes, 'metadata': metadata, 'name': os.path.basename(path) + '.cscz'}
        return jsonify({'token': token, 'metadata': metadata})
    finally:
        os.unlink(path)

@app.route('/api/details/<token>')
def api_details(token):
    rec = _store.get(token)
    if not rec:
        return jsonify({'error': 'token not found'}), 404
    return jsonify(rec['metadata'])

@app.route('/download/<token>')
def download_token(token):
    rec = _store.get(token)
    if not rec:
        return 'not found', 404
    bio = io.BytesIO(rec['bytes'])
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name=rec.get('name', f'{token}.bin'))

@app.route('/api/decompress', methods=['POST'])
def api_decompress():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'no file uploaded'}), 400
    data = f.read()
    try:
        csv_bytes, info = decompress_cscz_bytes(data)
    except Exception as e:
        return jsonify({'error': f'decompression failed: {e}'}), 500
    token = str(uuid.uuid4())
    _store[token] = {'bytes': csv_bytes, 'metadata': info, 'name': 'decompressed.csv'}
    return jsonify({'token': token, 'metadata': info})

@app.route('/api/verify', methods=['POST'])
def api_verify():
    a = request.files.get('a')
    b = request.files.get('b')
    if not a or not b:
        return jsonify({'error': 'two files (a and b) required'}), 400
    # write to temp files
    ta = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tb = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    try:
        a.save(ta.name); b.save(tb.name)
        rep = verify_csv_roundtrip(ta.name, tb.name, mode='semantic')
        return jsonify(rep)
    finally:
        try:
            ta.close(); tb.close()
            os.unlink(ta.name); os.unlink(tb.name)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# CLI wrappers (only compress/decompress/verify as requested)
# ---------------------------------------------------------------------------

def compress_csv_cli(input_csv: str, output_cscz: str):
    zip_bytes, metadata = compress_csv_inmemory_bytes(input_csv, show_progress=False)
    with open(output_cscz, 'wb') as f:
        f.write(zip_bytes)
    print(f'Wrote {output_cscz} ({len(zip_bytes)} bytes)')


def decompress_cscz_cli(input_cscz: str, output_csv: str):
    with open(input_cscz, 'rb') as f:
        data = f.read()
    csv_bytes, info = decompress_cscz_bytes(data)
    with open(output_csv, 'wb') as fo:
        fo.write(csv_bytes)
    print(f'Wrote {output_csv} ({len(csv_bytes)} bytes)')


def verify_cli(orig_csv: str, recon_csv: str):
    rep = verify_csv_roundtrip(orig_csv, recon_csv, mode='semantic')
    print(json.dumps(rep, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSV compressor - web+cli (only compress/decompress/verify CLI).')
    sub = parser.add_subparsers(dest='cmd')
    p1 = sub.add_parser('compress')
    p1.add_argument('input_csv')
    p1.add_argument('output_cscz')
    p2 = sub.add_parser('decompress')
    p2.add_argument('input_cscz')
    p2.add_argument('output_csv')
    p3 = sub.add_parser('verify')
    p3.add_argument('orig_csv')
    p3.add_argument('recon_csv')
    p4 = sub.add_parser('runserver')
    args = parser.parse_args()
    if args.cmd == 'compress':
        compress_csv_cli(args.input_csv, args.output_cscz)
    elif args.cmd == 'decompress':
        decompress_cscz_cli(args.input_cscz, args.output_csv)
    elif args.cmd == 'verify':
        verify_cli(args.orig_csv, args.recon_csv)
    elif args.cmd == 'runserver':
        # run Flask dev server
        print('Starting web server at http://127.0.0.1:5000')
        app.run(debug=False)
    else:
        parser.print_help()
