#!/usr/bin/env python
"""
This script contains the utility functions required to work with Xournal++ files (.xopp).
Xournal++ files are gzip compressed XML files.

Compression:
I wanted to use Xournal to create hand-written explanation slides for graphitout project.
Since the xopp files were binary, keeping them in the repo was a problem. I unzipped the
files to keep it in the original text (xml) format. But the file was too big. I looked
inside the file and saw that it was keeping raw x,y data. Hence I decided to remove some
precision and code the differences to reduce the size.
"""

import gzip
import lzma
import argparse

from typing import Callable

from lxml import etree as ET
from lxml.etree import _Element as Element

from common_funcs import xprint
from custom_base64 import encode_num2bytes, decode_bytes2num

# New tag to be used for compressed <stroke> elements
compressed_stroke_tag = "strk"

# Only compress if string size if above this
min_stroke_str_size = 20

# Scaling to be applied to both x and y values
xyscale = 32

# Char to split between x values and y values
xydelim = "|"

# Should be added at the top
first_line = b"""<?xml version="1.0" standalone="no"?>\n"""

perform_decompress_check = True
allowed_error_thr = 1 / xyscale / 2


def copy_all_attrib(src, dst):
    for name, value in src.attrib.items():
        dst.set(name, value)


def encode_float_values(values: list[float]) -> str:
    int_values = [int(round(val * xyscale)) for val in values]
    prev = 0
    result = []
    for ii in range(len(int_values)):
        current = int_values[ii]
        diff = current - prev
        converted = encode_num2bytes(diff).decode()
        result.append(converted)
        prev = current
    ret = "".join(result)
    return ret


def decode_float_values(text: str) -> list[float]:
    text_with_space = text.replace("+", " +").replace("-", " -")
    splits = text_with_space.encode().split()
    result = []
    prev = 0
    for val in splits:
        decoded_diff = decode_bytes2num(val)
        current = prev + decoded_diff
        result.append(current * 1.0 / xyscale)
        prev = current
    return result


def max_abs_diff(seq1: list[float], seq2: list[float]) -> float:
    return max(abs(p - q) for p, q in zip(seq1, seq2))


def compress_stroke_text(text: str) -> str | None:
    splits = text.split()
    if len(splits) % 2 != 0:
        raise RuntimeError(f"Expects count to be even. Got {len(splits)}")
    if len(splits) < 4:
        return None
    numbers = [float(val) for val in splits]
    xvalues = numbers[::2]
    yvalues = numbers[1::2]
    xvalues_encoded = encode_float_values(xvalues)
    yvalues_encoded = encode_float_values(yvalues)
    result = "".join([xvalues_encoded, xydelim, yvalues_encoded])

    if perform_decompress_check:
        # Ignore this part. Only for checks.
        check_values = [("X", xvalues, xvalues_encoded), ("Y", yvalues, yvalues_encoded)]
        for channel, real_values, encoded_values in check_values:
            decoded_values = decode_float_values(encoded_values)
            if len(decoded_values) != len(real_values):
                raise RuntimeError(
                    "Encode/decode size mistmatch: "
                    f"{len(decoded_values)} != {len(real_values)}"
                )
            max_error = max_abs_diff(decoded_values, real_values)
            if max_error > allowed_error_thr:
                raise RuntimeError(f"Max error {max_error} > {allowed_error_thr}")

    return result


def decompress_stroke_text(text: str) -> str:
    text = text.strip()
    xyparts = text.split(xydelim)
    if len(xyparts) != 2:
        raise RuntimeError(f"Expects text to have two parts split by {xydelim}")
    xpart, ypart = xyparts
    xvalues = decode_float_values(xpart)
    yvalues = decode_float_values(ypart)
    if len(xvalues) != len(yvalues):
        raise RuntimeError(
            "Length mistmatch between x and y values: " f"{len(xvalues)} != {len(yvalues)}"
        )
    parts = []
    for xv, yv in zip(xvalues, yvalues):
        parts.append(f"{xv:.6f} {yv:.6f}")
    return " ".join(parts)


def find_all_strokes_compress_replace(root: Element) -> Element:
    prev_attribs = None
    total_before = 0
    total_after = 0
    for stroke_elem in root.findall(".//stroke"):
        parent = stroke_elem.getparent()
        if parent is None:
            continue
        if stroke_elem.get("tool") != "pen":
            # we only want to compress the pen strokes
            continue
        if stroke_elem.text is None:
            continue
        text = stroke_elem.text.strip()
        if len(text) < min_stroke_str_size:
            # too small to process
            continue
        conv_text = compress_stroke_text(text)
        if conv_text is None:
            continue

        attribs = sorted([(key, value) for key, value in stroke_elem.attrib.items()])

        new_elem = ET.Element(compressed_stroke_tag)
        # Use attributes from previous compressed stroke
        if prev_attribs == attribs:
            new_elem.set("prev", "1")
        else:
            copy_all_attrib(stroke_elem, new_elem)
            prev_attribs = attribs
        new_elem.text = conv_text
        new_elem.tail = stroke_elem.tail
        parent.replace(stroke_elem, new_elem)
        total_before += len(text)
        total_after += len(conv_text)

    ratio = total_after / total_before * 100
    xprint(f"Stroke text compression ratio: {ratio:.2f}")
    return root


def decompress_all_strokes_replace(root: Element) -> Element:
    prev_attribs = None
    for compressed_elem in root.findall(f".//{compressed_stroke_tag}"):
        parent = compressed_elem.getparent()
        if parent is None:
            continue
        text = compressed_elem.text
        if text is None:
            continue
        text = text.strip()
        attribs = [(key, value) for key, value in compressed_elem.attrib.items()]
        if compressed_elem.get("prev", "0") == "1":
            if prev_attribs is None:
                raise RuntimeError("No previous attribute set to use")
            attribs = prev_attribs
        else:
            prev_attribs = attribs
        new_elem = ET.Element("stroke")
        for key, value in attribs:
            new_elem.set(key, value)
        new_elem.text = decompress_stroke_text(text)
        new_elem.tail = compressed_elem.tail
        parent.replace(compressed_elem, new_elem)
    return root


def remove_preview_element_from_root(root: Element) -> Element:
    for elem in root.findall("./preview"):
        xprint("Removing <preview>")
        root.remove(elem)
    return root


def get_opener_for_file(fname: str) -> Callable:
    opener: Callable = open
    if fname.endswith((".xopp", ".gz")):
        opener = gzip.open
    elif fname.endswith((".xz", ".lzma")):
        opener = lzma.open
    return opener


def read_file_bytes(fname: str) -> bytes:
    opener = get_opener_for_file(fname)
    with opener(fname, "rb") as reader:
        return reader.read()


def write_output(data: bytes, outfile: str) -> None:
    opener = get_opener_for_file(outfile)
    with opener(outfile, "wb") as writer:
        writer.write(data)


def process_compress_decompress(kind: str, infile: str, outfile: str) -> None:
    input_data = read_file_bytes(infile)
    root = ET.fromstring(input_data)
    if kind == "compress":
        remove_preview_element_from_root(root)
        find_all_strokes_compress_replace(root)
    elif kind == "decompress":
        decompress_all_strokes_replace(root)
    else:
        raise RuntimeError(f"Buggy! unknown {kind=}")
    processed_doc = ET.tostring(root)
    if kind == "decompress":
        processed_doc = first_line + processed_doc
    write_output(processed_doc, outfile)
    xprint("Done")


def handle_compress(args: argparse.Namespace) -> None:
    process_compress_decompress("compress", args.input, args.output)


def handle_decompress(args: argparse.Namespace) -> None:
    process_compress_decompress("decompress", args.input, args.output)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("compress", help="compress a .xopp file to .zml.lzma")
    subparser.set_defaults(func=handle_compress)
    opt = subparser.add_argument
    opt("-i", "--input", required=True, help="input .xopp file")
    opt("-o", "--output", required=True, help="output file (.xml or .xml.lzma)")

    subparser = subparsers.add_parser(
        "decompress", help="decompress a .xopp file to .zml.lzma"
    )
    subparser.set_defaults(func=handle_decompress)
    opt = subparser.add_argument
    opt("-i", "--input", required=True, help="input file (.xml or  (.xml or .xml.xz)")
    opt("-o", "--output", required=True, help="output file (.xopp or .xml)")

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
