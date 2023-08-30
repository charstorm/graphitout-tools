#!/usr/bin/env python
"""
This script handles compression of SVG

Xournal produces SVGs that are too big. This script compresses such SVGs by using relative
values in the "d" field of <path> elements.
"""

import argparse
from pathlib import Path

from lxml import etree as ET
from lxml.etree import _Element as Element

from scour.scour import scourString as scour_string
from scour.scour import sanitizeOptions as scour_sanitize_options
from scour.scour import parse_args as scour_parse_args

from common_funcs import xprint

xyscale = 10
inv_scaling = 1 / xyscale
input_suffix = ".svg"
output_suffix = "_opt.svg"

scour_options = scour_sanitize_options(
    scour_parse_args(["--indent=none", "--create-groups"])
)


def scale_stroke_width(elem: Element) -> None:
    style = elem.get("style", "")
    splits = style.split(";")
    parts = []
    for split in splits:
        if not split.startswith("stroke-width"):
            parts.append(f"{split};")
            continue
        key, value = split.split(":")
        value = str(float(value) * xyscale)
        parts.append(f"{key}:{value};")
    elem.set("style", "".join(parts))


def set_xy_scaling_for_path_elem(elem: Element) -> None:
    transform = elem.get("transform", "")
    if transform:
        transform += ";"
    transform += f"scale({inv_scaling})"
    elem.set("transform", transform)


def scale_and_int(val: float) -> int:
    return int(round(val * xyscale))


def encode_diff_with_scale(start: float, values: list[float]) -> list[int]:
    prev = scale_and_int(start)
    result = []
    for val_f in values:
        val = scale_and_int(val_f)
        diff = val - prev
        result.append(diff)
        prev = val
    return result


def compress_dstring(dstring: str) -> str | None:
    """
    We only compress SVGs produced by xournal, with a fixed pattern for dstring.

    If should be of the form: M num num L num num L num num ...
    """
    splits = dstring.split()
    if len(splits) % 3 != 0:
        return None
    commands = splits[::3]
    if len(commands) < 3:
        return None
    first_command = commands[0]
    rest_commands = list(set(commands[1:]))
    if first_command != "M":
        return None
    if len(rest_commands) != 1:
        return None
    if rest_commands[0] != "L":
        return None
    # We want to keep the first M num num separate
    first_3 = splits[:3]
    splits = splits[3:]
    x0 = float(first_3[1])
    y0 = float(first_3[2])
    xvalues = [float(el) for el in splits[1::3]]
    yvalues = [float(el) for el in splits[2::3]]
    xdiff = encode_diff_with_scale(x0, xvalues)
    ydiff = encode_diff_with_scale(y0, yvalues)
    assert len(xdiff) == len(ydiff)
    combined = [""] * (len(xdiff) * 2)
    combined[::2] = [f"{el:+d}" for el in xdiff]
    combined[1::2] = [f"{el:+d}" for el in ydiff]
    first_3[1] = str(x0 * xyscale)
    first_3[2] = str(y0 * xyscale)
    result = " ".join(first_3) + "l" + "".join(combined)
    return result


def process_path_elem(elem: Element) -> None:
    dstring = elem.get("d", "")
    if not dstring:
        return None
    compressed = compress_dstring(dstring)
    if compressed is None:
        return None
    elem.set("d", compressed)
    set_xy_scaling_for_path_elem(elem)
    scale_stroke_width(elem)


def find_all_path_compress_replace(root: Element) -> Element:
    for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
        process_path_elem(path)
    return root


def select_input_files(input_dir: str) -> list[str]:
    result = []
    for path in Path(input_dir).glob(f"*{input_suffix}"):
        path_str = str(path)
        if path_str.endswith(output_suffix):
            # we don't want output files from last execution to be selected
            continue
        result.append(path_str)
    if not result:
        raise ValueError(f"Found no svg files in {input_dir}")
    else:
        xprint(f"Number of SVG files found: {len(result)}")
    return result


def compress_file(input_file: str, output_file: str) -> None:
    with open(input_file, "rb") as reader:
        input_data = reader.read()

    root = ET.fromstring(input_data)
    root = find_all_path_compress_replace(root)
    output_data = ET.tostring(root).decode()
    output_data = scour_string(output_data, scour_options)

    with open(output_file, "w") as writer:
        writer.write(output_data)


def handle_compress(args: argparse.Namespace) -> None:
    input_files = select_input_files(args.input_dir)
    for input_file in input_files:
        try:
            output_file = input_file.replace(input_suffix, output_suffix)
            compress_file(input_file, output_file)
            xprint(f"Processed {input_file} => {output_file}")
        except Exception:
            xprint(f"Error while processing file '{input_file}'")
            raise


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("compress", help="compress svg file(s)")
    subparser.set_defaults(func=handle_compress)
    opt = subparser.add_argument
    opt("-i", "--input_dir", required=True, help="input directory to look for .svg files")

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
