# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the gfx936 W8A8 input-scale-first WGM8 code object.

The matching assembly generator is not distributed with this CO. This tool is
intentionally version-locked: only existing FP32 MUL register fields change.
Matrix instructions, register allocation, branches, memory ops, fixed selector
bits and ELF metadata remain byte-identical. Requires DTK llvm-objdump/llvm-mc.
Run from any directory, or use --check to verify the committed object.
"""

import argparse
import hashlib
import re
import shutil
import struct
import subprocess
from pathlib import Path

SOURCE_SHA = "7811b52cb012f7a209189722bffbfdf1a12d1b60f28f760b879ad090c4b2bbf6"
RESULT_SHA = "e3d9adfbbc518be344ac38e266fbfd9ed237a53d35653ac990588ec5d556f19b"
STEM = "gemm_w8a8_smooth_256x256x128_WGM8_TN_BF16"


def tool(name):
    path = shutil.which(name)
    if path:
        return path
    for directory in ("/opt/dtk/aillvm/bin", "/opt/dtk/llvm/bin", "/opt/dtk/bin"):
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"DTK tool not found: {name}")


def text_section(data):
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    assert header[0][:6] == b"\x7fELF\x02\x01" and header[2] == 224
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * header[11])
                for i in range(header[12])]
    names = sections[header[13]]
    strings = data[names[4]:names[4] + names[5]]
    for section in sections:
        if strings[section[0]:].split(b"\0", 1)[0] == b".text":
            return section[3], section[4]
    raise RuntimeError("missing ELF .text section")


def disassemble(path):
    text = subprocess.check_output(
        [tool("llvm-objdump"), "-d", "--mcpu=gfx936", str(path)], text=True)
    instructions = []
    for line in text.splitlines():
        match = re.match(r"\s*(.*?)\s*//\s*([0-9a-fA-F]+):\s*((?:[0-9a-fA-F]{8}\s*)+)$", line)
        if match:
            asm = match[1].strip()
            op, *rest = asm.split(None, 1)
            operands = re.split(r",(?![^\[]*\])", rest[0]) if rest else []
            raw = b"".join(int(word, 16).to_bytes(4, "little")
                           for word in match[3].split())
            instructions.append((int(match[2], 16), op,
                                 [x.strip() for x in operands], asm, raw))
    return instructions


def assemble(lines):
    text = subprocess.check_output(
        [tool("llvm-mc"), "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx936", "-show-encoding"],
        input="\n".join(lines) + "\n", text=True)
    encodings = re.findall(r"encoding:\s*\[([^\]]+)\]", text)
    assert len(encodings) == len(lines)
    return [bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", row))
            for row in encodings]


def generate(source):
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCE_SHA, "unsupported source CO version"
    address, offset = text_section(data)
    instructions = disassemble(source)
    multiplications = [x for x in instructions if x[1] == "v_pk_mul_f32"]
    assert len(multiplications) == 384
    pairs = []
    used = set()
    for i, first in enumerate(instructions):
        if first[1] != "v_pk_mul_f32" or first[0] in used:
            continue
        scratch, scale_weight, scale_input = first[2][:3]
        scale_input = scale_input.split()[0]
        assert scratch not in (scale_weight, scale_input)
        # Eight scale products can be interleaved before their accumulator
        # products. Locate the first actual use, rather than adjacent MULs.
        for second in instructions[i + 1:]:
            if scratch in second[2][1:]:
                break
            assert not second[2] or second[2][0] != scratch
        else:
            raise RuntimeError("scale-product register has no consumer")
        accumulator, temp, old_acc = second[2][:3]
        old_acc = old_acc.split()[0]
        assert second[1] == "v_pk_mul_f32"
        assert temp == scratch and old_acc == accumulator
        assert len({scratch, accumulator, scale_weight, scale_input}) == 4
        assert second[0] not in used
        used.update((first[0], second[0]))
        # The source SHA locks this dataflow and the original control flow.
        pairs.append((first, f"v_pk_mul_f32 {scratch}, {accumulator}, {scale_input}"))
        pairs.append((second, f"v_pk_mul_f32 {accumulator}, {scratch}, {scale_weight}"))
    assert len(used) == 384
    pairs.sort(key=lambda item: item[0][0])
    old_asm = [re.sub(r" op_sel_hi:\[[^]]+\]", "", old[3]) for old, _ in pairs]
    roundtrip = assemble(old_asm)
    selectors = {bytes(a ^ b for a, b in zip(encoded, old[4]))
                 for encoded, (old, _) in zip(roundtrip, pairs)}
    assert len(selectors) == 1
    selector = selectors.pop()
    assert len(selector) == 8 and int.from_bytes(selector[:4], "little") == 0
    assert int.from_bytes(selector[4:], "little") & ((1 << 18) - 1) == 0
    output = bytearray(data)
    allowed = set()
    for encoded, (old, _) in zip(assemble([new for _, new in pairs]), pairs):
        raw = bytes(a ^ b for a, b in zip(encoded, selector))
        assert len(raw) == len(old[4]) == 8
        # Only destination (word0[7:0]) and source-register fields (word1[17:0]).
        assert (int.from_bytes(raw[:4], "little") ^ int.from_bytes(old[4][:4], "little")) & ~255 == 0
        assert (int.from_bytes(raw[4:], "little") ^ int.from_bytes(old[4][4:], "little")) & ~((1 << 18) - 1) == 0
        start = offset + old[0] - address
        assert data[start:start + 8] == old[4]
        output[start:start + 8] = raw
        allowed.update(range(start, start + 8))
    changed = {i for i, (old, new) in enumerate(zip(data, output)) if old != new}
    assert changed and changed <= allowed and len(output) == len(data)
    assert hashlib.sha256(output).hexdigest() == RESULT_SHA, "unexpected generated CO"
    return output


def main():
    root = Path(__file__).resolve().parents[1] / "hsa/gfx936"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / (STEM + ".co"))
    parser.add_argument("--output", type=Path, default=root / (STEM + "_INPUT_SCALE_FIRST.co"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = generate(args.source)
    if args.check:
        assert args.output.read_bytes() == generated, "committed object differs"
    else:
        args.output.write_bytes(generated)
    print(f"Verified {len(generated)} bytes, SHA256 {RESULT_SHA}")


if __name__ == "__main__":
    main()
