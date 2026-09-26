#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only miner for Ryujinx's CPU tests (src/Ryujinx.Tests/Cpu, MIT).

usage: ryujinx_extract.py CPU_DIR {arrays|tests|coverage} [FILE...]
       ryujinx_extract.py CPU_DIR cpp File.cs:_Array_ ...   (C++ arrays for a port)
  arrays   : every `private static uint[] _Name_()` opcode list, with its comments
  tests    : every test method: name, value sources, fpcr handling, compare arguments
  coverage : per test method, the instruction names its opcode list covers
"""
import os
import re
import sys

ARRAY = re.compile(r'private static (?:uint|ulong)\[\] (_\w+_)\(\)\s*\{\s*return\s*\[(.*?)\];', re.S)
ENUM = re.compile(r'private static IEnumerable<ulong> (_\w+_)\(\)')
TEST = re.compile(r'public void (\w+)\((.*?)\)\s*\{(.*?)\n        \}', re.S)
ENTRY = re.compile(r'(0x[0-9A-Fa-f]+)u[l]?\s*,?\s*(?://\s*(.*))?')


def arrays(text):
    for m in ARRAY.finditer(text):
        items = []
        for line in m.group(2).splitlines():
            values = re.findall(r'0x[0-9A-Fa-f]+', line.split('//')[0])
            comment = line.split('//', 1)[1].strip() if '//' in line else ''
            for v in values:
                items.append((int(v, 16), comment))
        yield m.group(1), items


YIELDS = re.compile(r'private static IEnumerable<ulong> (_\w+_)\(\)\s*\{(.*?)\n        \}', re.S)


def yield_sets(text):
    """The fixed `yield return` values of the IEnumerable value sources (random tails dropped)."""
    for m in YIELDS.finditer(text):
        items = []
        for line in m.group(2).splitlines():
            y = re.search(r'yield return (0x[0-9A-Fa-f]+)u?l?;\s*(?://\s*(.*))?', line)
            if y:
                items.append((int(y.group(1), 16), (y.group(2) or '').strip()))
        yield m.group(1), items


def tests(text):
    for m in TEST.finditer(text):
        params = re.findall(r'\[(ValueSource\(nameof\((\w+)\)\)|Values[^\]]*|Random[^\]]*)\]\s*(\w+)\s+(\w+)', m.group(2))
        body = m.group(3)
        fpcr = re.findall(r'fpcr[^;]*;', body)
        compare = re.findall(r'CompareAgainstUnicorn\((.*?)\);', body)
        yield m.group(1), params, fpcr, compare, body


def emit_cpp(cpu_dir, specs):
    """specs: 'File.cs:_Array_' ... -> C++ constexpr arrays (kName), comments kept."""
    out = []
    for spec in specs:
        fname, arr = spec.split(':')
        text = open(os.path.join(cpu_dir, fname), encoding='utf-8-sig').read()
        table = dict(arrays(text))
        table.update(yield_sets(text))
        items = table[arr]
        decl = re.search(r'private static (\S+) ' + re.escape(arr) + r'\(\)', text).group(1)
        ctype = 'u32' if decl == 'uint[]' else 'u64'
        width = 8 if ctype == 'u32' else 16
        out.append(f'// {fname} {arr}')
        out.append(f'constexpr std::array<{ctype}, {len(items)}> k{arr.strip("_")}{{{{')
        for v, comment in items:
            out.append(f'    0x{v:0{width}X},{("  // " + comment) if comment else ""}')
        out.append('}};')
    return '\n'.join(out)


def main():
    cpu_dir, mode, *files = sys.argv[1:]
    if mode == 'cpp':
        print(emit_cpp(cpu_dir, files))
        return 0
    files = files or sorted(f for f in os.listdir(cpu_dir) if f.endswith('.cs'))
    for name in files:
        text = open(os.path.join(cpu_dir, name), encoding='utf-8-sig').read()
        if mode == 'arrays':
            for arr, items in arrays(text):
                print(f'{name}:{arr} ({len(items)})')
                for value, comment in items:
                    print(f'  0x{value:08X}  {comment}')
        elif mode in ('tests', 'coverage'):
            table = dict(arrays(text))
            for test, params, fpcr, compare, _ in tests(text):
                sources = [p[1] for p in params if p[1]]
                if mode == 'tests':
                    print(f'{name}:{test} sources={sources} fpcr={fpcr} compare={compare}')
                else:
                    names = []
                    for s in sources:
                        for _, comment in table.get(s, []):
                            names.append(comment.split(' ')[0] if comment else '?')
                    print(f'{name}:{test}: {" ".join(names)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
