#!/usr/bin/env python3
"""
bundle_deps.py

Usage:
    bundle_deps.py <entry_file> [--root <project_root>]

Reads the entry Python file, recursively finds local dependencies,
concatenates their contents with "# FILE: <path>" headers, and
copies the result to the clipboard.
"""

import ast
import os
import sys
from collections import deque

try:
    import pyperclip
except ImportError:
    print("Please install pyperclip: pip install pyperclip", file=sys.stderr)
    sys.exit(1)

def find_local_deps(entry_path, root):
    seen = set()
    queue = deque([os.path.normpath(entry_path)])
    deps = []

    while queue:
        path = queue.popleft()
        if path in seen or not path.endswith('.py'):
            continue
        seen.add(path)
        deps.append(path)

        # Parse imports
        with open(path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module
            elif isinstance(node, ast.Import):
                # handle only top-level imports
                mod = node.names[0].name.split('.')[0]
            else:
                continue

            # Only include local modules (inside root)
            mod_path = os.path.join(root, *mod.split('.'))
            for candidate in (mod_path + '.py', os.path.join(mod_path, '__init__.py')):
                if os.path.isfile(candidate):
                    queue.append(os.path.normpath(candidate))
                    break

    return deps

def bundle_files(file_list, root):
    parts = []
    for path in file_list:
        rel = os.path.relpath(path, root)
        parts.append(f"# FILE: {rel}")
        parts.append(open(path, 'r', encoding='utf-8').read())
        parts.append("")  # blank line
    return "\n".join(parts)

def filter_files(file_list, root):
    """Interactively filter files before bundling.

    :param file_list: List of absolute file paths to bundle.
    :param root: Project root for relative paths.
    :returns: Filtered list of files.
    """

    choice = input("Filter out some files? [y/N]: ").strip().lower()
    if choice not in ("y", "yes"):
        return file_list

    selected = []
    for path in file_list:
        rel = os.path.relpath(path, root)
        ans = input(f"Include {rel}? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            selected.append(path)
    return selected

def main():
    import argparse
    p = argparse.ArgumentParser(description="Bundle local Python deps into clipboard.")
    p.add_argument("entry_file", help="Path to the entry Python script")
    p.add_argument("--root", default=".", help="Project root directory")
    p.add_argument("--output", "-o", help="Output file path (if not specified, copies to clipboard)")
    args = p.parse_args()

    entry = os.path.abspath(args.entry_file)
    root = os.path.abspath(args.root)
    files = find_local_deps(entry, root)

    files = filter_files(files, root)
    
    # Print all files being bundled
    print("Files being bundled:")
    for file_path in files:
        rel_path = os.path.relpath(file_path, root)
        print(f"  {rel_path}")
    print()
    
    bundle = bundle_files(files, root)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(bundle)
        print(f"Bundled {len(files)} files to {args.output}")
    else:
        pyperclip.copy(bundle)
        print(f"Bundled {len(files)} files into clipboard.")

if __name__ == "__main__":
    main()

