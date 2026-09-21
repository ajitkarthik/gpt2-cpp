#!/usr/bin/env python3
"""Write gpt2_tokenizer.bin in llm.c's tokenizer format.

Standalone port of llm.c's write_tokenizer() with no torch dependency.

Format:
    256 x int32 little-endian header:
        [0] magic 20240328
        [1] version 2 (v2 adds the EOT token id at [3])
        [2] vocab size n
        [3] EOT token id
    then, for each token id 0..n-1:
        1 byte length, followed by that many raw bytes

Usage:  python3 write_tokenizer.py [output_path]
"""
import struct
import sys

import tiktoken


def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = [0] * 256
    header[0] = 20240328  # magic
    header[1] = 2  # version 2 includes the EOT token
    header[2] = n  # number of tokens
    header[3] = enc.eot_token  # EOT token id
    with open(filename, "wb") as f:
        f.write(struct.pack("<256i", *header))
        for i in range(n):
            b = enc.decode_bytes([i])
            assert len(b) < 256, f"token {i} length exceeds 255: {len(b)}"
            f.write(struct.pack("<B", len(b)))
            f.write(b)
    print(f"wrote {filename}: {n} tokens, eot={enc.eot_token}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "gpt2_tokenizer.bin"
    write_tokenizer(tiktoken.get_encoding("gpt2"), out)
