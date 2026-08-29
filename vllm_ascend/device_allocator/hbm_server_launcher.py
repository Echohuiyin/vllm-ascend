# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

_SERVER_BINARY = "vllm_ascend_hbm_server"


def native_server_path() -> Path:
    package_dir = Path(__file__).resolve().parents[1]
    candidates = (package_dir / _SERVER_BINARY, package_dir.parent / _SERVER_BINARY)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    paths = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"The native Famem HBM server was not installed. Expected an executable at one of: {paths}.")


def main(argv: Sequence[str] | None = None) -> int:
    server = native_server_path()
    arguments = list(sys.argv[1:] if argv is None else argv)
    os.execv(server, [str(server), *arguments])
    return 1  # pragma: no cover - os.execv only returns on an interpreter defect.


if __name__ == "__main__":
    raise SystemExit(main())
