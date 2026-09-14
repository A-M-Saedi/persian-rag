# Third-Party Notices

This project depends on third-party packages. They are **not distributed with
this repository** — they are installed by the user from PyPI, and each is
governed by its own licence, not by this project's [LICENSE](LICENSE).

This file is provided for attribution and transparency. Versions and licences
below were read from installed package metadata and match the pins in
`requirements.txt` / `requirements-pdf.txt`. If you install a different set,
re-verify with `pip-licenses`.

## Core dependencies

All 109 core dependencies are permissive, except the two MPL-2.0 entries noted
below. The principal packages:

| Package | Version | License |
|---|---|---|
| `chromadb` | 1.5.9 | Apache-2.0 |
| `sentence-transformers` | 5.1.2 | Apache-2.0 |
| `transformers` | 4.57.6 | Apache-2.0 |
| `tokenizers` | 0.22.2 | Apache-2.0 |
| `huggingface_hub` | 0.36.2 | Apache-2.0 |
| `requests` | 2.32.5 | Apache-2.0 |
| `langchain` | 0.3.30 | MIT |
| `langchain-core` | 0.3.86 | MIT |
| `langchain-text-splitters` | 0.3.11 | MIT |
| `ollama` | 0.6.2 | MIT |
| `pydantic` | 2.13.5 | MIT |
| `typer` | 0.23.2 | MIT |
| `rich` | 15.0.0 | MIT |
| `beautifulsoup4` | 4.15.0 | MIT |
| `torch` | 2.8.0 | BSD-3-Clause |
| `numpy` | 2.0.2 | BSD-3-Clause |
| `scipy` | 1.13.1 | BSD-3-Clause |
| `scikit-learn` | 1.6.1 | BSD-3-Clause |
| `httpx` | 0.28.1 | BSD-3-Clause |
| `certifi` | 2026.7.22 | **MPL-2.0** |
| `tqdm` | 4.70.1 | **MPL-2.0 AND MIT** |

MPL-2.0 is file-level copyleft: its obligations attach to the MPL-licensed
files themselves, which this project neither modifies nor redistributes. It
imposes no conditions on this project's own source.

A scan of all 114 installed distributions found no copyleft licences other
than the two MPL-2.0 entries above and the optional PDF dependency below.

## Optional dependency — PDF support

| Package | Version | License |
|---|---|---|
| `PyMuPDF` | 1.26.5 | **Dual licensed — GNU AFFERO GPL 3.0 or Artifex Commercial License** |

PyMuPDF is **not installed by default**. It is required only to read `.pdf`
files, and it is imported lazily — no PyMuPDF code is loaded unless a PDF is
actually opened. That boundary is enforced by a test:
`tests/test_sources.py::test_pymupdf_is_only_imported_when_a_pdf_is_actually_read`.

If you install `requirements-pdf.txt`, your use of PyMuPDF is governed by
AGPL-3.0 or by a commercial licence obtained from Artifex. That choice is
yours and is independent of this project's licence. If you intend to run this
as a network service with PDF support enabled, read AGPL-3.0 §13 carefully, or
contact Artifex for commercial terms.

- PyMuPDF: https://github.com/pymupdf/PyMuPDF
- Artifex commercial licensing: https://artifex.com/licensing/
