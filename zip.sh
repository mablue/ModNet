#!/bin/bash
# ============================================================
#  make_arxiv_zip.sh
#  Build a clean zip file for arXiv submission.
#
#  Usage:
#    chmod +x make_arxiv_zip.sh
#    ./make_arxiv_zip.sh
#
#  Output:
#    arxiv_submission.zip
# ============================================================

set -e

ZIP_NAME="ModNet"
BUILD_DIR="arxiv_build"

# ------------------------------------------------------------
#  1. Clean previous build
# ------------------------------------------------------------
if [ -d "$BUILD_DIR" ]; then
  echo "Removing old build directory..."
  rm -rf "$BUILD_DIR"
fi
if [ -f "${ZIP_NAME}.zip" ]; then
  echo "Removing old zip..."
  rm -f "${ZIP_NAME}.zip"
fi

mkdir -p "$BUILD_DIR"

# ------------------------------------------------------------
#  2. Copy main files
# ------------------------------------------------------------
echo "Copying main files..."

if [ ! -f "ModNet.tex" ]; then
  echo "Error: ModNet.tex not found."
  exit 1
fi

cp ModNet.tex "$BUILD_DIR/"

if [ -f "references.bib" ]; then
  cp references.bib "$BUILD_DIR/"
else
  echo "Warning: references.bib not found."
fi

if [ -f "ModNet.bbl" ]; then
  cp ModNet.bbl "$BUILD_DIR/"
  echo "  Included ModNet.bbl (fallback for arXiv)"
else
  echo "  Warning: ModNet.bbl not found. arXiv will need to run BibTeX."
fi

# ------------------------------------------------------------
#  3. Copy chapters
# ------------------------------------------------------------
if [ -d "chapters" ]; then
  echo "Copying chapters..."
  mkdir -p "$BUILD_DIR/chapters"
  cp chapters/*.tex "$BUILD_DIR/chapters/" 2>/dev/null || true
else
  echo "Warning: chapters/ directory not found."
fi

# ------------------------------------------------------------
#  4. Copy figure PDFs from results/
#  Only .pdf files are needed; skip .svg, .csv, .json, .txt
# ------------------------------------------------------------
if [ -d "results" ]; then
  echo "Copying figure PDFs from results/..."
  mkdir -p "$BUILD_DIR/results"

  # Copy directory structure preserving relative paths, but only PDFs
  find results -type f -name "*.pdf" | while read -r f; do
    rel_path="${f#results/}"
    dest_dir="$BUILD_DIR/results/$(dirname "$rel_path")"
    mkdir -p "$dest_dir"
    cp "$f" "$dest_dir/"
  done

  # Count what we copied
  n_pdf=$(find "$BUILD_DIR/results" -name "*.pdf" | wc -l)
  echo "  Copied $n_pdf PDF files."
else
  echo "Warning: results/ directory not found. No figures included."
fi

# ------------------------------------------------------------
#  5. Copy any additional top-level figure directory
# ------------------------------------------------------------
if [ -d "figures" ]; then
  echo "Copying figures/..."
  mkdir -p "$BUILD_DIR/figures"
  find figures -type f \( -name "*.pdf" -o -name "*.png" -o -name "*.jpg" \) | while read -r f; do
    rel_path="${f#figures/}"
    dest_dir="$BUILD_DIR/figures/$(dirname "$rel_path")"
    mkdir -p "$dest_dir"
    cp "$f" "$dest_dir/"
  done
fi

# ------------------------------------------------------------
#  6. Clean up build directory
#  Remove aux files that may have been accidentally copied
# ------------------------------------------------------------
echo "Cleaning build directory..."
find "$BUILD_DIR" -type f \( \
  -name "*.aux" -o \
  -name "*.log" -o \
  -name "*.out" -o \
  -name "*.toc" -o \
  -name "*.lof" -o \
  -name "*.lot" -o \
  -name "*.blg" -o \
  -name "*.synctex.gz" -o \
  -name "*.fls" -o \
  -name "*.fdb_latexmk" \
\) -delete

# ------------------------------------------------------------
#  7. Create the zip
# ------------------------------------------------------------
echo "Creating zip archive..."
cd "$BUILD_DIR"
zip -r "../${ZIP_NAME}.zip" . -x ".*" > /dev/null
cd ..

# ------------------------------------------------------------
#  8. Report
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
echo "  Output:     ${ZIP_NAME}.zip"
echo "  Size:       $(du -h ${ZIP_NAME}.zip | cut -f1)"
echo "  Files:"
zipinfo -1 "${ZIP_NAME}.zip" | head -40
echo ""
echo "  To clean up the build directory:"
echo "    rm -rf ${BUILD_DIR}"
echo "============================================================"
