#!/bin/bash
set -e

function help_text() {
    cat <<EOF
    Usage: $0 [ -i|--in-place-fixup FIX_FLAG ]
        --in-place-fixup	Fix imports (yes/no)
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    arg=$1
    case $arg in
    -h | --help)
        help_text
        ;;
    -i | --in-place-fixup)
        export FIX_FLAG="$2"
        shift
        shift
        ;;
    *)
        echo "ERROR: Unrecognised option: ${arg}"
        help_text
        exit 1
        ;;
    esac
done

echo "**********************"
if [ -z "${SRC_DIR}" ]; then
    SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
fi
echo "SRC_DIR: $SRC_DIR"
ls -lhFa $SRC_DIR


echo "**********************"
if [ -z "${FIX_FLAG}" ]; then
    FIX_FLAG="no"
fi

if [[ "$FIX_FLAG" == "yes" ]]; then
    ISORT_ARGS=""
    IBLACK_ARGS=""
else
    ISORT_ARGS="--diff --check-only"
    IBLACK_ARGS="--check"
fi

echo "Running isort"
isort \
    --multi-line=3 \
    --trailing-comma \
    --force-grid-wrap=0 \
    --use-parentheses \
    --line-width=120 \
    --section-default=THIRDPARTY \
    --project=realestate \
    --force-sort-within-sections \
    ${ISORT_ARGS} \
    ${SRC_DIR}

echo "**********************"
echo "Running black"
black ${IBLACK_ARGS} --line-length 120 ${SRC_DIR}

echo "**********************"
echo "Running pylint"
pylint -j 4 --rcfile=$SRC_DIR/.conda/.pylintrc $SRC_DIR/realestate $SRC_DIR/tests
