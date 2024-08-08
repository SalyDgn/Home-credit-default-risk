#! /bin/bash

EXECUTION_DATE=$(date "+%Y%m%d-%H%M")
YEAR=$(date "+%Y")
MONTH=$(date "+%m")

PROJECT_DIR=$PWD
LOGS_DIR=${PROJECT_DIR}/logs/${YEAR}/${MONTH}
echo $PROJECT_DIR
mkdir -p ${LOGS_DIR}

echo "================================== Start credit scoring training ====================================="

# Define an array of notebooks
notebooks=(
    "scoring_01_data_exploration.ipynb"
    "scoring_02_prepocessing.ipynb"
    "scoring_05_feature_selection_and_modeling.ipynb"
)

# Loop through each notebook and execute with Papermill
for notebook in "${notebooks[@]}"; do
    echo "Starting notebook: ${notebook}"
    python -m papermill "${PROJECT_DIR}/notebooks/${notebook}" \
    "${LOGS_DIR}/${EXECUTION_DATE}-${notebook%.ipynb}-artifact.ipynb" \
    --report-mode --log-output --no-progress-bar

    if [ $? != 0 ]; then
        echo "ERROR: failure during ${notebook} training!"
        exit 1
    fi
    echo "SUCCESS: Done ${notebook} training"
done

echo "================================ SUCCESS: All notebooks executed ==================================="
