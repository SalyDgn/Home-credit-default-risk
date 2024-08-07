#! /bin/bash

EXECUTION_DATE=$(date "+%Y%m%d-%H%M")
YEAR=$(date "+%Y")
MONTH=$(date "+%m")

PROJECT_DIR=$PWD
LOGS_DIR=${PROJECT_DIR}/logs/${YEAR}/${MONTH}

mkdir -p ${LOGS_DIR}

echo "================================== Start credit scoring training ====================================="

# Define an array of notebooks
notebooks=(
    "scoring_01_data_exploration.ipynp"
    "scoring_02_preprocessing.ipynb"
    "scoring_04_feature_selection_and_modeling.ipynb"
)

# Loop through each notebook and execute with Papermill
for notebook in "${notebooks[@]}"; then
    echo "Starting notebook: ${notebook}"
    papermill "notebooks/${notebook}" \
    "${LOGS_DIR}/${EXECUTION_DATE}-${notebook%.ipynb}-artifact.ipynb" \
    -k python39 --report-mode --log-output --no-progress-bar

    if [ $? != 0 ]; then
        echo "ERROR: failure during ${notebook} training!"
        exit 1
    fi
    echo "SUCCESS: Done ${notebook} training"
done

echo "================================ SUCCESS: All notebooks executed ==================================="
