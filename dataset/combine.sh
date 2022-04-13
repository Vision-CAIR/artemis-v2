#!/bin/bash

python combine.py offical_data/
python ../neural_speaker/sat/scripts/preprocess_artemis_data.py -save-out-dir full_combined/train/ -raw-artemis-data-csv  official_data/combined_artemis.csv --preprocess-for-deep-nets True
python ../neural_speaker/sat/scripts/preprocess_artemis_data.py -save-out-dir full_combined/analysis/ -raw-artemis-data-csv  official_data/combined_artemis.csv
