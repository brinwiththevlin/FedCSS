#!/usr/bin/env bash

set -e

DATASETS=("mnist")
LABEL_TAMPERING=("none" "random" "reverse" "zero")
IID=(-1 0.7)
WEIGHT_TAMPERING=("large_neg" "reverse" "random")
USERS=("25 0.4" "50 0.2" "100 0.1")

for dataset in "${DATASETS[@]}"; do
    for iid in "${IID[@]}"; do
        for label_tamp in "${LABEL_TAMPERING[@]}"; do
            for user in "${USERS[@]}"; do
                read -r num_users frac <<<"$user"
                if [ -f "$RESULTS_FILE" ]; then
                    echo "Results already exist at $RESULTS_FILE. Skipping..."
                else
                    echo "Running main.py with --dataset=$dataset --label_tampering=$label_tamp --weight_tampering=none"
                    python3 main.py \
                        --dataset_name $dataset \
                        --select_client \
                        --client_num $num_users \
                        --select_ratio $frac \
                        --corrupt_num 10 \
                        --train_type=meta \
                        --corruption_prob=1 \
                        --epochs 100 \
                        --batch_size 100 \
                        --corruption_type $label_tamp \
                        --weight_tampering "none"
                fi
            done
        done
        for weight_tamp in "${WEIGHT_TAMPERING[@]}"; do
            for user in "${USERS[@]}"; do
                RESULTS_FILE="${dataset}_${user}_cnn_1_ltnone_wt${weight_tamp}/results.csv"
                if [ -f "$RESULTS_FILE" ]; then
                    echo "Results already exist at $RESULTS_FILE. Skipping..."
                else
                    echo "Running main.py with --dataset=$dataset --label_tampering=none --weight_tampering=$weight_tamp"
                    python3 main.py \
                        --dataset_name $dataset \
                        --select_client \
                        --client_num $num_users \
                        --select_ratio $frac \
                        --corrupt_num 10 \
                        --train_type=meta \
                        --corruption_prob=1 \
                        --epochs 100 \
                        --batch_size 100 \
                        --corruption_type "none" \
                        --weight_tampering $weight_tamp
                fi
            done
        done
    done
done
