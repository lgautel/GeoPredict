#!/bin/bash

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYOPENGL_ERROR_CHECKING=0
export MESA_GL_VERSION_OVERRIDE=3.3

num_tasks=24
num_gpus=8
seeds="7"
base_video_out_path="./data"
num_trials_per_task=50
replan_steps=5
delay_minutes=2

for seed in $seeds; do
    echo "Starting experiments with seed: $seed"
    video_out_path="${base_video_out_path}/robocasa_main_${seed}"

    tasks_per_gpu=$((num_tasks / num_gpus))
    gpu_id=0
    while [ $gpu_id -lt $num_gpus ]; do
        start_task=$((gpu_id * tasks_per_gpu))
        end_task=$(((gpu_id + 1) * tasks_per_gpu - 1))
        
        task_idx=$start_task
        while [ $task_idx -le $end_task ]; do
            MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYOPENGL_ERROR_CHECKING=0 MESA_GL_VERSION_OVERRIDE=3.3 \
            CUDA_VISIBLE_DEVICES=$gpu_id \
            python tools/test_robocasa.py \
                --seed $seed \
                --suite_idx $task_idx \
                --video_out_path $video_out_path \
                --num_trials_per_task $num_trials_per_task \
                --replan_steps $replan_steps \
                --weight_path "./ckpts/GeoPredict_robocasa.pth" \
                --task_data_path "./ckpts/robocasa" \
                --gpu_id 0 &
            task_idx=$((task_idx + 1))

            sleep $((delay_minutes * 60))
        done

        gpu_id=$((gpu_id + 1))
    done

    wait
    echo "Finished experiments with seed: $seed"
done

echo "All experiments finished!"
