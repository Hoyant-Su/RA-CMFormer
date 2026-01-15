#!/bin/sh
trap "kill 0" SIGINT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$SCRIPT_DIR
cd $PYTHONPATH

CONFIG=config/exp_configs.yaml

exp=$(yq e '.exp' $CONFIG)
fold_index=$(yq e '.fold_index' $CONFIG)
fold_name=$(yq e '.fold_name' $CONFIG)
date=$(yq e '.date' $CONFIG)

lr=$(yq e '.settings.lr' $CONFIG)
epochs=$(yq e '.settings.epochs' $CONFIG)
warmup_epoch=$(yq e '.settings.warmup_epoch' $CONFIG)
model=$(yq e '.model' $CONFIG)


if [ "$exp" == "Breast_Colon" ]; then
    num_cls=2
elif [ "$exp" == "multi_cls_3_except_for_Colon" ]; then
    num_cls=3
fi

lrs=($lr)
folds=(1 2 3 4 5)
gpus=(0 1 2 3 4)
models=($model)


for model in "${models[@]}"; do
    if [[ "$model" == "resnet34" || "$model" == "uniformer_base_IL" ]]; then
        export FEAT_FUSION="false"
        script_name="train.py"
    else
        export FEAT_FUSION="true"
        script_name="train_concat.py"
    fi
    for lr in "${lrs[@]}"; do
        echo "============ LR = $lr, Epoch = $epochs, Warmup epoch = $warmup_epoch ============"
        for i in ${!folds[@]}; do
            fold=${folds[$i]}
            gpu=${gpus[$i]}

            ARGS=(
                --data_dir ../tumor_radiomics/Label/crop_tumor/total/image
                --train_anno_file ../tumor_radiomics/Label/exp/$exp/${fold_name}/fold_${fold}_train_tumor_label_dl.txt
                --val_anno_file ../tumor_radiomics/Label/exp/$exp/${fold_name}/fold_${fold}_val_tumor_label_dl.txt
                --batch-size 4
                --model $model
                --lr $lr
                --opt "adamw"
                --warmup-epochs $warmup_epoch 
                --epochs $epochs
                --num-classes $num_cls
                --img_size 20 160 160
                --crop_size 20 160 160
                --initial-checkpoint ./weights/uniformer_base_k400_8x8_partial.pth
                --output ./output/${date}/${exp}/lr_${lr}_epoch_${epochs}_warmup_${warmup_epoch}/fold_${fold}/train
            )
            if [[ "$FEAT_FUSION" == true ]]; then
                ARGS+=(
                    --feat_csv_dir ../tumor_radiomics/Label/exp/$exp/${fold_name}/fold_${fold}_scaled_features_dl.csv
                )
            fi
            CUDA_VISIBLE_DEVICES=$gpu python $script_name "${ARGS[@]}" &
        done
        wait
    done
wait
done

########################

