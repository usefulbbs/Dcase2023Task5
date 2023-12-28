#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: gwyan
 # @Date: 2023-12-20 14:36:05
 # @LastEditors: gwyan
 # @LastEditTime: 2023-12-25 21:12:50
### 
export PYTHON=/home/ygw/.conda/envs/gwyan/bin:$PYTHON

work_path=`pwd`
if [[ $(basename $work_path) != Dcase2023Task5 ]];then
    echo Please cd to the Projects, and execute the run.sh && exit 1
fi
IFS=,
export CUDA_VISIBLE_DEVICES=0,1,2,3

# training setting
meta_training=false
k_way=10
n_shot=5  # meta-training batch_size = k_way * n_shots

# post processing setting
val_path_=$work_path/Development/Validation_Set
evaluation_file=$work_path/src/output_csv/tim/Eval_out_tim.csv
new_evaluation_file=$work_path/src/output_csv/tim/Eval_out_tim_post.csv
    
if [ $# -lt 1 ];then
    echo -e 'Please select one option: feature/train/finetune/metric' && exit 1
    return
else
    case $1 in
    feature) 
        echo "******************************************************** start feature extraction ********************************************************"
        CurrentLog=$(date +%Y_%m_%d)_feature.log
        echo -e "\t\t\t\t The process will be recorded in $CurrentLog \t\t\t\t"
        nohup python -u src/main.py path.work_path=$work_path set.features=true \
                                    set.train=false set.eval=false set.test=false &> $CurrentLog &
        ;;
    train)
        echo "******************************************************** start training model ********************************************************"
        CurrentLog=$(date +%Y_%m_%d)_train.log
        echo -e "\t\t\t\t The process will be recorded in $CurrentLog \t\t\t\t"
        nohup python -u src/main.py path.work_path=$work_path set.features=false \
                                    set.train=true set.eval=false set.test=false \
                                    train.k_way=$k_way train.meta_training=$meta_training &> $CurrentLog &
        echo process PID:$! # echo PID
        ;;
    finetune) 
        echo "******************************************************** start fine-tuning  ********************************************************"
        CurrentLog=$(date +%Y_%m_%d)_fine-tuning.log
        echo -e "\t\t\t\t The process will be recorded in $CurrentLog \t\t\t\t"
        nohup python -u src/main.py path.work_path=$work_path set.features=false \
                                    set.train=false set.eval=true set.test=false \
                                     &> $CurrentLog &
        echo process PID:$! # echo PID
        ;;
    metric)
        echo "******************************************************** run post  ********************************************************"
        python src/utils/post_proc.py -val_path $val_path_ -evaluation_file $evaluation_file -new_evaluation_file $new_evaluation_file
        python evaluation_metrics/evaluation.py -pred_file=$new_evaluation_file -ref_files_path=$val_path_ -team_name=PKU_ADSP -dataset=VAL -savepath=$work_path/evaluation_metrics/
        ;;
    *) 
        echo Please select one option: feature/train/finetune/metric && exit 1
        return ;;
    esac
fi
