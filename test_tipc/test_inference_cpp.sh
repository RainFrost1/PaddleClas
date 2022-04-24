#!/bin/bash
source test_tipc/common_func.sh

function func_parser_key_cpp(){
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value_cpp(){
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

FILENAME=$1

dataline=$(cat ${FILENAME})
lines=(${dataline})

# parser params
dataline=$(awk 'NR==1, NR==14{print}'  $FILENAME)
IFS=$'\n'
lines=(${dataline})

# parser load config
use_gpu_line=$(sed "s/ //g" <<< ${lines[4]})
use_mkldnn_line=$(sed "s/ //g" <<< ${lines[5]})
use_tensorrt_line=$(sed "s/ //g" <<< ${lines[10]})
use_fp16_line=$(sed "s/ //g" <<< ${lines[8]})
use_gpu_key=$(func_parser_key "${use_gpu_line}")
use_gpu_value=$(func_parser_value "${use_gpu_line}")
use_mkldnn_key=$(func_parser_key "${use_mkldnn_line}")
use_mkldnn_value=$(func_parser_value "${use_mkldnn_line}")
use_tensorrt_key=$(func_parser_key "${use_tensorrt_line}")
use_tensorrt_value=$(func_parser_value "${use_tensorrt_line}")
use_fp16_key=$(func_parser_key "${use_fp16_line}")
use_fp16_value=$(func_parser_value "${use_fp16_line}")

LOG_PATH="./log/infer_cpp"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_infer_cpp.log"

function func_infer_cpp(){
    # inference cpp
    if [ "$use_gpu_value" = "True" ]; then
        if [ "$use_tensorrt_value" = "True" ]; then
            if [ "$use_fp16_value" = "True" ]; then
                _save_log_path="${LOG_PATH}/infer_cpp_${use_gpu_key}_${use_tensorrt_key}_${use_fp16_key}.log"
            else
                _save_log_path="${LOG_PATH}/infer_cpp_${use_gpu_key}_${use_tensorrt_key}.log"
            fi
        else
            _save_log_path="${LOG_PATH}/infer_cpp_${use_gpu_key}.log"
        fi
    else
        if [ "$use_mkldnn_value" = "True" ]; then
            _save_log_path="${LOG_PATH}/infer_cpp_use_cpu_${use_mkldnn_key}.log"
        else
            _save_log_path="${LOG_PATH}/infer_cpp_use_cpu.log"
        fi    
    fi
    # run infer cpp
    inference_cpp_cmd="./deploy/cpp/build/clas_system"
    infer_cpp_full_cmd="${inference_cpp_cmd} -c test_tipc/config/inference_cls.yaml > ${_save_log_path} 2>&1 "   
    eval $infer_cpp_full_cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${infer_cpp_full_cmd}" "${status_log}"
}

echo "################### run test cpp inference ###################"

func_infer_cpp