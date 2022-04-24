# Linux GPU/CPU C++ 推理功能测试

Linux GPU/CPU C++ 推理功能测试的主程序为`test_inference_cpp.sh`，可以测试基于C++预测引擎的推理功能。

## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | tensorrt | mkldnn |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  支持 | 支持 | 支持 | 支持 |

## 2. 测试流程

### 2.1 准备数据和推理模型

#### 2.1.1 准备数据

从验证集或者测试集中抽出至少一张图像，用于后续的推理过程验证。

#### 2.1.2 准备推理模型

* 如果已经训练好了模型，可以参考[模型导出](../../tools/export_model.py)，导出`inference model`，用于模型预测。得到预测模型后，假设模型文件放在`inference`目录下，则目录结构如下。

```
mobilenet_v3_small_infer/
|--inference.pdmodel
|--inference.pdiparams
|--inference.pdiparams.info
```
**注意**：上述文件中，`inference.pdmodel`文件存储了模型结构信息，`inference.pdiparams`文件存储了模型参数信息。注意文件存储的目录需要与[配置文件](../config/inference_cls.yaml)中的`inference_model_dir`参数对应一致。
其他参数修改，请直接修改[配置文件](../config/inference_cls.yaml)中的参数。包括测试图片路径、是否使用GPU等。

### 2.2 准备环境

C++推理功能编译，请参[C++编译文档](../../deploy/cpp_shitu/readme.md)。

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
# 注意：运行前请修改好`inference_cls.yaml`中的`inference_model_dir`参数
bash test_tipc/test_inference_cpp.sh test_tipc/config/inference_cls.yaml
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - ./deploy/inference_cpp/build/clas_system test_tipc/configs/mobilenet_v3_small/inference_cpp.txt ./images/demo.jpg > ./log/infer_cpp/infer_cpp_use_cpu_use_mkldnn.log 2>&1 !
```

最终log中会打印出结果，如下所示
```
img_file_list length: 1
result:
    class id: 8
    score: 0.9014719725
Current image path: deploy/images/ILSVRC2012_val_00000010.jpeg
Current time cost: 0.1409450000 s, average time cost in all: 0.1409450000 s.
    Top1: class_id: 259, score: 0.1844, label: Pomeranian
    Top2: class_id: 153, score: 0.1327, label: Maltese dog, Maltese terrier, Maltese
    Top3: class_id: 204, score: 0.1002, label: Lhasa, Lhasa apso
    Top4: class_id: 265, score: 0.0899, label: toy poodle
    Top5: class_id: 154, score: 0.0761, label: Pekinese, Pekingese, Peke
```
详细log位于`./log/infer_cpp/infer_cpp_use_cpu_use_mkldnn.log`中。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
