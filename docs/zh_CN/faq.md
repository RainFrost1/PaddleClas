# FAQ

>>
* Q: 启动训练后，为什么当前终端中的输出信息一直没有更新？
* A: 启动运行后，日志会实时输出到`mylog/workerlog.*`中，可以在这里查看实时的日志。


>>
* Q: 多卡评估时，为什么每张卡输出的精度指标不相同？
* A: 目前PaddleClas基于fleet api使用多卡，在多卡评估时，每张卡都是单独读取各自part的数据，不同卡中计算的图片是不同的，因此最终指标也会有微量差异，如果希望得到准确的评估指标，可以使用单卡评估。


>>
* Q: 在配置文件的`TRAIN`字段中配置了`mix`的参数，为什么`mixup`的数据增广预处理没有生效呢？
* A: 使用mixup时，数据预处理部分与模型输入部分均需要修改，因此还需要在配置文件中显式地配置`use_mix: True`，才能使得`mixup`生效。