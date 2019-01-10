
对 nets 中网络做了一些更改:
1. inception_v4/inception_v3/inception_resnet_v2 添加 global_pool 选项;
2. 最后一层不使用 global_pool 时，avg_pool2d 添加参数 stride=1 (默认为 2);
3. inception_resnet_v2 full_connected 改为 conv2d;
