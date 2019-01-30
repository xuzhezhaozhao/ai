
## 在公司 48 核机器运行比本机 6 核机器慢 50 倍
原因：可能是 blas 版本不对, 为什么会这样还不清楚，后续还需要进一步研究。
解决办法：将本机的 libblas.so.3, liblapack.so.3 (这里必须带上 .3 版本号，否则
不会生效)拷贝到公司机器上，由于没有 root 权限，需要指定运行时搜索动态库的路径，
可以使用链接参数 -Wl,-rpath=./lib 来指定搜索库路径. 可以用 ldd 来查看二进制文件
的动态库依赖. 或者使用 LD_LIBRARY_PATH 环境变量指定动态库搜索路径.

防止下次又遇到这个问题，备份一下这两个库：
链接: https://pan.baidu.com/s/1yr_0ScIcEk3J1imxhSOa0g 提取码: g43v


## 使用 openblas 时的性能优化
执行 export OMP_WAIT_POLICY=PASSIVE, 速度提高 2 倍
原因是 omp 和 openblas 多线程机制有冲突
参考：
https://github.com/facebookresearch/faiss/wiki/Troubleshooting
https://github.com/facebookresearch/faiss/issues/53
