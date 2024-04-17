## 记录什么
为了转成TensorRT，并对齐TensorRT推理结果和PyTorch推理结构做了什么，采了哪些坑？

## 做了什么

#### 1. PointPillars拆分
因为TensorRT不支持某些算子，如voxelization和NMS(最新版本好像支持了)，需要把可以转成TensorRT的部分和不可以转成TensorRT的部分进行拆分
- PointPillarsPre
- PointPillarsCore
- PointPillarsPos

#### 2. PointPillarsCore转成ONNX
这一步比较顺利，转成了ONNX并对齐了PyTorch的推理结果；值得注意的是，需要支持dynamic_axes，因为不同点云数据的pillars数量不一样

#### 3. ONNX转成TensorRT
这一步问题很多，卡了很长时间。大致是包括：
- 某些算子不支持，如在TensorRT 7.2.3.4中不支持scatterND，高本版支持该算子，因此需要解决算子重新编译的问题
- 某些操作不支持，如2D索引，转换成1D索引解决
- 在转TRT时，因为需要支持动态尺寸，因此在转换时需要加上`--xxShapes`等参数

#### 4. TensorRT推理
TensorRT推理时遇到了一下问题：
- 精度对不齐：定位发现TensorRT推理时，对于动态尺寸需要加些代码
- 速度慢：
    - 定位发现基于是voxelization的实现较慢(200+ms)；
    - 后来改成cuda实现，把gpu上voxelization的结果再copy到cpu上，速度有了较大提升，但还是速度慢(<100ms)
    - 后来注意到voxelization的结果也是在TRT(gpu)上完成推理，因此没有必要来回copy，最终voxelization的时间 <6ms
