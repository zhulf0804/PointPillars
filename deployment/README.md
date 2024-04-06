Code ToDo:
- TRT check accuracy
- Post processing (including nms)
- TRT runtime
- argc, argv

## 0. Performance comparison

- `GPU`: RTX 3080
- `Cuda`: 11.1
- `TensorRT`: 7.2.3.4
Similar detection results between PyTorch, ONNX and TRT inference are achieved. Detection results validated on val/000134.bin are summarized in `infer_results`. 

    Pictures(..)

    |  | inferTime |
    | :---: | :---: |
    | PyTorch (57af3f) | 29.00ms |
    | PyTorch () | 27.48ms |
    | ONNX | 24.00ms |
    | TensorRT | |
    | TensorRT(fp16) | | 

## 1. PyTorch2ONNX

#### 1.1 PyTorch2ONNX

```
cd PointPillars/deployment
python pytorch2onnx.py --ckpt ../pretrained/epoch_160.pth
```

#### 1.2 (Optional) ONNX inference

```
cd PointPillars/deployment
python onnx_infer.py --pc_path ../dataset/demo_data/val/000134.bin --onnx_path ../pretrained/model.onnx
```

#### 1.3 (Optional) Comaprison to Pytorch inference

```
cd PointPillars/deployment
python pytorch_infer.py --ckpt ../pretrained/epoch_160.pth --pc_path ../dataset/demo_data/val/000134.bin
```
## 2. ONNX2TRT
#### 2.1 ONNX2TRT
```
/your_path/TensorRT-7.2.3.4/bin/trtexec --onnx=../pretrained/model.onnx --saveEngine=../pretrained/model.trt --verbose --dumpProfile
```

#### 2.2 TRT inference
```
cd PointPillars/deployment/trt_infer
mkdir build 
cd build
cmake ..
make
```


## 3. Encountered problems
1. **INVALID_ARGUMENT: getPluginCreator could not find plugin ScatterND version 1**

    **Solution:** compile ScatterND plugin in TensorRT 7.2.3.4, which is supported in TensorRT >= 8.
    ```
    ## 1. Download TensorRT OSS
    git clone -b main https://github.com/nvidia/TensorRT TensorRT
    cd TensorRT
    git checkout release/7.2
    git submodule update --init --recursive

    ## 2. Add ScatterND plugin
    #### 2.1 Download scatterPlugin
    Download scatterPlugin from https://github.com/NVIDIA/TensorRT/tree/release/8.0/plugin/scatterPlugin, and then 
    mv scatterPlugin TensorRT/plugin
    #### 2.2 update plugin/InferPlugin.cpp  
    add line #include "scatterPlugin.h"
    add line initializePlugin<nvinfer1::plugin::ScatterNDPluginCreator>(logger, libNamespace);
    #### 2.3 update plugin/common/kernels/kernel.h
    add line pluginStatus_t scatterNDInference(cudaStream_t stream, int* outputDims, int nOutputDims, int sliceRank, int nRows, int rowSize, int CopySize, int sizeOfElementInBytes, const void* index, const void* updates, const void* data, void* output, void* workspace);
    #### 2.4 update plugin/CMakeLists.txt
    add line scatterPlugin

    ## 3. Compile
    cd TensorRT
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out

    ## 4. Replace 
    cp out/libnv* $TRT_LIBPATH
    ```

2. **While parsing node number 249 [Unsqueeze -> "383"]:**
--- Begin node ---
input: "382"
output: "383"
name: "Unsqueeze_249"
op_type: "Unsqueeze"
attribute {
  name: "axes"
  ints: -1
  type: INTS
}

    --- End node ---
    ERROR: /home/lifa/Drivers/TensorRT/parsers/onnx/onnx2trt_utils.cpp:188 In function convertAxis:
    [8] Assertion failed: axis >= 0 && axis < nbDims


    **Solution:**
    ```
    ## Update code from ##

    
    canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
    canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
    canvas = canvas.permute(2, 1, 0).contiguous()
        

    ## to ##

    canvas = torch.zeros((self.x_l * self.y_l, self.out_channel), dtype=torch.float32, device=device)
    cur_coors_flat = cur_coors[:, 2] * self.x_l + cur_coors[:, 1]  ## why select here.
    canvas[cur_coors_flat] = cur_features
    canvas = canvas.view(self.y_l, self.x_l, self.out_channel)
    canvas = canvas.permute(2, 0, 1).contiguous()
    ```