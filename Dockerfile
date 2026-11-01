FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# CUDA Driver API stubs for build-time linking (no host driver in container)
# llama.cpp links against libcuda.so which exports cu* Driver API functions
RUN mkdir -p /usr/local/cuda/lib64/stubs && \
    cat > /tmp/cuda_driver_stub.c << 'STUBEOF'
#include <stddef.h>
#include <stdint.h>
typedef int CUresult;
typedef int CUdevice;
typedef void *CUmemoryPool;
typedef int CUmemAllocationHandleType;
typedef unsigned int CUmemAccess_flags;
typedef long long CUmem_generic_alloc_t;

CUresult cuInit(unsigned int){return 0;}
CUresult cuGetExportTable(void**,void**){return 0;}
CUresult cuDriverGetVersion(int*){return 0;}
CUresult cuGetProcAddress(void**,char*,int,int,int*){return 0;}
CUresult cuGetProcAddressv(void**,void*){return 0;}
CUresult cuGetErrorName(CUresult,char**){return 0;}
CUresult cuGetErrorString(CUresult,char**){return 0;}

/* Context */
CUresult cuCtxCreate(void**,int,CUdevice){return 0;}
CUresult cuCtxDestroy(void*){return 0;}
CUresult cuCtxGetCurrent(void**){return 0;}
CUresult cuCtxSetCurrent(void*){return 0;}
CUresult cuCtxGetDevice(CUdevice*){return 0;}
CUresult cuCtxSynchronize(void){return 0;}
CUresult cuCtxGetApiVersion(int*,int*){return 0;}
CUresult cuCtxGetCacheConfig(void**){return 0;}
CUresult cuCtxSetCacheConfig(void*){return 0;}
CUresult cuCtxGetLimit(void*,size_t*){return 0;}
CUresult cuCtxSetLimit(void*,size_t){return 0;}
CUresult cuCtxGetSharedMemConfig(void**){return 0;}
CUresult cuCtxSetSharedMemConfig(void*){return 0;}
CUresult cuCtxGetFlags(unsigned int*){return 0;}
CUresult cuCtxAttach(int*,unsigned int){return 0;}
CUresult cuCtxDetach(void*){return 0;}
CUresult cuCtxGetStream(void**,int,void*){return 0;}
CUresult cuCtxGetIpcDevice(void**){return 0;}
CUresult cuCtxIpcSetMemPool(void*,CUmemoryPool*){return 0;}

/* Device */
CUresult cuDeviceGet(CUdevice*,int){return 0;}
CUresult cuDeviceGetCount(int*){return 0;}
CUresult cuDeviceGetAttribute(int*,int,CUdevice){return 0;}
CUresult cuDeviceGetNvCpuUsage(float*,CUdevice){return 0;}
CUresult cuDeviceGetDefaultMemPool(void**,CUdevice){return 0;}
CUresult cuDeviceGetMemPool(void**,CUdevice*,void*){return 0;}
CUresult cuDeviceSetMemPool(CUdevice*,CUmemoryPool*){return 0;}
CUresult cuDeviceGetLuid(char*,unsigned int*,CUdevice){return 0;}
CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t*,int,size_t,size_t){return 0;}
CUresult cuDeviceGetByName(CUdevice*,char*){return 0;}
CUresult cuDeviceGetByPCIBusId(CUdevice*,char*){return 0;}
CUresult cuDeviceGetName(char*,int,CUdevice){return 0;}
CUresult cuDeviceGetPCIBusId(char*,int,int,CUdevice){return 0;}
CUresult cuDevicePrimaryCtxGetState(CUdevice,unsigned int*,unsigned int*){return 0;}
CUresult cuDevicePrimaryCtxRetain(void**,CUdevice){return 0;}
CUresult cuDevicePrimaryCtxRelease(CUdevice){return 0;}
CUresult cuDevicePrimaryCtxSetFlags(CUdevice,unsigned int){return 0;}
CUresult cuDevicePrimaryCtxReset(CUdevice){return 0;}

/* Memory (legacy) */
CUresult cuMemAlloc(void**,size_t){return 0;}
CUresult cuMemFree(void*){return 0;}
CUresult cuMemAllocHost(void**,size_t){return 0;}
CUresult cuMemHostGetDevicePointer(void**,void*,unsigned int){return 0;}
CUresult cuMemHostRegister(void*,size_t,unsigned int){return 0;}
CUresult cuMemHostUnregister(void*){return 0;}
CUresult cuMemAllocAsync(void**,size_t,CUmemoryPool*,void*){return 0;}
CUresult cuMemFreeAsync(void*,void*){return 0;}
CUresult cuMemGetInfo(size_t*,size_t*){return 0;}
CUresult cuMemHostAlloc(void**,size_t,unsigned int){return 0;}

/* Memory (Unified/Address) */
CUresult cuMemAddressReserve(void**,size_t,size_t,unsigned int,CUmemoryPool*,unsigned int){return 0;}
CUresult cuMemAddressFree(void*,size_t){return 0;}
CUresult cuMemCreate(CUmemoryPool*,size_t,unsigned int,CUmemoryPool*,void*);
CUresult cuMemGetAllocationGranularity(size_t*,CUmemoryPool*,unsigned int);
CUresult cuMemGetAllocationHandleType(unsigned int*,CUmemoryPool*,void*,CUmemAllocationHandleType);
CUresult cuMemRetainAllocationHandle(CUmemoryPool**,void*,char*);
CUresult cuMemRelease(CUmemoryPool*);
CUresult cuMemUnmap(void**,size_t);
CUresult cuMemSetAccess(void**,size_t,CUmemoryPool*,CUmemAccess_flags,unsigned int);
CUresult cuMemMap(void**,size_t,unsigned int,CUmem_generic_alloc_t,CUmemoryPool*,unsigned int);
CUresult cuMemGetAddressRange(void**,size_t,void*);
CUresult cuMemPoolTrimTo(CUmemoryPool*,size_t);
CUresult cuMemPoolSetAccess(CUmemoryPool*,void*,size_t);
CUresult cuMemPoolGetAccess(unsigned int*,CUmemoryPool*,void*);
CUresult cuMemPoolSetAttribute(CUmemoryPool*,int,void*,size_t);
CUresult cuMemPoolGetAttribute(CUmemoryPool*,int,void*,size_t);
CUresult cuMemPoolExportPointer(void**,CUmemoryPool*,void*);
CUresult cuMemPoolImportPointer(CUmemoryPool*,void**);
CUresult cuMemPoolExportToShareLevel(void**,CUmemoryPool*,unsigned int,unsigned int);
CUresult cuMemPoolImportFromShareLevel(void**,void*,unsigned int,unsigned int);
CUresult cuMemPoolWaitForFreeEvent(CUmemoryPool*,void*,void*);
CUresult cuMemAllocFromPoolAsync(void**,size_t,CUmemoryPool*,void*){return 0;}

/* Memory Pool */
CUresult cuMemPoolCreate(CUmemoryPool*,void*){return 0;}
CUresult cuMemPoolDestroy(CUmemoryPool*){return 0;}

/* Memory Copy */
CUresult cuMemcpyHtoD(void*,void*,size_t){return 0;}
CUresult cuMemcpyDtoH(void*,void*,size_t){return 0;}
CUresult cuMemcpyHtoH(void*,void*,size_t){return 0;}
CUresult cuMemcpyDtoD(void*,void*,size_t){return 0;}
CUresult cuMemcpy2D(void*){return 0;}
CUresult cuMemcpy3D(void*){return 0;}
CUresult cuMemcpyHtoDAsync(void*,void*,size_t,void*){return 0;}
CUresult cuMemcpyDtoHAsync(void*,void*,size_t,void*){return 0;}
CUresult cuMemcpyHtoHAsync(void*,void*,size_t,void*){return 0;}
CUresult cuMemcpyDtoDAsync(void*,void*,size_t,void*){return 0;}

/* Memory Set */
CUresult cuMemsetD8(void*,unsigned int,size_t){return 0;}
CUresult cuMemsetD16(void*,unsigned short,size_t){return 0;}
CUresult cuMemsetD32(void*,unsigned int,size_t){return 0;}
CUresult cuMemsetD8Async(void*,unsigned int,size_t,void*){return 0;}
CUresult cuMemsetD16Async(void*,unsigned short,size_t,void*){return 0;}
CUresult cuMemsetD32Async(void*,unsigned int,size_t,void*){return 0;}

/* Stream */
CUresult cuStreamCreate(void**,unsigned int){return 0;}
CUresult cuStreamDestroy(void*){return 0;}
CUresult cuStreamSynchronize(void*){return 0;}
CUresult cuStreamQuery(void*){return 0;}
CUresult cuStreamWaitValue(void*,unsigned int*,unsigned int,unsigned int){return 0;}
CUresult cuStreamWriteValue(void*,unsigned int*,unsigned int,unsigned int){return 0;}
CUresult cuStreamCopyAttributes(void*,void*,unsigned int,void**,unsigned int*){return 0;}
CUresult cuStreamGetAttributes(void*,unsigned int*,void**,unsigned int*){return 0;}
CUresult cuStreamGetCtx(void**,void*){return 0;}

/* Event */
CUresult cuEventCreate(void**,unsigned int){return 0;}
CUresult cuEventDestroy(void*){return 0;}
CUresult cuEventSynchronize(void*){return 0;}
CUresult cuEventQuery(void*){return 0;}
CUresult cuEventRecord(void*,void*){return 0;}
CUresult cuIpcGetEventHandle(void*,void*){return 0;}
CUresult cuIpcOpenEventHandle(void**,void*){return 0;}
CUresult cuIpcCloseEventHandle(void*){return 0;}

/* Module / Function */
CUresult cuModuleLoad(CUresult*,char*){return 0;}
CUresult cuModuleLoadData(CUresult*,void*){return 0;}
CUresult cuModuleGetFunction(CUresult*,CUresult*,char*){return 0;}
CUresult cuModuleGetGlobal(CUresult*,int,CUresult*,void**){return 0;}
CUresult cuModuleGetTexRef(CUresult*,CUresult*,char*){return 0;}
CUresult cuModuleGetSurfRef(CUresult*,CUresult*,char*){return 0;}
CUresult cuModuleUnload(CUresult){return 0;}
CUresult cuFuncGetAttribute(int*,int,CUresult*){return 0;}
CUresult cuFuncSetAttribute(CUresult*,int,int){return 0;}
CUresult cuFuncSetCacheSize(CUresult*,unsigned int){return 0;}
CUresult cuFuncSetSharedSize(CUresult*,size_t){return 0;}
CUresult cuParamSetSize(CUresult*,unsigned int){return 0;}
CUresult cuParamSeti(CUresult*,int,unsigned int){return 0;}
CUresult cuParamSetf(CUresult*,int,float){return 0;}
CUresult cuParamSetv(CUresult*,int,void*,unsigned int){return 0;}
CUresult cuParamSetTexRef(CUresult*,int,CUresult*){return 0;}

/* Launch */
CUresult cuLaunchKernel(CUresult*,void**,void**,void**,size_t,size_t,size_t,size_t,void**){return 0;}
CUresult cuLaunchCooperativeKernel(CUresult*,void**,void**,size_t,size_t,size_t,void*){return 0;}

/* Occupancy */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int*,CUresult*,int,void*);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*,CUresult*,int,void*,unsigned int);
CUresult cuOccupancyMaxPotentialBlockSize(int*,int*,CUresult*,int,void*,int,unsigned int);
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int*,int*,CUresult*,int,void*,int,unsigned int,unsigned int);
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t*,CUresult*,int,void*);
CUresult cuOccupancyMaxPotentialClusterSize(int*,CUresult*,int,void*);
CUresult cuOccupancyMaxActiveClusters(int*,CUresult*,int,void*,int);

/* Pointer */
CUresult cuPointerSetAttribute(void*,void*,unsigned int){return 0;}
CUresult cuPointerGetAttributes(unsigned int*,int*,void**,void**){return 0;}
CUresult cuPointerGetAttribute(void*,int,void*,unsigned int){return 0;}

/* Array */
CUresult cuArrayCreate(void**,void*,int){return 0;}
CUresult cuArrayDestroy(void*){return 0;}
CUresult cuArray3DCreate(void**,void*,int){return 0;}
CUresult cuArrayGetPlane(void**,void*,void*){return 0;}
CUresult cuMipmappedArrayCreate(void**,void*,int){return 0;}
CUresult cuMipmappedArrayDestroy(void*){return 0;}
CUresult cuMipmappedArrayGetLevel(void**,void*,int){return 0;}

/* Texture / Surface Reference */
CUresult cuTexRefSetArray(CUresult*,void*,unsigned int){return 0;}
CUresult cuTexRefSetMatrix(CUresult*,void*){return 0;}
CUresult cuSurfRefSetArray(CUresult*,void*,unsigned int){return 0;}
CUresult cuTexRefSetFMT(CUresult*,int,int){return 0;}
CUresult cuTexRefSetMipmapTexRef(CUresult*,CUresult*,unsigned int){return 0;}
CUresult cuTexRefSetAddress(size_t*,CUresult*,size_t){return 0;}
CUresult cuTexRefSetAddress2D(CUresult*,void*,size_t,unsigned int,unsigned int,unsigned int){return 0;}
CUresult cuTexRefSetAddressMode(CUresult*,int,unsigned int){return 0;}
CUresult cuTexRefSetBorderColor(CUresult*,char*){return 0;}
CUresult cuTexRefSetFilterMode(CUresult*,void*){return 0;}
CUresult cuTexRefSetMaxMipmaps(CUresult*,int){return 0;}
CUresult cuTexRefSetMipmapFilterMode(CUresult*,void*){return 0;}
CUresult cuTexRefSetMipmapLevelBias(CUresult*,float,float){return 0;}
CUresult cuTexRefSetMipmapLevelClamp(CUresult*,float,float){return 0;}
CUresult cuTexRefSetMipmappedLevel(CUresult*,int){return 0;}
CUresult cuTexRefSetFlags(CUresult*,unsigned int){return 0;}
CUresult cuTexRefGetAddress(unsigned int*,CUresult*){return 0;}
CUresult cuTexRefGetArray(void**,CUresult*){return 0;}
CUresult cuTexRefGetAddressMode(int*,int,CUresult*){return 0;}
CUresult cuTexRefGetBorderColor(float*,CUresult*){return 0;}
CUresult cuTexRefGetFilterMode(void**,CUresult*){return 0;}
CUresult cuTexRefGetFlags(unsigned int*,CUresult*){return 0;}
CUresult cuTexRefGetFormat(int*,int*,CUresult*){return 0;}
CUresult cuTexRefGetMaxMipmaps(int*,CUresult*){return 0;}
CUresult cuTexRefGetMipmapFilterMode(int*,CUresult*){return 0;}
CUresult cuTexRefGetMipmapLevelBias(float*,float*,CUresult*){return 0;}
CUresult cuTexRefGetMipmapLevelClamp(float*,float*,CUresult*){return 0;}
CUresult cuTexRefGetMipmappedLevel(int*,CUresult*){return 0;}
CUresult cuSurfRefGetArray(void**,CUresult*){return 0;}

/* Graphics Interop */
CUresult cuGraphicsUnregisterResource(void*){return 0;}
CUresult cuGraphicsMapResources(int,void**,void*){return 0;}
CUresult cuGraphicsUnmapResources(int,void**,void*){return 0;}
CUresult cuGraphicsResourceGetMappedPointer(void**,size_t,void*){return 0;}
CUresult cuGraphicsResourceSetMapFlags(void*,unsigned int){return 0;}

/* External Memory / Semaphore */
CUresult cuImportExternalMemory(void**,void*){return 0;}
CUresult cuExternalMemoryGetMappedBuffer(void**,void*,void*){return 0;}
CUresult cuExternalMemoryGetMappedMipmappedArray(void**,void*,void*){return 0;}
CUresult cuDestroyExternalMemory(void*){return 0;}
CUresult cuImportExternalSemaphore(void**,void*){return 0;}
CUresult cuSignalExternalSemaphoresAsync(void**,unsigned int*,void*,void*,void*){return 0;}
CUresult cuWaitExternalSemaphoresAsync(void**,unsigned int*,void*,void*,void*){return 0;}
CUresult cuDestroyExternalSemaphore(void*){return 0;}

/* Stream Capture / Graph */
CUresult cuStreamBeginCapture(void*,unsigned int){return 0;}
CUresult cuStreamEndCapture(void**,void*){return 0;}
CUresult cuStreamIsCapturing(void*,unsigned int*){return 0;}
CUresult cuGraphCreate(void**,unsigned int){return 0;}
CUresult cuGraphUpload(void*,void*){return 0;}
CUresult cuGraphLaunch(void*,void*){return 0;}
CUresult cuGraphDestroy(void*){return 0;}
CUresult cuGraphAddKernelNode(void**,void*,unsigned int,void**,void*,void**){return 0;}
CUresult cuGraphAddMemcpyNode(void**,void*,unsigned int,void**,void*){return 0;}
CUresult cuGraphAddMemsetNode(void**,void*,unsigned int,void**,void*){return 0;}
CUresult cuGraphAddEmptyNode(void**,void*,unsigned int,void**){return 0;}
CUresult cuGraphAddMemAllocNode(void**,void*,unsigned int,void**){return 0;}
CUresult cuGraphAddMemFreeNode(void**,void*,unsigned int,void**,void*){return 0;}
CUresult cuGraphClone(void**,void*){return 0;}
CUresult cuGraphNodeFindInClone(void**,void*,void*){return 0;}
CUresult cuGraphNodeSetEnabled(void*,unsigned int){return 0;}
CUresult cuGraphChildGraphNodeGetChildGraphNode(void**,void*){return 0;}
CUresult cuGraphKernelNodeGetParams(void*,void*){return 0;}
CUresult cuGraphKernelNodeSetParams(void*,void**,void*,void**){return 0;}
CUresult cuGraphMemcpyNodeGetParams(void*,void*){return 0;}
CUresult cuGraphMemcpyNodeSetParams(void*,void*,void*,void**){return 0;}
CUresult cuGraphMemsetNodeGetParams(void*,void*){return 0;}
CUresult cuGraphMemsetNodeSetParams(void*,void*,void*,void**){return 0;}
CUresult cuGraphAddDependencies(void*,unsigned int,void**,void**){return 0;}
CUresult cuGraphDestroyNode(void*,void*){return 0;}
CUresult cuGraphExtractToDot(char*,void*){return 0;}
CUresult cuGraphDebugDotPrint(char*,char*,void*){return 0;}

/* IPC */
CUresult cuIpcGetMemHandle(void**,void*){return 0;}
CUresult cuIpcOpenMemHandle(void**,void*,unsigned int){return 0;}
CUresult cuIpcCloseMemHandle(void*){return 0;}

/* Linking */
CUresult cuLinkCreate(unsigned int,void**,char**,void*,void**){return 0;}
CUresult cuLinkAddData(void*,int,void*,size_t,char**,unsigned int,void*,void*){return 0;}
CUresult cuLinkAddFile(void*,int,char*,unsigned int,char**,unsigned int,void*,void*){return 0;}
CUresult cuLinkComplete(void*,void**,size_t*){return 0;}
CUresult cuLinkDestroy(void*){return 0;}

/* CUDA IPC Memory */
CUresult cuIpcGetMemHandle(void**,void*){return 0;}
CUresult cuIpcOpenMemHandle(void**,void*,unsigned int){return 0;}
CUresult cuIpcCloseMemHandle(void*){return 0;}

/* Texture 1D Linear */
CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t*,int,size_t,size_t){return 0;}
STUBEOF
    gcc -shared -fPIC -o /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda_driver_stub.c && \
    ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ARG LLAMA_TAG=b11869
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp /src

WORKDIR /src

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_RPC=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target rpc-server -j$(nproc)

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN rm -rf /usr/local/cuda/compat

COPY --from=build /src/build/bin/rpc-server /usr/local/bin/

EXPOSE 50052/tcp
ENTRYPOINT ["rpc-server"]
CMD ["-H", "0.0.0.0", "-p", "50052"]
