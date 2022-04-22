# NVIDIA CUPTI Bug reproducer
This packages reproduces memory and performance issues of running cupti captures for longer than a few seconds.
Build and run on a 8 GPU instance
```
docker build -t cupti_bug .
docker run --gpus all --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host --rm -v data:/workspace cupti_bug
```
To run agains CUDA 11.3 build the image with
```
docker build -f Dockerfile_cuda11.3 -t cupti_bug .
```
