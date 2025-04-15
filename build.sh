g++ -c -fPIC rkllm_wrapper.cpp -o rkllm_wrapper.o
g++ -shared -o librkllm_wrapper.so rkllm_wrapper.o  -lrkllmrt -lstdc++ -lpthread

