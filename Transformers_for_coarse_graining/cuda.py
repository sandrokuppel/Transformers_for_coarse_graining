import subprocess


def get_free_gpu():
    # Run nvidia-smi command and get the output
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    # Decode the output and split into lines
    gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
    # Convert memory values to integers
    gpu_memory = [int(x) for x in gpu_memory]
    # Get the index of the GPU with the most free memory
    best_gpu = gpu_memory.index(max(gpu_memory))
    return best_gpu