import torch


def cuda_test():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Device:", torch.cuda.get_device_name(0))
        return True
    else:
        print("CUDA is not available.")
    
    return False
    
    
if __name__ == "__main__":
    cuda_test()