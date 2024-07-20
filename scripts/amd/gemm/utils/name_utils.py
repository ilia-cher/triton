import torch
import triton
import triton.language as tl

import os

TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8': tl.float8e4b8,
    'bf8': tl.float8e5b16,
}


def get_filename_myKernels():
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/../myKernels.py"


def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def get_filename_compile_driver():
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/../compile_driver.py"


def get_filename_profile_driver(M, N, K, job_id):
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/../profile_driver_{M}x{N}x{K}_{job_id}.py"


def get_default_tuning_result_filename():
    git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
    git_branch_name = git_branch_name[0].decode()
    # handle branch name of "xxx/xxx" format
    git_branch_name = git_branch_name.replace('/', '_')
    git_commit_hash = run_bash_command("git rev-parse --short HEAD")
    git_commit_hash = git_commit_hash[0].decode()

    dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    path = os.path.dirname(os.path.abspath(__file__))
    defaultName = f"{path}/../tuning_results_{git_branch_name}@{git_commit_hash}_{dt_string}.yaml"
    return defaultName
