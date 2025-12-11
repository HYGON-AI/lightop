# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess


def adb_shell(cmd):
    # 执行cmd命令，如果成功，返回(0, 'xxx')；如果失败，返回(1, 'xxx')
    res = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 使用管道
    result = res.stdout.read()  # 获取输出结果
    res.wait()  # 等待命令执行完成
    res.stdout.close() # 关闭标准输出
    return result

@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        try:
            # rocminfo = adb_shell("rocminfo")
            # print(f"[lightop] rocminfo={rocminfo}")
            result = subprocess.run(
                ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            for line in output.split("\n"):
                if "gfx" in line.lower():
                    return line.split(":")[-1].strip()
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo failed {str(e)}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]
    return gfx


gfx = get_gfx()
LIGHTOP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# LIGHTOP_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
LIGHTOP_ASM_DIR = f"{LIGHTOP_ROOT_DIR}/hsa/{gfx}/"
os.environ["LIGHTOP_GPU_TARGET"] = gfx
os.environ["LIGHTOP_ASM_DIR"] = LIGHTOP_ASM_DIR
# print(f"[lightop] LIGHTOP_ASM_DIR={LIGHTOP_ASM_DIR}")


@functools.lru_cache(maxsize=1)
def get_CUs():
    try:
        result = subprocess.run(
            ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout
        for line in reversed(output.split("\n")):
            if "compute unit" in line.lower():
                return line.split(":")[-1].strip()
    except Exception as e:
        raise RuntimeError(f"Get DCU CUS from rocminfo failed {str(e)}")
CUs = get_CUs()
os.environ["LIGHTOP_GPU_CUS"] = CUs
