import os

class gemm_w8a8_deepgemm_masked_config:

    @staticmethod
    def get_deepgemm_w8a8_smooth_asm_dict():
        gemm_w8a8_smooth_asm_dict_80 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1001,
        }

        gemm_w8a8_smooth_asm_dict_72 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1001,
        }

        gemm_w8a8_smooth_asm_dict_64 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1001,
        }

        gemm_w8a8_smooth_asm_dict_48 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1001,
        }
        gpu_cus = int(os.environ.get("LIGHTOP_GPU_CUS", "80"))

        if gpu_cus == 80:
            gemm_w8a8_smooth_asm_dict = gemm_w8a8_smooth_asm_dict_80
        elif gpu_cus == 72:
            gemm_w8a8_smooth_asm_dict = gemm_w8a8_smooth_asm_dict_72
        elif gpu_cus == 64:
            gemm_w8a8_smooth_asm_dict = gemm_w8a8_smooth_asm_dict_64
        elif gpu_cus == 48:
            gemm_w8a8_smooth_asm_dict = gemm_w8a8_smooth_asm_dict_48
        else:
            raise ValueError(f"Unsupported GPU CUs: {gpu_cus}")
        return gemm_w8a8_smooth_asm_dict

