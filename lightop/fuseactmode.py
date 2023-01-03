from enum import Enum
class fuseactmode(Enum):
    miopen_sigmoid = 1      # 1 / (1 + e^-x)
    miopen_tanh = 2         # beta * tanh(alpha * x) torch no
    relu = 3                # native and miopen
    miopen_softrelu = 4     # log(1 + e^x)  torch no
    miopen_abs = 5          # abs(x)
    miopen_power = 6        # (alpha + beta * x )^gamma torch no
    miopen_clipped_relu = 7 # min(alpha, max(0, x)) torch no
    miopen_leaky_relu = 8   # alpha * x | x <= 0; x | x > 0
    miopen_elu = 9          # alpha * (e^x - 1) | x <= 0; x | x > 0
    native_silu = 13        # only natice