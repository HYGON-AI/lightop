import json
import os
import time
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm import _custom_ops as vllm_ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

from vllm.utils import FlexibleArgumentParser
from lmslim.quantize.quant_ops import lm_faster_rmsquant


DEFAULT_SEQUENCE_LENGTHS = [512]
HIDDEN_SIZE = 7168
BENCHMARK_DTYPE = torch.float16

CUDA_GRAPH_CAPTURE_CYCLES = 10


class CommunicatorBenchmark:
    def __init__(
        self, 
        rank: int,
        world_size: int,
        device: torch.device,
        cpu_group:ProcessGroup,
        sequence_lengths: list[int],
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.cpu_group = cpu_group

        max_seq_len = max(sequence_lengths)
        max_tensor_elements = max_seq_len * HIDDEN_SIZE
        self.max_size_override = max_tensor_elements * BENCHMARK_DTYPE.itemsize + 1

        self.custom_allreduce = None
        self._init_communicators()
    
    def _init_communicators(self):
        try:
            self.custom_all_reduce = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                max_size=self.max_size_override,
            )
            print()
        except Exception as e:
            self.custom_all_reduce = None
    
    def benchmark_allreduce(
        self,
        sequence_length: int,
        num_warmup: int,
        num_trials: int
    ):
        results = {}
        
        communicators = []
        if self.custom_all_reduce is not None:
            comm = self.custom_all_reduce
            communicators.append(
                (
                    "ca_stage2_fuse_norm_quant",
                    lambda t,d,w,e,q,r, c = comm: c.custom_all_reduce_fuse_norm_quant(t,d,w,e,q,r),
                    lambda t, c = comm: c.should_custom_ar(t),
                    comm.capture(),
                    "2stage_fuse_norm_quant",
                )
            )
            communicators.append(
                (
                    "ca_stage2",
                    lambda t,c = comm: c.custom_all_reduce(t),
                    lambda t,c = comm: c.should_custom_ar(t),
                    comm.capture(),
                    "2stage"
                )
            )
        out = {}
        latency_time = {}
        for name, allreduce_fn, should_use_fn, context, env_var in communicators:
            if env_var is not None:
                os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = env_var
            else:
                os.environ.pop("VLLM_CUSTOM_ALLREDUCE_ALGO", None)
            
            latency_time[name], out[name] = self.benchmark_allreduce_single(
                sequence_length,
                allreduce_fn,
                should_use_fn,
                context,
                num_warmup,
                num_trials,
            )

        #     print(f"latency : {latency}")
        # for name, out in out.items():
        #     torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)
        out1 = out["ca_stage2_fuse_norm_quant"]
        out2 = out["ca_stage2"]
        if not torch.allclose(out1, out2, rtol=1e-2, atol=5e-2):
            print("精度检查不合格！！！") 
            
        else:
            print("精度检查合格") 
        for name, latency in latency_time.items():
            print(f'{name}, seq_len: {sequence_length}, lantency: {latency}')
    
    def benchmark_allreduce_single(
        self,
        sequence_length: int,
        allreduce_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
        should_use_fn: Callable[[torch.Tensor], bool],
        context,
        num_warmup: int,
        num_trials: int,
    ):
        try:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            tensor = torch.randn(
                sequence_length, HIDDEN_SIZE, dtype=BENCHMARK_DTYPE, device=self.device
            )
            residual = torch.randn(
                sequence_length, HIDDEN_SIZE, dtype=BENCHMARK_DTYPE, device=self.device
            )
            rms_weight = torch.randn(
                sequence_length, HIDDEN_SIZE, dtype=BENCHMARK_DTYPE, device=self.device
            )
            eps = 1.0
            if not should_use_fn(tensor):
                return None
            # test acc 
            g_i = tensor.clone()
            if (os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] == "2stage"):
                pp = allreduce_fn(g_i)
                out1, s = lm_faster_rmsquant(pp, rms_weight, eps, torch.int8, residual=residual, update_input=False)
            else:
                out1, s, normout = allreduce_fn(g_i, HIDDEN_SIZE, rms_weight, eps, torch.int8, residual)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                graph_input = tensor.clone()

                with context:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(CUDA_GRAPH_CAPTURE_CYCLES):
                            if (os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] == "2stage"):
                                out = allreduce_fn(graph_input)
                                vllm_ops.rms_norm_dynamic_per_token_quant(out, rms_weight, eps, torch.int8, residual=residual)
                            else:
                                allreduce_fn(graph_input, HIDDEN_SIZE, rms_weight, eps, torch.int8, residual)
            torch.cuda.synchronize()
            for _ in range(num_warmup):
                graph.replay()
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(num_trials):
                graph.replay()
            torch.cuda.synchronize()

            end_time = time.perf_counter()

            return (
                (end_time - start_time) / num_trials / CUDA_GRAPH_CAPTURE_CYCLES * 1000
            ), out1
        except Exception as e:
            raise RuntimeError(
                f"CUDA graph benchmark failed for communicator: {e}"
            ) from e

def main():
    parser = FlexibleArgumentParser(description="Benchmark allreduce")
    
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Sequence lengths to benchmark (tensor shape: seq_len x hidden_size)",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="Number of warmup iterations"
    )

    parser.add_argument(
        "--num-trials", type=int, default=50, help="Number of benchmark trials"
    )

    args = parser.parse_args()
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    cpu_group = dist.new_group(backend="gloo")
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    benchmark = CommunicatorBenchmark(
        rank, world_size, device, cpu_group, args.sequence_lengths
    )

    for seq_len in args.sequence_lengths:
        if rank == 0:
            print(f"sequence length: {seq_len}, HIDDEN_SIZE: {HIDDEN_SIZE}")
        
        benchmark.benchmark_allreduce(
            sequence_length=seq_len,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
        )

if __name__ == "__main__":
    main()