import torch
from torch.library import Library, infer_schema
from collections.abc import Callable


DEFAULT_DISPATCH_KEY = "CUDA"


# 从vllm中迁移过来的，用于注册自定义op
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    if mutates_args is None:
        mutates_args = []

    if target_lib is None:
        raise TypeError(
            "direct_register_custom_op(...): target_lib must not be None. "
            "Create once: `from torch.library import Library` then "
            "`lmslim_lib = Library('lmslim', 'FRAGMENT')` and pass `target_lib=lmslim_lib`."
        )

    if dispatch_key is None:
        dispatch_key = DEFAULT_DISPATCH_KEY

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)