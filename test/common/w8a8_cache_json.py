# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any, Dict, List, Optional, Set


class W8a8GetCacheJSON:
    """Local fallback for the custom vllm.utils.W8a8GetCacheJSON helper."""

    _instance: Optional["W8a8GetCacheJSON"] = None

    def __new__(cls) -> "W8a8GetCacheJSON":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.triton_moejson_dict: Dict[str, Any] = {}
        self.moe_weight_shapes: List[Set[Any]] = []
        self.triton_json_dir = os.environ.get(
            "TRITON_JSON_DIR",
            os.environ.get("VLLM_TUNED_CONFIG_FOLDER", "./cache/moe"),
        )

    def get_moeint8_triton_cache(
        self,
        file_name: str,
        E: int,
        n1: int,
        n2: int,
        k: int,
        topk: int,
    ) -> Dict[str, Any]:
        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                cachedata = json.load(file)
        else:
            print("*********no bestconfig*******")
            cachedata = {}

        configs_dict: Dict[str, Any] = {}
        for key, value in cachedata.items():
            if not isinstance(value, dict):
                continue
            if "config1" in value and "config2" in value:
                configs_dict[str(key)] = value
                continue
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    configs_dict[f"{sub_key}_{key}"] = sub_value
        return configs_dict
