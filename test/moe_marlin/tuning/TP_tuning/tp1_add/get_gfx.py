# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])

