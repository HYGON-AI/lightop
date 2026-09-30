# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from lightop import op

def batch_linear_assignment(cost):
    worker, task = cost.shape
    cost = cost.reshape(1, worker, task)
    if task < worker:
        cost = cost.transpose(1, 2)
        col4row, row4col = op.batch_linear_assignment(cost.contiguous())
        row4col = row4col.long().squeeze(0)
        mask = (row4col >= 0).int() 
        valid_idx = torch.topk(mask, k=task, largest=True, sorted=False).indices
        row_ind = valid_idx
        col_ind = row4col[valid_idx] 
        return row_ind, col_ind
    else:
        col4row, row4col =  op.batch_linear_assignment(cost.contiguous())
        col4row = col4row.long().squeeze(0)
        mask = (col4row >= 0).int() 
        valid_idx = torch.topk(mask, k=worker, largest=True, sorted=False).indices
        row_ind = valid_idx
        col_ind = col4row[valid_idx] 
        return row_ind, col_ind

