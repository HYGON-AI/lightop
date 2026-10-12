// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef SQUEUE_H
#define SQUEUE_H

#ifdef __cplusplus
extern "C" {
#endif

void* create_queue(int limit);
void destroy_queue(void *_sq);
int push_queue(void *_sq,void *hdrdata,int notify_,int force = 0);
void *pop_queue(void *_sq,int force = 0);

#ifdef __cplusplus
}
#endif

#endif