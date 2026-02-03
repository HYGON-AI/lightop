      
#pragma once
#include <stdint.h>
#include <torch/all.h>

template <typename T>
struct hipFunctionArgs_mqa_logits{
  // Size of Tensor's packed dims, in elements
  uint64_t tensor2dSizeC;
  uint64_t tensor2dSizeA;
  uint64_t tensor2dSizeB;
  _Float16 * dataD;
  _Float16 * dataC;
  const _Float16 * dataA;
  const _Float16 * dataB;
  _Float16 alpha[2];
  _Float16 beta[2];
  unsigned int strideD1J;
  unsigned int strideD2K;
  unsigned int strideC1J; // = ldc
  unsigned int strideC2K; // = stridec
  unsigned int strideA1L; // = lda
  unsigned int strideA2K; // = stridea
  unsigned int strideB1L; // = ldb
  unsigned int strideB2K; // = strideb
  unsigned int sizeI;
  unsigned int sizeJ;
  unsigned int sizeK;
  unsigned int sizeL;
  int staggerUIter;
  unsigned int problemNumGroupTiles0;
  unsigned int problemNumGroupTiles1;
//  unsigned int magicNumberProblemNumGroupTiles0;
//  unsigned int gridNumWorkGroups0;
  unsigned int numFullBlocks;
  unsigned int wgmRemainder1;
  unsigned int use_fp16;
  unsigned int offsetd;
  unsigned int offsetc;
  unsigned int offseta;
  unsigned int offsetb;
  unsigned int pad;
  uint32_t* cu_seq_len_k_start;
  uint32_t* cu_seq_len_k_end;
  unsigned int num_heads;
};

    