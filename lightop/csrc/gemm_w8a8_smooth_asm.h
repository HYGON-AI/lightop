#pragma once

#include <torch/all.h>

template <typename T>
struct hipFunctionArgs_wa8a_smooth{
    // Size of Tensor's packed dims, in elements
    uint64_t tensor2dSizeC;
    uint64_t tensor2dSizeA;
    uint64_t tensor2dSizeB;
    T * dataD;
    T * dataC;
    const int8_t  * dataA;
    const int8_t  * dataB;
    int32_t alpha;
    int32_t beta;
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
    unsigned int magicNumberWgmRemainder1;
    unsigned int offsetd;
    unsigned int offsetc;
    unsigned int offseta;
    unsigned int offsetb;
      // unsigned int pad;
    float* dataAs;  
    float* dataBs;
  } ;

