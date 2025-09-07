#pragma once

#include <torch/all.h>

template <typename Ti,typename To,typename Tc,typename Talpha=Tc,typename Tbias=Tc>
struct __attribute__((packed)) GPUProb{
    // Size of Tensor's packed dims, in elements
    uint32_t size_0;//
    uint32_t size_1;//
    uint32_t size_2;//
    uint32_t size_3;//
    To * d ;//
    To * c ;//
    Ti * a ;//
    Ti * b ;//
    unsigned int strideD1 ;
    unsigned int strideD2 ;
    unsigned int strideC1 ; 
    unsigned int strideC2 ; 
    unsigned int strideA1 ; 
    unsigned int strideA2 ; 
    unsigned int strideB1 ; 
    unsigned int strideB2 ; 
    Tc  alpha;//
    Tc  beta;//
    Talpha * scaleAlphaVec;
    Tbias * bias;//
    unsigned int bias_type;//
    unsigned int strideBias;//
    Talpha activation_0;//
    Talpha activation_1;//
    unsigned int activationType;//
};
