// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0



#include <hip/hip_ext.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <mutex>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <unistd.h>
#include <queue>
#include <condition_variable>
#include "half.hpp"
#include "hytsf.h"
#include "time.h"
#include <sys/select.h>
#include "squeue.h"
using namespace std;


#define MAX_NUM (10)
#define MID_NUM (5.0)

#define TYPEF half_float::half//_Float16//half_float::half
#define tensile_half _Float16
#define THRESH (float)0.003

int loop_test(int thread_limit,int transa,int transb,int m,int n,int k);

typedef struct {
    int transa,transb;
    int m,n,k;
}gemm_input;

extern "C"
__global__ void AddC(
  _Float16 * C,
  _Float16 * D,
  int const m,  //1024
  int const n)
{
  uint64_t id = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;//hc_get_workitem_absolute_id(0);
  if (id >= (m*n))
    return;

  D[id] = C[id] + D[id];
}

std::mutex show_lck;
int result_cmp(std::vector<TYPEF> hd,std::vector<TYPEF> hd2,int len,int m,int result_show)
{
	int err_cnt = 0;
	uint16_t* dat0 = (uint16_t*)hd.data(),*dat1 = (uint16_t*)hd2.data();
	float d0,d1,t;
	for(int i = 0;i < len;i++){
		if(dat0[i] == dat1[i])
			continue;
		d0 = (float)hd[i];
		d1 = (float)hd2[i];
		t = max(d0,d1);
		if(t != 0.0){
			if(fabs(d0 - d1)/t > THRESH){
				err_cnt++;
				if(result_show)
					printf("    index:%d,m index:%d,n index:%d,  src:%8.9f(0x%x) dst:%8.9f(0x%x) \n",i,i%m,i/m,d0,dat0[i],d1,dat1[i]);
			}
		}else{
			err_cnt++;
			if(result_show)
				printf("    index:%d,m index:%d,n index:%d,  src:%8.9f(0x%x) dst:%8.9f(0x%x) \n",i,i%m,i/m,d0,dat0[i],d1,dat1[i]);
		}
		
	}
    std::unique_lock<std::mutex> lck;
	printf("c err_cnt %d\n",err_cnt);
	
	return err_cnt;
}


int main(int argc,char *argv[])
{
    int ret;
	void * rhandle = nullptr;
    TYPEF alpha = TYPEF(1.0f),beta = TYPEF(0.0f);
    int m = 1024, n = 1024, k = 1024;
    int lda, ldb, ldc, ldd, size_a, size_b, size_c;
	int transa = 0,transb = 0;
	int rocblas_only = 0;
	int result_show = 1;
	int gemm_bias = 1;
	int ch,debug=0;
	int loop_time = 100;
	int alignM;
	int test_self = 0;
	int localx,gpnum;
	int test_all = 0;

    while((ch = getopt(argc,argv,"a:b:m:n:k:A:B:o:d:s:v:c:l:D:h:t:"))!=-1)
    {
      switch(ch)
      {
          case 'a':
		    if(1 == atoi(optarg)){
				transa = 1;
			}
			else{
				transa = 0;
			}
            printf(" a %d,",transa);
            break;
          case 'b':
		    if(1 == atoi(optarg)){
				transb = 1;
			}
			else{
				transb = 0;
			}
			printf(" b %d,",transb);
            break;
		  case 'c':
		    loop_time = atoi(optarg);
			printf(" c %d,",loop_time);
		    break;
		  case 'd':
		    debug = atoi(optarg);
			printf(" d %d,",debug);
			break;
          case 'm':
		    m = atoi(optarg);
			printf(" m %d,",m);
            break;
		  case 'n':
		    n = atoi(optarg);
			printf(" n %d,",n);
			break;
          case 'k':
            k = atoi(optarg);
			printf(" k %d,",k);
            break;
		  case 'A':
		    alpha = (TYPEF)atof(optarg);
			printf(" A %f,",(float)alpha);
		    break;
		  case 'B':
		    beta = (TYPEF)atof(optarg);
			printf(" B %f,",(float)beta);
			break;
		  case 'o':
		    rocblas_only = atoi(optarg);
			printf(" o %d,",rocblas_only);
		    break;
		  case 'v':
		    result_show = atoi(optarg);
			printf(" v %d,",result_show);
		    break;
		  case 's':
		    gemm_bias = atoi(optarg);
			printf(" s %d,",gemm_bias);
		    break;
		  case 'l':
		   
			break;
		  case 'D':
		    test_self = atoi(optarg);
			printf(" D %d,",test_self);
		    break;
          case 't':
		    test_all = atoi(optarg);
			printf(" test all %d,",test_all);
		    break;
          case 'h':
            cout<<"cmd option:"<<endl;
            cout<<"      -a transA "<<endl;
            cout<<"      -b transB"<<endl;
            cout<<"      -m m"<<endl;
            cout<<"      -n n"<<endl;
			cout<<"      -k k"<<endl;
			cout<<"      -A alpha"<<endl;
			cout<<"      -B beta"<<endl;
			cout<<"      -d flag(debug)"<<endl;
			cout<<"      -o run rocblas only"<<endl;
			cout<<"      -s run gemm+bias or not"<<endl;
			cout<<"      -v result output"<<endl;
            cout<<"      -h (help)"<<endl;
			cout<<"      -D gemm test self"<<endl;
			cout<<"      -l use gemmbias library"<<endl;
            break;
          default:
            break;
      }
    }

    if(test_all){
        loop_test(test_all,transa,transb,m,n,k);
        return 0;
    }

    ret = gemmbias_forward_create_handle(&rhandle);
    printf("gemmbias_forward_create_handle ret %d\n",ret);
	
	hipEvent_t start = nullptr;
    hipEventCreate(&start);
	
	hipEvent_t stop = nullptr;
    hipEventCreate(&stop);
    
    if(transa == 0)
    {
        lda        = m;
        size_a     = k * lda;
        cout << "N";
    }
    else
    {
        lda        = k;
        size_a     = m * lda;
        cout << "T";
    }
    
    if(transb == 0)
    {
        ldb        = k;
        size_b     = n * ldb;
        cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b     = k * ldb;
        cout << "T: ";
    }
    ldc = m;
    ldd = ldc;
    size_c = n * ldc;
	
	gemm_bias = alignM = m;//((n+1)/2)*2;
	printf("align M:%d\n",alignM);
	
	cout << "m, n, k, " << m << ", " << n << ", " << k << std::endl;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<TYPEF> ha(size_a);
    std::vector<TYPEF> hb(size_b);
    std::vector<TYPEF> hc(size_c);
	std::vector<TYPEF> hc2(gemm_bias);//hc2(m);
	std::vector<TYPEF> hd(size_c);
	std::vector<TYPEF> hd2(size_c);
    std::vector<TYPEF> hc_gold(size_c);

    // initial data on host
    //srand(1);
	float tmp;
	srand((unsigned)time(NULL));
    for(int i = 0; i < size_a; ++i)
    {
		tmp = (MID_NUM - rand() % MAX_NUM)/4.0;
        ha[i] = (TYPEF)(tmp);
    }
    for(int i = 0; i < size_b; ++i)
    {
		tmp = (MID_NUM - rand() % MAX_NUM)/4.0;
        hb[i] = (TYPEF)(tmp);
    }
    for(int i = 0; i < n; ++i)
    {
		if(i == 0){
			for(int k = 0; k < m; ++k){
				tmp = (32.0 - rand() % 64)/1.0;//1.0 + k*1.0;
				//tmp = 1.0;
				hc[k + i*m] = (TYPEF)(tmp);//(rand() % MAX_NUM); 
			}
		}
		else
			for(int k = 0; k < m; ++k){
				hc[k + i*m] = hc[k];
			}
    }
	if(gemm_bias == size_c){
		for(int i = 0; i < n*m; ++i)
		{
			hc2[i] = hc[i];
		}
	}else{ //m
		for(int k = 0; k < m; ++k)
		{
			hc2[k] = hc[k];
		}
	}
    hc_gold = hc;

    // allocate memory on device
    TYPEF *da, *db, *dc,*dc2,*dd,*dd2;
    hipMalloc(&da, size_a * sizeof(TYPEF));
    hipMalloc(&db, size_b * sizeof(TYPEF));
    hipMalloc(&dc, size_c * sizeof(TYPEF));
	hipMalloc(&dc2, gemm_bias * sizeof(TYPEF));
	hipMalloc(&dd, size_c * sizeof(TYPEF));
	hipMalloc(&dd2, size_c * sizeof(TYPEF));

    // copy matrices from host to device
    hipMemcpy(da, ha.data(), sizeof(TYPEF) * size_a, hipMemcpyHostToDevice);
    hipMemcpy(db, hb.data(), sizeof(TYPEF) * size_b, hipMemcpyHostToDevice);
    hipMemcpy(dc, hc.data(), sizeof(TYPEF) * size_c, hipMemcpyHostToDevice);
	hipMemcpy(dc2, hc2.data(), sizeof(TYPEF) * gemm_bias, hipMemcpyHostToDevice);

    beta = (TYPEF)(1.0f);
    TYPEF tmp_beta = (TYPEF)(0.0);
	
	localx = 512;
	gpnum = (m*n+512-1)/512;

    printf("gpnum %d,localx %d\n",gpnum,localx);
	
	float ms0,ms1 = 0,ms2 = 0;
	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	if(rocblas_only == 0){
		
		ret = gemmbias_forward_half(rhandle, transa, transb, 
										m, n, k, 
										(const void*)(&alpha), 
										(const void*)da, (const void*)db, (const void*)(&tmp_beta), 
										(const void*)dc, (void*)dd);
        if(ret != 0){
            printf("gemmbias_forward_half return %d\n",ret);
        }
		hipLaunchKernelGGL(AddC, dim3(gpnum,1,1), dim3(localx,1,1), 0, nullptr, (_Float16*)dc, (_Float16*)dd, m , n);
		
		ms1 = 0.0;
		ms0 = 0.0;
		hipEventRecord(start, NULL);
		for(int i=0;i<loop_time;i++){
			gemmbias_forward_half(rhandle, transa, transb, 
										m, n, k, 
										(const void*)(&alpha), 
										(const void*)da, (const void*)db, (const void*)(&tmp_beta), 
										(const void*)dc, (void*)dd);
			hipLaunchKernelGGL(AddC, dim3(gpnum,1,1), dim3(localx,1,1), 0, nullptr, (_Float16*)dc, (_Float16*)dd, m , n);
		}
		
		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&ms2, start, stop);
		
		printf("gemm + bias    use %f ms\n",ms2/loop_time);
	
	}
	
	
	gemmbias_forward_half(rhandle, transa, transb, 
										m, n, k, 
										(const void*)(&alpha), 
										(const void*)da, (const void*)db, (const void*)(&beta), 
										(const void*)dc2, (void*)dd2);
							
	ms0 = 0.0;
	hipEventRecord(start, NULL);
	for(int i=0;i<loop_time;i++){
		gemmbias_forward_half(rhandle, transa, transb, 
										m, n, k, 
										(const void*)(&alpha), 
										(const void*)da, (const void*)db, (const void*)(&beta), 
										(const void*)dc2, (void*)dd2);
	}
	hipEventRecord(stop, NULL);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&ms2, start, stop);
	printf("gemmbias forward gpu use %f ms,\n",ms2/loop_time);
	
	if(0 == rocblas_only){
		hipEventRecord(start, NULL);
		for(int i=0;i<loop_time;i++){
			gemmbias_forward_half(rhandle, transa, transb, 
										m, n, k, 
										(const void*)(&alpha), 
										(const void*)da, (const void*)db, (const void*)(&tmp_beta), 
										(const void*)dc, (void*)dd);
			hipLaunchKernelGGL(AddC, dim3(gpnum,1,1), dim3(localx,1,1), 0, nullptr, (_Float16*)dc, (_Float16*)dd, m , n);
		}
		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&ms2, start, stop);
		
		printf("GEMM    use %f ms\n",ms2/loop_time);
	}
	
	cout << "m, n, k, lda, ldb, ldc, ldd = " << m << ", " << n << ", " << k << ", " << lda
                 << ", " << ldb << ", " << ldc << ", " << ldd << std::endl;

    // copy output from device to CPU
    hipMemcpy(hd.data(), dd, sizeof(TYPEF) * size_c, hipMemcpyDeviceToHost);
	hipMemcpy(hd2.data(), dd2, sizeof(TYPEF) * size_c, hipMemcpyDeviceToHost);

	if(0 == rocblas_only)
		result_cmp(hd,hd2,size_c,m,result_show);
	
	
	while(debug){
		char c = getchar();
		if(c == 'q')
			break;
	}
	
	
    hipFree(da);
    hipFree(db);
    hipFree(dc);
    gemmbias_forward_destroy_handle(rhandle);
	
	return 0;
}

void *g_sq;

void thread_test(void * rhandle)
{
    int transa, transb, m, n, k;
    TYPEF alpha = TYPEF(1.0f),beta = TYPEF(0.0f);
    int lda, ldb, ldc, ldd, size_a, size_b, size_c;
    int rocblas_only = 0;
    int result_show = 1;
    int gemm_bias = 1;
    int debug=0;
    int loop_time = 1;
    int alignM;
    int test_self = 0;
    int localx,gpnum;
    int ret;
        
    hipEvent_t start = nullptr;
    hipEventCreate(&start);
        
    hipEvent_t stop = nullptr;
    hipEventCreate(&stop);

    srand((unsigned)time(NULL));
        
    while(1){
        
        gemm_input *gi = (gemm_input*)pop_queue(g_sq,1);
        //std::cout<<"queue count :"<<cmd_que.queue.size()<<std::endl;

        if(gi == NULL)
            continue;
        else{
            transa = gi->transa;
            transb = gi->transb;
            m = gi->m;
            n = gi->n;
            k = gi->k;
        }
        
        loop_time = 1;
        result_show = 0;
        
        if(transa == 0)
        {
            lda        = m;
            size_a     = k * lda;
            cout << "N";
        }
        else
        {
            lda        = k;
            size_a     = m * lda;
            cout << "T";
        }
        
        if(transb == 0)
        {
            ldb        = k;
            size_b     = n * ldb;
            cout << "N: ";
        }
        else
        {
            ldb        = n;
            size_b     = k * ldb;
            cout << "T: ";
        }
        ldc = m;
        ldd = ldc;
        size_c = n * ldc;

        gemm_bias = alignM = m;//((n+1)/2)*2;
        

        // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
        std::vector<TYPEF> ha(size_a);
        std::vector<TYPEF> hb(size_b);
        std::vector<TYPEF> hc(size_c);
        std::vector<TYPEF> hc2(gemm_bias);//hc2(m);
        std::vector<TYPEF> hd(size_c);
        std::vector<TYPEF> hd2(size_c);
        std::vector<TYPEF> hc_gold(size_c);

        // initial data on host
        //srand(1);
        float tmp;
        
        for(int i = 0; i < size_a; ++i)
        {
            tmp = (MID_NUM - rand() % MAX_NUM)/4.0;
            ha[i] = (TYPEF)(tmp);
        }
        for(int i = 0; i < size_b; ++i)
        {
            tmp = (MID_NUM - rand() % MAX_NUM)/4.0;
            hb[i] = (TYPEF)(tmp);
        }
        for(int i = 0; i < n; ++i)
        {
            if(i == 0){
                for(int k = 0; k < m; ++k){
                    tmp = (32.0 - rand() % 64)/1.0;//1.0 + k*1.0;
                    
                    hc[k + i*m] = (TYPEF)(tmp);//(rand() % MAX_NUM); 
                }
            }
            else
                for(int k = 0; k < m; ++k){
                    hc[k + i*m] = hc[k];
                }
        }
        if(gemm_bias == size_c){
            for(int i = 0; i < n*m; ++i)
            {
                hc2[i] = hc[i];
            }
        }else{ //m
            for(int k = 0; k < m; ++k)
            {
                hc2[k] = hc[k];
            }
        }
        hc_gold = hc;

        // allocate memory on device
        TYPEF *da, *db, *dc,*dc2,*dd,*dd2;
        hipMalloc(&da, size_a * sizeof(TYPEF));
        hipMalloc(&db, size_b * sizeof(TYPEF));
        hipMalloc(&dc, size_c * sizeof(TYPEF));
        hipMalloc(&dc2, gemm_bias * sizeof(TYPEF));
        hipMalloc(&dd, size_c * sizeof(TYPEF));
        hipMalloc(&dd2, size_c * sizeof(TYPEF));

        // copy matrices from host to device
        hipMemcpy(da, ha.data(), sizeof(TYPEF) * size_a, hipMemcpyHostToDevice);
        hipMemcpy(db, hb.data(), sizeof(TYPEF) * size_b, hipMemcpyHostToDevice);
        hipMemcpy(dc, hc.data(), sizeof(TYPEF) * size_c, hipMemcpyHostToDevice);
        hipMemcpy(dc2, hc2.data(), sizeof(TYPEF) * gemm_bias, hipMemcpyHostToDevice);

        beta = (TYPEF)(1.0f);
        TYPEF tmp_beta = (TYPEF)(0.0);
        
        localx = 512;
        gpnum = (m*n+512-1)/512;

        printf("gpnum %d,localx %d\n",gpnum,localx);
        
        float ms0,ms1 = 0,ms2 = 0;
        ////////////////////////////////////////////////////
        ////////////////////////////////////////////////////
        if(rocblas_only == 0){
            
            ret = gemmbias_forward_half(rhandle, transa, transb, 
                                            m, n, k, 
                                            (const void*)(&alpha), 
                                            (const void*)da, (const void*)db, (const void*)(&tmp_beta), 
                                            (const void*)dc, (void*)dd);
            if(ret != 0){
                printf("gemmbias_forward_half return %d\n",ret);
            }
            hipLaunchKernelGGL(AddC, dim3(gpnum,1,1), dim3(localx,1,1), 0, nullptr, (_Float16*)dc, (_Float16*)dd, m , n);
            
            ms1 = 0.0;
            ms0 = 0.0;
            hipEventRecord(start, NULL);
            for(int i=0;i<loop_time;i++){
                gemmbias_forward_half(rhandle, transa, transb, 
                                            m, n, k, 
                                            (const void*)(&alpha), 
                                            (const void*)da, (const void*)db, (const void*)(&tmp_beta), 
                                            (const void*)dc, (void*)dd);
                hipLaunchKernelGGL(AddC, dim3(gpnum,1,1), dim3(localx,1,1), 0, nullptr, (_Float16*)dc, (_Float16*)dd, m , n);
            }
            
            hipEventRecord(stop, NULL);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms2, start, stop);
            
            printf("gemm + bias    use %f ms\n",ms2/loop_time);
        
        }
        
        
        gemmbias_forward_half(rhandle, transa, transb, 
                                            m, n, k, 
                                            (const void*)(&alpha), 
                                            (const void*)da, (const void*)db, (const void*)(&beta), 
                                            (const void*)dc2, (void*)dd2);
                                
        ms0 = 0.0;
        hipEventRecord(start, NULL);
        for(int i=0;i<loop_time;i++){
            gemmbias_forward_half(rhandle, transa, transb, 
                                            m, n, k, 
                                            (const void*)(&alpha), 
                                            (const void*)da, (const void*)db, (const void*)(&beta), 
                                            (const void*)dc2, (void*)dd2);
        }
        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&ms2, start, stop);
        printf("gemmbias forward gpu use %f ms,\n",ms2/loop_time);
        
        if(0 == rocblas_only){
            hipEventRecord(start, NULL);
            for(int i=0;i<loop_time;i++){
                gemmbias_forward_half(rhandle, transa, transb, 
                                            m, n, k, 
                                            (const void*)(&alpha), 
                                            (const void*)da, (const void*)db, (const void*)(&tmp_beta), 
                                            (const void*)dc, (void*)dd);
                hipLaunchKernelGGL(AddC, dim3(gpnum,1,1), dim3(localx,1,1), 0, nullptr, (_Float16*)dc, (_Float16*)dd, m , n);
            }
            hipEventRecord(stop, NULL);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms2, start, stop);
            
            printf("GEMM    use %f ms\n",ms2/loop_time);
        }

        {
            std::unique_lock<std::mutex> lck;
            cout << "transa,transb,m, n, k, lda, ldb, ldc, ldd = " << transa << ","<<transb <<","<< m << ", " << n << ", " << k << ", " << lda
                     << ", " << ldb << ", " << ldc << ", " << ldd << std::endl;
        }

        // copy output from device to CPU
        hipMemcpy(hd.data(), dd, sizeof(TYPEF) * size_c, hipMemcpyDeviceToHost);
        hipMemcpy(hd2.data(), dd2, sizeof(TYPEF) * size_c, hipMemcpyDeviceToHost);

        if(0 == rocblas_only)
            result_cmp(hd,hd2,size_c,m,result_show);

        
        
        hipFree(da);
        hipFree(db);
        hipFree(dc);
        hipFree(dc2);
        hipFree(dd);
        hipFree(dd2);

        delete gi;
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);
        
    pthread_exit(0);
}

int loop_test(int thread_limit,int _transa,int _transb,int m,int n,int k)
{
    int ret;
    int transa,transb;
    int flags=0;
  
    void * rhandle = nullptr;
    ret = gemmbias_forward_create_handle(&rhandle);
    printf("gemmbias_forward_create_handle ret %d\n",ret);
    g_sq = create_queue(1000);
   
    #if 0
    std::vector<int> _m={8,16,24,32,48,64,96,128,256,512,768,1024,2048,4096,8192,16384,30528,32256,32768};
    std::vector<int> _n={8,16,24,32,48,64,96,128,256,512,768,1024,2048,4096,8192,16384,30528,32256,32768};
    std::vector<int> _k={8,16,24,32,48,64,96,128,256,512,768,1024,2048,4096,8192,16384,30528,32256,32768};
    #else
    std::vector<int> _m={8,16,24,32,48,64,96,128,256,512,768,1024,2048,4096};
    std::vector<int> _n={8,16,24,32,48,64,96,128,256,512,768,1024,2048,4096};
    std::vector<int> _k={8,16,24,32,48,64,96,128,256,512,768,1024,2048,4096};
    #endif

    std::vector<int> vec_m;
    std::vector<int> vec_n;
    std::vector<int> vec_k;

    for(int i=0;i<_m.size();i++){
        
          vec_m.push_back(_m[i]-4);
        
          vec_m.push_back(_m[i]-1);
        
          vec_m.push_back(_m[i]);
        
        vec_m.push_back(_m[i]+1);
        if(i >= 3)//reduce repeat
            vec_m.push_back(_m[i]+4);
    }

    for(int i=0;i<_n.size();i++){
       
          vec_n.push_back(_n[i]-4);
        
          vec_n.push_back(_n[i]-1);
        
          vec_n.push_back(_n[i]);
       
        vec_n.push_back(_n[i]+1);
        if(i >= 3)//reduce repeat
            vec_n.push_back(_n[i]+4);
    }

    for(int i=0;i<_k.size();i++){
       
          vec_k.push_back(_k[i]-4);
        
          vec_k.push_back(_k[i]-1);
        
          vec_k.push_back(_k[i]);
        
          vec_k.push_back(_k[i]+1);
        if(i >= 3 )//reduce repeat
            vec_k.push_back(_k[i]+4);
    }

    for(int i=0;i<thread_limit;i++){
        auto thread = std::thread([=]() { thread_test(rhandle); });
        thread.detach();
    }

    //printf("%s,m size %d,n size %d,k size %d\n",__func__,vec_m.size(),vec_n.size(),vec_k.size());
    
    for(transa = _transa;transa <= 1;transa++)
      for(transb = _transb;transb <= 1;transb++ )
        for(int m_num=0;m_num<vec_m.size();m_num++)
           for(int n_num=0;n_num<vec_n.size();n_num++)
             for(int k_num=0;k_num<vec_k.size();k_num++){

                //printf("%s,m index %d,n index %d,k index %d\n",__func__,m_num,n_num,k_num);

                if(vec_m[m_num] == m && vec_n[n_num] == n && vec_k[k_num] == k){
                    flags = 1;
                }

                if(flags == 0)
                    continue;

                //printf("%s,m index %d,n index %d,k index %d have insert\n",__func__,m_num,n_num,k_num);
                    
                gemm_input *gi = new gemm_input;

                gi->transa = transa;
                gi->transb = transb;
                gi->m = vec_m[m_num];
                gi->n = vec_n[n_num];
                gi->k = vec_k[k_num];

                push_queue(g_sq,(void*)gi,1,1);

             }

    sleep(1000000);//wait thread 

    gemmbias_forward_destroy_handle(rhandle);
}
