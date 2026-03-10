#include <stdio.h>
#include <string.h>
#include <string.h>
#include <random>
#include "hip/hip_runtime.h"
#include <hip/hip_hcc.h>
//#include "internal/rocblas-auxiliary.h"
#define NUM 256

// #define fileName "gemm_w8a8_smooth_256x256x128_TN_FP16.co"
#define fileName "gemm_w8a8_smooth_256x256x128_WGM8_TN_FP16.co"
#define kernel_name  "Cijk_Alik_Bljk_I8II_BH_MT256x256x128_TT16x8_WG16x32x2_Squant" //hgemm_MT128x128x16_TT8x8_WG16x16" 

typedef signed char int8;

union FP32
{
    uint u;
    float f;
    struct
    {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
    };
};

union FP16
{
    unsigned short u;
    struct
    {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
    };
};

static _Float16 float_to_half(float fh)
{
	unsigned int x = *((unsigned int *)&fh);
	unsigned int sign = x & 0x80000000;
	unsigned int exponent_f32 = (x & 0x7f800000 )>>23;
	unsigned int mantissa_f32 = x & 0x007fffff;
	unsigned short y = (unsigned short)(sign>>16);
	unsigned int exponent_f16;
	unsigned int mantissa_f16;
	unsigned int hx;
	
	hx = x & 0x7fffffff;
	
	if(hx<0x33800000)
	{
		return *((_Float16*)&y);
	}
	
	if(hx>0x7f800000)
	{
		y = 0x7e00;
		return *((_Float16*)&y);
	}
	
	if (hx>=0x477fffff)
	{
		y |= 0x7c00;
		return *((_Float16*)&y);
	}
	
	mantissa_f16 = mantissa_f32 >> 13;
	
	if(exponent_f32>0x70)
	{
		exponent_f16 = exponent_f32-0x70;
	}
	else
	{
		exponent_f16=0;
		mantissa_f16 |= 0x400;
		mantissa_f16 = mantissa_f16>>(0x71-exponent_f32);
	}
	
	y = y | (unsigned short)(exponent_f16<<10) | (unsigned short)mantissa_f16;
	
	return *((_Float16*)&y);
}

static float half_to_float(_Float16 hf)
{
    FP16 h = *((FP16*)&hf);

    static const FP32 magic = { 113 << 23 };
    static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
}

void int8_buffer_load_cpu_2(int8 *src_a,int8 *src_b,int *src_c, _Float16 * dst_fp16, float *scaleA, float *scaleB, float *scale, int alpha,int beta,int size_m,int size_n,int size_k,int batch,int lda,int ldb,int ldc,int stridea,int strideb,int stridec)
{
	int i,j,k,n;
	int idx=size_m/256;
	int idy=size_n/256;
	int idn=1;	
	if(batch>20)
		idn=7;
			
	if(idx==0)
		idx=1;
			
	if(idy==0)
		idy=1;


	for(n=0;n<1;n+=1)
	{
		for(i=0;i<size_n;i+=idy)
		{
			for(j=0;j<size_m;j+=idx)
			{
				int sum=0;
				for(k=0;k<size_k;k++)
				{
					sum += (int)src_b[i*ldb+k+n*strideb] * (int)src_a[j*lda+k+n*stridea];
				}
				dst_fp16[i*ldc+j+n*stridec] = float_to_half(float(alpha*sum+beta*src_c[i*ldc+j+n*stridec]) * (scaleA[j]*scaleB[i]));
			}
		}
	}

}

void int8_buffer_load_cpu(int8 *src_a,int8 *src_b,int *src_c, _Float16 * dst_fp16, float *scaleA, float *scaleB, float *scale, int alpha,int beta,int size_m,int size_n,int size_k,int batch,int lda,int ldb,int ldc,int stridea,int strideb,int stridec)
{
	int i,j,k,n;
	int idx=size_m/256;
	int idy=size_n/256;
	int idn=1;	
	if(batch>20)
		idn=7;
			
	if(idx==0)
		idx=1;
			
	if(idy==0)
		idy=1;
	
	for(i=0;i<size_n;i++)
	{
		for(j=0;j<size_m;j++)
		{
			scale[i*size_m+j] = scaleA[j] * scaleB[i];
		}
	}

	for(n=0;n<1;n+=1)
	{
		for(i=0;i<size_n;i+=idy)
		{
			for(j=0;j<size_m;j+=idx)
			{
				int sum=0;
				for(k=0;k<size_k;k++)
				{
					sum += (int)src_b[i*ldb+k+n*strideb] * (int)src_a[j*lda+k+n*stridea];
				}
				dst_fp16[i*ldc+j+n*stridec] = float_to_half(float(alpha*sum+beta*src_c[i*ldc+j+n*stridec]) * scale[i*size_m+j]);
			}
		}
	}

}

int main(int argc,char **argv)
{
	//printf("m,n,k,groupmapping,depth,su,num");
	int8 *src_a;
	int8 *src_b;
	int *src_c,*c_device;
	int8 *a_device,*b_device;
	_Float16 *out_cpu;
//	int8 *rocblas_a, *rocblas_b, *rocblas_c;
	_Float16 *out_gpu;
	_Float16 *d_device;
	
	float *scaleA,*scaleB,*scale;
	float *scaleA_device,*scaleB_device;
	
	int alpha = 1;
	int beta=0;
	size_t size_m =atoi(argv[1]);
	size_t size_n =atoi(argv[2]);
	size_t size_k =atoi(argv[3]);
	size_t batch = 1;
	
	size_t lda=size_k;;
	size_t ldb=size_k;;//max(256,size_n);
	size_t ldc=size_m;;
	
	size_t stridea=lda*size_m;;
	size_t strideb=ldb*size_n;;//max(256,size_n);
	size_t stridec=ldc*size_n;

	size_t i,j,n;
	int error=0;
	int workgroupmapping = 8;
	int depth= 128;//atoi(argv[8]);
	int SU = 0;//atoi(argv[9]);
	int lds= 0;//atoi(argv[10]);
	int num = 5;//atoi(argv[7]);
	int pad=0;
	

	
	size_t strideC1J = ldc;
	size_t strideC2K = stridec;
	size_t strideA1L = lda;
	size_t strideA2K = stridea;
	size_t strideB1L = ldb;
	size_t strideB2K = strideb;
	
int flag_init=atoi(argv[4]);
	
	FILE *fp;
	size_t sizeA = max(lda*size_m,stridea*batch);
	size_t sizeB = max(ldb*size_n,strideb*batch);
	//for(int kk=atoi(argv[2]);kk<atoi(argv[3]);kk+=256)
	{
		//size_m = kk;
		//size_n = 256;

	src_a = (int8 *)malloc(sizeA*sizeof(int8));
	src_b = (int8 *)malloc(sizeB*sizeof(int8));
	src_c = (int *)malloc(batch*ldc*size_n*sizeof(int));
	
	out_cpu = (_Float16 *)malloc(batch*ldc*size_n*sizeof(_Float16));
	out_gpu = (_Float16 *)malloc(batch*ldc*size_n*sizeof(_Float16));
	
	scaleA = (float *)malloc(size_m*sizeof(float));
	scaleB = (float *)malloc(size_n*sizeof(float));
	scale = (float *)malloc(size_m*size_n*sizeof(float));
	

	hipMalloc((void**)&a_device,sizeA*sizeof(int8));
	hipMalloc((void**)&b_device,sizeB*sizeof(int8));
	hipMalloc((void**)&c_device,batch*ldc*size_n*sizeof(int));
	hipMalloc((void**)&d_device,batch*ldc*size_n*sizeof(_Float16));
	
	hipMalloc((void**)&scaleA_device,size_m*sizeof(float));
	hipMalloc((void**)&scaleB_device,size_n*sizeof(float));

	if(a_device==NULL || b_device==NULL || c_device==NULL)
		printf("malloc error\n");
	
	if(src_a==NULL || src_b==NULL || out_cpu==NULL || out_gpu==NULL)
		printf("cpu malloc error\n");

	int rand_data = sqrt((65535/size_k));
	if(rand_data>256)
		rand_data=256;
	if(flag_init)
	{
		
	int rand_data = 7;//log((65535/size_k));
	srand(7);
	using namespace std;
	std::random_device rd;  // 用于获取随机数种子
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-3.0f, 3.0f);
		
		for(n=0;n<batch;n++)
		{
			for(i=0;i<size_k;i++)
			{
				for(j=0;j<size_m;j++)
				{
					//src_a[i+j*lda+n*stridea] = (int8)rand() / (RAND_MAX+1.0);
					src_a[i+j*lda+n*stridea] = static_cast<int8>(dis(gen));//(rand()%7)-3;
				}
			}
		}
		//printf("\n");
		for(n=0;n<batch;n++)
		{
			for(i=0;i<size_k;i++)
			{
				for(j=0;j<size_n;j++)
				{
					//src_b[i+j*ldb+n*strideb] = (int8)rand() / (RAND_MAX+1.0);
					src_b[i+j*ldb+n*strideb] = static_cast<int8>(dis(gen));//(rand()%7)-3;
				}
			}
		}
		//printf("\n");

		for(n=0;n<batch;n++)
		{
			for(i=0;i<size_n;i++)
			{
				for(j=0;j<size_m;j++)
					//out_gpu[i*ldc+j+n*stridec] = out_cpu[i*ldc+j+n*stridec] = (int8)rand() / (RAND_MAX+1.0);// (rand()%rand_data)+1;
					src_c[i*ldc+j+n*stridec] = (rand()%rand_data)+1;
			}
		}
		
		for(i=0;i<size_m;i++)
			scaleA[i] = (int)(dis(gen));//(rand()%7)-3;;//(rand()%2);
		
		for(i=0;i<size_n;i++)
			scaleB[i] = (int)(dis(gen));//(rand()%7)-3;;//(rand()%2);

	}

	hipInit(0);
	hipDevice_t device;
	hipCtx_t context;
	hipDeviceGet(&device,0);
	hipCtxCreate(&context,0,device);

	hipMemcpy(a_device,src_a,sizeA*sizeof(int8),hipMemcpyHostToDevice);
	hipMemcpy(b_device,src_b,sizeB*sizeof(int8),hipMemcpyHostToDevice);
	hipMemcpy(c_device,src_c,batch*ldc*size_n*sizeof(int),hipMemcpyHostToDevice);
	
	hipMemcpy(scaleA_device,scaleA,size_m*sizeof(float),hipMemcpyHostToDevice);
	hipMemcpy(scaleB_device,scaleB,size_n*sizeof(float),hipMemcpyHostToDevice);

//	hipMemcpy(rocblas_a,src_a,stridea*size_k*sizeof(int8),hipMemcpyHostToDevice);
//	hipMemcpy(rocblas_b,src_b,strideb*size_k*sizeof(int8),hipMemcpyHostToDevice);
//    hipMemcpy(rocblas_c,out_cpu,stridec*size_n*sizeof(int8),hipMemcpyHostToDevice);
	
	hipModule_t Module;
	hipFunction_t Function;

	hipModuleLoad(&Module,fileName);
	hipModuleGetFunction(&Function,Module,kernel_name);
	
	  /* module function args */
  struct {
    // Size of Tensor's packed dims, in elements
    uint64_t tensor2dSizeC;
    uint64_t tensor2dSizeA;
    uint64_t tensor2dSizeB;
    _Float16 * dataD;
    int * dataC;
    const int8 * dataA;
    const int8 * dataB;
    int alpha;
    int beta;
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
	float *scaleA;
	float *scaleB;
    unsigned int pad;
  } hipFunctionArgs;
  
   unsigned int staggerUIter = SU; // how many stride-sized clicks to stagger start offset
  int unrollLoopIters = size_k/depth/1; // /DepthU/GSU
  int tmp;
  if(depth==8)
	  tmp=16;
  else if(depth==16)
	  tmp=8;
  while (staggerUIter>1) {
    if (unrollLoopIters >= (staggerUIter*tmp)) {
      break;}
    staggerUIter /= 2; // step down to smaller stagger
  }
  if (staggerUIter>=1) staggerUIter -= 1;
  
   /* grid sizes */
  const unsigned int workDim = 3;
  const unsigned int threadTile[2] = { 16, 8 };
  const unsigned int groupSize[2] = { 16, 32 };
  size_t localWorkSize[3] = { 768, 1, 1 };
  size_t globalWorkSize[1][3];
  globalWorkSize[0][2] = 1;
  globalWorkSize[0][2] *= batch;
  unsigned int sizeOfC0 = size_m;
  unsigned int sizeOfC1 = size_n;
  unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
  unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);
  unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;
  unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;
  // b/c single kernel, add extra work-group here if edge needed
  if (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }
  if (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }
  unsigned int problemNumGroupTiles0 = totalWorkGroups0;
  unsigned int problemNumGroupTiles1 = totalWorkGroups1;
  const unsigned smallNumMagicShift = 31; // bozo, review
  unsigned magicNumberProblemNumGroupTiles0 = (1L<<smallNumMagicShift) / problemNumGroupTiles0 + 1; // bozo, review
  unsigned numFullBlocks =  problemNumGroupTiles1 / workgroupmapping; // divide by WorkGroupMapping
  unsigned wgmRemainder1 =  problemNumGroupTiles1 % workgroupmapping;
  if (wgmRemainder1 == 0) wgmRemainder1 = workgroupmapping;
  unsigned magicNumberWgmRemainder1 = ((1L<<smallNumMagicShift) / wgmRemainder1 + 1);
  globalWorkSize[0][0] = totalWorkGroups0;
  globalWorkSize[0][1] = totalWorkGroups1;
  
  uint64_t tensor2dSizeC = 1 * std::max(size_m, strideC1J) * max(size_n, strideC2K);
  uint64_t tensor2dSizeA = 1;
  uint64_t tensor2dSizeAStride = 0;
  uint64_t tensor2dSizeAOffset = 0;
  tensor2dSizeAStride = max(tensor2dSizeA*size_k, (uint64_t)strideA1L);
  tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_k;
  tensor2dSizeA = tensor2dSizeAStride;
  tensor2dSizeAStride = max(tensor2dSizeA*size_m, (uint64_t)strideA2K);
  tensor2dSizeAOffset += tensor2dSizeAStride - tensor2dSizeA*size_m;
  tensor2dSizeA = tensor2dSizeAStride;
  tensor2dSizeA -= tensor2dSizeAOffset;

  uint64_t tensor2dSizeB = 1;
  uint64_t tensor2dSizeBStride = 0;
  uint64_t tensor2dSizeBOffset = 0;
  tensor2dSizeBStride = max(tensor2dSizeB*size_k, (uint64_t)strideB1L);
  tensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB*size_k;
  tensor2dSizeB = tensor2dSizeBStride;
  tensor2dSizeBStride = max(tensor2dSizeB*size_n, (uint64_t)strideB2K);
  tensor2dSizeBOffset += tensor2dSizeBStride - tensor2dSizeB*size_n;
  tensor2dSizeB = tensor2dSizeBStride;
  tensor2dSizeB -= tensor2dSizeBOffset;
  
  hipFunctionArgs.tensor2dSizeC = tensor2dSizeC;
  hipFunctionArgs.tensor2dSizeB = tensor2dSizeB;
  hipFunctionArgs.tensor2dSizeA = tensor2dSizeA;
  hipFunctionArgs.dataD = d_device;
  hipFunctionArgs.dataC = c_device;
  hipFunctionArgs.dataA = a_device;
  hipFunctionArgs.dataB = b_device;
  hipFunctionArgs.alpha = alpha;
  hipFunctionArgs.beta = beta;
  hipFunctionArgs.strideD1J = strideC1J;
  hipFunctionArgs.strideD2K = strideC2K;
  hipFunctionArgs.strideC1J = strideC1J;
  hipFunctionArgs.strideC2K = strideC2K;
  hipFunctionArgs.strideA1L = strideA1L;
  hipFunctionArgs.strideA2K = strideA2K;
  hipFunctionArgs.strideB1L = strideB1L;
  hipFunctionArgs.strideB2K = strideB2K;
  hipFunctionArgs.sizeI = size_m;
  hipFunctionArgs.sizeJ = size_n;
  hipFunctionArgs.sizeK = batch;
  hipFunctionArgs.sizeL = size_k;
  hipFunctionArgs.offseta = 0;
  hipFunctionArgs.offsetb =0;
  hipFunctionArgs.offsetc=0;
  hipFunctionArgs.offsetd=0;
  hipFunctionArgs.scaleA=scaleA_device;
  hipFunctionArgs.scaleB=scaleB_device;
  hipFunctionArgs.pad=pad;
  
  hipFunctionArgs.staggerUIter = staggerUIter;
  hipFunctionArgs.problemNumGroupTiles0 = problemNumGroupTiles0;
  hipFunctionArgs.problemNumGroupTiles1 = problemNumGroupTiles1;
//  hipFunctionArgs.magicNumberProblemNumGroupTiles0=magicNumberProblemNumGroupTiles0;
//  hipFunctionArgs.gridNumWorkGroups0 = globalWorkSize[0][0];
  hipFunctionArgs.numFullBlocks = numFullBlocks;
  hipFunctionArgs.wgmRemainder1=wgmRemainder1;
  hipFunctionArgs.magicNumberWgmRemainder1 =magicNumberWgmRemainder1;
 
// printf("staggerUIter=%d,\n",staggerUIter);
//printf("problemNumGroupTiles0=%d,problemNumGroupTiles1=%d\n",problemNumGroupTiles0,problemNumGroupTiles1);
//printf("magicNumberProblemNumGroupTiles0=%d,\n",magicNumberProblemNumGroupTiles0);
//printf("gridNumWorkGroups0=%d\n",globalWorkSize[0][0]);
//printf("numFullBlocks=%d,wgmRemainder1=%d,magicNumberWgmRemainder1=%d\n",numFullBlocks,wgmRemainder1,magicNumberWgmRemainder1);

	
	size_t size = sizeof(hipFunctionArgs);
	//printf("size %d\n",size);
	void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,&hipFunctionArgs,
				HIP_LAUNCH_PARAM_BUFFER_SIZE,&size,HIP_LAUNCH_PARAM_END};
	
		//time
	float time_elapsed=0;
	hipEvent_t start,stop;

	float total_time=0;
	printf("start\n");
	for(int kkm = 0; kkm<num; kkm++)
	{

		hipEventCreateWithFlags(&start,hipEventDefault);
		hipEventCreateWithFlags(&stop, hipEventDefault);
		
		//gridx*blockx,gridy*blocky,gridz*blockz,blockx,blocky,blockz
		hipExtModuleLaunchKernel(Function, globalWorkSize[0][0]*localWorkSize[0],
		  globalWorkSize[0][1]*localWorkSize[1],
		  globalWorkSize[0][2]*localWorkSize[2],
		  localWorkSize[0],
		  localWorkSize[1],
		  localWorkSize[2],0,0,NULL,(void**)&config,start,stop,0);

		while (hipEventQuery(stop) != hipSuccess);

		hipEventElapsedTime(&time_elapsed,start,stop);
		
		printf("%d %d %d time is %f(ms),GFlops is %f GFlops\n",size_m,size_n,size_k,time_elapsed,((float)2.0*size_m*size_n*size_k)/(time_elapsed*1000000));
		
		if(num>1)
		{
			if(kkm>0)
				total_time += time_elapsed; 
		}
		else
			total_time = time_elapsed;
		
		if(kkm==0)
		  hipMemcpy(out_gpu,d_device,size_n*ldc*batch*sizeof(_Float16),hipMemcpyDeviceToHost);

	}
	printf("end\n");
	if(flag_init)
	{
		int8_buffer_load_cpu(src_a,src_b,src_c,out_cpu,scaleA,scaleB,scale,alpha,beta,size_m,size_n,size_k,batch,lda,ldb,ldc,stridea,strideb,stridec);
		//int8_buffer_load_cpu_2(src_a,src_b,src_c,out_gpu,scaleA,scaleB,scale,alpha,beta,size_m,size_n,size_k,batch,lda,ldb,ldc,stridea,strideb,stridec);
		/*rocblas_handle handle;
	    rocblas_create_handle(&handle);
	
		rocblas_operation  transA;
	    rocblas_operation  transB;

		transA=rocblas_operation_transpose;	
		transB=rocblas_operation_none;
		rocblas_hgemm_strided_batched(handle,transA,transB,size_m,size_n,size_k,(rocblas_half*)(&alpha),(rocblas_half*)(a_device),lda,stridea,(rocblas_half*)(b_device),ldb,strideb,(rocblas_half*)(&beta),(rocblas_half*)(c_device),ldc,stridec,batch);
		hipMemcpy(out_gpu,c_device,size_n*ldc*batch*sizeof(int8),hipMemcpyDeviceToHost);
		*/
		//
		{
			int i,j,k,n;
			int idx=size_m/256;
			int idy=size_n/256;
			int idn=1;	
			if(batch>20)
				idn=7;
			
			if(idx==0)
				idx=1;
			
			if(idy==0)
				idy=1;
			
			for(n=0;n<batch;n+=1)
			{
				for(i=0;i<size_n;i+=idy)
				{
					for(j=0;j<size_m;j+=idx)
					{
						if(fabs((half_to_float(out_gpu[i*ldc+j+n*stridec]) - half_to_float(out_cpu[i*ldc+j+n*stridec])))>1e-3)
						{
							error++;
							printf("pp %d,%d,%d,%f,%f\n",n,i,j,(half_to_float(out_gpu[i*ldc+j+n*stridec])),half_to_float((out_cpu[i*ldc+j+n*stridec])));
							break;
						}
					}
					
					if(error!=0)
						break;
				}
				
					if(error!=0)
						break;
			}
		}

	}
	for(i=0;i<1;i++)
	{
		printf("%d,%f,%f\n",i,half_to_float(out_gpu[i]),half_to_float(out_cpu[i]));
	//	printf("%x\n",((int*)out_gpu)[0]);
	//	printf("%f,%f,\n\n",half_to_float(out_gpu[i]),half_to_float(out_gpu[i+2]));
	//	printf("%f,%f,\n\n",half_to_float(out_cpu[i]),half_to_float(out_cpu[i+2]));
	}
	//printf("\n");
	//printf("%f,\n",((float*)(out_gpu))[0]);
	//printf("\n");
	
	//for(i=0;i<80;i++)
	//	printf("%f,",((float*)out_gpu)[i]);
	//printf("\n");
	//for(i=80;i<256;i++)
	//	printf("%f,",((float*)out_gpu)[i]);	
	
	if(error==0)
	{
		total_time /= (num-1);
		printf("PASS %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", size_m, size_n, size_k,batch,lda,ldb,ldc,stridea,strideb,stridec);
		printf("MT%dx%dx%d_su%d_wgm%d_lds%d,",macroTile0,macroTile1,depth,SU,workgroupmapping,lds);
		printf("%f,%f\n",total_time*1000,((double)2.0*size_m*size_n*size_k*batch)/(total_time*1000000));
	}
	else
	{
		total_time /= (num-1);//total_time = 2147483647.0;
		printf("FAILED %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", size_m, size_n, size_k,batch,lda,ldb,ldc,stridea,strideb,stridec);
		printf("MT%dx%dx%d_su%d_wgm%d_lds%d,",macroTile0,macroTile1,depth,SU,workgroupmapping,lds);
		printf("%f,%f\n",total_time*1000,((double)2.0*size_m*size_n*size_k*batch)/(total_time*1000000));
	}
		
//	printf("%f,%f\n",half_to_float(out_gpu[0]),half_to_float(out_gpu[1]));
	
	hipEventDestroy(start);
	hipEventDestroy(stop);
	hipFree(a_device);
	hipFree(b_device);
	hipFree(c_device);
	hipFree(d_device);
	hipFree(scaleA_device);
	hipFree(scaleB_device);
	
	free(src_a);
	free(src_b);
	free(src_c);
	free(scale);
	free(scaleA);
	free(scaleB);
	free(out_cpu);
	free(out_gpu);
	
	
	
	}
	

	
}
