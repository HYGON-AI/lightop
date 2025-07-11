1, 新建/opt/rocm/rocblas/lib/librarygemmbiasf目录,
   将Kernels.so-000-gfx900.hsaco,Kernels.so-000-gfx906.hsaco,TensileLibrary.dat,TensileLibrary_gfx906.co,TensileManifest.txt
   放到/opt/rocm/rocblas/lib/librarygemmbiasf下面
   
2, 编程需要引用头文件 hytsf.h,步骤如下
   1,新建调用handle
     void * rhandle = nullptr;
     gemmbias_forward_create_handle(&rhandle);
	 
   2,调用gemm+bias功能,beta必须设置为1.0,若为0则表示只有gemm功能
     gemmbias_forward_half(rhandle, transa, transb, 
                             m, n, k, 
                            (const void*)(&alpha), 
                            (const void*)da, (const void*)db, (const void*)(&tmp_beta), 
                            (const void*)dc, (void*)dd);
	
     dc是bias的地址,长度为m	
	
   3,销毁调用handle
	 gemmbias_forward_destroy_handle(rhandle);