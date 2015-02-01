/*
* mandel_GPU.cu
*
*  Created on: 17.12.2014
*      Author: raub
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "mandel_GPU.cuh"

dim3 ThreadsPerBlock(16,16);
dim3 BlocksPerGrid;

//o~--------------------------------------------------------------------~o//
__global__ void mandel_gpu(uchar4* img, CRange range, unsigned int MaxIter)
//o~--------------------------------------------------------------------~o//
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < range.window.x && y < range.window.y)
    {
        unsigned int i = x + y*blockDim.y*gridDim.x;

        float cx = range.xmin + range.xstep * (float)x;
        float cy = range.ymin + range.ystep * (float)y;

        float px = 0;
        float py = 0;
        float tmp = 0;

        unsigned int n = 0;
        while(n++ <= MaxIter && px*px + py*py < 4)
        {
            tmp = px*px - py*py + cx;
            py = 2 * px * py + cy;
            px = tmp;
        }

        img[i].w = 255;
        if(n >= MaxIter)
        {
            img[i].x = 255;
            img[i].y = 255;
            img[i].z = 255;
        }
        else
        {
            img[i].x = 0;
            img[i].y = 0;
            img[i].z = 0;
        }
    }
}


//o~--------------------------------------------------------------------~o//
__global__ void simpleframe(uchar4* img, CRange range)
//o~--------------------------------------------------------------------~o//
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    unsigned int i = x + y*blockDim.y*gridDim.x;

    img[i].x = 0;
    img[i].y = 0;
    img[i].z = 0;
    img[i].w = 255;

    if(x == 0 || x == range.window.x-1 || y == 0 || y == range.window.y-1)
    {
        img[i].x = 255;
    }
}


//o~--------------------------------------------------------------------~o//
void calcGrid(dim3 windowSize)
//o~--------------------------------------------------------------------~o//
{
    if(ThreadsPerBlock.x >= windowSize.x)
    {
        BlocksPerGrid.x = 1;
    }
    else
    {
        BlocksPerGrid.x = (windowSize.x + (ThreadsPerBlock.x-1)) / ThreadsPerBlock.x;
    }

    if(ThreadsPerBlock.y >= windowSize.y)
    {
        BlocksPerGrid.y = 1;
    }
    else
    {
        BlocksPerGrid.y = (windowSize.y + (ThreadsPerBlock.y-1)) / ThreadsPerBlock.y;
    }

    //  printf("Window: %d x %d => Grid: (%d x %d) x (%d x %d)\n", windowSize.x, windowSize.y,
    //                                                             BlocksPerGrid.x, BlocksPerGrid.y,
    //                                                             ThreadsPerBlock.x, ThreadsPerBlock.y);
}



//o~--------------------------------------------------------------------~o//
void render(uchar4* img, CRange range, unsigned int MaxIter)
//o~--------------------------------------------------------------------~o//
{
    mandel_gpu<<<BlocksPerGrid, ThreadsPerBlock>>>(img, range, MaxIter);
    //simpleframe<<<BlocksPerGrid, ThreadsPerBlock>>>(img, range);
}
