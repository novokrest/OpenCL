#define SWAP(a, b)         { __local float* tmp = a; a = b; b = tmp; }
//#define GLOBAL_SWAP(a, b) { __global float* tmp = a; a = b; b = tmp; }

__kernel void inclusive_scan_hillis_steele(__global float* input, __global float* lasts, __global float* output, __local float* a, __local float* b, int offset, int is_final)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint bid = get_group_id(0);
    uint block_size  = get_local_size(0);
    uint blocks_count = get_num_groups(0);
    uint gsize = get_global_size(0);
    
    if (is_final) {
        
        int8 offsets;
        offsets[0] = 0;
        offsets[1] = gsize;
        offsets[2] = offsets[1] / block_size;
        offsets[3] = offsets[2] / block_size;
        offsets[4] = offsets[3] / block_size;
        //    int k = 1;
        //    int offset = gsize / block_size;
        //    while(offset) {
        //        offsets[k + 1] = offsets[k] + offset;
        //        offset /= block_size;
        //        ++k;
        //    }
        
        int8 decomp;
        
        int id = gid + 1;
        int ds = 0;
        while(id) {
            int q = id / block_size;
            int r = id % block_size;
            id = q;
            decomp[ds] = r;
            ++ds;
        }
        
        float v = 0;
        int p = 0;
        for (uint i = 3; i > 0; i-=1) {
            p = p * block_size + decomp[i];
            int ind = offsets[i] + p;
            if (ind > 15 || ind < 0) {
                output[gid] = 1000;
                return;
            }
            v += lasts[ind];
        }
        

        
        output[gid] =1; v;
        return;
    }
    

    
    a[lid] = b[lid] = lasts[gid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = 1; s < block_size; s <<= 1) {
        if (lid > (s - 1)) {
            b[lid] = a[lid] + a[lid - s];
        }
        else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a, b);
    }

    if (lid == block_size - 1) {
        lasts[bid + gsize + offset] = a[lid];
    }
    
    lasts[gid + offset] = a[lid];
}

//__kernel void result_scan(__global float* lasts, __global float* output)
//{
//    uint gid = get_global_id(0);
//    uint block_size = get_local_size(0);
//    uint gsize = get_global_size(0);
//    
//    int8 offsets;
//    offsets[0] = 0;
//    offsets[1] = gsize;
//    offsets[2] = offsets[1] / block_size;
//    offsets[3] = offsets[2] / block_size;
//    offsets[4] = offsets[3] / block_size;
////    int k = 1;
////    int offset = gsize / block_size;
////    while(offset) {
////        offsets[k + 1] = offsets[k] + offset;
////        offset /= block_size;
////        ++k;
////    }
//    
//    int4 decomp;
//    
//    int id = gid + 1;
//    int ds = 0;
//    while(id) {
//        int q = id / block_size;
//        int r = id % block_size;
//        id = q;
//        decomp[ds] = r;
//        ++ds;
//    }
//    
//    if (ds - 1 > 3) {
//        output[gid] = -1;
//        return;
//    }
//    
//    float v = 0;
//    int index;
//    int p = 0;
//    for (uint i = ds - 1; i > 0; i-=1) {
//        p = p * block_size + decomp[i];
//        index = offsets[i];
//        if (i < 0) {
//            output[gid] = 1000;
//            return;
//        }
//        v += decomp[i];
//    }
//    
//    //output[gid] = index;
//}
//
//__kernel void global_inclusive_scan_hillis_steele(__global float* input, __global float* output, __global float* a, __global float* b, int gsize)
//{
//    uint gid = get_global_id(0);
//    //uint gsize = get_global_size(0);
//    
//    a[gid] = b[gid] = input[gid];
//    barrier(CLK_GLOBAL_MEM_FENCE);
//    
//    //a[gid] = gsize;
//    
//    for (uint s = 1; s < 128; s <<= 1) {
////        a[gid] = 2;
////        continue;
//        if (gid > (s - 1)) {
//            b[gid] = a[gid] + a[gid - s];
//        }
//        else {
//            b[gid] = a[gid];
//        }
//        barrier(CLK_GLOBAL_MEM_FENCE);
//       // GLOBAL_SWAP(a, b);
//    }
//    
//    output[gid] = a[gid];
//}