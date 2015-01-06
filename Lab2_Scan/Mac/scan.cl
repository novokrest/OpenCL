#define SWAP(a, b)   { __local float* tmp = a; a = b; b = tmp; }

__kernel void inclusive_scan_hillis_steele(__global float* input, __global float* lasts, __global float* output, 
                                           __local float* a, __local float* b, int st, int is_final)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint bid = get_group_id(0);
    uint block_size  = get_local_size(0);
    uint blocks_count = get_num_groups(0);
    uint gsize = get_global_size(0);
    
    if (is_final) {
        int offsets[8];
        offsets[0] = 0;
        int offset = gsize;
        int k = 1;
        while(offset) {
            offsets[k] = offsets[k - 1] + offset;
            offset /= block_size;
            ++k;
        }

        int id = gid + 1;
        int decomp[8];
        int ds = 0;
        while(id) {
            int q = id / block_size;
            int r = id % block_size;
            id = q;
            decomp[ds] = r;
            ++ds;
        }
        
        float res = 0;
        int p = 0;
        int ind = 0;
        for (int i = ds - 1; i > -1; i -= 1) {
            if (decomp[i] == 0) {
                continue;
            }
            p = p * block_size + decomp[i];
            ind = offsets[i] + p - 1;
            res += lasts[ind];
        }
        
        output[gid] = res;
        return;
    }
    
    a[lid] = b[lid] = lasts[gid + st];
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
        lasts[bid + gsize + st] = a[lid];
    }
    
    lasts[gid + st] = a[lid];
}
