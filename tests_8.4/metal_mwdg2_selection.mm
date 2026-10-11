#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

struct LevelArgs { uint32_t offset, count, level, maxLevel; float epsilon; };
struct RegulariseArgs { uint32_t parentOffset, childOffset, parentCount; };
struct ExtraArgs {
    uint32_t parentOffset, childOffset, parentCount, level, maxLevel;
    float epsilon, extraFactor;
};
struct DecodeArgs { uint32_t parentOffset, childOffset, childCount; };
struct LeafArgs { uint32_t nodeCount, finestOffset; };

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        auto make = [&](NSUInteger bytes) {
            return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };
        auto run = [&](NSString *name, std::vector<id<MTLBuffer>> buffers,
                       uint32_t count, const void *args, size_t argsSize) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error(error.description.UTF8String);
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            for (NSUInteger i=0; i<buffers.size(); ++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
            if (args && argsSize) [enc setBytes:args length:argsSize atIndex:buffers.size()];
            [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        try {
            // DetailMW::get_norm_detail uses max(abs(detail))/globalMax per
            // field. Verify mixed scales and the explicit zero-normalizer rule.
            constexpr uint32_t n = 5;
            auto details = make(4*n*sizeof(float)), maxima = make(4*sizeof(float));
            auto normalized = make(n*sizeof(float));
            float *d = (float *)details.contents, *m = (float *)maxima.contents;
            for (unsigned i=0;i<4*n;++i) d[i]=0;
            m[0]=2; m[1]=0; m[2]=4; m[3]=0;
            d[0*n+0]=-1; d[1*n+1]=0; d[1*n+2]=0.25f;
            d[1*n+3]=2; d[2*n+4]=3; d[3*n+4]=0.0f;
            run(@"mwdg2_normalize_detail_maxima", {details,maxima,normalized}, n, &n, sizeof(n));
            float *norm=(float *)normalized.contents;
            if (fabs(norm[0]-.5f)>1e-7f || norm[1]!=0 || !std::isinf(norm[2]) ||
                !std::isinf(norm[3]) || fabs(norm[4]-.75f)>1e-7f) return 1;
            printf("PASS MWDG2 detail normalization including zero maxima\n");

            // CUDA's epsilon scaling and preflag OR. epsilon_local = .8/2=.4.
            auto thresholdNorm=make(16*sizeof(float)), pre=make(16), sig=make(16);
            float *tn=(float *)thresholdNorm.contents; uint8_t *pf=(uint8_t *)pre.contents;
            uint8_t *sg=(uint8_t *)sig.contents;
            for(unsigned i=0;i<16;++i){tn[i]=0;pf[i]=0;sg[i]=0;}
            tn[8]=.399f; tn[9]=.4f; tn[10]=.8f; pf[8]=1;
            LevelArgs la{8,4,2,3,.8f};
            run(@"mwdg2_threshold_level",{thresholdNorm,pre,sig},4,&la,sizeof(la));
            if(sg[8]!=1 || sg[9]!=1 || sg[10]!=1 || sg[11]!=0) return 1;
            printf("PASS CUDA-equivalent level threshold and preflags\n");

            // Upward closure: each parent is ORed with its four children.
            auto reg=make(15); auto *r=(uint8_t *)reg.contents;
            for(unsigned i=0;i<15;++i)r[i]=0;
            r[3]=1; r[11]=1; // hit parent 0 and parent 2
            RegulariseArgs ra{0,3,3};
            run(@"mwdg2_regularise_upward",{reg},3,&ra,sizeof(ra));
            if(r[0]!=1 || r[1]!=0 || r[2]!=1) return 1;
            printf("PASS upward significance regularisation\n");

            // CUDA M_BAR=1.5, so extra threshold multiplier is 2^(2.5).
            auto extraSig=make(21), extraNorm=make(21*sizeof(float));
            auto *es=(uint8_t *)extraSig.contents; float *en=(float *)extraNorm.contents;
            for(unsigned i=0;i<21;++i){es[i]=0;en[i]=0;}
            es[1]=es[2]=1; en[1]=1.2f; en[2]=1.1f;
            ExtraArgs ea{1,5,2,1,3,.8f,std::pow(2.0f,2.5f)};
            run(@"mwdg2_propagate_extra_significance",{extraSig,extraNorm},2,&ea,sizeof(ea));
            for(unsigned i=5;i<9;++i) if(es[i]!=1) return 1;
            for(unsigned i=9;i<13;++i) if(es[i]!=0) return 1;
            printf("PASS CUDA M_BAR extra-significance propagation\n");

            // Decode topology from root through two levels. A split node is
            // reachable as a child; only its own significance opens children.
            auto topology=make(21), reachable=make(21), leaves=make(21);
            auto *ts=(uint8_t *)topology.contents,*tr=(uint8_t *)reachable.contents;
            auto *tl=(uint8_t *)leaves.contents;
            for(unsigned i=0;i<21;++i){ts[i]=tr[i]=tl[i]=0;}
            ts[0]=1; ts[1]=1; ts[5]=1; tr[0]=1;
            DecodeArgs da1{0,1,4};
            run(@"mwdg2_decode_topology_level",{topology,reachable},4,&da1,sizeof(da1));
            DecodeArgs da2{1,5,16};
            run(@"mwdg2_decode_topology_level",{topology,reachable},16,&da2,sizeof(da2));
            LeafArgs leafArgs{21,5};
            run(@"mwdg2_mark_reachable_leaves",{topology,reachable,leaves},21,&leafArgs,sizeof(leafArgs));
            unsigned leafCount=0;
            for(unsigned i=0;i<21;++i) leafCount += tl[i]!=0;
            if (leafCount!=7 || tr[0]!=1 || tr[1]!=1 || tr[2]!=1 || tr[3]!=1 ||
                tr[4]!=1 || tr[5]!=1 || tr[8]!=1 || tr[9]!=0 || tl[0]!=0 || tl[1]!=0 ||
                tl[2]!=1 || tl[4]!=1 || tl[5]!=1 || tl[8]!=1) return 1;
            printf("PASS decode traversal and reachable-leaf topology\n");
        } catch (const std::exception &e) {
            fprintf(stderr,"MWDG2 selection test failed: %s\n",e.what()); return 1;
        }
    }
    return 0;
}
