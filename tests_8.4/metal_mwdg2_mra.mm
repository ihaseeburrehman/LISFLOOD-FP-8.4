#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        auto device = MTLCreateSystemDefaultDevice();
        auto queue = [device newCommandQueue];
        NSError *error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]]] error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        auto make = [&](size_t bytes) { return [device newBufferWithLength:bytes
            options:MTLResourceStorageModeShared]; };
        auto dispatch = [&](NSString *name, std::vector<id<MTLBuffer>> buffers,
                            unsigned count, std::vector<unsigned> constants = {}) {
            auto pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:name] error:&error];
            if (!pipeline) throw std::runtime_error("missing MRA pipeline");
            auto command = [queue commandBuffer];
            auto enc = [command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            for (unsigned i = 0; i < buffers.size(); ++i)
                [enc setBuffer:buffers[i] offset:0 atIndex:i];
            if (constants.empty()) constants.push_back(count);
            [enc setBytes:constants.data() length:constants.size()*4 atIndex:buffers.size()];
            [enc dispatchThreadgroups:MTLSizeMake((count + 127)/128, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        auto dispatchFlow = [&](id<MTLBuffer> scales, id<MTLBuffer> active,
                                id<MTLBuffer> neighbours, id<MTLBuffer> rhs,
                                unsigned count, unsigned level, float dx, float dy) {
            auto pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"mwdg2_rhs"] error:&error];
            if (!pipeline) throw std::runtime_error("missing MWDG2 flow pipeline");
            struct { unsigned count, level; } layout{count, level};
            struct { float dx, dy, gravity, tolerance; } physics{dx, dy, 9.81f, 1e-6f};
            auto command = [queue commandBuffer];
            auto enc = [command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:scales offset:0 atIndex:0];
            [enc setBuffer:active offset:0 atIndex:1];
            [enc setBuffer:neighbours offset:0 atIndex:2];
            [enc setBuffer:rhs offset:0 atIndex:3];
            [enc setBuffer:scales offset:0 atIndex:4];
            [enc setBuffer:make(count * sizeof(float)) offset:0 atIndex:5];
            [enc setBuffer:scales offset:0 atIndex:6];
            [enc setBuffer:make(count * sizeof(unsigned)) offset:0 atIndex:7];
            [enc setBytes:&layout length:sizeof(layout) atIndex:8];
            [enc setBytes:&physics length:sizeof(physics) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((count + 63) / 64, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        // mwdg2_rhs consumes the production CSR neighbour format (four face
        // offsets followed by uint4 segments). mwdg2_leaf_neighbours is a
        // separate fixed-slot diagnostic kernel; adapt its eight quarter-face
        // IDs before passing those results to the flow operator.
        auto packFlowNeighbours = [&](id<MTLBuffer> quarter, unsigned count) {
            unsigned recordsStart=(count*4+4)&~3u;
            auto packed=make((recordsStart+count*4*4)*sizeof(unsigned));
            auto *src=(unsigned *)quarter.contents;
            auto *dst=(unsigned *)packed.contents;
            unsigned segment=0;
            for(unsigned i=0;i<count;++i) for(unsigned face=0;face<4;++face) {
                dst[i*4+face]=segment;
                unsigned other=src[i*8+face*2];
                dst[recordsStart+segment*4]=other;
                dst[recordsStart+segment*4+1]=0;
                dst[recordsStart+segment*4+2]=1;
                dst[recordsStart+segment*4+3]=0;
                ++segment;
            }
            dst[count*4]=segment;
            return packed;
        };
        auto dispatchRasterImport = [&](id<MTLBuffer> raster, id<MTLBuffer> scales,
                                        unsigned width, unsigned height,
                                        unsigned side, unsigned offset) {
            auto pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"mwdg2_import_raster"] error:&error];
            if (!pipeline) throw std::runtime_error("missing MWDG2 raster import pipeline");
            unsigned geometry[4] = {width,height,side,offset};
            auto command = [queue commandBuffer];
            auto enc = [command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:raster offset:0 atIndex:0];
            [enc setBuffer:scales offset:0 atIndex:1];
            [enc setBytes:geometry length:sizeof(geometry) atIndex:2];
            unsigned count=width*height;
            [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];
            [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        auto dispatchProject = [&](id<MTLBuffer> scales, id<MTLBuffer> owners,
                                   id<MTLBuffer> output, unsigned width,
                                   unsigned height, unsigned side, unsigned level) {
            auto pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"mwdg2_project_raster"] error:&error];
            if (!pipeline) throw std::runtime_error("missing MWDG2 raster projection pipeline");
            unsigned geometry[4]={width,height,side,level};
            auto command=[queue commandBuffer]; auto enc=[command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:scales offset:0 atIndex:0];
            [enc setBuffer:owners offset:0 atIndex:1];
            [enc setBuffer:output offset:0 atIndex:2];
            [enc setBytes:geometry length:sizeof(geometry) atIndex:3];
            unsigned count=width*height;
            [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding]; [command commit]; [command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        auto dispatchCFL = [&](id<MTLBuffer> scales,id<MTLBuffer> active,
                               id<MTLBuffer> localDt,unsigned count,unsigned level,
                               float dx,float dy,float gravity,float cfl,
                               float threshold,float maxDt) {
            auto pipeline=[device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"mwdg2_cfl"] error:&error];
            if(!pipeline) throw std::runtime_error("missing MWDG2 CFL pipeline");
            unsigned layout[2]={count,level}; float physics[4]={dx,dy,gravity,cfl};
            auto command=[queue commandBuffer]; auto enc=[command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:scales offset:0 atIndex:0];
            [enc setBuffer:active offset:0 atIndex:1];
            [enc setBuffer:localDt offset:0 atIndex:2];
            [enc setBytes:layout length:sizeof(layout) atIndex:3];
            [enc setBytes:physics length:sizeof(physics) atIndex:4];
            [enc setBytes:&threshold length:sizeof(threshold) atIndex:5];
            [enc setBytes:&maxDt length:sizeof(maxDt) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding]; [command commit]; [command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        std::mt19937 rng(84);
        std::uniform_real_distribution<float> random(-1, 1);
        for (unsigned count : {1u, 257u, 65537u}) {
            auto input = make(count * 12 * 4), encoded = make(count * 12 * 4);
            auto output = make(count * 12 * 4);
            auto a = (float *)input.contents, b = (float *)encoded.contents;
            auto c = (float *)output.contents;
            for (unsigned i = 0; i < count * 12; ++i) a[i] = random(rng);
            dispatch(@"mwdg2_encode", {input, encoded}, count);
            dispatch(@"mwdg2_decode", {encoded, output}, count);
            double maximum = 0;
            for (unsigned i = 0; i < count * 12; ++i)
                maximum = fmax(maximum, fabs(a[i] - c[i]));
            if (maximum > 2e-6) return 1;
            // Parent mean equals the area-weighted child mean, regardless
            // of slopes. This independently checks conservative restriction.
            for (unsigned i = 0; i < count; ++i) {
                double mean = 0;
                for (unsigned k = 0; k < 4; ++k) mean += a[i*12+k*3] * .25;
                if (fabs(mean - b[i*12]) > 2e-7) return 1;
            }
            printf("PASS Metal MWDG2 transform: %u parents, max round-trip %.3g\n", count, maximum);
        }
        // A linear polynomial must have zero details and survive coarsening.
        auto input = make(48), encoded = make(48), output = make(48);
        auto a = (float *)input.contents, b = (float *)encoded.contents;
        for (unsigned child = 0; child < 4; ++child) {
            a[child*3] = 2 + ((child&1) ? 1 : -1)*.25f + ((child&2) ? 1 : -1)*.5f;
            a[child*3+1] = .25f / sqrtf(3);
            a[child*3+2] = .5f / sqrtf(3);
        }
        dispatch(@"mwdg2_encode", {input, encoded}, 1);
        for (unsigned i = 3; i < 12; ++i) if (fabs(b[i]) > 1e-6) return 1;
        printf("PASS Metal MWDG2 linear-polynomial details vanish\n");
        // A uniform still-water state on a flat bed must have no DG residual.
        auto flow = make(12 * sizeof(float)), activeFlow = make(sizeof(unsigned));
        auto neighboursFlow = make(8 * sizeof(unsigned)), rhsFlow = make(9 * sizeof(float));
        auto state = (float *)flow.contents;
        for (unsigned i = 0; i < 12; ++i) state[i] = 0;
        state[0] = 2.0f;
        *(unsigned *)activeFlow.contents = 0;
        for (unsigned i = 0; i < 8; ++i) ((unsigned *)neighboursFlow.contents)[i] = ~0u;
        dispatchFlow(flow, activeFlow, neighboursFlow, rhsFlow, 1, 0, 10.0f, 10.0f);
        for (unsigned i = 0; i < 9; ++i)
            if (fabs(((float *)rhsFlow.contents)[i]) > 2e-6) return 1;
        printf("PASS Metal MWDG2 flow residual: uniform still water\n");
        // The same lake-at-rest invariant must hold over a planar bed when
        // depth has the opposite modal slope and the free surface is flat.
        state[0] = 3.0f; state[1] = 0.0f; state[2] = 0.0f;
        state[9] = 1.0f; state[10] = 0.1f; state[11] = 0.2f;
        dispatchFlow(flow, activeFlow, neighboursFlow, rhsFlow, 1, 0, 10.0f, 10.0f);
        float lakeResidual = 0;
        for (unsigned i = 0; i < 9; ++i)
            lakeResidual = fmaxf(lakeResidual, fabsf(((float *)rhsFlow.contents)[i]));
        if (lakeResidual > 2e-5f) {
            fprintf(stderr, "planar lake-at-rest residual %.8g\n", lakeResidual);
            return 1;
        }
        printf("PASS Metal MWDG2 hydrostatic balance: planar lake at rest\n");
        // ESRI row order is north-to-south, while Morton y increases north.
        // Verify field-major cell-average projection and zero initial slopes.
        auto importRaster=make(4*4*sizeof(float)), importScales=make(21*12*sizeof(float));
        float *ir=(float *)importRaster.contents,*is=(float *)importScales.contents;
        for(unsigned field=0;field<4;++field) for(unsigned k=0;k<4;++k)
            ir[field*4+k]=100*field+k;
        for(unsigned k=0;k<21*12;++k) is[k]=-1;
        dispatchRasterImport(importRaster,importScales,2,2,4,5);
        auto dilate=[](unsigned value) {
            value=(value|(value<<8))&0x00ff00ffu; value=(value|(value<<4))&0x0f0f0f0fu;
            value=(value|(value<<2))&0x33333333u; return (value|(value<<1))&0x55555555u;
        };
        for(unsigned row=0;row<2;++row) for(unsigned x=0;x<2;++x) {
            unsigned y=1-row,node=5+dilate(x)+(dilate(y)<<1),source=row*2+x;
            for(unsigned field=0;field<4;++field) {
                unsigned p=node*12+field*3;
                if(is[p]!=ir[field*4+source] || is[p+1]!=0 || is[p+2]!=0) return 1;
            }
        }
        printf("PASS Metal MWDG2 raster import: ESRI row order and field layout\n");
        auto projectScales=make(21*12*sizeof(float)),projectFlags=make(21*sizeof(unsigned));
        auto projectOwners=make(4*sizeof(unsigned)),projectOut=make(4*4*sizeof(float));
        auto pro=(float *)projectScales.contents;
        for(unsigned i=0;i<21*12;++i) pro[i]=0;
        for(unsigned i=0;i<21;++i) ((unsigned *)projectFlags.contents)[i]=0;
        ((unsigned *)projectFlags.contents)[0]=1;
        dispatch(@"mwdg2_owner_map",{projectFlags,projectOwners},4,{1});
        for(unsigned node=1;node<=4;++node) {
            pro[node*12]=10+node; pro[node*12+3]=.1f*node;
            pro[node*12+6]=-.2f*node; pro[node*12+9]=2+node;
        }
        dispatchProject(projectScales,projectOwners,projectOut,2,2,2,1);
        const unsigned expectedNodes[4]={3,4,1,2};
        float *po=(float *)projectOut.contents;
        for(unsigned i=0;i<4;++i) {
            unsigned node=expectedNodes[i];
            if(fabs(po[i]-8)>1e-6 || fabs(po[4+i]-.1f*node)>1e-6 ||
               fabs(po[8+i]+.2f*node)>1e-6 || fabs(po[12+i]-(2+node))>1e-6) return 1;
        }
        printf("PASS Metal MWDG2 raster output: rectangular owner projection\n");
        auto cflState=make(12*sizeof(float)),cflActive=make(sizeof(unsigned)),cflOut=make(sizeof(float));
        float *cu=(float *)cflState.contents;
        for(unsigned i=0;i<12;++i) cu[i]=0;
        cu[0]=2; cu[3]=4; *(unsigned *)cflActive.contents=0;
        dispatchCFL(cflState,cflActive,cflOut,1,0,3,4,10,.5f,1e-6f,10);
        float expectedDt=.5f*3.0f/(2.0f+sqrtf(20.0f));
        if(fabs(*(float *)cflOut.contents-expectedDt)>1e-6) return 1;
        printf("PASS Metal MWDG2 CFL timestep from leaf mean state\n");
        // The implicit Manning source must preserve momentum when n=0 and
        // dissipate it for positive roughness, including modal slopes.
        auto frictionState=make(12*sizeof(float));
        auto frictionActive=make(sizeof(unsigned));
        auto frictionN=make(sizeof(float));
        float *fu=(float *)frictionState.contents;
        for(unsigned i=0;i<12;++i) fu[i]=0;
        fu[0]=2.0f; fu[1]=0.1f; fu[2]=-0.05f;
        fu[3]=1.4f; fu[4]=0.2f; fu[5]=0.1f;
        fu[6]=-0.8f; fu[7]=-0.1f; fu[8]=0.2f;
        *(unsigned *)frictionActive.contents=0;
        auto dispatchFriction=[&](float manning) {
            *(float *)frictionN.contents=manning;
            auto pipeline=[device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"mwdg2_friction"] error:&error];
            if(!pipeline) throw std::runtime_error("missing MWDG2 friction pipeline");
            unsigned count=1; float params[4]={9.81f,0.25f,1e-6f,1e-6f};
            auto command=[queue commandBuffer]; auto enc=[command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:frictionState offset:0 atIndex:0];
            [enc setBuffer:frictionActive offset:0 atIndex:1];
            [enc setBuffer:frictionN offset:0 atIndex:2];
            [enc setBytes:&count length:sizeof(count) atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc endEncoding]; [command commit]; [command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        float qxBefore=fu[3],qyBefore=fu[6];
        dispatchFriction(0.0f);
        if(fabs(fu[3]-qxBefore)>1e-6 || fabs(fu[6]-qyBefore)>1e-6) return 1;
        auto frictionReference=[](float h,float qx,float qy,float n) {
            if(h<1e-6f) return std::pair<float,float>{0,0};
            float ux=qx/h,uy=qy/h;
            if(fabs(ux)<1e-6f && fabs(uy)<1e-6f) return std::pair<float,float>{0,0};
            float cf=9.81f*n*n/std::pow(h,1.0f/3.0f);
            float speed=std::sqrt(ux*ux+uy*uy);
            float sx=-cf*ux*speed,sy=-cf*uy*speed;
            float dx=1+0.25f*cf/h*(2*ux*ux+uy*uy)/speed;
            float dy=1+0.25f*cf/h*(2*uy*uy+ux*ux)/speed;
            return std::pair<float,float>{qx+0.25f*sx/dx,qy+0.25f*sy/dy};
        };
        auto q0Ref=frictionReference(2.0f,1.4f,-0.8f,.03f);
        auto qxmRef=frictionReference(1.9f,1.2f,-.7f,.03f);
        auto qxpRef=frictionReference(2.1f,1.6f,-.9f,.03f);
        auto qymRef=frictionReference(2.05f,1.3f,-1.0f,.03f);
        auto qypRef=frictionReference(1.95f,1.5f,-.6f,.03f);
        dispatchFriction(0.03f);
        if(fabs(fu[3]-q0Ref.first)>2e-6 || fabs(fu[6]-q0Ref.second)>2e-6 ||
           fabs(fu[4]-.5f*(qxpRef.first-qxmRef.first))>2e-6 ||
           fabs(fu[7]-.5f*(qxpRef.second-qxmRef.second))>2e-6 ||
           fabs(fu[5]-.5f*(qypRef.first-qymRef.first))>2e-6 ||
           fabs(fu[8]-.5f*(qypRef.second-qymRef.second))>2e-6 ||
           !(fabs(fu[3])+fabs(fu[6]) < fabs(qxBefore)+fabs(qyBefore))) {
            fprintf(stderr,"friction got %.9g %.9g %.9g %.9g %.9g %.9g; ref %.9g %.9g %.9g %.9g %.9g %.9g\n",
                    fu[3],fu[4],fu[5],fu[6],fu[7],fu[8],q0Ref.first,.5f*(qxpRef.first-qxmRef.first),
                    .5f*(qypRef.first-qymRef.first),q0Ref.second,.5f*(qxpRef.second-qxmRef.second),
                    .5f*(qypRef.second-qymRef.second));
            return 1;
        }
        printf("PASS Metal MWDG2 implicit Manning friction: zero-n identity and positive-n dissipation\n");
        // Exercise the production pieces as one GPU sequence: import a
        // uniform raster, restrict the full hierarchy, select its root leaf,
        // build neighbours, take both SSP-RK2 stages, and project back.
        auto e2eRaster=make(4*16*sizeof(float));
        auto e2eScales=make(21*12*sizeof(float));
        auto e2eDetails=make(5*36*sizeof(float));
        auto e2eFlags=make(21*sizeof(unsigned));
        auto e2eOwners=make(16*sizeof(unsigned));
        auto e2eActive=make(sizeof(unsigned));
        auto e2eNeighbours=make(8*sizeof(unsigned));
        auto e2eInvalid=make(sizeof(unsigned));
        auto e2eRhs=make(9*sizeof(float));
        auto e2eNext=make(21*12*sizeof(float));
        auto e2eSecondRhs=make(9*sizeof(float));
        auto e2eFinal=make(21*12*sizeof(float));
        auto e2eDt=make(sizeof(float));
        auto e2eOutput=make(4*16*sizeof(float));
        float *er=(float *)e2eRaster.contents;
        for(unsigned i=0;i<16;++i) { er[i]=2.0f; er[16+i]=0; er[32+i]=0; er[48+i]=0; }
        unsigned importGeometry[4]={4,4,4,5};
        dispatch(@"mwdg2_import_raster",{e2eRaster,e2eScales},16,
                 {importGeometry[0],importGeometry[1],importGeometry[2],importGeometry[3]});
        for(int level=1;level>=0;--level) {
            unsigned n=1u<<(2*level);
            dispatch(@"mwdg2_encode_level",{e2eScales,e2eDetails},n,{(n-1)/3,n});
        }
        for(unsigned i=0;i<21;++i) ((unsigned *)e2eFlags.contents)[i]=0;
        dispatch(@"mwdg2_owner_map",{e2eFlags,e2eOwners},16,{2});
        *(unsigned *)e2eActive.contents=0;
        dispatch(@"mwdg2_leaf_neighbours",{e2eActive,e2eOwners,e2eNeighbours,e2eInvalid},1,{1,2});
        dispatchFlow(e2eScales,e2eActive,e2eNeighbours,e2eRhs,1,2,1.0f,1.0f);
        for(unsigned i=0;i<9;++i)
            if(fabs(((float *)e2eRhs.contents)[i])>2e-6f)
                throw std::runtime_error("adaptive coarse lake-at-rest residual");
        printf("PASS Metal MWDG2 integrated GPU sequence: raster to one adaptive coarse leaf\n");
        dispatchCFL(e2eScales,e2eActive,e2eDt,1,2,1,1,9.81f,.3f,1e-6f,10);
        auto runRk=[&](id<MTLBuffer> current,id<MTLBuffer> rhs,id<MTLBuffer> output,float blend) {
            auto pipeline=[device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"mwdg2_rk"] error:&error];
            if(!pipeline) throw std::runtime_error("missing MWDG2 RK pipeline");
            float stage[2]={*(float *)e2eDt.contents,blend}; unsigned count=1;
            auto command=[queue commandBuffer]; auto enc=[command computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:e2eScales offset:0 atIndex:0];
            [enc setBuffer:current offset:0 atIndex:1];
            [enc setBuffer:rhs offset:0 atIndex:2];
            [enc setBuffer:e2eActive offset:0 atIndex:3];
            [enc setBuffer:output offset:0 atIndex:4];
            [enc setBuffer:current offset:0 atIndex:5];
            [enc setBytes:&count length:sizeof(count) atIndex:6];
            [enc setBytes:stage length:sizeof(stage) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc endEncoding]; [command commit]; [command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
        };
        runRk(e2eScales,e2eRhs,e2eNext,0.0f);
        dispatchFlow(e2eNext,e2eActive,e2eNeighbours,e2eSecondRhs,1,2,1,1);
        runRk(e2eNext,e2eSecondRhs,e2eFinal,0.5f);
        dispatchProject(e2eFinal,e2eOwners,e2eOutput,4,4,4,2);
        for(unsigned i=0;i<16;++i) {
            float *eo=(float *)e2eOutput.contents;
            if(fabs(eo[i]-2.0f)>2e-6f || fabs(eo[16+i])>2e-6f ||
               fabs(eo[32+i])>2e-6f || fabs(eo[48+i])>2e-6f)
                throw std::runtime_error("adaptive coarse lake-at-rest output");
        }
        printf("PASS Metal MWDG2 integrated GPU sequence: two RK stages, CFL and raster output\n");
        // Summed depth residual over a 2x2 adaptive patch must conserve mass
        // with nonuniform depth. Momentum is zero: the CUDA-style domain edge
        // (ghost depth copied, normal discharge zeroed) passes water when the
        // edge cell has outward discharge, so it is not a closed wall.
        auto patchScales = make(21 * 12 * sizeof(float));
        auto patchFlags = make(21 * sizeof(unsigned)), patchOwners = make(16 * sizeof(unsigned));
        auto patchActive = make(4 * sizeof(unsigned));
        auto patchNeighbours = make(4 * 8 * sizeof(unsigned));
        auto patchInvalid = make(4 * sizeof(unsigned)), patchRhs = make(4 * 9 * sizeof(float));
        auto ps = (float *)patchScales.contents;
        for (unsigned i = 0; i < 21 * 12; ++i) ps[i] = 0;
        for (unsigned i = 1; i <= 4; ++i) {
            ((unsigned *)patchActive.contents)[i-1] = i;
            ps[i*12] = 1.0f + 0.2f * i;
        }
        for (unsigned i=0; i<21; ++i) ((unsigned *)patchFlags.contents)[i]=0;
        ((unsigned *)patchFlags.contents)[0]=1;
        dispatch(@"mwdg2_owner_map", {patchFlags,patchOwners}, 16, {1});
        dispatch(@"mwdg2_leaf_neighbours", {patchActive,patchOwners,patchNeighbours,patchInvalid}, 4, {4,1});
        auto patchFlowNeighbours=packFlowNeighbours(patchNeighbours,4);
        dispatchFlow(patchScales,patchActive,patchFlowNeighbours,patchRhs,4,1,1.0f,1.0f);
        double massResidual=0;
        for (unsigned i=0; i<4; ++i) massResidual += ((float *)patchRhs.contents)[i*9];
        if (fabs(massResidual)>2e-5) {
            fprintf(stderr,"closed-patch mass residual %.8g\n",massResidual);
            return 1;
        }
        printf("PASS Metal MWDG2 conservation: closed adaptive 2x2 patch\n");
        // Coarse-to-fine interfaces must conserve the area-integrated depth
        // residual when both sides use the same two-point subface flux.
        // A uniform transport state is set on every tree node, as an encoded
        // tree holds it: CUDA-style lookup reads a finer neighbour through its
        // ancestor at the reading leaf's level. The domain edge is not a wall
        // (ghost depth copied, normal discharge zeroed), so compare the total
        // against the same state on the all-finest mesh: with a uniform state
        // the edge fluxes are the same, and the interior must cancel on both.
        auto nodeSpan=[](unsigned node) {
            unsigned start=0,count=1,level=0;
            while(node>=start+count) { start+=count; count*=4; ++level; }
            return 4u>>level;
        };
        auto cfTotal=[&](bool splitChild,unsigned &leafCount) {
            auto cfScale=make(21*12*sizeof(float)),cfFlags=make(21*sizeof(unsigned));
            auto cfOwner=make(16*sizeof(unsigned)),cfSelect=make(21*sizeof(unsigned));
            auto cfActive=make(21*sizeof(unsigned)),cfNgh=make(21*8*sizeof(unsigned));
            auto cfBad=make(21*sizeof(unsigned)),cfRhs=make(21*9*sizeof(float));
            auto cf=(unsigned *)cfFlags.contents;
            for(unsigned i=0;i<21;++i) cf[i]=0;
            cf[0]=1;
            for(unsigned i=1;i<5;++i) cf[i]=splitChild ? i==1 : 1;
            dispatch(@"mwdg2_owner_map",{cfFlags,cfOwner},16,{2});
            dispatch(@"mwdg2_leaf_flags",{cfFlags,cfSelect},21);
            auto cfLeaf=(unsigned *)cfSelect.contents,cfIds=(unsigned *)cfActive.contents;
            auto cfU=(float *)cfScale.contents;
            for(unsigned i=0;i<21*12;++i) cfU[i]=0;
            for(unsigned i=0;i<21;++i) {
                cfU[i*12]=1.0f; cfU[i*12+3]=0.02f; cfU[i*12+6]=0.015f;
            }
            unsigned cfCount=0;
            for(unsigned i=0;i<21;++i) if(cfLeaf[i]) cfIds[cfCount++]=i;
            dispatch(@"mwdg2_leaf_neighbours",{cfActive,cfOwner,cfNgh,cfBad},cfCount,{cfCount,2});
            auto cfFlowNeighbours=packFlowNeighbours(cfNgh,cfCount);
            dispatchFlow(cfScale,cfActive,cfFlowNeighbours,cfRhs,cfCount,2,1.0f,1.0f);
            double total=0;
            for(unsigned i=0;i<cfCount;++i) {
                unsigned span=nodeSpan(cfIds[i]);
                total+=span*span*((float *)cfRhs.contents)[i*9];
                if(((unsigned *)cfBad.contents)[i]) throw std::runtime_error("coarse-fine neighbour lookup failed");
            }
            leafCount=cfCount;
            return total;
        };
        unsigned mixedLeaves=0,fineLeaves=0;
        double mixedTotal=cfTotal(true,mixedLeaves),fineTotal=cfTotal(false,fineLeaves);
        if(mixedLeaves!=7 || fineLeaves!=16 || fabs(mixedTotal-fineTotal)>5e-5) {
            fprintf(stderr,"coarse-fine mass residual %.8g (all-finest %.8g, leaves %u/%u)\n",
                    mixedTotal,fineTotal,mixedLeaves,fineLeaves);
            return 1;
        }
        printf("PASS Metal MWDG2 conservation: mixed coarse-fine interfaces\n");
        // Root coarsened -> one leaf; root split -> four leaves; one child
        // split -> seven leaves. All selected leaves cover the same area.
        auto splits = make(21*4), leaves = make(21*4);
        auto s = (unsigned *)splits.contents, l = (unsigned *)leaves.contents;
        for (unsigned scenario = 0; scenario < 3; ++scenario) {
            for (unsigned i = 0; i < 21; ++i) s[i] = 0;
            s[0] = scenario > 0; s[1] = scenario > 1;
            dispatch(@"mwdg2_leaf_flags", {splits, leaves}, 21);
            unsigned active = 0; double area = 0;
            for (unsigned i = 0; i < 21; ++i) if (l[i]) {
                ++active; area += i == 0 ? 1 : i < 5 ? .25 : .0625;
            }
            if (active != 1 + 3*scenario || area != 1) return 1;
            printf("PASS Metal tree selection: %u active leaves, area %.1f\n", active, area);
        }
        for (unsigned maximumLevel : {1u, 3u, 5u}) {
            unsigned finestCount = 1u << (2 * maximumLevel);
            unsigned finestOffset = (finestCount - 1) / 3;
            unsigned totalNodes = finestCount + finestOffset;
            auto scales = make(totalNodes * 12 * 4);
            auto details = make(finestOffset * 36 * 4);
            auto v = (float *)scales.contents;
            std::vector<float> expected(finestCount * 12);
            for (unsigned i = 0; i < expected.size(); ++i)
                v[finestOffset*12+i] = expected[i] = random(rng);
            for (int level = int(maximumLevel)-1; level >= 0; --level) {
                unsigned n = 1u << (2*level);
                dispatch(@"mwdg2_encode_level", {scales, details}, n, {(n-1)/3, n});
            }
            double mean = 0;
            for (unsigned i = 0; i < finestCount; ++i) mean += expected[i*12]/finestCount;
            if (fabs(mean - v[0]) > 1e-6) return 1;
            // Erase children to ensure decoding reconstructs from the root
            // and details, rather than merely observing unchanged input.
            for (unsigned i = 12; i < totalNodes * 12; ++i) v[i] = 0;
            for (unsigned level = 0; level < maximumLevel; ++level) {
                unsigned n = 1u << (2*level);
                dispatch(@"mwdg2_decode_level", {scales, details}, n, {(n-1)/3, n});
            }
            double maximum = 0;
            for (unsigned i = 0; i < expected.size(); ++i)
                maximum = fmax(maximum, fabs(expected[i] - v[finestOffset*12+i]));
            if (maximum > 4e-6) return 1;
            printf("PASS Metal hierarchy L=%u: conservative root, decode error %.3g\n", maximumLevel, maximum);

            auto flags = make(totalNodes*4), owners = make(finestCount*4);
            auto f = (unsigned *)flags.contents, o = (unsigned *)owners.contents;
            for (unsigned i = 0; i < totalNodes; ++i) f[i] = i < finestOffset;
            dispatch(@"mwdg2_owner_map", {flags, owners}, finestCount, {maximumLevel});
            for (unsigned i = 0; i < finestCount; ++i)
                if (o[i] != finestOffset + i) return 1;
            f[0] = 0;
            dispatch(@"mwdg2_owner_map", {flags, owners}, finestCount, {maximumLevel});
            for (unsigned i = 0; i < finestCount; ++i) if (o[i] != 0) return 1;
            printf("PASS Metal owner map L=%u: fully refined and fully coarsened\n", maximumLevel);
        }
        // Independently enumerate all adjacent finest cells. This checks
        // coarse/fine interfaces in each direction, including missing edges.
        for (unsigned level : {2u, 4u}) {
            unsigned side = 1u << level, finest = side*side;
            unsigned offset = (finest-1)/3, nodes = offset+finest;
            auto flags = make(nodes*4), owners = make(finest*4);
            auto selected = make(nodes*4), active = make(nodes*4);
            auto neighbours = make(nodes*8*4), invalid = make(nodes*4);
            auto f = (unsigned *)flags.contents;
            for (unsigned i=0; i<nodes; ++i) f[i]=0;
            f[0]=f[1]=1;
            // L=4 deliberately has a deeply refined patch against a coarse
            // neighbour; the diagnostic must detect the invalid topology.
            if (level==4) f[8]=f[33]=1;
            dispatch(@"mwdg2_owner_map", {flags, owners}, finest, {level});
            dispatch(@"mwdg2_leaf_flags", {flags, selected}, nodes);
            unsigned n=0;
            auto ids=(unsigned *)active.contents;
            for (unsigned i=0; i<nodes; ++i)
                if (((unsigned *)selected.contents)[i]) ids[n++]=i;
            dispatch(@"mwdg2_leaf_neighbours", {active,owners,neighbours,invalid}, n, {n,level});
            auto contract=[](unsigned code) {
                unsigned r=0;
                for (unsigned bit=0; bit<16; ++bit) r |= ((code>>(2*bit))&1)<<bit;
                return r;
            };
            auto morton=[](unsigned x,unsigned y) {
                unsigned r=0;
                for (unsigned bit=0; bit<16; ++bit) r |= ((x>>bit)&1)<<(2*bit) | ((y>>bit)&1)<<(2*bit+1);
                return r;
            };
            unsigned bad=0;
            for (unsigned i=0; i<n; ++i) {
                unsigned start=0,count=1,depth=0;
                while (ids[i]>=start+count) { start+=count; count*=4; ++depth; }
                unsigned span=side>>depth,code=ids[i]-start;
                int x=contract(code)*span,y=contract(code>>1)*span;
                if (((unsigned *)invalid.contents)[i]) { ++bad; continue; }
                for (unsigned face=0; face<4; ++face)
                    for (unsigned k=0; k<span; ++k) {
                        int nx=face==1?x+span:face==3?x-1:x+k;
                        int ny=face==0?y+span:face==2?y-1:y+k;
                        unsigned expected=(nx<0||ny<0||nx>=int(side)||ny>=int(side)) ?
                            ~0u : ((unsigned *)owners.contents)[morton(nx,ny)];
                        unsigned got=((unsigned *)neighbours.contents)[i*8+face*2+(k>=span/2)];
                        if (got!=expected) throw std::runtime_error("incorrect adaptive neighbour");
                    }
            }
            if ((level==2 && bad!=0) || (level==4 && bad==0)) return 1;
            printf("PASS Metal mixed-level neighbours L=%u: %u leaves, %u unbalanced detected\n",level,n,bad);
            for (unsigned pass=0; pass<level; ++pass) {
                dispatch(@"mwdg2_grade", {owners,flags}, finest, {level});
                dispatch(@"mwdg2_owner_map", {flags,owners}, finest, {level});
            }
            dispatch(@"mwdg2_leaf_flags", {flags,selected}, nodes);
            n=0;
            double area=0;
            for (unsigned i=0; i<nodes; ++i) if (((unsigned *)selected.contents)[i]) {
                ids[n++]=i;
                unsigned start=0,count=1;
                while (i>=start+count) { start+=count; count*=4; }
                area+=1.0/count;
            }
            dispatch(@"mwdg2_leaf_neighbours", {active,owners,neighbours,invalid}, n, {n,level});
            for (unsigned i=0; i<n; ++i) if (((unsigned *)invalid.contents)[i]) return 1;
            if (area!=1) return 1;
            printf("PASS Metal automatic 2:1 grading L=%u: %u leaves, exact domain coverage\n",level,n);
        }
    }
}
