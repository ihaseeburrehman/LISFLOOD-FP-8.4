#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct LevelArgs { uint32_t offset, count, level, maxLevel; float epsilon; };
struct RegulariseArgs { uint32_t parentOffset, childOffset, parentCount; };
struct ExtraArgs {
    uint32_t parentOffset, childOffset, parentCount, level, maxLevel;
    float epsilon, extraFactor;
};
struct DecodeArgs { uint32_t parentOffset, childOffset, childCount; };
struct LeafArgs { uint32_t nodeCount, finestOffset; };

static uint32_t levelOffset(uint32_t level) {
    return (uint32_t)((((uint64_t)1 << (2 * level)) - 1) / 3);
}
static uint32_t levelCount(uint32_t level) { return 1u << (2 * level); }
static uint32_t dilate(uint32_t v) {
    v = (v | (v << 8)) & 0x00ff00ffu;
    v = (v | (v << 4)) & 0x0f0f0f0fu;
    v = (v | (v << 2)) & 0x33333333u;
    return (v | (v << 1)) & 0x55555555u;
}
static uint32_t morton(uint32_t x, uint32_t y) { return dilate(x) | (dilate(y) << 1); }
static void geometry(uint32_t node, uint32_t maxLevel, uint32_t &x, uint32_t &y,
                     uint32_t &span, uint32_t &level) {
    uint32_t start=0,count=1; level=0;
    while(node>=start+count){start+=count;count*=4;++level;}
    uint32_t code=node-start, cx=0,cy=0;
    for(uint32_t b=0;b<16;++b){cx|=((code>>(2*b))&1u)<<b;cy|=((code>>(2*b+1))&1u)<<b;}
    span=1u<<(maxLevel-level);x=cx*span;y=cy*span;
}

class Runner {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::vector<id<MTLComputePipelineState>> pipelines;
    std::vector<std::string> names;

    explicit Runner(NSString *path) {
        device=MTLCreateSystemDefaultDevice();
        if(!device) throw std::runtime_error("Metal device unavailable");
        queue=[device newCommandQueue]; NSError *error=nil;
        library=[device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if(!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    }
    id<MTLBuffer> buffer(NSUInteger bytes) {
        auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if(!b) throw std::runtime_error("Metal buffer allocation failed");
        std::memset(b.contents,0,bytes); return b;
    }
    id<MTLComputePipelineState> pipeline(const char *name) {
        for(size_t i=0;i<names.size();++i) if(names[i]==name) return pipelines[i];
        NSError *error=nil;
        id<MTLFunction> fn=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if(!fn) throw std::runtime_error(std::string("Missing Metal function: ")+name);
        auto p=[device newComputePipelineStateWithFunction:fn error:&error];
        if(!p) throw std::runtime_error(error.localizedDescription.UTF8String);
        names.emplace_back(name); pipelines.push_back(p); return p;
    }
    void run(const char *name, const std::vector<id<MTLBuffer>> &buffers,
             const void *args, size_t argBytes, uint32_t count, uint32_t width=64) {
        auto p=pipeline(name); id<MTLCommandBuffer> command=[queue commandBuffer];
        auto enc=[command computeCommandEncoder]; [enc setComputePipelineState:p];
        for(NSUInteger i=0;i<buffers.size();++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
        if(args&&argBytes) [enc setBytes:args length:argBytes atIndex:buffers.size()];
        [enc dispatchThreadgroups:MTLSizeMake((count+width-1)/width,1,1)
            threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [enc endEncoding]; [command commit]; [command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusError)
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
    void scan(const std::vector<id<MTLBuffer>> &buffers, uint32_t count) {
        uint32_t tiles=(count+255)/256;
        auto p=pipeline("tree_scan_tiles"); id<MTLCommandBuffer> command=[queue commandBuffer];
        auto enc=[command computeCommandEncoder]; [enc setComputePipelineState:p];
        for(NSUInteger i=0;i<buffers.size();++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
        [enc setBytes:&count length:sizeof(count) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(tiles,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding]; [command commit]; [command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusError) throw std::runtime_error(command.error.localizedDescription.UTF8String);
        if(tiles!=1) throw std::runtime_error("mini-case scan assumes a single tile");
    }
    void flow(id<MTLBuffer> state,id<MTLBuffer> active,id<MTLBuffer> neighbours,
              id<MTLBuffer> rhs,uint32_t count,uint32_t level,const float physics[4]) {
        auto p=pipeline("mwdg2_rhs");id<MTLCommandBuffer> command=[queue commandBuffer];
        auto enc=[command computeCommandEncoder];[enc setComputePipelineState:p];
        [enc setBuffer:state offset:0 atIndex:0];[enc setBuffer:active offset:0 atIndex:1];
        [enc setBuffer:neighbours offset:0 atIndex:2];[enc setBuffer:rhs offset:0 atIndex:3];
        [enc setBuffer:state offset:0 atIndex:4];
        id<MTLBuffer> edge=[device newBufferWithLength:count*4+4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> flags=[device newBufferWithLength:count*4+4 options:MTLResourceStorageModeShared];
        [enc setBuffer:edge offset:0 atIndex:5];[enc setBuffer:state offset:0 atIndex:6];
        [enc setBuffer:flags offset:0 atIndex:7];
        uint32_t layout[2]={count,level};[enc setBytes:layout length:sizeof(layout) atIndex:8];
        [enc setBytes:physics length:4*sizeof(float) atIndex:9];
        const float wave[4]={0,0,0,0};[enc setBytes:wave length:sizeof(wave) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
    void cfl(id<MTLBuffer> state,id<MTLBuffer> active,id<MTLBuffer> output,
             uint32_t count,uint32_t level,const float physics[4],const float control[2]) {
        auto p=pipeline("mwdg2_mini_cfl");id<MTLCommandBuffer> command=[queue commandBuffer];
        auto enc=[command computeCommandEncoder];[enc setComputePipelineState:p];
        [enc setBuffer:state offset:0 atIndex:0];[enc setBuffer:active offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        uint32_t layout[2]={count,level};[enc setBytes:layout length:sizeof(layout) atIndex:3];
        [enc setBytes:physics length:4*sizeof(float) atIndex:4];
        [enc setBytes:control length:2*sizeof(float) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
};

struct Mesh {
    id<MTLBuffer> split, owners, leafFlags, prefix, active, nodes, neighbours;
    uint32_t activeCount=0;
    std::vector<uint32_t> activeHost;
};

static Mesh buildMesh(Runner &r, id<MTLBuffer> significant, uint32_t maxLevel,
                      uint32_t nodeCount, uint32_t finestCount) {
    Mesh m;
    m.split=r.buffer(nodeCount*sizeof(uint32_t));
    auto sig=(const uint8_t *)significant.contents; auto split=(uint32_t *)m.split.contents;
    for(uint32_t i=0;i<nodeCount;++i) split[i]=sig[i]?1u:0u;
    m.owners=r.buffer(finestCount*sizeof(uint32_t));
    uint32_t level=maxLevel;
    r.run("mwdg2_owner_map",{m.split,m.owners},&level,sizeof(level),finestCount);
    m.leafFlags=r.buffer(nodeCount*sizeof(uint32_t));
    r.run("mwdg2_leaf_flags",{m.split,m.leafFlags},&nodeCount,sizeof(nodeCount),nodeCount);
    m.prefix=r.buffer(nodeCount*sizeof(uint32_t));
    auto sums=r.buffer(sizeof(uint32_t));
    r.scan({m.leafFlags,m.prefix,sums},nodeCount);
    m.activeCount=*(uint32_t *)sums.contents;
    if(!m.activeCount || m.activeCount>finestCount) throw std::runtime_error("invalid selected leaf count");
    m.active=r.buffer(m.activeCount*sizeof(uint32_t));
    m.nodes=r.buffer(nodeCount*sizeof(uint32_t));
    for(uint32_t i=0;i<nodeCount;++i)((uint32_t *)m.nodes.contents)[i]=i;
    r.run("tree_compact_nodes",{m.leafFlags,m.prefix,m.nodes,m.active},&nodeCount,sizeof(nodeCount),nodeCount);
    m.activeHost.assign((uint32_t *)m.active.contents,(uint32_t *)m.active.contents+m.activeCount);
    // Match the production ungraded CUDA topology. Each face stores its
    // neighbor at the coarser of the two face levels, in packed CSR form.
    uint32_t faceCount=m.activeCount*4, recordsStart=(faceCount+4u)&~3u;
    std::vector<uint32_t> offsets(recordsStart,0),records;
    auto owner=(const uint32_t *)m.owners.contents;
    for(uint32_t i=0;i<m.activeCount;++i) {
        uint32_t x0,y0,span,leafLevel; geometry(m.activeHost[i],maxLevel,x0,y0,span,leafLevel);
        for(uint32_t face=0;face<4;++face) {
            offsets[i*4+face]=uint32_t(records.size()/4);
            int x=int(x0+span/2),y=int(y0+span/2);
            if(face==0)y=int(y0+span); if(face==1)x=int(x0+span);
            if(face==2)y=int(y0)-1; if(face==3)x=int(x0)-1;
            uint32_t id=0xffffffffu;
            if(x>=0&&y>=0&&x<int(1u<<maxLevel)&&y<int(1u<<maxLevel)) {
                id=owner[morton(uint32_t(x),uint32_t(y))];
                uint32_t nx,ny,nspan,nlevel; geometry(id,maxLevel,nx,ny,nspan,nlevel);
                while(nlevel>leafLevel) { id=(id-1u)/4u; geometry(id,maxLevel,nx,ny,nspan,nlevel); }
            }
            records.push_back(id); records.push_back(0); records.push_back(span); records.push_back(0);
        }
    }
    offsets[faceCount]=uint32_t(records.size()/4);
    m.neighbours=r.buffer((offsets.size()+records.size())*sizeof(uint32_t));
    auto packed=(uint32_t *)m.neighbours.contents;
    std::memcpy(packed,offsets.data(),offsets.size()*sizeof(uint32_t));
    std::memcpy(packed+offsets.size(),records.data(),records.size()*sizeof(uint32_t));
    return m;
}

static void encodeHierarchy(Runner &r, id<MTLBuffer> scales, id<MTLBuffer> details,
                            uint32_t maxLevel) {
    for(int level=(int)maxLevel-1;level>=0;--level){
        uint32_t offset=levelOffset(level), count=levelCount(level), args[2]={offset,count};
        r.run("mwdg2_encode_level",{scales,details},args,sizeof(args),count);
    }
}

static Mesh selectMesh(Runner &r,id<MTLBuffer> scales,id<MTLBuffer> details,
                       id<MTLBuffer> preflags,uint32_t maxLevel,uint32_t nodeCount,
                       uint32_t finestCount,float epsilon) {
    encodeHierarchy(r,scales,details,maxLevel);
    uint32_t internalCount=levelOffset(maxLevel),finestOffset=internalCount;
    const float *s=(const float *)scales.contents,*d=(const float *)details.contents;
    float normalizers[4]={0,0,0,0};
    for(uint32_t code=0;code<finestCount;++code){
        uint32_t node=finestOffset+code;
        for(uint32_t f=0;f<4;++f) normalizers[f]=std::max(normalizers[f],std::fabs(s[node*12+f*3]));
    }
    auto maxima=r.buffer(4*internalCount*sizeof(float));
    float *cm=(float *)maxima.contents;
    for(uint32_t n=0;n<internalCount;++n) for(uint32_t f=0;f<4;++f){
        float v=0;for(uint32_t k=0;k<9;++k)v=std::max(v,std::fabs(d[n*36+f*9+k]));
        cm[f*internalCount+n]=v;
    }
    auto norm=r.buffer(nodeCount*sizeof(float));
    auto norms=r.buffer(4*sizeof(float)); std::copy(normalizers,normalizers+4,(float *)norms.contents);
    r.run("mwdg2_normalize_detail_maxima",{maxima,norms,norm},&internalCount,sizeof(internalCount),internalCount);
    auto sig=r.buffer(nodeCount); auto pre=r.buffer(nodeCount);
    std::memcpy(pre.contents,preflags.contents,nodeCount);
    for(int level=(int)maxLevel-1;level>=0;--level){
        LevelArgs a{levelOffset(level),levelCount(level),(uint32_t)level,maxLevel,epsilon};
        r.run("mwdg2_threshold_level",{norm,pre,sig},&a,sizeof(a),a.count);
    }
    for(int level=(int)maxLevel-1;level>=0;--level){
        RegulariseArgs a{levelOffset(level),levelOffset(level+1),levelCount(level)};
        r.run("mwdg2_regularise_upward",{sig},&a,sizeof(a),a.parentCount);
    }
    for(uint32_t level=0;level+1<maxLevel;++level){
        ExtraArgs a{levelOffset(level),levelOffset(level+1),levelCount(level),level,maxLevel,
                    epsilon,std::pow(2.0f,2.5f)};
        r.run("mwdg2_propagate_extra_significance",{sig,norm},&a,sizeof(a),a.parentCount);
    }
    auto reachable=r.buffer(nodeCount); ((uint8_t *)reachable.contents)[0]=1;
    for(uint32_t level=0;level<maxLevel;++level){
        DecodeArgs a{levelOffset(level),levelOffset(level+1),4*levelCount(level)};
        r.run("mwdg2_decode_topology_level",{sig,reachable},&a,sizeof(a),a.childCount);
    }
    auto reachableLeaves=r.buffer(nodeCount);
    LeafArgs leafArgs{nodeCount,finestOffset};
    r.run("mwdg2_mark_reachable_leaves",{sig,reachable,reachableLeaves},&leafArgs,sizeof(leafArgs),nodeCount);
    auto mesh=buildMesh(r,sig,maxLevel,nodeCount,finestCount);
    uint32_t explicitLeaves=0;auto reachableFlags=(const uint8_t *)reachableLeaves.contents;
    for(uint32_t n=0;n<nodeCount;++n)explicitLeaves+=reachableFlags[n]!=0;
    if(explicitLeaves!=mesh.activeCount)throw std::runtime_error("split and reachable leaf sets disagree");
    return mesh;
}

static double waterMass(const Mesh &mesh,const float *hydro,uint32_t maxLevel) {
    double total=0;
    for(uint32_t node:mesh.activeHost){uint32_t x,y,span,level;geometry(node,maxLevel,x,y,span,level);
        total+=(double)(hydro[node*12]-hydro[node*12+9])*(double)span*(double)span;}
    return total;
}

static void runRK(Runner &r,id<MTLBuffer> initial,id<MTLBuffer> current,id<MTLBuffer> rhs,
                  id<MTLBuffer> active,id<MTLBuffer> output,uint32_t count,float dt,float blend) {
    auto p=r.pipeline("mwdg2_rk");id<MTLCommandBuffer> command=[r.queue commandBuffer];
    auto enc=[command computeCommandEncoder];[enc setComputePipelineState:p];
    [enc setBuffer:initial offset:0 atIndex:0];[enc setBuffer:current offset:0 atIndex:1];
    [enc setBuffer:rhs offset:0 atIndex:2];[enc setBuffer:active offset:0 atIndex:3];
    [enc setBuffer:output offset:0 atIndex:4];[enc setBuffer:current offset:0 atIndex:5];
    [enc setBytes:&count length:4 atIndex:6];
    float stage[2]={dt,blend};[enc setBytes:stage length:sizeof(stage) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1)
        threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(command.error.localizedDescription.UTF8String);
}

int main(int argc,char **argv) {
@autoreleasepool {
    if(argc!=2){fprintf(stderr,"usage: %s combined.metallib\n",argv[0]);return 2;}
    try {
        Runner r([NSString stringWithUTF8String:argv[1]]);
        constexpr uint32_t L=3, side=1u<<L, cells=side*side;
        constexpr uint32_t finestOffset=(cells-1)/3,nodeCount=finestOffset+cells;
        constexpr float gravity=9.81f,tolerance=1e-6f,epsilon=0.5f;
        std::vector<float> raster(4*cells,0.0f);
        for(uint32_t y=0;y<side;++y)for(uint32_t x=0;x<side;++x){
            uint32_t p=y*side+x; bool wet=(x>=1&&x<=4&&y>=1&&y<=4);
            raster[p]=wet?(0.12f+0.015f*float((x+2*y)%4)):0.0f;
            raster[cells+p]=wet?0.006f:0.0f;
            raster[2*cells+p]=wet?0.002f:0.0f;
            raster[3*cells+p]=0.002f*float(x)+0.001f*float(y);
        }
        auto input=r.buffer(raster.size()*sizeof(float));
        std::memcpy(input.contents,raster.data(),raster.size()*sizeof(float));
        auto projected=r.buffer(cells*12*sizeof(float));
        uint32_t inputGeometry[4]={side,side,0,0};
        r.run("mwdg2_project_input",{input,projected},inputGeometry,sizeof(inputGeometry),cells);
        auto scales=r.buffer(nodeCount*12*sizeof(float));
        const float *rowModes=(const float *)projected.contents; float *sv=(float *)scales.contents;
        for(uint32_t y=0;y<side;++y)for(uint32_t x=0;x<side;++x){
            uint32_t src=(y*side+x)*12,dst=(finestOffset+morton(x,y))*12;
            std::copy(rowModes+src,rowModes+src+12,sv+dst);
        }
        auto preflags=r.buffer(nodeCount); auto pf=(uint8_t *)preflags.contents;
        const float *initialFine=(const float *)scales.contents;
        for(uint32_t node=0;node<finestOffset;++node){
            uint32_t x0,y0,span,level;geometry(node,L,x0,y0,span,level);
            bool wet=false,dry=false;
            for(uint32_t y=y0;y<y0+span;++y)for(uint32_t x=x0;x<x0+span;++x){
                uint32_t fine=finestOffset+morton(x,y);
                bool current=initialFine[fine*12]-initialFine[fine*12+9]>=tolerance;
                wet|=current;dry|=!current;
            }
            pf[node]=(wet&&dry)?1:0;
        }
        auto details=r.buffer(nodeCount*36*sizeof(float));
        Mesh mesh=selectMesh(r,scales,details,preflags,L,nodeCount,cells,epsilon);
        double area=0; for(uint32_t node:mesh.activeHost){uint32_t x,y,span,l;geometry(node,L,x,y,span,l);area+=(double)span*span/cells;}
        if(std::fabs(area-1.0)>1e-12)throw std::runtime_error("initial adaptive leaves do not cover the domain");
        if(mesh.activeCount<=1||mesh.activeCount>=cells)throw std::runtime_error(std::string("mini-case did not select a nontrivial adaptive mesh: ")+std::to_string(mesh.activeCount)+" leaves");
        uint32_t level2Leaves=0,level3Leaves=0;
        for(uint32_t node:mesh.activeHost){uint32_t x,y,span,l;geometry(node,L,x,y,span,l);level2Leaves+=l==2;level3Leaves+=l==3;}
        if(!level2Leaves||!level3Leaves)throw std::runtime_error("mini-case did not retain both coarse and finest leaves");
        printf("PASS raster projection + CUDA-style MRA selection: %u leaves / %u finest cells, area %.0f%%\n",
               mesh.activeCount,cells,100.0*area);
        printf("PASS nonuniform adaptive resolution: %u level-2 and %u level-3 leaves\n",level2Leaves,level3Leaves);

        // The CUDA-compatible MWDG2 state is eta=h+z; mwdg2_rhs and the
        // positivity kernel recover depth as eta-z internally.
        id<MTLBuffer> hydro=scales;
        const float *hv=(const float *)hydro.contents;
        auto cfls=r.buffer(mesh.activeCount*sizeof(float));
        uint32_t flowLayout[2]={mesh.activeCount,L};
        float cflPhysics[4]={1.0f,1.0f,gravity,0.0f},cflControl[2]={0.45f,tolerance};
        r.cfl(hydro,mesh.active,cfls,mesh.activeCount,L,cflPhysics,cflControl);
        float dt=INFINITY;const float *dtv=(const float *)cfls.contents;
        for(uint32_t i=0;i<mesh.activeCount;++i)dt=std::min(dt,dtv[i]);
        if(!std::isfinite(dt)||dt<=0)throw std::runtime_error("CFL reduction produced invalid timestep");
        dt=std::min(dt,0.005f);
        double massBefore=waterMass(mesh,hv,L);

        auto initial=r.buffer(nodeCount*12*sizeof(float));
        auto stage1=r.buffer(nodeCount*12*sizeof(float));
        auto stage2=r.buffer(nodeCount*12*sizeof(float));
        auto rhs1=r.buffer(mesh.activeCount*9*sizeof(float));
        auto rhs2=r.buffer(mesh.activeCount*9*sizeof(float));
        auto invalid=r.buffer(mesh.activeCount*sizeof(uint32_t));
        std::memcpy(initial.contents,hydro.contents,nodeCount*12*sizeof(float));
        float physics[4]={1.0f,1.0f,gravity,tolerance};
        r.flow(hydro,mesh.active,mesh.neighbours,rhs1,mesh.activeCount,L,physics);
        runRK(r,initial,hydro,rhs1,mesh.active,stage1,mesh.activeCount,dt,0.0f);
        uint32_t positivityArgs=mesh.activeCount;
        float positivityTolerance=tolerance;
        // Positivity has two constants after its three buffers.
        auto runPositivity=[&](id<MTLBuffer> state){
            auto p=r.pipeline("mwdg2_positivity");id<MTLCommandBuffer> command=[r.queue commandBuffer];
            auto enc=[command computeCommandEncoder];[enc setComputePipelineState:p];
            [enc setBuffer:state offset:0 atIndex:0];[enc setBuffer:mesh.active offset:0 atIndex:1];
            [enc setBuffer:invalid offset:0 atIndex:2];[enc setBytes:&positivityArgs length:4 atIndex:3];
            [enc setBytes:&positivityTolerance length:4 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((mesh.activeCount+63)/64,1,1)
                threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [enc endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(command.error.localizedDescription.UTF8String);
        };
        runPositivity(stage1);
        r.flow(stage1,mesh.active,mesh.neighbours,rhs2,mesh.activeCount,L,physics);
        runRK(r,initial,stage1,rhs2,mesh.active,stage2,mesh.activeCount,dt,0.5f);
        runPositivity(stage2);
        for(uint32_t i=0;i<mesh.activeCount;++i) if(((uint32_t *)invalid.contents)[i])
            throw std::runtime_error("DG2 RK2 produced invalid/negative MWDG2 mode state");
        double massAfter=waterMass(mesh,(const float *)stage2.contents,L);
        double relativeMass=std::fabs(massAfter-massBefore)/std::max(1e-12,std::fabs(massBefore));
        if(relativeMass>3e-3)throw std::runtime_error("DG2 RK2 closed-domain mass balance exceeded tolerance");
        printf("PASS native Metal DG2 SSP-RK2 + CFL: dt %.6g, relative mass change %.3g\n",dt,relativeMass);

        uint32_t expandLayout[2]={cells,L};
        r.run("mwdg2_mini_expand_finest",{stage2,mesh.owners,scales},expandLayout,sizeof(expandLayout),cells);
        // Refresh wet/dry shoreline preflags from the post-step finest-cell
        // eta-z state before selecting the next topology.
        const float *updatedFine=(const float *)scales.contents;
        for(uint32_t node=0;node<finestOffset;++node){
            uint32_t x0,y0,span,level;geometry(node,L,x0,y0,span,level);
            bool wet=false,dry=false;
            for(uint32_t y=y0;y<y0+span;++y)for(uint32_t x=x0;x<x0+span;++x){
                uint32_t fine=finestOffset+morton(x,y);
                bool current=updatedFine[fine*12]-updatedFine[fine*12+9]>=tolerance;
                wet|=current;dry|=!current;
            }
            pf[node]=(wet&&dry)?1:0;
        }
        Mesh updated=selectMesh(r,scales,details,preflags,L,nodeCount,cells,epsilon);
        std::vector<uint32_t> oldLeaves=mesh.activeHost,newLeaves=updated.activeHost;
        std::sort(oldLeaves.begin(),oldLeaves.end());std::sort(newLeaves.begin(),newLeaves.end());
        std::vector<uint32_t> oldOnly,newOnly;
        std::set_difference(oldLeaves.begin(),oldLeaves.end(),newLeaves.begin(),newLeaves.end(),std::back_inserter(oldOnly));
        std::set_difference(newLeaves.begin(),newLeaves.end(),oldLeaves.begin(),oldLeaves.end(),std::back_inserter(newOnly));
        if(oldOnly.empty()||newOnly.empty())throw std::runtime_error("post-step MRA selection did not change active leaf IDs");
        printf("PASS post-step topology changed: %zu leaves replaced by %zu new leaves\n",oldOnly.size(),newOnly.size());
        area=0;for(uint32_t node:updated.activeHost){uint32_t x,y,span,l;geometry(node,L,x,y,span,l);area+=(double)span*span/cells;}
        if(std::fabs(area-1)>1e-12)throw std::runtime_error("regridded leaves do not cover the domain");
        auto output=r.buffer(4*cells*sizeof(float));uint32_t outputLayout[2]={cells,L};
        r.run("mwdg2_mini_project_output",{scales,output},outputLayout,sizeof(outputLayout),cells);
        const float *out=(const float *)output.contents;double outputMass=0;
        for(uint32_t i=0;i<cells;++i){
            if(!std::isfinite(out[i])||out[i]<-5e-5f||!std::isfinite(out[cells+i])) {
                char message[192];
                std::snprintf(message,sizeof(message),
                    "raster projection invalid at %u: h=%.9g eta=%.9g",
                    i,double(out[i]),double(out[cells+i]));
                throw std::runtime_error(message);
            }
            outputMass+=std::max(0.0,(double)out[i]);
        }
        double rasterRelative=std::fabs(outputMass-massAfter)/std::max(1e-12,std::fabs(massAfter));
        if(rasterRelative>4e-3)throw std::runtime_error("projected raster does not conserve volume");
        printf("PASS adaptive update/re-encode/reselect + raster output: %u leaves, raster mass error %.3g\n",
               updated.activeCount,rasterRelative);
        printf("PASS MWDG2 adaptive mini-case completed on %s\n",r.device.name.UTF8String);
        return 0;
    } catch(const std::exception &e) {fprintf(stderr,"FAIL MWDG2 adaptive mini-case: %s\n",e.what());return 1;}
}}
