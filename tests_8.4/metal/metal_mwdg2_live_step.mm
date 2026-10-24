#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../../source/metal/adaptive/metal_mwdg2_mesh.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t dilate(uint32_t v){v=(v|(v<<8))&0x00ff00ffu;v=(v|(v<<4))&0x0f0f0f0fu;v=(v|(v<<2))&0x33333333u;return(v|(v<<1))&0x55555555u;}
static uint32_t morton(uint32_t x,uint32_t y){return dilate(x)|(dilate(y)<<1);}
static constexpr uint32_t levelOffset(uint32_t l){return((1u<<(2*l))-1u)/3u;}

struct Runtime {
  id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
  std::vector<std::string> names; std::vector<id<MTLComputePipelineState>> pipes;
  explicit Runtime(NSString *path){
    device=MTLCreateSystemDefaultDevice(); if(!device)throw std::runtime_error("Metal unavailable");
    NSError *e=nil; library=[device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
    if(!library)throw std::runtime_error(e.localizedDescription.UTF8String); queue=[device newCommandQueue];
  }
  id<MTLBuffer> buffer(size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("buffer allocation failed");std::memset(b.contents,0,bytes);return b;}
  id<MTLComputePipelineState> pipeline(const char *name){
    for(size_t i=0;i<names.size();++i)if(names[i]==name)return pipes[i];
    NSError *e=nil; id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if(!f)throw std::runtime_error(std::string("missing kernel ")+name);
    auto p=[device newComputePipelineStateWithFunction:f error:&e];if(!p)throw std::runtime_error(e.localizedDescription.UTF8String);
    names.emplace_back(name);pipes.push_back(p);return p;
  }
  void run(const char *name,std::vector<id<MTLBuffer>> b,const void *args,size_t argBytes,uint32_t n,uint32_t width=64){
    auto p=pipeline(name);id<MTLCommandBuffer> c=[queue commandBuffer];auto e=[c computeCommandEncoder];[e setComputePipelineState:p];
    for(NSUInteger i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i];if(args&&argBytes)[e setBytes:args length:argBytes atIndex:b.size()];
    [e dispatchThreadgroups:MTLSizeMake((n+width-1)/width,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];
    if(c.status==MTLCommandBufferStatusError)throw std::runtime_error(c.error.localizedDescription.UTF8String);
  }
};

static double mass(const lis::metal::mwdg2::AdaptiveMesh &mesh,const float *u,uint32_t maxLevel){
  double m=0;for(uint32_t node:mesh.activeHost){uint32_t start=0,count=1,l=0;while(node>=start+count){start+=count;count*=4;++l;}uint32_t span=1u<<(maxLevel-l);m+=(u[node*12]-u[node*12+9])*double(span)*span;}return m;
}

static void rk(Runtime &r,id<MTLBuffer> initial,id<MTLBuffer> current,id<MTLBuffer> rhs,
               id<MTLBuffer> active,id<MTLBuffer> output,uint32_t count,float dt,float blend){
  uint32_t n=count;float stage[2]={dt,blend};
  // mwdg2_rk has two constant argument blocks, so encode it directly.
  auto p=r.pipeline("mwdg2_rk");id<MTLCommandBuffer> c=[r.queue commandBuffer];auto e=[c computeCommandEncoder];[e setComputePipelineState:p];
  [e setBuffer:initial offset:0 atIndex:0];[e setBuffer:current offset:0 atIndex:1];[e setBuffer:rhs offset:0 atIndex:2];[e setBuffer:active offset:0 atIndex:3];[e setBuffer:output offset:0 atIndex:4];[e setBuffer:current offset:0 atIndex:5];
  [e setBytes:&n length:4 atIndex:6];[e setBytes:stage length:sizeof(stage) atIndex:7];[e dispatchThreadgroups:MTLSizeMake((count+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];
  if(c.status==MTLCommandBufferStatusError)throw std::runtime_error(c.error.localizedDescription.UTF8String);
}

int main(int argc,char **argv){@autoreleasepool{
  if(argc!=2){std::fprintf(stderr,"usage: %s combined.metallib\n",argv[0]);return 2;}
  try{
    Runtime r([NSString stringWithUTF8String:argv[1]]);
    constexpr uint32_t L=4,side=1u<<L,cells=side*side,finest=levelOffset(L),nodes=finest+cells;
    constexpr float g=9.81f,depthTol=1e-6f,epsilon=.15f;
    id<MTLBuffer> scales=r.buffer(size_t(nodes)*12*sizeof(float));
    id<MTLBuffer> details=r.buffer(size_t(nodes)*36*sizeof(float));
    id<MTLBuffer> pre=r.buffer(nodes);
    auto *u=(float *)scales.contents;
    for(uint32_t y=0;y<side;++y)for(uint32_t x=0;x<side;++x){uint32_t p=(finest+morton(x,y))*12;bool wet=x>=4&&x<12&&y>=4&&y<12;u[p]=wet?.20f:0;u[p+3]=wet?.006f:0;u[p+6]=wet?.002f:0;u[p+9]=.0005f*x+.0002f*y;u[p]+=u[p+9];}
    lis::metal::mwdg2::AdaptiveMeshBuilder builder(r.device,r.library,r.queue);
    auto mesh=builder.select(scales,details,pre,L,nodes,epsilon);
    const float dx=1,dy=1,cfl=.35f,maxDt=.005f;
    id<MTLBuffer> dts=r.buffer(size_t(mesh.activeCount)*sizeof(float));uint32_t layout[2]={mesh.activeCount,L};float physics[4]={dx,dy,g,cfl};
    // The CFL kernel also receives physics, depth tolerance, and max dt.
    auto cp=r.pipeline("mwdg2_cfl");id<MTLCommandBuffer> cc=[r.queue commandBuffer];auto ce=[cc computeCommandEncoder];[ce setComputePipelineState:cp];
    [ce setBuffer:scales offset:0 atIndex:0];[ce setBuffer:mesh.active offset:0 atIndex:1];[ce setBuffer:dts offset:0 atIndex:2];[ce setBytes:layout length:sizeof(layout) atIndex:3];[ce setBytes:physics length:sizeof(physics) atIndex:4];[ce setBytes:&depthTol length:4 atIndex:5];[ce setBytes:&maxDt length:4 atIndex:6];[ce dispatchThreadgroups:MTLSizeMake((mesh.activeCount+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[ce endEncoding];[cc commit];[cc waitUntilCompleted];
    float dt=maxDt;auto *dtp=(float *)dts.contents;for(uint32_t i=0;i<mesh.activeCount;++i)dt=std::min(dt,dtp[i]);if(!std::isfinite(dt)||dt<=0)throw std::runtime_error("invalid CFL dt");
    auto neighbours=mesh.neighbours;auto rhs1=r.buffer(size_t(mesh.activeCount)*9*sizeof(float)),rhs2=r.buffer(size_t(mesh.activeCount)*9*sizeof(float));
    auto initial=r.buffer(size_t(nodes)*12*sizeof(float)),stage1=r.buffer(size_t(nodes)*12*sizeof(float)),stage2=r.buffer(size_t(nodes)*12*sizeof(float));
    auto invalid=r.buffer(size_t(mesh.activeCount)*sizeof(uint32_t));std::memcpy(initial.contents,scales.contents,size_t(nodes)*12*sizeof(float));
    double before=mass(mesh,u,L);
    auto runRhs=[&](id<MTLBuffer> state,id<MTLBuffer> rhs){auto p=r.pipeline("mwdg2_rhs");id<MTLCommandBuffer> c=[r.queue commandBuffer];auto e=[c computeCommandEncoder];[e setComputePipelineState:p];[e setBuffer:state offset:0 atIndex:0];[e setBuffer:mesh.active offset:0 atIndex:1];[e setBuffer:neighbours offset:0 atIndex:2];[e setBuffer:rhs offset:0 atIndex:3];[e setBuffer:state offset:0 atIndex:4];id<MTLBuffer> edge=[r.device newBufferWithLength:mesh.activeCount*4+4 options:MTLResourceStorageModeShared];id<MTLBuffer> flags=[r.device newBufferWithLength:mesh.activeCount*4+4 options:MTLResourceStorageModeShared];[e setBuffer:edge offset:0 atIndex:5];[e setBuffer:state offset:0 atIndex:6];[e setBuffer:flags offset:0 atIndex:7];[e setBytes:layout length:8 atIndex:8];float f[4]={dx,dy,g,depthTol};[e setBytes:f length:16 atIndex:9];float wave[4]={0,0,0,0};[e setBytes:wave length:16 atIndex:10];[e dispatchThreadgroups:MTLSizeMake((mesh.activeCount+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];if(c.status==MTLCommandBufferStatusError)throw std::runtime_error(c.error.localizedDescription.UTF8String);};
    auto runPos=[&](id<MTLBuffer> state){auto p=r.pipeline("mwdg2_positivity");id<MTLCommandBuffer> c=[r.queue commandBuffer];auto e=[c computeCommandEncoder];[e setComputePipelineState:p];[e setBuffer:state offset:0 atIndex:0];[e setBuffer:mesh.active offset:0 atIndex:1];[e setBuffer:invalid offset:0 atIndex:2];[e setBytes:&layout[0] length:4 atIndex:3];[e setBytes:&depthTol length:4 atIndex:4];[e dispatchThreadgroups:MTLSizeMake((mesh.activeCount+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];if(c.status==MTLCommandBufferStatusError)throw std::runtime_error(c.error.localizedDescription.UTF8String);};
    runRhs(scales,rhs1);rk(r,initial,scales,rhs1,mesh.active,stage1,mesh.activeCount,dt,0);runPos(stage1);runRhs(stage1,rhs2);rk(r,initial,stage1,rhs2,mesh.active,stage2,mesh.activeCount,dt,.5f);runPos(stage2);
    auto *bad=(uint32_t *)invalid.contents;for(uint32_t i=0;i<mesh.activeCount;++i)if(bad[i])throw std::runtime_error("invalid DG2 positivity result");
    double after=mass(mesh,(float *)stage2.contents,L);double rel=std::fabs(after-before)/std::max(1e-12,std::fabs(before));if(rel>5e-3)throw std::runtime_error("adaptive closed-domain mass residual exceeds tolerance");
    uint32_t expand[2]={cells,L};r.run("mwdg2_mini_expand_finest",{stage2,mesh.owners,scales},expand,sizeof(expand),cells);
    auto next=builder.select(scales,details,pre,L,nodes,epsilon,stage2,mesh.active,mesh.activeCount);
    if(next.activeCount==0)throw std::runtime_error("reselection yielded empty mesh");
    std::printf("PASS adaptive DG2 runtime step: dt=%.7g, leaves=%u->%u, relative mass residual=%.3g\n",dt,mesh.activeCount,next.activeCount,rel);
    return 0;
  }catch(const std::exception &e){std::fprintf(stderr,"FAIL adaptive DG2 runtime step: %s\n",e.what());return 1;}
}}
