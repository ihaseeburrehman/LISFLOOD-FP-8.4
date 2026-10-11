#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../source/metal/adaptive/metal_mwdg2_mesh.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static uint32_t dilate(uint32_t v) {
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  v = (v | (v << 2)) & 0x33333333u;
  return (v | (v << 1)) & 0x55555555u;
}
static uint32_t morton(uint32_t x, uint32_t y) { return dilate(x) | (dilate(y) << 1); }
static constexpr uint32_t levelOffset(uint32_t l) { return (((1u << (2*l))-1u)/3u); }

int main(int argc, char **argv) {
@autoreleasepool {
  if (argc != 2) { std::fprintf(stderr,"usage: %s mwdg2.metallib\n",argv[0]); return 2; }
  try {
    constexpr uint32_t L=4, side=1u<<L, cells=side*side;
    constexpr uint32_t finestOffset=levelOffset(L), nodes=finestOffset+cells;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device)throw std::runtime_error("Metal device unavailable");
    NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLCommandQueue> queue=[device newCommandQueue];
    lis::metal::mwdg2::AdaptiveMeshBuilder builder(device,library,queue);
    id<MTLBuffer> scales=[device newBufferWithLength:size_t(nodes)*12*sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> details=[device newBufferWithLength:size_t(nodes)*36*sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> preflags=[device newBufferWithLength:nodes
                                                options:MTLResourceStorageModeShared];
    std::memset(scales.contents,0,size_t(nodes)*12*sizeof(float));
    std::memset(details.contents,0,size_t(nodes)*36*sizeof(float));
    std::memset(preflags.contents,0,nodes);
    auto *state=static_cast<float *>(scales.contents);
    for(uint32_t y=0;y<side;++y) for(uint32_t x=0;x<side;++x) {
      uint32_t node=finestOffset+morton(x,y), p=node*12;
      state[p]=(x>=4&&x<12&&y>=4&&y<12)?0.2f:0.0f;
      state[p+3]=0.01f; state[p+6]=-0.005f;
    }
    auto mesh=builder.select(scales,details,preflags,L,nodes,0.15f);
    if(mesh.activeCount<=1||mesh.activeCount>=cells)
      throw std::runtime_error("selection did not produce a compressed adaptive mesh");
    double area=0;
    for(uint32_t node:mesh.activeHost) {
      uint32_t start=0,count=1,level=0;
      while(node>=start+count){start+=count;count*=4;++level;}
      uint32_t span=1u<<(L-level);
      area+=double(span)*span;
    }
    if(std::abs(area-double(cells))>1e-9)
      throw std::runtime_error("adaptive leaves do not cover the full finest domain");
    // Production meshes now retain CUDA's arbitrary level gaps. Validate the
    // packed CSR face table instead of checking a removed 2:1 grading mask.
    if(!mesh.neighbours)throw std::runtime_error("missing adaptive face-neighbor table");
    const uint32_t faceCount=mesh.activeCount*4;
    const auto *packed=static_cast<const uint32_t *>(mesh.neighbours.contents);
    const uint32_t recordsStart=(faceCount+4u)&~3u;
    const uint32_t recordCount=static_cast<uint32_t>(mesh.neighbours.length/sizeof(uint32_t)-recordsStart)/4;
    if(packed[faceCount]!=recordCount)
      throw std::runtime_error("adaptive face-neighbor CSR has an invalid terminal offset");
    for(uint32_t i=0;i<faceCount;++i) {
      if(packed[i]>packed[i+1] || packed[i+1]>recordCount)
        throw std::runtime_error("adaptive face-neighbor CSR offsets are invalid");
      for(uint32_t r=packed[i];r<packed[i+1];++r) {
        const uint32_t neighbor=packed[recordsStart+4*r];
        if(neighbor!=0xffffffffu && neighbor>=nodes)
          throw std::runtime_error("adaptive face-neighbor ID is outside the tree");
      }
    }
    std::printf("PASS reusable adaptive MWDG2 mesh: %u leaves from %u finest cells on %s\n",
                mesh.activeCount,cells,device.name.UTF8String);
    return 0;
  } catch(const std::exception &e) {
    std::fprintf(stderr,"FAIL reusable MWDG2 mesh: %s\n",e.what()); return 1;
  }
}}
