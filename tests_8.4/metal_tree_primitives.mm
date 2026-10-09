#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> scanPipeline, offsetPipeline, compactPipeline;

static id<MTLBuffer> buffer(size_t count) {
    return [device newBufferWithLength:sizeof(unsigned) * count
                              options:MTLResourceStorageModeShared];
}

static void finish(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s\n", command.error.description.UTF8String);
        exit(1);
    }
}

static void scan(id<MTLBuffer> input, id<MTLBuffer> output, unsigned count) {
    unsigned groups = (count + 255) / 256;
    id<MTLBuffer> sums = buffer(groups);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:scanPipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:sums offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    finish(command);
    if (groups <= 1) return;
    id<MTLBuffer> offsets = buffer(groups);
    scan(sums, offsets, groups);
    command = [queue commandBuffer];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:offsetPipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder setBuffer:offsets offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    finish(command);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
            error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        scanPipeline = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"tree_scan_tiles"] error:&error];
        offsetPipeline = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"tree_add_tile_offsets"] error:&error];
        compactPipeline = [device newComputePipelineStateWithFunction:
            [library newFunctionWithName:@"tree_compact_nodes"] error:&error];
        if (!scanPipeline || !offsetPipeline || !compactPipeline) return 1;
        std::mt19937 generator(8431);
        for (unsigned count : {1u, 255u, 256u, 257u, 65537u, 1048579u}) {
            id<MTLBuffer> flags = buffer(count), prefix = buffer(count);
            id<MTLBuffer> nodes = buffer(count), active = buffer(count);
            auto *f = (unsigned *)flags.contents;
            auto *p = (unsigned *)prefix.contents;
            auto *n = (unsigned *)nodes.contents;
            auto *a = (unsigned *)active.contents;
            for (unsigned i = 0; i < count; ++i) { f[i] = generator() % 2; n[i] = i; }
            scan(flags, prefix, count);
            unsigned total = 0;
            for (unsigned i = 0; i < count; ++i) {
                if (p[i] != total) return 1;
                total += f[i];
            }
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:compactPipeline];
            [encoder setBuffer:flags offset:0 atIndex:0];
            [encoder setBuffer:prefix offset:0 atIndex:1];
            [encoder setBuffer:nodes offset:0 atIndex:2];
            [encoder setBuffer:active offset:0 atIndex:3];
            [encoder setBytes:&count length:sizeof(count) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            finish(command);
            for (unsigned i = 0; i < count; ++i)
                if (f[i] && a[p[i]] != i) return 1;
            printf("PASS native Metal scan and stable compaction: %u nodes, %u active\n", count, total);
        }
    }
}
