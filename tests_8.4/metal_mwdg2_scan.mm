#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../source/metal/adaptive/metal_mwdg2_scan.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
@autoreleasepool {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s tree.metallib\n", argv[0]);
    return 2;
  }
  try {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device unavailable");
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    lis::metal::mwdg2::TreeScan scan(device, library, queue);

    // Cross three recursive scan levels and include a partial final tile.
    constexpr uint32_t count = 700013;
    std::vector<uint32_t> values(count), expected(count);
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
      values[i] = ((i * 37u + i / 17u) % 101u) < 29u ? 1u : 0u;
      expected[i] = total;
      total += values[i];
    }
    id<MTLBuffer> input = [device newBufferWithBytes:values.data()
        length:values.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:
        values.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    uint32_t actualTotal = scan.scan(input, output, count);
    const auto *actual = static_cast<const uint32_t *>(output.contents);
    if (actualTotal != total) throw std::runtime_error("scan total mismatch");
    for (uint32_t i = 0; i < count; ++i)
      if (actual[i] != expected[i])
        throw std::runtime_error("exclusive scan mismatch at index " + std::to_string(i));
    std::printf("PASS recursive Metal tree scan: %u flags, %u selected on %s\n",
                count, actualTotal, device.name.UTF8String);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL Metal MWDG2 scan: %s\n", e.what());
    return 1;
  }
}}
