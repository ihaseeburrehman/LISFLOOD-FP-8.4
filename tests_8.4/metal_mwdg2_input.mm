#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../source/metal/adaptive/mwdg2_input_projection.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "Metal device unavailable\n"); return 1; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        id<MTLFunction> function = [library newFunctionWithName:@"mwdg2_project_input"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }

        auto project = [&](const std::vector<float> &raster, unsigned width, unsigned height) {
            const unsigned cells = width * height;
            id<MTLBuffer> in = [device newBufferWithBytes:raster.data()
                length:raster.size() * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> out = [device newBufferWithLength:cells * 12 * sizeof(float)
                options:MTLResourceStorageModeShared];
            unsigned geometry[4] = {width, height, 0, 0};
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:in offset:0 atIndex:0];
            [encoder setBuffer:out offset:0 atIndex:1];
            [encoder setBytes:geometry length:sizeof(geometry) atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake((cells + 63) / 64, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error(command.error.description.UTF8String);
            const float *values = static_cast<const float *>(out.contents);
            return std::vector<float>(values, values + cells * 12);
        };

        // Constant h/qx/qy/z gives constant eta and zero x/y coefficients.
        std::vector<float> constant(4 * 9);
        for (unsigned i = 0; i < 9; ++i) {
            constant[i] = 2.0f; constant[9+i] = 3.0f;
            constant[18+i] = -4.0f; constant[27+i] = 10.0f;
        }
        auto c = project(constant, 3, 3);
        for (unsigned i = 0; i < 9; ++i) {
            require(std::fabs(c[i*12] - 12.0f) < 1e-6f, "constant eta mean");
            require(std::fabs(c[i*12+1]) < 1e-6f && std::fabs(c[i*12+2]) < 1e-6f,
                    "constant eta modes");
            require(std::fabs(c[i*12+3] - 3.0f) < 1e-6f, "constant qx");
            require(std::fabs(c[i*12+6] + 4.0f) < 1e-6f, "constant qy");
            require(std::fabs(c[i*12+9] - 10.0f) < 1e-6f, "constant bed");
        }
        printf("PASS Metal MWDG2 input: constant fields and eta=h+z\n");

        // Planar samples; interior cell coefficients recover the exact
        // orthonormal DG2 projection of corner values.
        std::vector<float> plane(4 * 16, 0.0f);
        for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
            unsigned k = y*4+x;
            plane[k] = 5.0f + 2.0f*x - 3.0f*y;       // h
            plane[3*16+k] = 1.0f + 4.0f*x + 2.0f*y; // z
        }
        auto p = project(plane, 4, 4);
        const float inv4sqrt3 = 1.0f/(4.0f*std::sqrt(3.0f));
        unsigned interior = 1*4+1;
        float etaSw = plane[interior]+plane[3*16+interior];
        float etaSe = plane[interior+1]+plane[3*16+interior+1];
        float etaNw = plane[interior+4]+plane[3*16+interior+4];
        float etaNe = plane[interior+5]+plane[3*16+interior+5];
        require(std::fabs(p[interior*12+1]-(etaNe-etaNw+etaSe-etaSw)*inv4sqrt3)<1e-6f,
                "planar eta x coefficient");
        require(std::fabs(p[interior*12+2]-(etaNe-etaSe+etaNw-etaSw)*inv4sqrt3)<1e-6f,
                "planar eta y coefficient");
        require(std::fabs(p[interior*12+10] - 2.0f/std::sqrt(3.0f)) < 1e-6f,
                "planar bed x coefficient");
        printf("PASS Metal MWDG2 input: planar x/y modal coefficients\n");

        // ESRI file rows are north-to-south; host parsing reverses them to
        // CUDA's south-to-north raster storage and converts -9999 to wall.
        NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"mwdg2_projection_%d.asc", getpid()]];
        {
            std::ofstream f(path.fileSystemRepresentation);
            f << "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n"
              << "10 20\n30 40\n";
        }
        auto esri = mwdg2_input::readEsriAscii(path.fileSystemRepresentation, 0.0f, 99.0f);
        require(esri.width == 2 && esri.height == 2, "ESRI dimensions");
        require(esri.values[0] == 30 && esri.values[1] == 40 &&
                esri.values[2] == 10 && esri.values[3] == 20, "ESRI row orientation");
        std::remove(path.fileSystemRepresentation);
        auto esriFields = mwdg2_input::packFields(esri,
            mwdg2_input::defaultPlane(2,2,0), mwdg2_input::defaultPlane(2,2,0),
            mwdg2_input::defaultPlane(2,2,0));
        auto oriented = project(esriFields, 2, 2);
        require(std::fabs(oriented[1] - 20.0f/(4.0f*std::sqrt(3.0f))) < 1e-6f,
                "ESRI orientation x mode");
        // CUDA duplicates the southern outer row at interface row zero. The
        // first modal cell therefore has no y slope; the next row spans the
        // southern and northern raster samples and carries the signed slope.
        require(std::fabs(oriented[2]) < 1e-6f, "ESRI south-edge y padding");
        require(std::fabs(oriented[2*12+2] + 40.0f/(4.0f*std::sqrt(3.0f))) < 1e-6f,
                "ESRI orientation y mode");
        printf("PASS Metal MWDG2 input: ESRI row orientation\n");

        // No-data becomes the configured wall height. The last-column and
        // last-row corners copy the nearest sample, so their outward slope is
        // zero. Also check absent start data defaults to zero.
        {
            std::ofstream f(path.fileSystemRepresentation);
            f << "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n"
              << "-9999 7\n1 2\n";
        }
        auto withWall = mwdg2_input::readEsriAscii(path.fileSystemRepresentation, 0.0f, 50.0f);
        require(withWall.values[2] == 50.0f, "no-data wall replacement");
        auto defaults = mwdg2_input::readEsriAscii("/no/such/start.asc", 0.0f, 50.0f);
        require(defaults.values.size() == 1 && defaults.values[0] == 0.0f,
                "missing start default");
        std::remove(path.fileSystemRepresentation);
        auto wallFields = mwdg2_input::packFields(withWall,
            mwdg2_input::defaultPlane(2,2,0), mwdg2_input::defaultPlane(2,2,0),
            mwdg2_input::defaultPlane(2,2,50));
        auto w = project(wallFields, 2, 2);
        // Cell (x=1,y=1) repeats its own corner in both directions.
        unsigned edge = 3;
        require(std::fabs(w[edge*12+10]) < 1e-6f && std::fabs(w[edge*12+11]) < 1e-6f,
                "nearest-copy outer padding");
        printf("PASS Metal MWDG2 input: no-data walls, missing start, edge padding\n");

        NSString *base = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"mwdg2_start_%d", getpid()]];
        NSString *demPath = [base stringByAppendingString:@".dem"];
        NSString *startPath = [base stringByAppendingString:@".start"];
        auto writeRaster = [](NSString *file, const char *values) {
            std::ofstream f(file.fileSystemRepresentation);
            f << "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n"
              << values;
        };
        writeRaster(demPath, "5 5\n5 5\n");
        writeRaster(startPath, "1 1\n1 1\n");
        writeRaster([startPath stringByAppendingString:@".Qx"], "2 2\n2 2\n");
        writeRaster([startPath stringByAppendingString:@".Qy"], "-3 -3\n-3 -3\n");
        auto initial = mwdg2_input::loadInitialFields(startPath.fileSystemRepresentation,
            demPath.fileSystemRepresentation, 2, 2, 99.0f, true);
        require(initial[0] == 1 && initial[1] == 1 && initial[2] == 1 && initial[3] == 1,
                "start file load");
        require(initial[4] == 2 && initial[5] == 2 && initial[6] == 2 && initial[7] == 2,
                "Qx suffix load");
        require(initial[8] == -3 && initial[9] == -3 && initial[10] == -3 && initial[11] == -3,
                "Qy suffix load");
        require(initial[12] == 5 && initial[13] == 5 && initial[14] == 5 && initial[15] == 5,
                "DEM load");
        auto missing = mwdg2_input::loadInitialFields("", "/no/such/dem", 2, 2, 99.0f, false);
        for (unsigned i = 0; i < 4; ++i) {
            require(missing[i] == 0 && missing[4+i] == 0 && missing[8+i] == 0 && missing[12+i] == 99,
                    "absent input defaults");
        }
        std::remove(demPath.fileSystemRepresentation);
        std::remove(startPath.fileSystemRepresentation);
        std::remove([[startPath stringByAppendingString:@".Qx"] fileSystemRepresentation]);
        std::remove([[startPath stringByAppendingString:@".Qy"] fileSystemRepresentation]);
        printf("PASS Metal MWDG2 input: .start/.dem and optional .Qx/.Qy files\n");
        return 0;
    }
}
