#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Host-side raster contract for metal_mwdg2_input.metal. Values are stored
// south-to-north, x-fastest, after reading the ESRI ASCII rows (which are
// north-to-south). A missing optional file returns its supplied default.
namespace mwdg2_input {
struct RasterPlane {
    unsigned width = 0;
    unsigned height = 0;
    std::vector<float> values;
};

inline RasterPlane readEsriAscii(const std::string &path, float defaultValue,
                                 float wallHeight) {
    std::ifstream input(path);
    if (!input) return {0, 0, {defaultValue}};

    unsigned width = 0, height = 0;
    float noData = -9999.0f;
    std::string key;
    float value = 0.0f;
    for (unsigned line = 0; line < 6; ++line) {
        if (!(input >> key >> value))
            throw std::runtime_error("Malformed ESRI ASCII raster header: " + path);
        if (key == "ncols" || key == "NCOLS") width = static_cast<unsigned>(value);
        else if (key == "nrows" || key == "NROWS") height = static_cast<unsigned>(value);
        else if (key == "NODATA_value" || key == "nodata_value") noData = value;
    }
    if (!width || !height) throw std::runtime_error("Invalid raster dimensions: " + path);

    RasterPlane raster{width, height, std::vector<float>(size_t(width) * height)};
    // ESRI's first data row is north. CUDA stores row j at height-1-j, so
    // internal y=0 is the southernmost row.
    for (unsigned row = 0; row < height; ++row) {
        for (unsigned x = 0; x < width; ++x) {
            float sample = 0.0f;
            if (!(input >> sample)) throw std::runtime_error("Truncated raster data: " + path);
            // The legacy adaptive CUDA reader recognizes -9999 explicitly,
            // independent of the header's NODATA_value declaration.
            const bool nodata = std::fabs(sample + 9999.0f) < 1e-10f;
            raster.values[size_t(height - 1 - row) * width + x] =
                nodata ? wallHeight : sample;
        }
    }
    return raster;
}

inline RasterPlane defaultPlane(unsigned width, unsigned height, float value) {
    return {width, height, std::vector<float>(size_t(width) * height, value)};
}

inline RasterPlane readOrDefault(const std::string &path, unsigned width,
                                 unsigned height, float defaultValue,
                                 float wallHeight) {
    RasterPlane raster = readEsriAscii(path, defaultValue, wallHeight);
    if (raster.width == 0) return defaultPlane(width, height, defaultValue);
    if (raster.width != width || raster.height != height)
        throw std::runtime_error("Raster dimensions do not match MWDG2 domain: " + path);
    return raster;
}

// Pack h, qx, qy, z as four consecutive planes. Missing start/discharge data
// are zero; missing DEM data are wallHeight, matching the CUDA caller's
// initialisation. Existing rasters must share dimensions; no silent resampling
// is performed. The Metal kernel copies the nearest edge sample when it forms
// the outer corner of each cell.
inline std::vector<float> packFields(const RasterPlane &h, const RasterPlane &qx,
                                     const RasterPlane &qy, const RasterPlane &z) {
    if (!h.width || !h.height || h.values.size() != size_t(h.width) * h.height)
        throw std::runtime_error("Invalid water-depth raster");
    const size_t cells = size_t(h.width) * h.height;
    const RasterPlane *fields[] = {&h, &qx, &qy, &z};
    std::vector<float> packed(4 * cells);
    for (unsigned field = 0; field < 4; ++field) {
        if (fields[field]->width != h.width || fields[field]->height != h.height ||
            fields[field]->values.size() != cells)
            throw std::runtime_error("MWDG2 input raster dimensions do not match");
        std::copy(fields[field]->values.begin(), fields[field]->values.end(),
                  packed.begin() + field * cells);
    }
    return packed;
}

inline std::vector<float> loadInitialFields(const std::string &startPath,
                                            const std::string &demPath,
                                            unsigned width, unsigned height,
                                            float wallHeight, bool startq2d) {
    RasterPlane h = readOrDefault(startPath, width, height, 0.0f, wallHeight);
    RasterPlane z = readOrDefault(demPath, width, height, wallHeight, wallHeight);
    RasterPlane qx = defaultPlane(width, height, 0.0f);
    RasterPlane qy = defaultPlane(width, height, 0.0f);
    if (startq2d && !startPath.empty()) {
        qx = readOrDefault(startPath + ".Qx", width, height, 0.0f, wallHeight);
        qy = readOrDefault(startPath + ".Qy", width, height, 0.0f, wallHeight);
    }
    return packFields(h, qx, qy, z);
}
} // namespace mwdg2_input
