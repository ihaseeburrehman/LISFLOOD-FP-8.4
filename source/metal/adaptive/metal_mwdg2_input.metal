#include <metal_stdlib>
using namespace metal;

// Project four cell-centred raster planes into the three orthonormal DG2
// modes used by the adaptive MWDG2 state. Input planes are field-major in the
// order h, qx, qy, z; rows increase south-to-north (the CUDA reader's internal
// layout after it reverses ESRI's north-to-south rows). Output is cell-major,
// then variable-major in the order eta=h+z, qx, qy, z, then mode 0, x, y.
// Each cell uses the four nearest cell samples as its corners. CUDA's
// interface construction duplicates the first internal row at interface
// rows 0 and 1, so cell y uses samples max(y-1,0) and y. At the outer raster
// boundary the nearest edge sample is copied. The host replaces -9999 with
// wall_height before upload, as read_raster_file does.
kernel void mwdg2_project_input(device const float *raster [[buffer(0)]],
                                device float *modes [[buffer(1)]],
                                constant uint4 &geometry [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
    const uint width = geometry.x;
    const uint height = geometry.y;
    const uint cells = width * height;
    if (index >= cells) return;

    const uint x = index % width;
    const uint y = index / width;
    // read_and_project_modes_dg2: the node on the raster's east edge takes the
    // edge cell (geometry.z = raster width), not the wall padding
    const uint validWidth = geometry.z ? geometry.z : width;
    const uint xp = x + 1 == validWidth ? x : min(x + 1, width - 1);
    const uint ym = y == 0 ? 0 : y - 1;

    const uint nw = y * width + x;
    const uint ne = y * width + xp;
    const float inv4sqrt3 = 0.14433756729740643f;

    float corner[4][4]; // variable (h,qx,qy,z), corner (SW,SE,NW,NE)
    for (uint variable = 0; variable < 4; ++variable) {
        const device float *plane = raster + variable * cells;
        corner[variable][0] = plane[ym * width + x];
        corner[variable][1] = plane[ym * width + xp];
        corner[variable][2] = plane[nw];
        corner[variable][3] = plane[ne];
    }

    // The state stores free surface rather than depth. Projecting eta from
    // nodal sums preserves lake-at-rest cancellation with the independently
    // projected bed coefficients.
    corner[0][0] += corner[3][0];
    corner[0][1] += corner[3][1];
    corner[0][2] += corner[3][2];
    corner[0][3] += corner[3][3];

    for (uint variable = 0; variable < 4; ++variable) {
        const float swv = corner[variable][0];
        const float sev = corner[variable][1];
        const float nwv = corner[variable][2];
        const float nev = corner[variable][3];
        const uint base = index * 12 + variable * 3;
        modes[base] = 0.25f * (swv + sev + nwv + nev);
        modes[base + 1] = (nev - nwv + sev - swv) * inv4sqrt3;
        modes[base + 2] = (nev - sev + nwv - swv) * inv4sqrt3;
    }
}
