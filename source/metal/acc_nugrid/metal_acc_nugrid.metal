// LISFLOOD-FP 8.4: ACC on the static non-uniform grid (ACC_NUGRID), ported
// from cuda/acc_nugrid (compute_q.cu, update_h.cu, calculate_dt.cu, rain.cu,
// rain_uniform.cu, load_interface_q_vol.cu) and the acc_nugrid loss kernel of
// cuda_losses.cu. The mesh, neighbour records and interfaces are built once
// on the host (metal_acc_nugrid_simulate.mm).
//
// Leaves: h, z, n, dx (cell size), li/lj (cell index at the leaf's level,
// from the south-west), x0/y0/span (finest cells), recStart/recCount.
// Records (one per leaf and neighbouring face, as find_nonuniform_neighbours):
// owner, nghbr (leaf index or -1), dir (1 N, 2 E, 3 S, 4 W), length, q
// (volumetric, m3/s), flags, interface range, boundary side/bound.
#include <metal_stdlib>
using namespace metal;

struct NugLeaf {
    float h, z, n, dx;
    uint li, lj, x0, y0;
    uint span, recStart, recCount, level;
};

struct NugRecord {
    uint owner;
    int nghbr;          // leaf index, or -1 at the domain edge
    uint dir;           // 1 N, 2 E, 3 S, 4 W
    float length;       // neighbour cell size (CUDA dx of the record)
    float ownerFlag, nghbrFlag;
    uint itfStart, itfCount;
    float dxSum;
    int side;           // edge records: 0 N, 1 E, 2 S, 3 W; else -1
    uint bound;         // edge record inside that side's prescribed range
    uint pad;
};

// params: dt, g, DepthThresh, FPn
// bc: types[4] (0 closed, 1 free, 2 hfix, 3 hvar, 4 qfix, 5 qvar), values[4]
struct NugBc { int4 type; float4 value; };

kernel void nug_drain(device NugLeaf *leaf [[buffer(0)]],
                      constant uint &count [[buffer(1)]],
                      constant float &nodata [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    if (abs(leaf[i].z - nodata) < 1e-6f) leaf[i].h = 0.0f;
}

// update_uniform_rain: depth += rate * dt (previous step), except NoData
kernel void nug_rain_uniform(device NugLeaf *leaf [[buffer(0)]],
                             constant uint &count [[buffer(1)]],
                             constant float2 &p [[buffer(2)]],   // cell rain (m), nodata
                             uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    if (abs(leaf[i].z - p.y) < 1e-6f) return;
    leaf[i].h += p.x;
}

// update_rain / rate_at_cell (dynamic rain); g = {xsz_rain, ysz_rain}, f =
// {dt, nodata, tly_rain gap (tly - tly_rain), dx_rain, dy_rain}
kernel void nug_rain_dynamic(device NugLeaf *leaf [[buffer(0)]],
                             device const float *rain [[buffer(1)]],
                             constant uint &count [[buffer(2)]],
                             constant int2 &g [[buffer(3)]],
                             constant float4 &f [[buffer(4)]],
                             constant float &dyRain [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    if (abs(leaf[i].z - f.y) < 1e-6f) return;
    float dx = leaf[i].dx;
    int tileI = int(float(leaf[i].li) * dx / f.w);
    int tileJ = int((float(leaf[i].lj) - f.z / dx) * dx / dyRain);
    float rate = 0.0f;
    if (tileI < g.x && tileJ >= 0 && tileJ < g.y) rate = rain[tileJ * g.x + tileI];
    leaf[i].h += rate * f.x;
}

// calculate_dt: cfl*dx/sqrt(g h) for wet leaves, initial_tstep otherwise;
// p = {initial_tstep, cfl, g, DepthThresh}
kernel void nug_cfl(device const NugLeaf *leaf [[buffer(0)]],
                    device float *out [[buffer(1)]],
                    constant uint &count [[buffer(2)]],
                    constant float4 &p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float h = leaf[i].h;
    out[i] = h < p.w ? p.x : min(p.x, p.y * leaf[i].dx / sqrt(p.z * h));
}

// load_interface_q_vol: the perpendicular discharges of the previous step
kernel void nug_load_qvol(device const uint *loadIdx [[buffer(0)]],
                          device const float *q [[buffer(1)]],
                          device float *qvol [[buffer(2)]],
                          constant uint &count [[buffer(3)]],
                          uint k [[thread_position_in_grid]]) {
    if (k >= count) return;
    qvol[k] = q[loadIdx[k]];
}

// compute_q.cu (one thread per record); p = {dt, g, DepthThresh, FPn},
// nonUniformN: Manning from the leaves (nfile) instead of FPn
kernel void nug_compute_q(device const NugLeaf *leaf [[buffer(0)]],
                          device const NugRecord *rec [[buffer(1)]],
                          device float *q [[buffer(2)]],
                          device const float *qvol [[buffer(3)]],
                          device float *vRec [[buffer(4)]],
                          constant uint &count [[buffer(5)]],
                          constant float4 &p [[buffer(6)]],
                          constant NugBc &bc [[buffer(7)]],
                          constant uint &nonUniformN [[buffer(8)]],
                          uint r [[thread_position_in_grid]]) {
    if (r >= count) return;
    NugRecord R = rec[r];
    float dt = p.x, g = p.y, thresh = p.z;
    float length = R.length;
    NugLeaf O = leaf[R.owner];
    float hOwner = O.h, zOwner = O.z;
    float qOld = q[r] / length;
    if (R.nghbr < 0) {
        int sign = (R.dir == 1 || R.dir == 4) ? -1 : 1;
        int dir = (R.dir == 1 || R.dir == 3) ? 1 : 2;
        int edge = int(R.dir);
        int type = R.bound ? bc.type[R.side] : 0;
        float value = R.bound ? bc.value[R.side] : 0.0f;
        float manning = nonUniformN ? O.n : p.w;
        float out = 0.0f;
        float hflow = hOwner;   // CUDA leaves it unset for QFIX/QVAR
        if (type == 1) {            // FREE: value is the friction slope
            hflow = hOwner;
            float Sf = value;
            if (hflow > thresh) {
                out = float(sign) * (abs(qOld) + abs(g * dt * hflow * Sf)) /
                      (1.0f + g * dt * hflow * manning * manning * abs(qOld) / pow(hflow, 10.0f / 3.0f)) * length;
                if (dir == 1 && sign == -1 && out > 0.0f) out = 0.0f;
                if (dir == 1 && sign == 1 && out < 0.0f) out = 0.0f;
                if (dir == 2 && sign == -1 && out > 0.0f) out = 0.0f;
                if (dir == 2 && sign == 1 && out < 0.0f) out = 0.0f;
            }
        } else if (type == 2 || type == 3) {   // HFIX, HVAR
            hflow = max(hOwner, value - zOwner);
            if (hflow >= thresh) {
                float dh = value - zOwner - hOwner;
                float Sf = (edge == 1 || edge == 4) ? -dh / length : dh / length;
                out = (qOld - g * dt * hflow * Sf) /
                      (1.0f + g * dt * hflow * manning * manning * abs(qOld) / pow(hflow, 10.0f / 3.0f)) * length;
            }
        } else if (type == 4 || type == 5) {   // QFIX, QVAR
            out = -value * float(sign) * length;
        }
        q[r] = out;
        vRec[r] = out != 0.0f ? out / length / hflow : 0.0f;
        return;
    }
    NugLeaf N = leaf[R.nghbr];
    float hNghbr = N.h, zNghbr = N.z;
    float manning = nonUniformN ? 0.5f * (O.n + N.n) : p.w;
    float out = 0.0f, hflow = 0.0f;
    if (hOwner > thresh || hNghbr > thresh) {
        float qOpp = 0.0f;
        for (uint k = 0; k < R.itfCount; ++k) qOpp += qvol[R.itfStart + k];
        float qOppAvg = qOpp / R.dxSum;
        float qvect = sqrt(qOld * qOld + qOppAvg * qOppAvg);
        float yOwner = (zOwner + hOwner) * R.ownerFlag;
        float yNghbr = (zNghbr + hNghbr) * R.nghbrFlag;
        float dh = yOwner + yNghbr;
        float Sf = -dh / length;
        hflow = max(zOwner + hOwner, zNghbr + hNghbr) - max(zOwner, zNghbr);
        hflow = max(hflow, 0.0f);
        hflow = min(hflow, 10.0f);
        if (hflow > thresh)
            out = (qOld - g * dt * hflow * Sf) /
                  (1.0f + g * dt * hflow * manning * manning * abs(qvect) / pow(hflow, 10.0f / 3.0f)) * length;
    }
    q[r] = out;
    vRec[r] = out != 0.0f ? out / length / hflow : 0.0f;
}

// update_h.cu (one thread per leaf). Point sources: srcLeaf[k] leaf index,
// srcType/srcValue; p = {dt, DepthThresh, unused, unused}.
kernel void nug_update_h(device NugLeaf *leaf [[buffer(0)]],
                         device const NugRecord *rec [[buffer(1)]],
                         device float *q [[buffer(2)]],
                         device const int *srcLeaf [[buffer(3)]],
                         device const int *srcType [[buffer(4)]],
                         device const float *srcValue [[buffer(5)]],
                         device const float *vRec [[buffer(6)]],
                         device float *vLeaf [[buffer(7)]],
                         constant uint2 &count [[buffer(8)]],     // leaves, sources
                         constant float4 &p [[buffer(9)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count.x) return;
    float dt = p.x, thresh = p.y;
    uint start = leaf[i].recStart, n = leaf[i].recCount;
    float flux = 0.0f;
    for (uint k = 0; k < n; ++k) {
        uint d = rec[start + k].dir;
        float flag = (d == 1 || d == 4) ? 1.0f : -1.0f;
        flux += flag * q[start + k];
    }
    float dV = dt * flux;
    float dx = leaf[i].dx;
    float H = leaf[i].h;
    float cv = H * (dx * dx);
    if (H > thresh && cv + dV < 0.0f) {
        float weight = -cv / dV;
        for (uint k = 0; k < n; ++k) {
            uint d = rec[start + k].dir;
            float v = q[start + k];
            if ((d == 4 && v < 0.0f) || (d == 2 && v > 0.0f) ||
                (d == 3 && v > 0.0f) || (d == 1 && v < 0.0f))
                q[start + k] = v * weight;
        }
    }
    flux = 0.0f;
    float vDir[5] = {0, 0, 0, 0, 0};
    float nDir[5] = {0, 0, 0, 0, 0};
    for (uint k = 0; k < n; ++k) {
        uint d = rec[start + k].dir;
        float flag = (d == 1 || d == 4) ? 1.0f : -1.0f;
        flux += flag * q[start + k];
        vDir[d] += flag * vRec[start + k];
        nDir[d] += 1.0f;
    }
    float vn = vDir[1] / nDir[1], ve = vDir[2] / nDir[2], vs = vDir[3] / nDir[3], vw = vDir[4] / nDir[4];
    float vx = max(abs(vw), abs(ve)), vy = max(abs(vn), abs(vs));
    vLeaf[i] = sqrt(vx * vx + vy * vy);
    H += dt * flux / (dx * dx);
    if (H < 0.0f) H = 0.0f;
    for (uint s = 0; s < count.y; ++s) {
        if (srcLeaf[s] != int(i)) continue;
        int type = srcType[s];
        if (type == 2 || type == 3) H = max(0.0f, srcValue[s] - leaf[i].z);
        else if (type == 4 || type == 5) H = max(0.0f, H + srcValue[s] * dt / dx);
    }
    leaf[i].h = H;
}

// Leaf depth on every raster cell (row 0 = north) for losses and output.
kernel void nug_project(device const NugLeaf *leaf [[buffer(0)]],
                        device const int *owner [[buffer(1)]],
                        device float *depth [[buffer(2)]],
                        constant uint &cells [[buffer(3)]],
                        uint r [[thread_position_in_grid]]) {
    if (r >= cells) return;
    int o = owner[r];
    depth[r] = o >= 0 ? leaf[o].h : 0.0f;
}

// apply_losses_acc_nugrid: the soil laws ran on every raster cell (depth now
// holds the depth after losses); restrict the change to the leaf average and
// keep the rounding correction on the raster cells. layout = {leaves, xsz,
// ysz}; nodata. removed[i]: depth removed times valid cell count.
kernel void nug_restrict_losses(device NugLeaf *leaf [[buffer(0)]],
                                device const float *depth [[buffer(1)]],
                                device const float *dem [[buffer(2)]],
                                device float *residual [[buffer(3)]],
                                device float *removed [[buffer(4)]],
                                constant uint4 &layout [[buffer(5)]],
                                constant float &nodata [[buffer(6)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= layout.x) return;
    removed[i] = 0.0f;
    uint xsz = layout.y, ysz = layout.z;
    uint x0 = leaf[i].x0, y0 = leaf[i].y0, span = leaf[i].span;
    float h = leaf[i].h;
    float loss = 0.0f;
    uint valid = 0;
    for (uint y = y0; y < y0 + span && y < ysz; ++y)
        for (uint x = x0; x < x0 + span && x < xsz; ++x) {
            uint r = (ysz - 1 - y) * xsz + x;
            if (abs(dem[r] - nodata) < 1e-6f) continue;
            loss += h - depth[r];
            ++valid;
        }
    if (valid == 0) return;
    float next = h - loss / float(valid);
    float correction = (loss - float(valid) * (h - next)) / float(valid);
    for (uint y = y0; y < y0 + span && y < ysz; ++y)
        for (uint x = x0; x < x0 + span && x < xsz; ++x) {
            uint r = (ysz - 1 - y) * xsz + x;
            if (abs(dem[r] - nodata) < 1e-6f) continue;
            residual[r] += correction;
        }
    removed[i] = (h - next) * float(valid);
    leaf[i].h = next;
}
