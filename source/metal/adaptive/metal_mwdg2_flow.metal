#include "metal_mwdg2_mra.metal"

constant float mw_sqrt3 = 1.7320508075688772f;

inline float3 mw_basis(uint4 cell, float2 point) {
    float2 local = 2.0f * (point-float2(cell.xy))/float(cell.z)-1.0f;
    return float3(1.0f,mw_sqrt3*local);
}

inline float mw_bed(device const float *scales, uint node, float3 basis) {
    uint p=node*12+9;
    return dot(float3(scales[p],scales[p+1],scales[p+2]),basis);
}

// MRA stores free-surface elevation eta (not depth h), together with qx, qy
// and bed z. Recover the conserved depth used by the SWE operator as eta-z.
inline float3 mw_state(device const float *scales, uint node, float3 basis) {
    float3 state;
    uint p=node*12;
    state.x=dot(float3(scales[p],scales[p+1],scales[p+2]),basis)-
            mw_bed(scales,node,basis);
    state.y=dot(float3(scales[p+3],scales[p+4],scales[p+5]),basis);
    state.z=dot(float3(scales[p+6],scales[p+7],scales[p+8]),basis);
    return state;
}

// CUDA's adaptive neighbor lookup promotes a finer face neighbor to its
// ancestor at the current leaf's level. Coarse leaves therefore use the
// common parent polynomial along the entire face, rather than extrapolating
// whichever fine segment happened to be stored first.
inline uint mw_neighbor_at_local_level(uint other, uint localLevel,
                                      uint maxLevel) {
    if (other >= 0xfffffffeu) return other;
    while (node_geometry(other, maxLevel).w > localLevel)
        other = (other - 1u) / 4u;
    return other;
}

inline float3 mw_physical(float3 u, uint axis, float g, float tolerance) {
    if (u.x<tolerance) return float3(0.0f);
    float h=u.x;
    float velocity=u[axis+1]/h;
    float3 flux=float3(u[axis+1],u.y*velocity,u.z*velocity);
    flux[axis+1] += .5f*g*h*h;
    return flux;
}

inline float3 mw_star(float3 u, float bed, float common_bed, float tolerance) {
    float h=max(0.0f,u.x+bed-common_bed);
    return float3(h,u.yz*(u.x>tolerance?h/u.x:0.0f));
}

inline float2 mw_friction_point(float h,float2 q,float n,float g,float dt,
                                float depthTolerance,float speedTolerance) {
    if(h<depthTolerance) return float2(0);
    float2 velocity=q/h;
    if(abs(velocity.x)<speedTolerance && abs(velocity.y)<speedTolerance) return float2(0);
    float speed=length(velocity);
    float cf=g*n*n/pow(h,1.0f/3.0f);
    float2 source=-cf*velocity*speed;
    float dx=1+dt*cf/h*(2*velocity.x*velocity.x+velocity.y*velocity.y)/speed;
    float dy=1+dt*cf/h*(2*velocity.y*velocity.y+velocity.x*velocity.x)/speed;
    return q+dt*float2(source.x/dx,source.y/dy);
}

// CUDA adaptive flux_HLL.cuh: two-rarefaction star-state wave speeds and
// contact-speed selection for transverse momentum, with the same one-sided
// branches and wet/dry depth threshold.
inline float3 mw_hll_x(float3 neg, float3 pos, float g, float tolerance) {
    if (neg.x<tolerance && pos.x<tolerance) return float3(0.0f);
    float hn=neg.x<tolerance?0.0f:neg.x;
    float un=neg.x<tolerance?0.0f:neg.y/neg.x;
    float vn=neg.x<tolerance?0.0f:neg.z/neg.x;
    float hp=pos.x<tolerance?0.0f:pos.x;
    float up=pos.x<tolerance?0.0f:pos.y/pos.x;
    float vp=pos.x<tolerance?0.0f:pos.z/pos.x;
    float an=sqrt(g*hn), ap=sqrt(g*hp);
    float starTerm=0.5f*(an+ap)+0.25f*(un-up);
    float hstar=starTerm*starTerm/g;
    float ustar=0.5f*(un+up)+an-ap;
    float astar=sqrt(g*hstar);
    float sneg=hn<tolerance?up-2.0f*ap:min(un-an,ustar-astar);
    float spos=hp<tolerance?un+2.0f*an:max(up+ap,ustar+astar);
    float3 fn=float3(neg.y,un*neg.y+0.5f*g*hn*hn,neg.z*un);
    float3 fp=float3(pos.y,up*pos.y+0.5f*g*hp*hp,pos.z*up);
    if (sneg>=0.0f) return fn;
    if (spos<0.0f) return fp;
    float3 flux;
    flux.x=(spos*fn.x-sneg*fp.x+sneg*spos*(hp-hn))/(spos-sneg);
    flux.y=(spos*fn.y-sneg*fp.y+sneg*spos*(pos.y-neg.y))/(spos-sneg);
    float smid=(sneg*hp*(up-spos)-spos*hn*(un-sneg)) /
               (hp*(up-spos)-hn*(un-sneg));
    flux.z=flux.x*(smid>=0.0f?vn:vp);
    return flux;
}

inline float3 mw_hll(float3 neg, float3 pos, uint axis, float g, float tolerance) {
    if (axis==0) return mw_hll_x(neg,pos,g,tolerance);
    float3 negRot=float3(neg.x,neg.z,-neg.y);
    float3 posRot=float3(pos.x,pos.z,-pos.y);
    float3 flux=mw_hll_x(negRot,posRot,g,tolerance);
    return float3(flux.x,-flux.z,flux.y);
}

// CUDA-compatible DG2 operator; CUDA samples each face at its midpoint.
// The first Legendre residual is formed from opposite-face flux sums. Keep
// this stencil: multi-point face quadrature creates transverse flux modes
// absent from CUDA's update and breaks 1-D symmetry over long runs.
// Scales contain h, qx, qy, z (three orthonormal Legendre modes each).
// CUDA reads a leaf's own coefficients from its assembled solution (after
// ghost-cell slope clearing, friction and limiting) but its neighbours from
// the scale-coefficient tree loaded before those changes: `scales` is the
// stage view, `tree` the hierarchy.
kernel void mwdg2_rhs(device const float *scales [[buffer(0)]],
                      device const uint *active [[buffer(1)]],
                      device const uint *neighbours [[buffer(2)]],
                      device float *rhs [[buffer(3)]],
                      device const float *tree [[buffer(4)]],
                      device float *edgeRate [[buffer(5)]],
                      device const float *plainTree [[buffer(6)]],
                      device const uint *frictioned [[buffer(7)]],
                      constant uint2 &layout [[buffer(8)]],
                      constant float4 &physics [[buffer(9)]],
                      uint index [[thread_position_in_grid]]) {
    if (index>=layout.x) return;
    // friction_implicit.cu updates a leaf's neighbour copies only when the
    // leaf itself is above tol_h and tol_s: read the neighbours from the
    // tree with friction applied (tree) or without (plainTree) accordingly.
    if (!frictioned[index]) tree=plainTree;
    uint node=active[index];
    uint4 cell=node_geometry(node,layout.y);
    float dx=physics.x*cell.z,dy=physics.y*cell.z,g=physics.z,tolerance=physics.w;
    float3 residual[3]={float3(0),float3(0),float3(0)};
    uint faceCount=layout.x*4u;
    uint recordsStart=(faceCount+4u)&~3u;
    device const uint4 *segments=(device const uint4 *)(neighbours+recordsStart);
    uint other[4];
    for(uint face=0;face<4;++face) {
        uint begin=neighbours[index*4+face];
        other[face]=begin<neighbours[index*4+face+1]
            ? mw_neighbor_at_local_level(segments[begin].x,cell.w,layout.y)
            : 0xffffffffu;
    }
    float2 pN=float2(float(cell.x)+.5f*cell.z,float(cell.y+cell.z));
    float2 pE=float2(float(cell.x+cell.z),float(cell.y)+.5f*cell.z);
    float2 pS=float2(float(cell.x)+.5f*cell.z,float(cell.y));
    float2 pW=float2(float(cell.x),float(cell.y)+.5f*cell.z);
    float3 bN=mw_basis(cell,pN),bE=mw_basis(cell,pE),bS=mw_basis(cell,pS),bW=mw_basis(cell,pW);
    float3 n=mw_state(scales,node,bN),e=mw_state(scales,node,bE);
    float3 s=mw_state(scales,node,bS),w=mw_state(scales,node,bW);
    float zn=mw_bed(scales,node,bN),ze=mw_bed(scales,node,bE);
    float zs=mw_bed(scales,node,bS),zw=mw_bed(scales,node,bW);
    float3 nn=n,ee=e,ss=s,ww=w;
    float znn=zn,zee=ze,zss=zs,zww=zw;
    // CUDA add_ghost_cells uses the cell means for a default closed-boundary
    // ghost and clears all ghost slopes; it also zeros the outward discharge.
    // Using the local face trace here looks more conservative, but changes the
    // CUDA boundary Riemann problem and produces a different long-run volume.
    uint local=node*12;
    float3 meanState=float3(scales[local]-scales[local+9],
                            scales[local+3],scales[local+6]);
    float meanBed=scales[local+9];
    if(other[0]==0xffffffffu) { nn=meanState; nn.z=0.0f; znn=meanBed; }
    else if(other[0]==0xfffffffeu) { nn=meanState; nn.x=pow(meanState.z*meanState.z/g,1.0f/3.0f); znn=meanBed; }
    else {uint4 c=node_geometry(other[0],layout.y);float3 b=mw_basis(c,pN);nn=mw_state(tree,other[0],b);znn=mw_bed(tree,other[0],b);}
    if(other[1]==0xffffffffu) { ee=meanState; ee.y=0.0f; zee=meanBed; }
    else if(other[1]==0xfffffffeu) { ee=meanState; ee.x=pow(meanState.y*meanState.y/g,1.0f/3.0f); zee=meanBed; }
    else {uint4 c=node_geometry(other[1],layout.y);float3 b=mw_basis(c,pE);ee=mw_state(tree,other[1],b);zee=mw_bed(tree,other[1],b);}
    if(other[2]==0xffffffffu) { ss=meanState; ss.z=0.0f; zss=meanBed; }
    else if(other[2]==0xfffffffeu) { ss=meanState; ss.x=pow(meanState.z*meanState.z/g,1.0f/3.0f); zss=meanBed; }
    else {uint4 c=node_geometry(other[2],layout.y);float3 b=mw_basis(c,pS);ss=mw_state(tree,other[2],b);zss=mw_bed(tree,other[2],b);}
    if(other[3]==0xffffffffu) { ww=meanState; ww.y=0.0f; zww=meanBed; }
    else if(other[3]==0xfffffffeu) { ww=meanState; ww.x=pow(meanState.y*meanState.y/g,1.0f/3.0f); zww=meanBed; }
    else {uint4 c=node_geometry(other[3],layout.y);float3 b=mw_basis(c,pW);ww=mw_state(tree,other[3],b);zww=mw_bed(tree,other[3],b);}

    // dg2_update.cu updates only leaves whose own or a neighbour's mean depth
    // (the ghost's for a boundary face) reaches tol_h; the others keep their
    // state (zero residual here).
    bool wet=meanState.x>=tolerance;
    for(uint face=0;face<4;++face) {
        float hMean;
        if(other[face]==0xffffffffu) hMean=meanState.x;
        else if(other[face]==0xfffffffeu) {
            float qn=(face==0||face==2)?meanState.z:meanState.y;
            hMean=pow(qn*qn/g,1.0f/3.0f);
        } else hMean=tree[other[face]*12]-tree[other[face]*12+9];
        wet=wet||hMean>=tolerance;
    }
    edgeRate[index]=0.0f;
    if(!wet) {
        for(uint k=0;k<9;++k) rhs[index*9+k]=0.0f;
        return;
    }
    float zne=max(zn,znn),zeeI=max(ze,zee),zse=max(zs,zss),zwe=max(zw,zww);
    float3 nStar=mw_star(n,zn,zne,tolerance),nnStar=mw_star(nn,znn,zne,tolerance);
    float3 eStar=mw_star(e,ze,zeeI,tolerance),eeStar=mw_star(ee,zee,zeeI,tolerance);
    float3 sStar=mw_star(s,zs,zse,tolerance),ssStar=mw_star(ss,zss,zse,tolerance);
    float3 wStar=mw_star(w,zw,zwe,tolerance),wwStar=mw_star(ww,zww,zwe,tolerance);
    float3 Fn=mw_hll(nStar,nnStar,1,g,tolerance);
    float3 Fe=mw_hll(eStar,eeStar,0,g,tolerance);
    float3 Fs=mw_hll(ssStar,sStar,1,g,tolerance);
    float3 Fw=mw_hll(wwStar,wStar,0,g,tolerance);
    // CUDA evaluates the numerical flux directly from the hydrostatic star
    // states. The bed-source terms below provide the balance; adding a second
    // hydrostatic pressure correction here double-counts that source.
    // volume rate out through domain-edge faces (closed, FREE or boundary)
    edgeRate[index]=(other[0]>=0xfffffffeu?Fn.x*dx:0.0f)-(other[2]>=0xfffffffeu?Fs.x*dx:0.0f)
                   +(other[1]>=0xfffffffeu?Fe.x*dy:0.0f)-(other[3]>=0xfffffffeu?Fw.x*dy:0.0f);
    float3 u0x=.5f*(eStar+wStar),u1x=(eStar-wStar)/(2.0f*mw_sqrt3);
    float3 u0y=.5f*(nStar+sStar),u1y=(nStar-sStar)/(2.0f*mw_sqrt3);
    residual[0]=-(Fe-Fw)/dx-(Fn-Fs)/dy;
    residual[1]=-mw_sqrt3/dx*(Fe+Fw-mw_physical(u0x-u1x,0,g,tolerance)-mw_physical(u0x+u1x,0,g,tolerance));
    residual[2]=-mw_sqrt3/dy*(Fs+Fn-mw_physical(u0y-u1y,1,g,tolerance)-mw_physical(u0y+u1y,1,g,tolerance));

    float etaE=e.x+ze,etaW=w.x+zw,etaN=n.x+zn,etaS=s.x+zs;
    float zStarE=zeeI-max(0.0f,-(etaE-zeeI));
    float zStarW=zwe-max(0.0f,-(etaW-zwe));
    float zStarN=zne-max(0.0f,-(etaN-zne));
    float zStarS=zse-max(0.0f,-(etaS-zse));
    float z1x=(zStarE-zStarW)/(2.0f*mw_sqrt3);
    float z1y=(zStarN-zStarS)/(2.0f*mw_sqrt3);
    residual[0].y += -2.0f*mw_sqrt3*g*u0x.x*z1x/dx;
    residual[1].y += -2.0f*mw_sqrt3*g*u1x.x*z1x/dx;
    residual[0].z += -2.0f*mw_sqrt3*g*u0y.x*z1y/dy;
    residual[2].z += -2.0f*mw_sqrt3*g*u1y.x*z1y/dy;
    for (uint field=0; field<3; ++field) for (uint mode=0; mode<3; ++mode)
        rhs[index*9+field*3+mode]=residual[mode][field];
}

// Instantaneous outward volume rate on tagged north FREE faces. The host
// combines the two SSP-RK2 stage rates with equal weights for mass accounting.
kernel void mwdg2_boundary_flux(device const float *scales [[buffer(0)]],
                               device const uint *active [[buffer(1)]],
                               device const uint *neighbours [[buffer(2)]],
                               device float *rate [[buffer(3)]],
                               constant uint2 &layout [[buffer(4)]],
                               constant float4 &physics [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
    if (index>=layout.x) return;
    uint node=active[index];
    uint faceCount=layout.x*4;
    uint begin=neighbours[index*4],end=neighbours[index*4+1];
    uint recordsStart=(faceCount+4u)&~3u;
    device const uint4 *segments=(device const uint4 *)(neighbours+recordsStart);
    uint4 cell=node_geometry(node,layout.y);
    float dy=physics.y*cell.z;
    float total=0.0f;
    for (uint segment=begin; segment<end; ++segment) if(segments[segment].x==0xfffffffeu)
      for (uint gauss=0; gauss<2; ++gauss) {
        float segmentStart=float(segments[segment].y),segmentLength=float(segments[segment].z-segments[segment].y);
        float tangent=(segmentStart+segmentLength*(.5f+(gauss?1.0f:-1.0f)/(2*mw_sqrt3)))/float(cell.z);
        float2 point=float2(float(cell.x)+tangent*float(cell.z),float(cell.y+cell.z));
        float3 basis=mw_basis(cell,point);
        float3 local=mw_state(scales,node,basis);
        float3 remote=local;
        remote.x=pow(local.z*local.z/physics.z,1.0f/3.0f);
        float bed=mw_bed(scales,node,basis);
        float3 ls=mw_star(local,bed,bed,physics.w);
        float3 rs=mw_star(remote,bed,bed,physics.w);
        float massFlux=mw_hll(ls,rs,1,physics.z,physics.w).x;
        total += massFlux*dy*segmentLength/(2.0f*float(cell.z));
    }
    rate[index]=total;
}

// CUDA's adaptive loop clears water on NoData leaves before rainfall and the
// flux update. NoData transitions are protected to finest cells by the host
// preflags, so the cell-centre lookup identifies the complete leaf.
kernel void mwdg2_drain_nodata(device float *scales [[buffer(0)]],
                               device const uint *active [[buffer(1)]],
                               device const float *dem [[buffer(2)]],
                               device float *drainedVolume [[buffer(3)]],
                               constant uint4 &layout [[buffer(4)]],
                               constant float2 &physics [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
    if (index >= layout.x) return;
    drainedVolume[index]=0.0f;
    uint node=active[index];
    uint4 cell=node_geometry(node,layout.y);
    uint x=cell.x+cell.z/2, y=cell.y+cell.z/2;
    if (x>=layout.z || y>=layout.w) return;
    uint row=layout.w-1-y;
    if (abs(dem[row*layout.z+x]-physics.x)<1.0f) {
        uint p=node*12;
        float h=scales[p]-scales[p+9];
        drainedVolume[index]=h*physics.y*float(cell.z*cell.z);
        scales[p]=scales[p+9]; scales[p+1]=scales[p+10]; scales[p+2]=scales[p+11];
        scales[p+3]=scales[p+4]=scales[p+5]=0.0f;
        scales[p+6]=scales[p+7]=scales[p+8]=0.0f;
    }
}

// SSP RK2 stage on compact leaves. Bed coefficients never evolve: they are
// taken from the tree.
kernel void mwdg2_rk(device const float *initial [[buffer(0)]],
                     device const float *current [[buffer(1)]],
                     device const float *rhs [[buffer(2)]],
                     device const uint *active [[buffer(3)]],
                     device float *output [[buffer(4)]],
                     device const float *bed [[buffer(5)]],
                     constant uint &count [[buffer(6)]],
                     constant float2 &stage [[buffer(7)]],
                     uint index [[thread_position_in_grid]]) {
    if (index>=count) return;
    uint node=active[index];
    for (uint mode=0; mode<12; ++mode) {
        float value=mode<9 ? current[node*12+mode] : bed[node*12+mode];
        if (mode<9) value=stage.y*initial[node*12+mode]+(1-stage.y)*(value+stage.x*rhs[index*9+mode]);
        output[node*12+mode]=value;
    }
}

inline float mw_minmod(float a, float b, float c) {
    if (a*b>0.0f && a*c>0.0f)
        return copysign(min(abs(a),min(abs(b),abs(c))),a);
    return 0.0f;
}
inline float3 mw_component(device const float *u, uint node, uint field) {
    uint p=node*12+field*3;
    return float3(u[p],u[p+1],u[p+2]);
}
inline float mw_eval_component(float3 c, float3 basis) { return dot(c,basis); }

// CUDA adaptive limit_slopes/get_limited_slopes: only finest leaves are
// limited; the Krivodonova jump detector decides whether to apply minmod.
// Results are written separately so adjacent leaves read an immutable state,
// matching CUDA's assembled neighbour snapshot.
kernel void mwdg2_limit_slopes(device const float *u [[buffer(0)]],
                               device const uint *active [[buffer(1)]],
                               device const uint *neighbours [[buffer(2)]],
                               device float *limited [[buffer(3)]],
                               device const float *tree [[buffer(4)]],
                               device const float *plainTree [[buffer(5)]],
                               device const uint *frictioned [[buffer(6)]],
                               constant uint4 &layout [[buffer(7)]],
                               constant float4 &physics [[buffer(8)]],
                               constant float &maxDepth [[buffer(9)]],
                               uint index [[thread_position_in_grid]]) {
    if (index>=layout.x) return;
    if (!frictioned[index]) tree=plainTree;
    uint node=active[index];
    uint4 cell=node_geometry(node,layout.y);
    uint out=index*6;
    uint p=node*12;
    // CUDA limit_slopes only operates on finest-level cells and skips padding.
    if (cell.w!=layout.y || cell.x>=layout.z || cell.y>=layout.w) {
        limited[out]=u[p+1]; limited[out+1]=u[p+2];
        limited[out+2]=u[p+4]; limited[out+3]=u[p+5];
        limited[out+4]=u[p+7]; limited[out+5]=u[p+8];
        return;
    }
    float3 local[3]={mw_component(u,node,0),mw_component(u,node,1),mw_component(u,node,2)};
    float3 z=float3(u[p+9],u[p+10],u[p+11]);
    float3 remoteMean[4][3], remoteTrace[4][3];
    float remoteH[4];
    float2 points[4]={float2(float(cell.x)+0.5f*cell.z,float(cell.y+cell.z)),
                      float2(float(cell.x+cell.z),float(cell.y)+0.5f*cell.z),
                      float2(float(cell.x)+0.5f*cell.z,float(cell.y)),
                      float2(float(cell.x),float(cell.y)+0.5f*cell.z)};
    for(uint face=0;face<4;++face) {
        uint faceCount=layout.x*4;
        uint begin=neighbours[index*4+face];
        uint recordsStart=(faceCount+4u)&~3u;
        device const uint4 *segments=(device const uint4 *)(neighbours+recordsStart);
        uint other=mw_neighbor_at_local_level(segments[begin].x,cell.w,layout.y);
        if(other<0xfffffffeu) {
            for(uint f=0;f<3;++f) remoteMean[face][f]=mw_component(tree,other,f);
            uint4 otherCell=node_geometry(other,layout.y);
            float3 b=mw_basis(otherCell,points[face]);
            for(uint f=0;f<3;++f)
                remoteTrace[face][f]=float3(mw_eval_component(remoteMean[face][f],b),0,0);
            remoteH[face]=remoteMean[face][0].x-tree[other*12+9];
        } else {
            // CUDA adaptive ghost mean. For FREE, use the critical-depth
            // ghost and retain momentum; default boundaries zero the normal
            // momentum in the corresponding flux branch below.
            float h=local[0].x-z.x;
            float eta=local[0].x;
            bool freeFace=other==0xfffffffeu;
            if(freeFace) {
                float qn=(face==0||face==2)?local[2].x:local[1].x;
                h=pow(qn*qn/physics.z,1.0f/3.0f), eta=h+z.x;
            }
            remoteMean[face][0]=float3(eta,0,0);
            remoteMean[face][1]=float3((!freeFace&&(face==1||face==3))?0.0f:local[1].x,0,0);
            remoteMean[face][2]=float3((!freeFace&&(face==0||face==2))?0.0f:local[2].x,0,0);
            for(uint f=0;f<3;++f) remoteTrace[face][f]=remoteMean[face][f];
            remoteH[face]=h;
        }
    }
    float3 faceBasis[4]={float3(1,0,mw_sqrt3),float3(1,mw_sqrt3,0),
                         float3(1,0,-mw_sqrt3),float3(1,-mw_sqrt3,0)};
    float minHx=min(local[0].x-z.x,min(remoteH[1],remoteH[3]));
    float minHy=min(local[0].x-z.x,min(remoteH[0],remoteH[2]));
    bool limitX=0.05f*maxDepth<minHx;
    bool limitY=0.05f*maxDepth<minHy;
    for(uint field=0;field<3;++field) {
        float3 c=local[field];
        float normX=max(abs(c.x-c.y),abs(c.x+c.y));
        float normY=max(abs(c.x-c.z),abs(c.x+c.z));
        float dse=normX>1e-12f?abs(remoteTrace[1][field].x-mw_eval_component(c,faceBasis[1]))/(0.5f*physics.x*normX):0.0f;
        float dsw=normX>1e-12f?abs(remoteTrace[3][field].x-mw_eval_component(c,faceBasis[3]))/(0.5f*physics.x*normX):0.0f;
        float dsn=normY>1e-12f?abs(remoteTrace[0][field].x-mw_eval_component(c,faceBasis[0]))/(0.5f*physics.y*normY):0.0f;
        float dss=normY>1e-12f?abs(remoteTrace[2][field].x-mw_eval_component(c,faceBasis[2]))/(0.5f*physics.y*normY):0.0f;
        float sx=(max(dse,dsw)<physics.w)?c.y:mw_minmod(c.y,c.x-remoteMean[3][field].x,remoteMean[1][field].x-c.x);
        float sy=(max(dsn,dss)<physics.w)?c.z:mw_minmod(c.z,c.x-remoteMean[2][field].x,remoteMean[0][field].x-c.x);
        if(field==0) {
            // CUDA limits eta in its assembled h+z view, then converts the
            // limited slopes back to depth for that temporary view. This
            // persistent Metal tree stores eta, so keep eta_limited here;
            // subtracting z here would store h while later kernels subtract
            // z again when recovering depth.
            limited[out]=limitX?sx:c.y;
            limited[out+1]=limitY?sy:c.z;
        } else {
            uint k=out+2+(field-1)*2;
            limited[k]=limitX?sx:c.y;
            limited[k+1]=limitY?sy:c.z;
        }
    }
}

kernel void mwdg2_apply_limited_slopes(device float *u [[buffer(0)]],
                                      device const uint *active [[buffer(1)]],
                                      device const float *limited [[buffer(2)]],
                                      constant uint &count [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
    if(index>=count) return;
    uint p=active[index]*12, q=index*6;
    u[p+1]=limited[q]; u[p+2]=limited[q+1];
    u[p+4]=limited[q+2]; u[p+5]=limited[q+3];
    u[p+7]=limited[q+4]; u[p+8]=limited[q+5];
}

// Match CUDA adaptive MWDG2: preserve updated h and slopes exactly, and clear
// momentum coefficients when the cell-mean depth falls below tol_h.
kernel void mwdg2_positivity(device float *scales [[buffer(0)]],
                             device const uint *active [[buffer(1)]],
                             device uint *invalid [[buffer(2)]],
                             constant uint &count [[buffer(3)]],
                             constant float &tolerance [[buffer(4)]],
                             uint index [[thread_position_in_grid]]) {
    if (index>=count) return;
    uint p=active[index]*12;
    float mean=scales[p]-scales[p+9];
    bool bad=!isfinite(mean);
    for (uint k=0; k<12; ++k) bad |= !isfinite(scales[p+k]);
    invalid[index]=bad;
    if (mean<tolerance) {
        for (uint k=3; k<9; ++k) scales[p+k]=0;
    }
}

// Reconstruct the four cell-centred output fields on the rectangular LISFLOOD
// raster from the adaptive leaves. Output planes are H, HU, HV and DEM; the
// `owners` map is built over the padded 2^L square, while only xsz*ysz cells
// are written. `geometry = {xsz, ysz, side, maxLevel}`.
// Prepare raster-scale loss fields from current leaf means. CUDA adaptive
// losses evaluate the nonlinear soil law at every finest cell using that
// owning leaf's mean depth, then restrict the depth change to the leaf.
kernel void mwdg2_loss_prepare(device const float *state [[buffer(0)]],
                               device const uint *owners [[buffer(1)]],
                               device float *depth [[buffer(2)]],
                               device float *qx [[buffer(3)]],
                               device float *qy [[buffer(4)]],
                               constant uint4 &layout [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
    const uint cells = layout.z * layout.w;
    if (index >= cells) return;
    const uint x = index % layout.z;
    const uint row = index / layout.z;
    const uint y = layout.w - 1 - row;
    const uint code = morton_dilate(x) | (morton_dilate(y) << 1);
    const uint node = owners[code];
    const uint p = node * 12;
    depth[index] = max(0.0f, state[p] - state[p + 9]);
    qx[index] = state[p + 3];
    qy[index] = state[p + 6];
}

// Restrict raster-scale infiltration/evaporation to each active leaf and
// apply the same dry-face momentum rules as CUDA apply_losses_adaptive.
kernel void mwdg2_restrict_losses(device float *state [[buffer(0)]],
                                 device const uint *active [[buffer(1)]],
                                 device const float *depth [[buffer(2)]],
                                 device const float *dem [[buffer(3)]],
                                 device float *residual [[buffer(4)]],
                                 constant uint4 &layout [[buffer(5)]], // count,L,xsz,ysz
                                 constant float2 &physics [[buffer(6)]], // nodata,threshold
                                 uint index [[thread_position_in_grid]]) {
    if (index >= layout.x) return;
    const uint node = active[index];
    const uint4 cell = node_geometry(node, layout.y);
    const uint xEnd = min(cell.x + cell.z, layout.z);
    const uint yEnd = min(cell.y + cell.z, layout.w);
    const uint p = node * 12;
    const float h = state[p] - state[p + 9];
    float loss = 0.0f;
    uint valid = 0;
    for (uint y = cell.y; y < yEnd; ++y) {
        const uint row = layout.w - 1 - y;
        for (uint x = cell.x; x < xEnd; ++x) {
            const uint raster = row * layout.z + x;
            if (abs(dem[raster] - physics.x) < 1e-6f) continue;
            loss += h - depth[raster];
            ++valid;
        }
    }
    if (valid == 0) return;
    const float next = max(0.0f, h - loss / float(valid));
    const float correction = (loss - float(valid) * (h - next)) / float(valid);
    for (uint y = cell.y; y < yEnd; ++y) {
        const uint row = layout.w - 1 - y;
        for (uint x = cell.x; x < xEnd; ++x) {
            const uint raster = row * layout.z + x;
            if (abs(dem[raster] - physics.x) < 1e-6f) continue;
            residual[raster] += correction;
        }
    }
    if (next == h) return;
    const float oldDepth = max(h, 0.0f);
    const float scale = next > physics.y && oldDepth > 0.0f ? next / oldDepth : 0.0f;
    state[p] = state[p + 9] + next;
    const float h1x = state[p + 1] - state[p + 10];
    const float h1y = state[p + 2] - state[p + 11];
    const bool dry = next <= physics.y || mw_sqrt3 * abs(h1x) >= next ||
                     mw_sqrt3 * abs(h1y) >= next;
    if (dry) {
        state[p + 1] = state[p + 10];
        state[p + 2] = state[p + 11];
        state[p + 4] = state[p + 5] = 0.0f;
        state[p + 7] = state[p + 8] = 0.0f;
    } else {
        state[p + 4] *= scale; state[p + 5] *= scale;
        state[p + 7] *= scale; state[p + 8] *= scale;
    }
    state[p + 3] *= scale;
    state[p + 6] *= scale;
}

// Rain forcing is raster-scale and applied before the next flow CFL/step,
// using the previously selected dt (the first increment is deliberately 0).
// Per-leaf restriction keeps heterogeneous dynamic rain conservative.
kernel void mwdg2_apply_rain(device float *state [[buffer(0)]],
                             device const uint *active [[buffer(1)]],
                             device const float *rainRate [[buffer(2)]],
                             device const float *dem [[buffer(3)]],
                             constant uint4 &layout [[buffer(4)]], // count,L,xsz,ysz
                             constant float3 &forcing [[buffer(5)]], // dt,nodata,clampNegative
                             uint index [[thread_position_in_grid]]) {
    if (index >= layout.x || forcing.x <= 0.0f) return;
    const uint node = active[index];
    const uint4 cell = node_geometry(node, layout.y);
    const uint xEnd = min(cell.x + cell.z, layout.z);
    const uint yEnd = min(cell.y + cell.z, layout.w);
    const uint p = node * 12;
    const float h = max(0.0f, state[p] - state[p + 9]);
    float incrementSum = 0.0f;
    uint valid = 0;
    for (uint y = cell.y; y < yEnd; ++y) {
        const uint row = layout.w - 1 - y;
        for (uint x = cell.x; x < xEnd; ++x) {
            const uint raster = row * layout.z + x;
            if (abs(dem[raster] - forcing.y) < 1e-6f) continue;
            float increment = rainRate[raster] * forcing.x;
            if (forcing.z > 0.5f) increment = max(-h, increment);
            else increment = max(0.0f, increment);
            incrementSum += increment;
            ++valid;
        }
    }
    if (valid) state[p] += incrementSum / float(valid);
}

kernel void mwdg2_project_raster(device const float *scales [[buffer(0)]],
                                 device const uint *owners [[buffer(1)]],
                                 device float *output [[buffer(2)]],
                                 constant uint4 &geometry [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    uint cells=geometry.x*geometry.y;
    if(index>=cells) return;
    uint x=index%geometry.x;
    uint row=index/geometry.x;
    uint y=geometry.y-1-row;
    uint code=morton_dilate(x)+(morton_dilate(y)<<1);
    uint node=owners[code];
    uint4 cell=node_geometry(node,geometry.w);
    float2 point=float2(float(x)+.5f,float(y)+.5f);
    float3 basis=mw_basis(cell,point);
    uint p=node*12;
    float eta=dot(float3(scales[p],scales[p+1],scales[p+2]),basis);
    float qx=dot(float3(scales[p+3],scales[p+4],scales[p+5]),basis);
    float qy=dot(float3(scales[p+6],scales[p+7],scales[p+8]),basis);
    float z=mw_bed(scales,node,basis);
    output[index]=eta-z;
    output[cells+index]=qx;
    output[2*cells+index]=qy;
    output[3*cells+index]=z;
}

// CUDA DG2 CFL estimate evaluated from each adaptive leaf's mean state. CUDA
// seeds each leaf with initial_tstep, then overwrites wet leaves with their
// local CFL step; dry leaves retain the seed and can cap the global minimum.
kernel void mwdg2_cfl(device const float *scales [[buffer(0)]],
                      device const uint *active [[buffer(1)]],
                      device float *localDt [[buffer(2)]],
                      constant uint2 &layout [[buffer(3)]], // count, maxLevel
                      constant float4 &physics [[buffer(4)]], // dx, dy, g, CFL
                      constant float &depthThreshold [[buffer(5)]],
                      constant float &maxDt [[buffer(6)]],
                      uint index [[thread_position_in_grid]]) {
    if(index>=layout.x) return;
    uint node=active[index];
    uint4 cell=node_geometry(node,layout.y);
    uint p=node*12;
    float h=scales[p]-scales[p+9];
    if(!isfinite(h) || h<depthThreshold) { localDt[index]=maxDt; return; }
    float qx=scales[p+3],qy=scales[p+6];
    float speed=sqrt(max(physics.z*h,0.0f));
    float dx=physics.x*cell.z,dy=physics.y*cell.z;
    float dtX=physics.w*dx/(abs(qx/h)+speed);
    float dtY=physics.w*dy/(abs(qy/h)+speed);
    localDt[index]=min(dtX,dtY);
}

// friction_implicit.cu for the leaves themselves: a leaf whose mean depth is
// below tol_h or mean speed below tol_s only has its mean discharge zeroed (its
// slopes are kept) and does not update its neighbour copies (flag 0); the
// others get the full update below (flag 1). params = {g, dt, tol_h, tol_s}.
kernel void mwdg2_friction_own(device float *scales [[buffer(0)]],
                               device const uint *active [[buffer(1)]],
                               device const float *manning [[buffer(2)]],
                               device uint *frictioned [[buffer(3)]],
                               constant uint &count [[buffer(4)]],
                               constant float4 &params [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
    if(index>=count) return;
    uint p=active[index]*12;
    float h=scales[p]-scales[p+9];
    bool belowDepth=h<params.z;
    float ux=belowDepth?0.0f:scales[p+3]/h, uy=belowDepth?0.0f:scales[p+6]/h;
    bool belowSpeed=abs(ux)<params.w && abs(uy)<params.w;
    frictioned[index]=!(belowDepth||belowSpeed);
    if(belowDepth||belowSpeed) { scales[p+3]=0.0f; scales[p+6]=0.0f; return; }
    float n=manning[index],g=params.x,dt=params.y;
    float z0=scales[p+9],zx=scales[p+10],zy=scales[p+11];
    float h0=scales[p]-z0,hx=scales[p+1]-zx,hy=scales[p+2]-zy;
    float2 q0=mw_friction_point(h0,float2(scales[p+3],scales[p+6]),n,g,dt,params.z,params.w);
    float2 qxm=mw_friction_point(h0-hx,float2(scales[p+3]-scales[p+4],scales[p+6]-scales[p+7]),n,g,dt,params.z,params.w);
    float2 qxp=mw_friction_point(h0+hx,float2(scales[p+3]+scales[p+4],scales[p+6]+scales[p+7]),n,g,dt,params.z,params.w);
    float2 qym=mw_friction_point(h0-hy,float2(scales[p+3]-scales[p+5],scales[p+6]-scales[p+8]),n,g,dt,params.z,params.w);
    float2 qyp=mw_friction_point(h0+hy,float2(scales[p+3]+scales[p+5],scales[p+6]+scales[p+8]),n,g,dt,params.z,params.w);
    scales[p+3]=q0.x; scales[p+6]=q0.y;
    scales[p+4]=.5f*(qxp.x-qxm.x); scales[p+7]=.5f*(qxp.y-qxm.y);
    scales[p+5]=.5f*(qyp.x-qym.x); scales[p+8]=.5f*(qyp.y-qym.y);
}

// CUDA's implicit Manning source is applied to the mean and the four DG2
// directional Gauss pairs, then the momentum slopes are reconstructed.
kernel void mwdg2_friction(device float *scales [[buffer(0)]],
                           device const uint *active [[buffer(1)]],
                           device const float *manning [[buffer(2)]],
                           constant uint &count [[buffer(3)]],
                           constant float4 &params [[buffer(4)]],
                           uint index [[thread_position_in_grid]]) {
    if(index>=count) return;
    uint p=active[index]*12;
    float n=manning[index],g=params.x,dt=params.y;
    float tolH=params.z,tolSpeed=params.w;
    float z0=scales[p+9],zx=scales[p+10],zy=scales[p+11];
    float h0=scales[p]-z0,hx=scales[p+1]-zx,hy=scales[p+2]-zy;
    float2 q0=mw_friction_point(h0,float2(scales[p+3],scales[p+6]),n,g,dt,tolH,tolSpeed);
    float2 qxm=mw_friction_point(h0-hx,float2(scales[p+3]-scales[p+4],scales[p+6]-scales[p+7]),n,g,dt,tolH,tolSpeed);
    float2 qxp=mw_friction_point(h0+hx,float2(scales[p+3]+scales[p+4],scales[p+6]+scales[p+7]),n,g,dt,tolH,tolSpeed);
    float2 qym=mw_friction_point(h0-hy,float2(scales[p+3]-scales[p+5],scales[p+6]-scales[p+8]),n,g,dt,tolH,tolSpeed);
    float2 qyp=mw_friction_point(h0+hy,float2(scales[p+3]+scales[p+5],scales[p+6]+scales[p+8]),n,g,dt,tolH,tolSpeed);
    scales[p+3]=q0.x; scales[p+6]=q0.y;
    scales[p+4]=.5f*(qxp.x-qxm.x); scales[p+7]=.5f*(qxp.y-qxm.y);
    scales[p+5]=.5f*(qyp.x-qym.x); scales[p+8]=.5f*(qyp.y-qym.y);
}

// ---------------------------------------------------------------------------
// HWFV1 (cuda/adaptive with solver_type HWFV1): Haar wavelets on the cell
// means and a first-order finite-volume update. The tree keeps the MWDG2
// layout (12 floats per node) with all slope modes zero.

// encode_and_thresh_flow (HWFV1): parent mean = average of the four children;
// details alpha, beta, gamma (encode_details.cuh), stored in the alpha_0,
// beta_0 and gamma_0 slots of the MWDG2 detail layout.
kernel void hwfv1_encode_level(device float *scales [[buffer(0)]],
                               device float *details [[buffer(1)]],
                               constant uint2 &level [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
    if (index >= level.y) return;
    uint node = level.x + index;
    for (uint field = 0; field < 4; ++field) {
        float c0 = scales[(node*4+1)*12 + field*3], c1 = scales[(node*4+2)*12 + field*3];
        float c2 = scales[(node*4+3)*12 + field*3], c3 = scales[(node*4+4)*12 + field*3];
        scales[node*12 + field*3] = .25f*((c0 + c2) + (c1 + c3));
        scales[node*12 + field*3 + 1] = scales[node*12 + field*3 + 2] = 0.0f;
        device float *d = details + node*36 + field*9;
        for (uint k = 0; k < 9; ++k) d[k] = 0.0f;
        d[0] = .25f*((c0 - c2) + (c1 - c3));
        d[3] = .25f*((c0 + c2) - (c1 + c3));
        d[6] = .25f*((c0 - c2) - (c1 - c3));
    }
}

// fv1_update.cu: leaves whose own or neighbour mean depth reaches tol_h are
// updated; the others keep their state and initial_tstep as CFL value.
// view: the leaf's own state after friction; tree: neighbour means (friction
// applied, as CUDA's d_neighbours). params = {dt, initial_tstep, CFL, tol_q}.
kernel void hwfv1_update(device const float *view [[buffer(0)]],
                         device const uint *active [[buffer(1)]],
                         device const uint *neighbours [[buffer(2)]],
                         device float *out [[buffer(3)]],
                         device const float *tree [[buffer(4)]],
                         device float *localDt [[buffer(5)]],
                         device float *edgeRate [[buffer(6)]],
                         device const float *plainTree [[buffer(7)]],
                         device const uint *frictioned [[buffer(8)]],
                         constant uint2 &layout [[buffer(9)]],
                         constant float4 &physics [[buffer(10)]],
                         constant float4 &params [[buffer(11)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= layout.x) return;
    if (!frictioned[index]) tree = plainTree;
    uint node = active[index];
    uint4 cell = node_geometry(node, layout.y);
    float dx = physics.x*cell.z, dy = physics.y*cell.z, g = physics.z, tol = physics.w;
    uint p = node*12;
    for (uint k = 0; k < 12; ++k) out[p+k] = view[p+k];
    localDt[index] = params.y;
    edgeRate[index] = 0.0f;
    float z = view[p+9], h = view[p] - z;
    float3 U = float3(h, view[p+3], view[p+6]);
    uint recordsStart = (layout.x*4u + 4u) & ~3u;
    device const uint4 *segments = (device const uint4 *)(neighbours + recordsStart);
    float3 Un[4]; float zn[4];
    bool wet = h >= tol;
    for (uint face = 0; face < 4; ++face) {
        uint begin = neighbours[index*4 + face];
        uint other = begin < neighbours[index*4 + face + 1]
            ? mw_neighbor_at_local_level(segments[begin].x, cell.w, layout.y) : 0xffffffffu;
        uint normal = (face == 0 || face == 2) ? 2u : 1u;  // N/S: qy, E/W: qx
        if (other == 0xffffffffu) {            // add_ghost_cells: closed default
            Un[face] = U; Un[face][normal] = 0.0f; zn[face] = z;
        } else if (other == 0xfffffffeu) {     // FREE: critical depth, momentum kept
            Un[face] = U; Un[face].x = pow(U[normal]*U[normal]/g, 1.0f/3.0f); zn[face] = z;
        } else {
            uint o = other*12;
            zn[face] = tree[o+9];
            Un[face] = float3(tree[o] - tree[o+9], tree[o+3], tree[o+6]);
        }
        wet = wet || Un[face].x >= tol;
    }
    if (!wet) return;
    float eta = h + z;
    float zN = max(z, zn[0]), zE = max(z, zn[1]), zS = max(z, zn[2]), zW = max(z, zn[3]);
    float3 nNeg = mw_star(U, z, zN, tol), nPos = mw_star(Un[0], zn[0], zN, tol);
    float3 eNeg = mw_star(U, z, zE, tol), ePos = mw_star(Un[1], zn[1], zE, tol);
    float3 sNeg = mw_star(Un[2], zn[2], zS, tol), sPos = mw_star(U, z, zS, tol);
    float3 wNeg = mw_star(Un[3], zn[3], zW, tol), wPos = mw_star(U, z, zW, tol);
    float3 Fn = mw_hll(nNeg, nPos, 1, g, tol), Fe = mw_hll(eNeg, ePos, 0, g, tol);
    float3 Fs = mw_hll(sNeg, sPos, 1, g, tol), Fw = mw_hll(wNeg, wPos, 0, g, tol);
    uint edge[4];
    for (uint face = 0; face < 4; ++face) {
        uint begin = neighbours[index*4 + face];
        edge[face] = begin < neighbours[index*4 + face + 1] ? segments[begin].x : 0xffffffffu;
    }
    edgeRate[index] = (edge[0] >= 0xfffffffeu ? Fn.x*dx : 0.0f) - (edge[2] >= 0xfffffffeu ? Fs.x*dx : 0.0f)
                    + (edge[1] >= 0xfffffffeu ? Fe.x*dy : 0.0f) - (edge[3] >= 0xfffffffeu ? Fw.x*dy : 0.0f);
    // get_bed_src_x / get_bed_src_y
    float zStarW = zW - max(0.0f, -(eta - zW)), zStarE = zE - max(0.0f, -(eta - zE));
    float zStarS = zS - max(0.0f, -(eta - zS)), zStarN = zN - max(0.0f, -(eta - zN));
    float bedX = g*0.5f*(wPos.x + eNeg.x)*(zStarE - zStarW)/dx;
    float bedY = g*0.5f*(sPos.x + nNeg.x)*(zStarN - zStarS)/dy;
    float dt = params.x;
    float qx = U.y, qy = U.z;
    h  = h  - dt*(Fe.x - Fw.x)/dx;
    qx = qx - dt*((Fe.y - Fw.y)/dx + bedX);
    qy = qy - dt*(Fe.z - Fw.z)/dx;
    h  = h  - dt*(Fn.x - Fs.x)/dy;
    qx = qx - dt*(Fn.y - Fs.y)/dy;
    qy = qy - dt*((Fn.z - Fs.z)/dy + bedY);
    bool belowDepth = h < tol;
    bool belowDischarge = abs(qx) < params.w && abs(qy) < params.w;
    if (belowDepth || belowDischarge) { qx = 0.0f; qy = 0.0f; }
    if (!belowDepth) {
        float c = sqrt(g*h);
        localDt[index] = min(params.z*dx/(abs(qx/h) + c), params.z*dy/(abs(qy/h) + c));
    }
    out[p] = h + z; out[p+3] = qx; out[p+6] = qy;
}

// add_ghost_cells.cu, prescribed boundaries on finest leaves at the domain
// edge: HFIX/HVAR set the leaf's own mean depth to max(0, level - z0); QFIX/
// QVAR set it to q*dt/dx_finest (as CUDA does). cell[i] = type*4 + side, or
// -1 (sides N,E,S,W = 0..3; types as metal_adaptive_bc.h). delta[i] receives
// the volume added (m3). params = {dt, dx_finest, cell area, unused}.
kernel void adaptive_boundary_cells(device float *view [[buffer(0)]],
                                    device const uint *active [[buffer(1)]],
                                    device const int *cell [[buffer(2)]],
                                    device float *delta [[buffer(3)]],
                                    constant uint &count [[buffer(4)]],
                                    constant float4 &inlet [[buffer(5)]],
                                    constant float4 &params [[buffer(6)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    delta[index] = 0.0f;
    int c = cell[index];
    if (c < 0) return;
    uint p = active[index]*12;
    int type = c / 4, side = c % 4;
    float z = view[p+9], h = view[p] - z, value = inlet[side];
    float hNew = (type == 2 || type == 3) ? max(0.0f, value - z) : value*params.x/params.y;
    view[p] = hNew + z;
    delta[index] = (hNew - h)*params.z;
}
