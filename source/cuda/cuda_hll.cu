#include "cuda_hll.cuh"
#include "cuda_device_functions.cuh"
#include "cuda_solver.cuh"

__device__ lis::cuda::FlowVector lis::cuda::HLL::y
(
	const FlowVector& u_neg,
	const FlowVector& u_pos
)
{
	return detail::hll_y(u_neg, u_pos);
}

__device__ lis::cuda::FlowVector lis::cuda::HLL::x
(
	const FlowVector& u_neg,
	const FlowVector& u_pos
)
{
	return detail::hll_x(u_neg, u_pos);
}
