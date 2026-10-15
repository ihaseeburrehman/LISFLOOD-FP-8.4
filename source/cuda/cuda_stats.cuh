#pragma once
#include "../lisflood.h"
#include "../geometry.h"
#include "stats.h"

namespace lis
{
namespace cuda
{

class StatsCollector
{
public:
	StatsCollector
	(
		Geometry& geometry,
		NUMERIC_TYPE DepthThresh,
		int acceleration,
		int element_offset = 0,
		int element_count = 0
	);

	void zero_instantaneous_mass();

	void accumulate_mass();
	void accumulate_mass(NUMERIC_TYPE step_dt);

	StatsEntry create_entry
	(
		::Solver& solver,
		NUMERIC_TYPE* H
	);

	MassStats* instantaneous_mass();

	~StatsCollector();

	NUMERIC_TYPE previous_volume = C(0.0);
	MassStats* d_instantaneous_mass; /**< through BCs/PSs during timestep*/

	/** infiltration and evaporation (cuda_losses), m3: set before create_entry */
	double interval_loss = 0.0; /**< since the previous entry */
	double total_loss = 0.0;    /**< since the start */

private:
	NUMERIC_TYPE area
	(
		NUMERIC_TYPE* H
	);

	NUMERIC_TYPE volume
	(
		NUMERIC_TYPE* H
	);

	void zero_cumulative_mass();

	Geometry& geometry;
	int element_offset;
	int elements;
	bool distributed;
	NUMERIC_TYPE* histogram_levels;
	int* histogram_counts;
	void* area_temp;
	size_t area_bytes;
	void* volume_temp;
	size_t volume_bytes;
	NUMERIC_TYPE* d_volume;
	MassStats* d_cumulative_mass; /**< through BCs/PSs since last write */
	NUMERIC_TYPE cumulative_time;
//	MassStats* d_instantaneous_mass; /**< through BCs/PSs during timestep*/
//	NUMERIC_TYPE previous_volume = C(0.0);
};

}
}
