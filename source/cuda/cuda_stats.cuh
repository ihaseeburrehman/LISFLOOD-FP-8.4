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

	/** volume of the initial water: the first entry's error is measured from it */
	void start(NUMERIC_TYPE* H);

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
	/** water volume (m3) and wet area (m2) of the real cells, all ranks */
	void measure
	(
		NUMERIC_TYPE* H,
		NUMERIC_TYPE& volume,
		NUMERIC_TYPE& area
	);

	void zero_cumulative_mass();

	Geometry& geometry;
	NUMERIC_TYPE depth_threshold;
	bool ghost;        /**< FV1/FV2/DG2 layout with a ghost-cell ring */
	bool distributed;
	double* d_sums;    /**< depth sum and wet cells, from domain_sums */
	double previous_rain = 0.0; /**< rain volume at the previous entry */
	MassStats* d_cumulative_mass; /**< through BCs/PSs since last write */
	NUMERIC_TYPE cumulative_time;
//	MassStats* d_instantaneous_mass; /**< through BCs/PSs during timestep*/
//	NUMERIC_TYPE previous_volume = C(0.0);
};

}
}
