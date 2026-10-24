#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "../../lisflood.h"

namespace lis
{
namespace cuda
{
namespace acc_nugrid
{

typedef struct RainfallUniform
{
	NUMERIC_TYPE        value;
	char                timeseries[32] = { '\0' };
	int                 row = 0;

	RainfallUniform
	(
		const NUMERIC_TYPE& value
	)
	{
		this->value = value;
	}

	// lines of the rainfall file, as fgets reads them
	std::vector<std::string> lines;
	bool loaded = false;

	void load
	(
		const char* rainfilename
	)
	{
		FILE* fp = fopen(rainfilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening time varying rainfall %s, file: %s, line: %d.\n", rainfilename, __FILE__, __LINE__);
			exit(-1);
		}

		char str[255];
		while (fgets(str, sizeof(str), fp) != NULL) lines.push_back(str);
		fclose(fp);
		loaded = true;
	}

	void update_rainfall
	(
		const char* rainfilename,
		const NUMERIC_TYPE& time_now
	)
	{
		char str[255];
		char buf[32] = { '\0' };

		// the file is read once; each step replays its lines as fgets gave
		// them (a read past the end leaves str as it was, as fgets does)
		if (!loaded) load(rainfilename);
		size_t next = 0;
		auto read_line = [&](char* s, int) { if (next < lines.size()) strcpy(s, lines[next].c_str()); next++; };

		char* timeseriesptr = timeseries;

		read_line(str, sizeof(str));
		sscanf(str, "%s", buf);

		NUMERIC_TYPE value_1 = C(0.0);
		NUMERIC_TYPE value_2 = C(0.0);

		NUMERIC_TYPE t_1 = C(0.0);
		NUMERIC_TYPE t_2 = C(0.0);

		int num_rows_timeseries = 0;
		int time_multiplier = 1;

		read_line(str, sizeof(str));

		sscanf(str, "%d %s", &num_rows_timeseries, buf);

		time_multiplier = (!strncmp(buf, "seconds", 7)) ? 1 : (!strncmp(buf, "minutes", 7)) ? 60 : 3600;

		for (int i = 0; i < row + 1; i++) read_line(str, sizeof(str));

		sscanf(str, "%" NUM_FMT " %" NUM_FMT, &value_1, &t_1);

		read_line(str, sizeof(str));

		sscanf(str, "%" NUM_FMT " %" NUM_FMT, &value_2, &t_2);

		if (time_now > t_2*time_multiplier)
		{
			row++;

			t_1 = t_2*time_multiplier;
			value_1 = value_2;

			read_line(str, sizeof(str));

			sscanf(str, "%" NUM_FMT " %" NUM_FMT, &value_2, &t_2);
		}

		if (row + 1 >= num_rows_timeseries)
		{
			row--;
			value = C(0.0);
		}
		else {
			value = value_1 + (value_2 - value_1) / ((t_2 - t_1) * time_multiplier) * (time_now - t_1 * time_multiplier);
			value /= (C(1000.0) * C(3600.0));
		}
	}

} RainfallUniform;

}
}
}