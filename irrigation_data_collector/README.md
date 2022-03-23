# Climate_Scripts

Routines for collecting GRIDMET climate data for shapefiles 

Note, paths will need to be modified depending on the location of pilot study area shapefiles and desired output directory structure

/data/ subdirectory will need to be unzipped (too large to push unless zipped)

collect_all.py: collects climate data for pilot study areas

GridMETDataCollector3.py: climate data collector

make_pickles.py: creates nested data structure

postprocess.py: processes climate data into separate CSVs

stats.py: derives monthly and annual statistics, as well as vapor pressure deficit 
