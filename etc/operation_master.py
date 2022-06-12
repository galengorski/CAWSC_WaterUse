"""

(1) CensusFromSBToWB.py
path: CensusFromSBToWB.py
use: copy data from census data collector and put them in the census folder.

(2) join swud and non-swud data
path: C:\work\water_use\add_no_swud_data_to_annual_training.py
use: joind sud and non-swud data in one file

(3) Generate annual data (census + climate)
path: C:\work\water_use\CAWSC_WaterUse\etc\generate_annual_training_dataset.py
use: Use it for annual only. For Monthly, use annual census and then compute climate from the following function.


(4) we can obtain annual, monthly, and daily climate from
path:C:\work\water_use\CAWSC_WaterUse\etc\assemble_climate_only.py
or C:\work\water_use\CAWSC_WaterUse\etc\assemble_climate_from_one_folder.py

(5) correct population
path: C:\work\water_use\ml_experiments\annual_v_0_0\0_population_auditing_fixed2.py

(5) run pr_training script

(6) run
"""