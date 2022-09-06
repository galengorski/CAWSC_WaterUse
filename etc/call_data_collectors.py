# Created by dherbert at 12:53 PM 5/21/2020 using PyCharm
# Description: Runs a test case for the climate & census data collectors

import sys

sys.path.insert(
    0,
    r"C:\Users\dherbert\Desktop\pycharm\CAWSC_WaterUse\census_data_collector\census-data-collector",
)
sys.path.insert(
    0,
    r"C:\Users\dherbert\Desktop\pycharm\CAWSC_WaterUse\climate_data_collector",
)
import GridMETDataCollector as gmet
import geopandas
import shapefile
from censusdc.utils import CensusTimeSeries


def get_training_data(
    huc2=13,
    service_area_id="1852627",
    apikey="1234",
    thread_num=8,
    start_year=2015,
    end_year=2016,
):
    """
    Runs the climate & census data collectors
    huc2 = int value of huc2 you want to assess (default=13)
    service_area_id = int value of service area you want to assess (default=1852627), eventually all service areas incl.
    thread_num = number of threads you want to use for ts.get_timeseries()
    apikey = unique apikey (default is 1234). You have to update this.
    years = list of range of years you want to assess, default=list(range(2015,2016))
    """
    # (1) read in service area shapefile attribute table
    shp_file = r"shapefiles\serviceArea_1852627.shp"
    service_area_df = geopandas.read_file(shp_file)

    # (2) Get all Service areas that exist within HUC2 & get unique values
    curr_huc2 = service_area_df.loc[service_area_df["HUC2"] == str(huc2)]
    service_area_list = curr_huc2["GNIS_ID"].unique()
    print(service_area_list)

    # (3) format data/add data for census collector
    shp = shapefile.Reader(shp_file)
    polygon = shp.shape(0)
    ts = CensusTimeSeries(shp_file, apikey, field="GNIS_ID")

    # (4) Loop over service areas & download census data (right now we're just testing 1 service area)
    # for serv_area in service_area_list:
    serv_area = service_area_id
    df = ts.get_timeseries(
        serv_area,
        verbose=2,
        polygons=polygon,
        multithread=True,
        thread_pool=thread_num,
        years=list(range(start_year, end_year)),
    )
    print(df)
    df.to_csv("1852627_census.csv")

    # (4) Download climate data
    yearFilter = str(start_year) + "-" + str(end_year)
    climateFilter = ["pet", "pr", "tmmx"]
    zoneShpPath = shp_file
    climateData = gmet.get_data(
        zoneShpPath,
        zone_field="GNIS_ID",
        year_filter=yearFilter,
        climate_filter=climateFilter,
        multiprocessing=True,
    )
    print(climateData)

    # (5) Link data to Water use data
    # pass


# ------------------------------------
# Download HUC12 data
# -----------------------------------

# test function
get_training_data(
    huc2=13,
    service_area_id="1852627",
    apikey="1234",
    start_year=2010,
    end_year=2019,
)
