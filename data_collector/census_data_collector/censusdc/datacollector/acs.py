from .cbase import CensusBase


class AcsVariables(object):
    """
    Small listing of common census variable names for querying data
    """
    population = 'B01003_001E'
    households = 'B11001_001E'
    households2 = "B19001_001E"
    income_lt_10k = "B19001_002E"
    income_10K_15k = "B19001_003E"
    income_15k_20k = "B19001_004E"
    income_20k_25k = "B19001_005E"
    income_25k_30k = "B19001_006E"
    income_30k_35k = "B19001_007E"
    income_35k_40k = "B19001_008E"
    income_40k_45k = "B19001_009E"
    income_45k_50k = "B19001_010E"
    income_50k_60k = "B19001_011E"
    income_60k_75k = "B19001_012E"
    income_75k_100k = "B19001_013E"
    income_100k_125k = "B19001_014E"
    income_125k_150k = "B19001_015E"
    income_150k_200k = "B19001_016E"
    income_gt_200k = "B19001_017E"
    median_income = "B19013_001E"
    tot_h_age = "B25034_001E"
    h_age_newer_2005 = "B25034_002E"
    h_age_2000_2004 = "B25034_003E"
    h_age_1990_1999 = "B25034_004E"
    h_age_1980_1989 = "B25034_005E"
    h_age_1970_1979 = "B25034_006E"
    h_age_1960_1969 = "B25034_007E"
    h_age_1950_1959 = "B25034_008E"
    h_age_1940_1949 = "B25034_009E"
    h_age_older_1939 = "B25034_010E"
    median_h_year = "B25036_001E"


AcsHR = {v: k for k, v in AcsVariables.__dict__.items()
         if not k.startswith("__")}


class Acs1(CensusBase):
    """
    Class to collect data from the Acs1 census using geojson features
    from TigerWeb

    Parameters
    ----------
    features: dict
        features from TigerWeb data collections
        {polygon_name: [geojson, geojson,...]}
    year : int
        census data year of the TigerWeb data
    apikey : str
        users specific census apikey (obtained from
        https://api.census.gov/data/key_signup.html)

    """
    def __init__(self, features, year, apikey):
        super(Acs1, self).__init__(features, year, apikey, 'acs1')

    def get_data(self, level='finest', variables=(), retry=100, verbose=True,
                 multiproc=False, multithread=False, thread_pool=4):
        """
        Method to get data from the Acs1 servers and set it to feature
        properties!

        Parameters
        ----------
        level : str
            determines the geographic level of data queried
            default is 'finest' available based on census dataset and
            the geoJSON feature information
        variables : list, tuple
            user specified Acs1 variables, default pulls variables from
            the AcsVariables class
        retry : int
            number of retries for HTTP connection issues before failure
        verbose : bool
            verbose operation mode
        multithread : bool
            boolean flag to allow multithreading of data collection
        multiproc : bool
            multiprocessing support using ray, linux only!
        thread_pool : int
            number of CPU threads to use during multithread operations

        """
        super(Acs1, self).get_data(level=level, variables=variables,
                                   retry=retry, verbose=verbose,
                                   multiproc=multiproc,
                                   multithread=multithread,
                                   thread_pool=thread_pool)


class Acs5(CensusBase):
    """
    Class to collect data from the Acs5 census using geojson features
    from TigerWeb

    Parameters
    ----------
    features: dict
        features from TigerWeb data collections
        {polygon_name: [geojson, geojson,...]}
    year : int
        census data year of the TigerWeb data
    apikey : str
        users specific census apikey (obtained from
        https://api.census.gov/data/key_signup.html)

    """
    def __init__(self, features, year, apikey):
        super(Acs5, self).__init__(features, year, apikey, 'acs5')

    def get_data(self, level='finest', variables=(), retry=100, verbose=True,
                 multiproc=False, multithread=False, thread_pool=4):
        """
        Method to get data from the Acs5 servers and set it to feature
        properties!

        Parameters
        ----------
        level : str
            determines the geographic level of data queried
            default is 'finest' available based on census dataset and
            the geoJSON feature information
        variables : list, tuple
            user specified Acs5 variables, default pulls variables from
            the AcsVariables class
        retry : int
            number of retries for HTTP connection issues before failure
        verbose : bool
            verbose operation mode
        multiproc : bool
            multiprocessing support using ray, linux only!
        multithread : bool
            boolean flag to allow multithreading of data collection
        thread_pool : int
            number of CPU threads to use during multithread operations
        """
        super(Acs5, self).get_data(level=level, variables=variables,
                                   retry=retry, verbose=verbose,
                                   multiproc=multiproc,
                                   multithread=multithread,
                                   thread_pool=thread_pool)
