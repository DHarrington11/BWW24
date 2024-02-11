import os
import pickle

import numpy as np
from odc.stac import stac_load
import pandas as pd
import planetary_computer as pc
import pystac_client
from tqdm import tqdm
import xarray as xr


def get_time_slice(csv_path):
    """
    Load a CSV of data and extract appropriate time slice.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file.

    Returns
    -------
    pandas.Series
        Series of appropriate time slices.
    """

    df = pd.read_csv(csv_path)

    # Challenge 1 Latitude and Longitudes are stored differently
    if "Season(SA = Summer Autumn, WS = Winter Spring)" in df.columns:
        df["Season(SA = Summer Autumn, WS = Winter Spring)"].replace(
            "SA", "2022-06-01/2023-12-01", inplace=True
        )
        df["Season(SA = Summer Autumn, WS = Winter Spring)"].replace(
            "WS", "2021-12-01/2022-06-01", inplace=True
        )
        return df["Season(SA = Summer Autumn, WS = Winter Spring)"]

    else:
        return pd.Series(["2022-01-01/2023-01-01" for _ in range(len(df))])


def get_long_lats(csv_path):
    """
    Load a CSV of latitude and longitude pairs into a pandas dataframe.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the latitude and longitude values.
    """

    df = pd.read_csv(csv_path)

    # Challenge 1 Latitude and Longitudes are stored differently
    if "Latitude and Longitude" in df.columns:
        return pd.DataFrame(
            [eval(row) for row in df["Latitude and Longitude"]],
            columns=["Latitude", "Longitude"],
        )

    else:
        return df[["Latitude", "Longitude"]]


def get_sentinel_data(
    bbox, collection, time_slice="2020-03-20/2020-03-21", n_samples=12
):
    """
    Load Sentinel satellite data from the Planetary Computer STAC API.

    Parameters
    ----------
    bbox : tuple
        A bounding box in the form (minx, miny, maxx, maxy).
    collection : str
        The choosen collection either Sentinel 1 RTC or Sentinel 2 L2A
    time_slice : str, optional
        A time slice in the format "start_time/end_time", by default "2020-03-20/2020-03-21".
    n_samples : int, optional
        The number of evenly spaced samples to take from the time frame

    Returns
    -------
    xarray.Dataset
        A dataset containing the loaded Sentinel data.

    """

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=time_slice,
    )

    items = list(search.get_all_items())
    items = [items[i] for i in np.linspace(0, len(items) - 1, n_samples).astype(int)]

    # Looks like you should be able to load them all at once but alas
    ds = stac_load(
        items,
        patch_url=pc.sign,
        bbox=bbox,
        crs="EPSG:4326",
        resolution=11132**-1,
    )

    return ds


def create_bbox(lat_long_row, size=15):
    """
    Create a bounding box around a given latitude and longitude point.

    Parameters
    ----------
    lat_long_row : tuple
        A tuple containing the latitude and longitude values.
    size : int, optional
        The size of the bounding box in units of 0.0001 degrees, by default 15.

    Returns
    -------
    tuple
        A bounding box in the form (minx, miny, maxx, maxy).

    """

    lat, long = lat_long_row[0], lat_long_row[1]

    # Default size of 15 gives roughly 34x34
    lat_0 = lat + 0.0001 * size
    lat_1 = lat - 0.0001 * size
    long_0 = long + 0.0001 * size
    long_1 = long - 0.0001 * size

    return (long_1, lat_1, long_0, lat_0)


def main(csv_path, output_path):
    """
    Load Sentinel satellite data for each latitude and longitude pair in a CSV file
    and save the resulting data as a set of pickled Xarray datasets.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing the latitude and longitude pairs.
    output_path : str
        The directory in which to save the pickled datasets.

    """

    # Load lat longs
    lat_longs = get_long_lats(csv_path)
    time_series = get_time_slice(csv_path)

    # Make directory for data
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Iterate over rows to save data
    for idx, row in tqdm(enumerate(lat_longs.values)):
        # create bbox
        bbox = create_bbox(row)

        # get data
        ds1 = get_sentinel_data(bbox, "sentinel-1-rtc", time_series.iloc[idx])
        ds2 = get_sentinel_data(bbox, "sentinel-2-l2a", time_series.iloc[idx])

        # Doesn't merge so well on dates so will be NaNs in output
        ds = xr.merge([ds1, ds2], compat="override")

        # Save data as pickle
        with open(f"{output_path}/{idx}.pickle", "wb") as handle:
            pickle.dump(ds, handle)


if __name__ == "__main__":
    print("Loading training data for challenge 1...")
    main("./data/challenge1_data.csv", "./data/challenge1/train")

    print("\nLoading training data for challenge 2...")
    main("./data/challenge2_data.csv", "./data/challenge2/train")

    print("\nLoading test data for challenge 1...")
    main("./data/challenge1_submission.csv", "./data/challenge1/test")

    print("\nLoading test data for challenge 1...")
    main("./data/challenge2_submission.csv", "./data/challenge2/test")