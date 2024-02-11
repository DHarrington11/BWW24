import numpy as np
from torch import tensor
import torch.nn.functional as nnf
import xarray as xr

# Useful website for features
# https://custom-scripts.sentinel-hub.com


# This function doesn't seem to work right now
def mask_data(data):
    """
    Masks out pixels with specific Sentinel-2 Scene Classification (SCL) values in the input data.

    Parameters:
    data (xarray Dataset): Input data containing the SCL variable.

    Returns:
    xarray Dataset: The input data with the masked pixels removed.
    """
    # 0  - NODATA
    # 1  - Saturated or Defective
    # 2  - Dark Areas
    # 3  - Cloud Shadow
    # 4  - Vegetation
    # 5  - Bare Ground
    # 6  - Water
    # 7  - Unclassified
    # 8  - Cloud
    # 9  - Definitely Cloud
    # 10 - Thin Cloud
    # 11 - Snow or Ice

    cloud_mask = (
        (data.SCL != 0)
        & (data.SCL != 1)
        & (data.SCL != 3)
        & (data.SCL != 6)
        & (data.SCL != 8)
        & (data.SCL != 9)
        & (data.SCL != 10)
    )

    cleaned_data = data.where(cloud_mask, data.mean(dim="time"))
    return cleaned_data


def rvi(data):
    """
    Calculates the Radar Vegetation Index (RVI) using the vertical and horizontal polarization data.

    Parameters:
    data (xarray Dataset): Input data containing the vertical and horizontal polarization data.

    Returns:
    numpy array: The RVI image.
    """
    vh = data["vh"]  # Vertical polarization data
    vv = data["vv"]  # Horizontal polarization data

    dop = vv / (vv + vh)
    rvi = (np.sqrt(dop)) * ((4 * vh) / (vv + vh))

    return rvi


# Create a mask for no data, saturated data, clouds, cloud shadows, and water


def smi(data):
    """
    Calculates the Soil Moisture Index (SMI) using the vertical and horizontal polarization data.

    Parameters:
    data (xarray Dataset): Input data containing the vertical and horizontal polarization data.

    Returns:
    numpy array: The SMI image.
    """
    # Calculate SMI
    vh = data["vh"]  # Vertical polarization data
    vv = data["vv"]  # Horizontal polarization data

    smi = (vh - vv) / (vh + vv)

    return smi


def ndvi(data):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) using the Sentinel-2 red and near-infrared bands.

    Parameters:
    data (xarray Dataset): Input data containing the red and near-infrared bands.

    Returns:
    numpy array: The NDVI image.
    """
    b08 = data["B08"]
    b04 = data["B04"]

    ndvi_image = (b08 - b04) / (b08 + b04)

    return ndvi_image


def msi(data):
    """
    Calculates the Moisture Stress Index (MSI) using Sentinel-2 bands 8 and 11.

    Parameters:
    data (xarray Dataset): Input data containing the bands 8 and 11.

    Returns:
    numpy array: The MSI image.
    """
    b08 = data["B08"]
    b11 = data["B11"]

    msi = b11 / b08

    return msi


def ari(data):
    """
    Calculates the Atmospherically Resistant Vegetation Index (ARVI) using Sentinel-2 bands 3 and 5.

    Parameters:
    data (xarray Dataset): Input data containing the bands 3 and 5.

    Returns:
    numpy array: The ARVI image.
    """
    b03 = data["B03"]
    b05 = data["B05"]

    ari = (1 / b03) - (1 / b05)

    return ari


def evi(data):
    """
    Calculates the Enhanced Vegetation Index (EVI) using the Sentinel-2 red, near-infrared, and blue bands.

    Parameters:
    data (xarray Dataset): Input data containing the red, near-infrared, and blue bands.

    Returns:
    numpy array: The EVI image.
    """
    # Calculate EVI
    red = data["B04"]  # Red band
    nir = data["B08"]  # Near-infrared band
    blue = data["B02"]  # Blue band

    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

    return evi


def feature_engineering(data):
    """
    Run feature engineering stack on data.

    Parameters:
    data (xarray Dataset): Input data containing all bands.

    Returns:
    xarray Dataset: The data with all features computed
    """
    data["RVI"] = rvi(data)
    data["SMI"] = smi(data)
    data["NDVI"] = ndvi(data)
    data["MSI"] = msi(data)
    data["ARI"] = ari(data)
    data["EVI"] = evi(data)

    return data


def xarray_to_tensor(xarray):
    """
    Convert an xarray DataArray into a PyTorch tensor.

    Args:
        xarray (xarray.DataArray): A DataArray containing the data to be converted.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    data_arr = np.vstack([xarray[i].dropna(dim="time").values for i in xarray])
    return tensor(data_arr)


def resize_tensor(tensor, x=32, y=32):
    """
    Resize a PyTorch tensor using bilinear interpolation.

    Args:
        tensor (torch.Tensor): The tensor to be resized.
        x (int): The new height of the tensor.
        y (int): The new width of the tensor.

    Returns:
        torch.Tensor: The resized tensor.
    """
    channels = tensor.shape[0]
    tensor = nnf.interpolate(tensor.expand(1, -1, -1, -1), (x, y))
    return tensor.resize(channels, x, y)


def pipeline(data):
    """
    A pipeline for processing satellite image data.

    Args:
        data (pd.DataFrame): The data to be processed.

    Returns:
        torch.Tensor: The processed tensor.
    """
    features = feature_engineering(data)

    s1_data = features[["RVI", "SMI"]]
    s2_data = features[["NDVI", "MSI", "ARI", "EVI", "SCL"]]

    s2_masked = mask_data(s2_data)
    s2_masked = s2_masked[["NDVI", "MSI", "ARI", "EVI"]]

    merged = xr.merge([s1_data, s2_masked])
    tensor = xarray_to_tensor(merged)
    resized_tensor = resize_tensor(tensor)
    return resized_tensor
