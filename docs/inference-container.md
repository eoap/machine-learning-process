# Inference container:

This module enables users to create an inference pipeline that takes a Sentinel-2 STAC Item from the [Planetary Computer](https://planetarycomputer.microsoft.com/api/stac/v1/collections), and generates a binary mask TIFF image using a pre-trained CNN model. For details on how the model was trained, refer to the [training container documentation](./training-container.md).



## **`Make Inference` Module:**

**Inputs**:
 
- `input_reference`: A list of Sentinel-2 product references from [Planetary Computer](https://planetarycomputer.microsoft.com/api/stac/v1/collections). Note: the inference application provides accurate results only when the Sentinel-2 product has low or no cloud cover. High cloud coverage may significantly reduce prediction accuracy.

**Outputs**:

- `{STAC_ITEM_ID}_classified.tif`: A binary `.tif` image in `COG` format containing the full-resolution land cover classification predicted by the model, with each pixel assigned to a land cover class as defined in the table below. 
- `overview_{STAC_ITEM_ID}_classified.tif`: A binary `.tif` image in `COG` format containing lower-resolution overview of the classification result, generated to support fast visualisation and efficient browsing across zoom levels. 
- `STAC objects`: STAC objects related to the provided masks, including STAC Catalog and STAC Item.

*Land Cover Classes*
| Class ID | Class Name            |
|----------|-----------------------|
| 0        | AnnualCrop            |
| 1        | Forest                |
| 2        | HerbaceousVegetation  |
| 3        | Highway               |
| 4        | Industrial            |
| 5        | Pasture               |
| 6        | PermanentCrop         |
| 7        | Residential           |
| 8        | River                 |
| 9        | SeaLake               |
| 10       | No Data               |


## How the Application Works

The application begins by reading the input Sentinel-2 STAC Item(s) from the [Planetary Computer](https://planetarycomputer.microsoft.com/api/stac/v1/collections) and then extracting the 12 common Sentinel-2 spectral bands (see table below), ordered to match those expected by the trained ML model. 

*Sentinel-2 Spectral Bands*
| Index | Asset Key  | Asset Common Name |
|-------|------------|-------------------|
| 1     | B01        | Coastal           |
| 2     | B02        | Blue              |
| 3     | B03        | Green             |
| 4     | B04        | Red               |
| 5     | B05        | Red Edge          |
| 6     | B06        | Red Edge          |
| 7     | B07        | Red Edge          |
| 8     | B08        | NIR               |
| 9     | B8A        | Narrow NIR        |
| 10    | B09        | Water Vapor       |
| 11    | B11        | SWIR 1 (16)       |
| 12    | B12        | SWIR 2 (22)       |

As part of the preprocessing, all selected bands are resampled to a consistent spatial resolution of 10 meters.

The pipeline then proceeds with a sliding window approach: it reads and stacks small image chips from the resampled bands (in the specified order), forming multi-band input arrays. These image chips are fed to the trained CNN model, which predicts the corresponding LC class for each chip.

At the end of the process, the application generates:
- The LC classification prediction map (COG mask)
- A visual overview image
- An updated STAC Catalog and Item containing metadata and references to the output files.