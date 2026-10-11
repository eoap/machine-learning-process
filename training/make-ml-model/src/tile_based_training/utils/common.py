import base64
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

# import boto3  # noqa: F401
import duckdb
import joblib

# from fs_s3fs import S3FS
import numpy as np
import rasterio
import requests
import tensorflow as tf
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from rasterio.env import Env
from rasterio.errors import RasterioIOError

logger = logging.getLogger(__name__)
logging.getLogger("httpchecksum").setLevel(logging.INFO)


def configure_gpu():
    """Enable GPU memory growth and return device name."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ GPU memory growth enabled.")
            return "/GPU:0"
        except RuntimeError as e:
            logger.info(f"❌ Error enabling GPU memory growth: {e}")
            return "/CPU:0"

    return "/CPU:0"


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as error:
        raise ValueError("yaml file is empty") from error
    except Exception as e:
        raise e


def write_yaml(path_to_yaml: Path, **args) -> None:
    """Writes a dictionary to a YAML file.

    Args:
        path_to_yaml (Path): Path to the YAML file.
        data (dict): Data to be written to the YAML file.

    Raises:
        ValueError: If the provided data is empty.
        Exception: If any other error occurs.
    """

    # data = dict(args["args"])
    args = args["args"]
    data = {
        "BATCH_SIZE": args["BATCH_SIZE"],
        "CLASSES": args["CLASSES"],
        "DECAY": args["DECAY"],  ### float
        "EPOCHS": args["EPOCHS"],
        "EPSILON": args["EPSILON"],
        "LEARNING_RATE": args["LEARNING_RATE"],
        "LOSS": args["LOSS"],
        "MEMENTUM": args["MEMENTUM"],
        "OPTIMIZER": args["OPTIMIZER"],
        "REGULARIZER": args["REGULARIZER"],
        "SAMPLES_PER_CLASS": args["SAMPLES_PER_CLASS"],
        "stac_reference": args["stac_reference"],
    }
    try:
        with path_to_yaml.open("w") as yaml_file:
            yaml.dump(data, yaml_file)

        logger.info(f"YAML file: {path_to_yaml} written successfully")

    except Exception as e:
        logger.error(f"Error writing to YAML file {path_to_yaml}: {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_txt(path: Path) -> list:
    """load txt files data

    Args:
        path (Path): path to txt file

    Returns:
        List: data as list of items' id
    """
    with open(path) as file:
        data = [line.strip() for line in file]
        random.shuffle(data)
    return data


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


TOKEN_CACHE = {"token": None, "expires_at": 0}


def get_token(url, **kwargs):
    data = {**kwargs}

    response = requests.post(url, data=data, timeout=30)

    if response.status_code == 200:
        json_data = response.json()
        access_token = json_data.get("access_token")
        return access_token
    else:
        logger.error(f"Request for a token failed with status code {response.status_code}")


def get_headers():
    payload = {
        "client_id": "ai-extensions",
        "username": "ai-extensions-user",
        "password": os.environ.get("IAM_PASSWORD"),
        "grant_type": "password",
    }
    token = get_token(url=os.environ.get("IAM_URL"), **payload)
    headers = {"Authorization": f"Bearer {token}"}
    return headers


class UserSettings:
    """

    This code defines a UserSettings class that has methods for reading JSON files,
    matching regular expressions, and setting environment variables for an S3 service.
    The set_s3_environment method sets environment variables for an S3 service based
    on a given URL.

    class for reading JSON files, matching regular expressions,
    and setting environment variables for an S3 service"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.settings = self.read_json_file(file_path)

    @staticmethod
    def load_json(path: Path) -> ConfigBox:
        """load json files data

        Args:
            path (Path): path to json file

        Returns:
            ConfigBox: data as class attributes instead of dict
        """
        with open(path) as f:
            content = json.load(f)

        logger.info(f"json file loaded succesfully from: {path}")
        return ConfigBox(content)

    @staticmethod
    def read_json_file(file_path):
        """read a JSON file and return the contents as a dictionary"""
        with open(file_path, encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def match_regex(regex, string):
        """match a regular expression to a string and return the match object"""
        return re.search(regex, string)

    @staticmethod
    def set_s3_vars(s3_service):
        """set environment variables for an S3 service"""
        os.environ["AWS_ACCESS_KEY_ID"] = s3_service["AccessKey"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = s3_service["SecretKey"]
        os.environ["AWS_DEFAULT_REGION"] = s3_service["Region"]
        os.environ["AWS_REGION"] = s3_service["Region"]
        os.environ["AWS_S3_ENDPOINT"] = s3_service["ServiceURL"]

    def set_s3_environment(self, url):
        """set environment variables for an S3 service based on a given URL"""
        for _, s3_service in self.settings["S3"]["Services"].items():
            self.set_s3_vars(s3_service)
            # if self.match_regex(s3_service["UrlPattern"], url):
            #     self.set_s3_vars(s3_service)
            #     break


# class CustomStacIO(DefaultStacIO):
#     def __init__(self):
#         session = boto3.session.Session()

#         self.s3 = session.client(
#             service_name="s3",
#             region_name=settings.region_name,  # type: ignore
#             use_ssl=True,
#             endpoint_url=f"https://{settings.endpoint_url}",
#             aws_access_key_id=settings.aws_access_key_id,  # type: ignore
#             aws_secret_access_key=settings.aws_secret_access_key,
#         )

#     def read_text(self, source, *args, **kwargs):
#         parsed = urlparse(source)
#         if parsed.scheme == "s3":
#             bucket = parsed.netloc
#             key = parsed.path[1:]

#             return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
#         else:
#             return super().read_text(source, *args, **kwargs)

#     def write_text(self, dest, txt, *args, **kwargs):
#         parsed = urlparse(dest)
#         if parsed.scheme == "s3":
#             bucket = parsed.netloc
#             key = parsed.path[1:]
#             self.s3.put_object(
#                 Body=txt.encode("UTF-8"),
#                 Bucket=bucket,
#                 Key=key,
#                 ContentType="application/geo+json",
#             )
#         else:
#             super().write_text(dest, txt, *args, **kwargs)


# def s3_config_notebook(usersetting_path):
#     user = UserSettings(str(usersetting_path))
#     settings = user.settings
#     # print(settings.settings)

#     for key, value in settings["S3"]["Services"].items():
#         bucket_name = urlparse(value["UrlPattern"]).netloc
#         break
#     user.set_s3_environment(f"s3://{bucket_name}")
#     if os.environ.get("AWS_REGION"):
#         logger.info("S3 bucket is configured")
#     else:
#         logger.error("S3 bucket's configuration is failed")
#         sys.exit(1)


# def s3_bucket_config():
#     bucket_name = os.environ.get("BUCKET_NAME")
#     # for key, value in settings.S3.Services.items():
#     #     bucket_name = urlparse(value['UrlPattern']).netloc
#     #     break

#     logger.info(f"Bucket name is {bucket_name}")
#     ######## Configuration to access aws bucket to read images directly with gdal

#     # settings = UserSettings(str(usersetting_path))
#     # settings.set_s3_environment(f"s3://{bucket_name}")
#     if os.environ.get("AWS_REGION"):
#         logger.info("S3 bucket is configured")
#     else:
#         logger.error("S3 bucket's configuration is failed")
#         sys.exit(1)


# def rasterio_read(image_path: str):
#     full_path = f"/vsizip/vsicurl/{image_path}" if ".zip/" in image_path and image_path.startswith("http") else image_path

#     with Env(GDAL_DISABLE_READDIR_ON_OPEN='TRUE', VSI_CACHE='TRUE'):
#         with rasterio.open(full_path) as src:
#             total_bands = src.count
#             band_index_to_exclude = [11] if total_bands == 13 else []
#             band_indices = [b for b in range(1, total_bands + 1) if b not in band_index_to_exclude]

#             # Read the specified bands
#             data = src.read(band_indices)
#             data = data.astype("float32")

#     return data


def rasterio_read(image_path: str, max_retries: int = 3, backoff_base: int = 5):
    """
    Attempts to read a GeoTIFF file using rasterio with retries and exponential backoff.

    Parameters:
        image_path (str): Path to the file (excluding the /vsizip/vsicurl/ prefix).
        max_retries (int): Number of attempts before giving up.
        backoff_base (int): Base seconds for backoff (e.g., 5 → 5s, 10s, 20s...).

    Returns:
        data (np.ndarray): The read image data as a NumPy array (float32).
    """
    # print(os.getcwd())
    full_path = (
        f"/vsizip/vsicurl/{image_path}"
        if ".zip/" in image_path and image_path.startswith("http")
        else image_path
    )
    attempt = 0

    while attempt < max_retries:
        try:
            with (
                Env(
                    VSI_CACHE="TRUE",
                    GDAL_CACHEMAX=67108864,
                    CPL_VSIL_CURL_USE_HEAD="NO",
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_HTTP_TIMEOUT="30",
                ),
                rasterio.open(full_path) as src,
            ):
                total_bands = src.count
                band_index_to_exclude = [11] if total_bands == 13 else []
                band_indices = [
                    b for b in range(1, total_bands + 1) if b not in band_index_to_exclude
                ]
                data = src.read(band_indices).astype("float32")
                return data
        except RasterioIOError:
            wait_time = backoff_base * (2**attempt) + 5
            time.sleep(wait_time)
            attempt += 1

    raise RuntimeError(f"[ERROR] Unable to read {full_path} after {max_retries} attempts.")


# def rasterio_read(image_path: str):

#     read_url = f'/vsizip/vsicurl/{image_path}'

#     with rasterio.open(read_url) as src:
#         total_bands = src.count
#         band_index_to_exclude = [11] if total_bands == 13 else [] # Exclude band 11(cirrus band)
#         band_indices = [b for b in range(1, total_bands + 1) if b not in band_index_to_exclude]
#         data = src.read(band_indices)
#         data = data.astype("float32")

#     return data


def augmentation(data):
    # Random rotation (90, 180, 270 degrees)
    k = random.choice([0, 1, 2, 3])  # 0 means no rotation
    data = np.rot90(data, k=k, axes=(1, 2))

    # Random flip
    if random.choice([True, False]):
        data = np.flip(data, axis=1)  # vertical flip
    if random.choice([True, False]):
        data = np.flip(data, axis=2)  # horizontal flip
    data = data / 10000.0  # Normalize
    return data

    #### For GDAL
    # settings = S3Settings(
    #     region_name=region_name,
    #     endpoint_url=endpoint_url,
    #     aws_access_key_id=aws_access_key_id,
    #     aws_secret_access_key=aws_secret_access_key,
    # )

    # gdal.SetConfigOption("AWS_REGION", settings.region_name)
    # gdal.SetConfigOption("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    # gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
    # gdal.SetConfigOption("AWS_S3_ENDPOINT", settings.endpoint_url)
    # gdal.SetConfigOption("AWS_HTTPS", "YES")
    # gdal.SetConfigOption("AWS_VIRTUAL_HOSTING", "FALSE")


def duckdb_config():
    duckdb.install_extension("spatial")
    duckdb.load_extension("spatial")
    # s3_aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    # s3_aws_access_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    # s3_aws_region = os.environ.get("AWS_REGION")
    # s3_aws_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    # duckdb_s3_endpoint_url = s3_aws_endpoint_url.replace("https://", "")
    # Set environment variable for httpfs
    os.environ["BUILD_HTTPFS"] = "1"

    # Connect to DuckDB
    # connection = duckdb.connect()
    # # connection.execute("INSTALL httpfs")
    # # connection.execute("LOAD httpfs")
    # # connection.execute("INSTALL spatial")
    # # connection.execute("LOAD spatial")
    # # # Set correct S3 credentials
    # # connection.execute(f"SET s3_region = '{s3_aws_region}';")
    # # connection.execute(f"SET s3_access_key_id = '{s3_aws_access_key_id}';")
    # # connection.execute(f"SET s3_secret_access_key = '{s3_aws_access_secret_key}';")
    # # connection.execute(f"SET s3_endpoint = '{duckdb_s3_endpoint_url}';")  # No extra "https://"
    # # connection.execute("SET s3_use_ssl = true;")
    # connection.execute("SET enable_progress_bar = false;")

    # return connection


# def sql_generator(class_name,  geoparquet_asset_path, geometry= None):
#     # Use the geometry variable inside the CQL2 filter
#     cql2_filter = {
#         # "op": "and",
#         "args": [
#             {
#                 "op": "like",
#                 "args": [{"property": "id"}, f"{class_name}%"],
#             },
#             # {
#             #     "op": "s_intersects",
#             #     "args": [
#             #         {"property": "geometry"},
#             #         geometry,
#             #     ],
#             # },
#         ],
#     }

#     # Convert CQL2 filter to SQL WHERE clause
#     sql_where = to_sql_where(json_parse(cql2_filter), IdempotentDict())

#     # Define the SQL query with dynamically inserted sql_where

#     sql_query = f"SELECT * EXCLUDE(geometry), ST_AsWKB(geometry) as geometry FROM read_parquet('{geoparquet_asset_path}') WHERE {sql_where}"
#     return sql_query


def sql_generator(class_name, geoparquet_asset_path, samples_per_class=100):
    if class_name not in (
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ):
        raise ValueError("Unknown training class")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    query = "SELECT * EXCLUDE(geometry), ST_AsWKB(geometry) as geometry FROM read_parquet(?) WHERE id LIKE ? ORDER BY id LIMIT ?"
    return query, [geoparquet_asset_path, class_name + "%", int(samples_per_class)]
