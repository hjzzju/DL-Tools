import csv
import os
import logging
from pathlib import Path
import time
import requests

from pyxctools.constants import XC_BASE_URL


class XenoCanto:

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def query(self,
              search_terms: str = "",
              genus: str = None,
              recordist: str = None,
              country: str = None,
              location: str = None,
              remarks: str = None,
              latitude: float = None,
              longitude: float = None,
              box: str = None,
              background_species: str = None,
              type: str = None,
              catalogue_number: str = None,
              license: str = None,
              quality: str = None,
              area: str = None,
              since: str = None,
              year: str = None,
              month: str = None,
              page: int = None) -> dict:

        """
        Returns JSON from the API call with the given search terms.

        For details of each parameter, see https://www.xeno-canto.org/help/search.

        :return: A dictionary that represents the JSON returned by the xeno-canto API.
        """

        query_params = {"gen": genus,
                        "rec": recordist,
                        "cnt": country,
                        "loc": f'"{location}"' if location else None,  # Location must be wrapped in double quotes.
                        "rmk": remarks,
                        "lat": latitude,
                        "lon": longitude,
                        "box": box,
                        "also": background_species,
                        "type": type,
                        "nr": catalogue_number,
                        "lic": license,
                        "q": quality,
                        "area": area,
                        "since": since,
                        "year": year,
                        "month": month}

        # Build query in the weird format that the xeno-canto API takes.
        query = " ".join([search_terms] + [f"{name}:{var}" for name, var in query_params.items() if var])
        payload = {"query": query, "page": page}
        self.logger.debug(f"Sending request with parameters {payload}")

        r = requests.get(XC_BASE_URL, params=payload)
        r.raise_for_status()

        file_data = r.json()

        self.logger.info(f"Found {file_data['numRecordings']} recordings with "
                         f"{file_data['numSpecies']} species over "
                         f"{file_data['numPages']} pages.")

        if int(file_data["numRecordings"]) <= 0:
            raise Warning("No results!")

        self.logger.debug("Request successful.")

        return file_data

    def _get_dir_path(self, dir: str) -> Path:
        """
        Returns the absolute path to dir. If the directory does not exist, it is created. If it is none, dir is taken as
        the current working current working directory.

        :param dir: The directory to get the path to.
        :return: A path object representing the absolute path to dir.
        """

        if not dir:
            dir = os.getcwd()

        # Raises a FileNotFoundError if the directory does not exist.
        path = Path(dir).resolve()

        if not os.path.exists(path):
            self.logger.debug(f"Created new directory at {path}.")
            os.makedirs(path)

        return path

    def save_metadata(self, file_data: dict, dir: str = None) -> None:
        """
        Saves the result of a query to a CSV.

        :param file_data: Data to save, usually the value returned by the query method.
        :param dir: Directory to save to. If None, the current working directory is used.
        :return:
        """
        if int(file_data["numRecordings"]) <= 0:
            raise Exception("Empty metadata!")

        path = self._get_dir_path(dir)

        keys = file_data["recordings"][0].keys()

        # Save metadata
        with open(path / "metadata.csv", "w") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(file_data["recordings"])

        self.logger.info("Downloaded metadata.")

    def download_files(self,
                       page: int = None,
                       search_terms: str = "",
                       genus: str = None,
                       recordist: str = None,
                       country: str = None,
                       location: str = None,
                       remarks: str = None,
                       latitude: float = None,
                       longitude: float = None,
                       box: str = None,
                       background_species: str = None,
                       type: str = None,
                       catalogue_number: str = None,
                       license: str = None,
                       quality: str = None,
                       area: str = None,
                       since: str = None,
                       year: str = None,
                       month: str = None,

                       dir: str = "sounds",
                       save_metadata: bool = True) -> None:
        """
        Downloads audio files and metadata returned by xeno-canto with the given search_terms.

        For details of each parameter, see https://www.xeno-canto.org/help/search.

        :param save_metadata: Downloads and saves the metadata if true.
        :param dir: The name of the directory to download to.
        :return:
        """

        path = self._get_dir_path(dir)

        file_data = self.query(search_terms=search_terms,
                               genus=genus,
                               recordist=recordist,
                               country=country,
                               location=location,
                               remarks=remarks,
                               latitude=latitude,
                               longitude=longitude,
                               box=box,
                               background_species=background_species,
                               type=type,
                               catalogue_number=catalogue_number,
                               license=license,
                               quality=quality,
                               area=area,
                               since=since,
                               year=year,
                               month=month,
                               page=page)

        if int(file_data["numRecordings"]) <= 0:
            return

        # print(f"Downloading {file_data['numRecordings']} files...")

        # Download recording and write metadata
        for i,recording in enumerate(file_data["recordings"]):
            try:
                with requests.get(f"http:{recording['file']}", allow_redirects=True, stream=True) as r:
                    # Note that xeno-canto only supports mp3s
                    try:
                        with open(f"{path / recording['id']}.mp3", "wb") as f:
                            f.write(r.content)
                    except KeyboardInterrupt:
                        exit(0)
                    except:
                        print("err in write")
                        time.sleep(3)
            except KeyboardInterrupt:
                exit(0)
            except:
                print("err in page")
                print(page)
                time.sleep(3)
        # print(f"Finished downloads.")

        if save_metadata:
            self.save_metadata(file_data, dir=dir)
