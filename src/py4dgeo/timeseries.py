from py4dgeo.epoch import Epoch, as_epoch
from py4dgeo.UpdateableZipFile import UpdateableZipFile
from py4dgeo.util import Py4DGeoError, find_file

import datetime
import json
import logging
import os
import tempfile
import zipfile

import numpy as np
from scipy.spatial import cKDTree


logger = logging.getLogger("py4dgeo")

# This version controls the on-disk format of a spatial time-series archive.
# It is intentionally independent of py4dgeo's package version.
SPATIAL_TIMESERIES_FILE_FORMAT_VERSION = 0


class SpatialTimeseries:
    """A time series of elevations sampled at fixed spatial locations.

    Parameters
    ----------
    filename : str or os.PathLike
        Archive filename. A new archive is created when the file does not
        exist.
    compress : bool, default=True
        Compress the NumPy arrays stored in the archive. Compression reduces
        disk use but increases the cost of appending epochs. Therefore, this
        is a tradeoff decision between disk space and runtime.
    force : bool, default=False
        Replace an existing archive with a new, empty time series.
    """

    def __init__(self, filename, compress=True, force=False):
        self.filename = find_file(filename, fatal=False)
        self.compress = bool(compress)

        self._corepoints = None
        self._elevations = None
        self._timestamps = None

        # If the filename does not exist, we create a new archive
        if force or not os.path.exists(self.filename):
            logger.info("Creating spatial time-series file %s", self.filename)
            with zipfile.ZipFile(self.filename, mode="w") as zf:
                zf.writestr(
                    "SPATIAL_TIMESERIES_FILE_FORMAT",
                    str(SPATIAL_TIMESERIES_FILE_FORMAT_VERSION),
                )
                zf.writestr("USE_COMPRESSION", str(self.compress))

        # Validate the archive and restore its storage configuration.
        try:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                version = int(zf.read("SPATIAL_TIMESERIES_FILE_FORMAT").decode())
                if version != SPATIAL_TIMESERIES_FILE_FORMAT_VERSION:
                    raise Py4DGeoError(
                        "Spatial time-series file format is out of date!"
                    )

                compression_setting = zf.read("USE_COMPRESSION").decode()
                if compression_setting not in ("True", "False"):
                    raise Py4DGeoError(
                        "Invalid compression setting in spatial time-series file"
                    )
                self.compress = compression_setting == "True"
        except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
            raise Py4DGeoError(
                f"Could not read spatial time-series file '{self.filename}'"
            ) from exc

    @property
    def corepoints(self):
        """The fixed spatial locations at which elevations are sampled."""
        if self._corepoints is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                # Double check that the reference has already been set
                if "corepoints.zip" not in zf.namelist():
                    raise Py4DGeoError("Corepoints for time series not yet set")

                # Extract it from the archive
                with tempfile.TemporaryDirectory() as tmp_dir:
                    corepoints_file = zf.extract("corepoints.zip", path=tmp_dir)
                    self._corepoints = Epoch.load(corepoints_file)

        return self._corepoints

    @corepoints.setter
    def corepoints(self, corepoints):
        """
        Set the corepoints for this timeseries (only possible once).
        They may be supplied as an :class:`Epoch` or as an ``(n, 3)`` NumPy array.
        """
        corepoints = as_epoch(corepoints)
        locations = np.asarray(corepoints.cloud)

        if locations.ndim != 2 or locations.shape[1] != 3:
            raise Py4DGeoError("Corepoints must be an array with shape (n, 3)")
        if not np.isfinite(locations[:, :2]).all():
            raise Py4DGeoError("Corepoint X/Y coordinates must be finite")
        if locations.shape[0] == 0:
            raise Py4DGeoError("At least one core point is required")

        with zipfile.ZipFile(self.filename, mode="a") as zf:
            # If we already have corepoints in the archive, the user should start a 
            # new timeseries instead
            if "corepoints.zip" in zf.namelist():
                raise Py4DGeoError(
                    "Corepoints cannot be changed - please start a new time series"
                )

            # Ensure that the search trees are built
            corepoints._validate_search_tree()

            # Write the corepoints into the archive
            with tempfile.TemporaryDirectory() as tmp_dir:
                corepoints_filename = os.path.join(tmp_dir, "corepoints.zip")
                corepoints.save(corepoints_filename)
                zf.write(corepoints_filename, arcname="corepoints.zip")

        self._corepoints = corepoints

    @corepoints.deleter
    def corepoints(self):
        self._corepoints = None

    @property
    def timestamps(self):
        """Absolute timestamps of the epochs, in elevation-column order."""
        if self._timestamps is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                if "timestamps.json" not in zf.namelist():
                    return []
                serialized = json.loads(zf.read("timestamps.json").decode())
                self._timestamps = [
                    datetime.datetime.fromisoformat(value) for value in serialized
                ]

        return self._timestamps

    @timestamps.setter
    def timestamps(self, values):
        """Set timestamps manually on an otherwise empty time series."""
        values = list(values)
        if not all(isinstance(value, datetime.datetime) for value in values):
            raise Py4DGeoError("Timestamps must be datetime.datetime instances")

        with zipfile.ZipFile(self.filename, mode="a") as zf:
            # If we already have timestamps in the archive, this is not possible
            if "timestamps.json" in zf.namelist():
                raise Py4DGeoError(
                    "Timestamps can only be set on a freshly created time series"
                )

            elevations_filename = self._numpy_filename("elevations")
            if elevations_filename in zf.namelist():
                raise Py4DGeoError(
                    "Timestamps must be set before assigning elevations"
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                timestamps_file = os.path.join(tmp_dir, "timestamps.json")
                with open(timestamps_file, "w", encoding="utf-8") as f:
                    json.dump([value.isoformat() for value in values], f)
                zf.write(timestamps_file, arcname="timestamps.json")

        self._timestamps = values

    @property
    def elevations(self):
        """Elevation time series with shape ``(n_corepoints, n_epochs)``."""
        if self._elevations is None:
            with zipfile.ZipFile(self.filename, mode="r") as zf:
                filename = self._numpy_filename("elevations")
                if filename not in zf.namelist():
                    return np.empty(
                        (self.corepoints.cloud.shape[0], 0), dtype=np.float64
                    )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    elevations_file = zf.extract(filename, path=tmp_dir)
                    if self.compress:
                        with np.load(elevations_file, allow_pickle=False) as data:
                            self._elevations = data["arr_0"]
                    else:
                        self._elevations = np.load(
                            elevations_file, allow_pickle=False
                        )

        return self._elevations

    @elevations.setter
    def elevations(self, values):
        """Set a precomputed elevation matrix once."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise Py4DGeoError("Elevations must be a two-dimensional array")
        if values.shape[0] != self.corepoints.cloud.shape[0]:
            raise Py4DGeoError(
                "The number of elevation rows must match the number of corepoints"
            )

        timestamps = self.timestamps
        if values.shape[1] != len(timestamps):
            raise Py4DGeoError(
                "Set timestamps first; the number of elevation columns must match "
                "the number of timestamps"
            )

        with zipfile.ZipFile(self.filename, mode="a") as zf:
            filename = self._numpy_filename("elevations")
            # If we already have elevations in the archive, this is not possible
            if filename in zf.namelist():
                raise Py4DGeoError(
                    "Elevations can only be set once; use add_epochs to append data"
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                elevations_file = os.path.join(tmp_dir, filename)
                self._write_numpy(elevations_file, values)
                zf.write(elevations_file, arcname=filename)

        self._elevations = values

    @elevations.deleter
    def elevations(self):
        self._elevations = None

    def add_epochs(self, *epochs):
        """
        Sample and append one or more timestamped epochs.

        Elevations are sampled at the fixed X/Y positions of ``corepoints``.
        Epochs are stored in chronological order. If epochs are added in a
        later call with timestamps preceding already stored data, the stored
        columns are reordered accordingly.
        """
        if not epochs:
            return

        corepoints = self.corepoints
        corepoint_xy = np.asarray(corepoints.cloud[:, :2], dtype=np.float64)

        normalized_epochs = []
        for epoch in epochs:
            epoch = as_epoch(epoch)
            if epoch.timestamp is None:
                raise Py4DGeoError(
                    "Epochs need to define a timestamp to be added to a SpatialTimeseries"
                )
            normalized_epochs.append(epoch)

        # Stable sorting preserves the order of epochs that have equal
        # timestamps within this call.
        normalized_epochs.sort(key=lambda epoch: epoch.timestamp)

        new_timestamps = [epoch.timestamp for epoch in normalized_epochs]
        new_elevations = [
            self._sample_elevations(epoch, corepoint_xy)
            for epoch in normalized_epochs
        ]
        new_elevations = np.column_stack(new_elevations)

        old_timestamps = list(self.timestamps)
        old_elevations = self.elevations
        if old_elevations.shape[1] != len(old_timestamps):
            raise Py4DGeoError(
                "The stored timestamps and elevations do not have matching lengths"
            )

        combined_timestamps = old_timestamps + new_timestamps
        combined_elevations = np.concatenate(
            (old_elevations, new_elevations), axis=1
        )

        # Keep timestamps and their corresponding elevation columns aligned.
        order = sorted(
            range(len(combined_timestamps)), key=combined_timestamps.__getitem__
        )
        combined_timestamps = [combined_timestamps[index] for index in order]
        combined_elevations = combined_elevations[:, order]

        self._write_timeseries(combined_timestamps, combined_elevations)

        # Invalidate then release caches so subsequent access reloads the
        # committed archive contents.
        self._timestamps = combined_timestamps
        self._elevations = combined_elevations

    @staticmethod
    def _sample_elevations(epoch, corepoint_xy):
        """Sample the Z coordinate of the nearest point in XY for each corepoint."""
        cloud = np.asarray(epoch.cloud, dtype=np.float64)
        elevations = np.full(corepoint_xy.shape[0], np.nan, dtype=np.float64)

        if cloud.shape[0] == 0:
            return elevations

        valid = np.isfinite(cloud[:, :3]).all(axis=1)
        if not np.any(valid):
            return elevations

        valid_cloud = cloud[valid]
        tree = cKDTree(valid_cloud[:, :2])
        _, nearest_indices = tree.query(corepoint_xy, k=1)
        elevations[:] = valid_cloud[nearest_indices, 2]
        return elevations

    def _write_timeseries(self, timestamps, elevations):
        """Replace the timestamp and elevation entries in the ZIP."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            timestamps_file = os.path.join(tmp_dir, "timestamps.json")
            with open(timestamps_file, "w", encoding="utf-8") as f:
                json.dump([timestamp.isoformat() for timestamp in timestamps], f)

            elevations_filename = self._numpy_filename("elevations")
            elevations_file = os.path.join(tmp_dir, elevations_filename)
            self._write_numpy(elevations_file, elevations)

            with UpdateableZipFile(self.filename, mode="a") as zf:
                for filename in ("timestamps.json", elevations_filename):
                    if filename in zf.namelist():
                        zf.remove(filename)

                zf.write(timestamps_file, arcname="timestamps.json")
                zf.write(elevations_file, arcname=elevations_filename)

    def _write_numpy(self, filename, values):
        if self.compress:
            np.savez_compressed(filename, values)
        else:
            np.save(filename, values)

    def _numpy_filename(self, name):
        extension = "npz" if self.compress else "npy"
        return f"{name}.{extension}"
