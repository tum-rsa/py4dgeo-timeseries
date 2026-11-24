import datetime

import numpy as np
import pytest

from py4dgeo.epoch import Epoch
from py4dgeo.util import Py4DGeoError
from py4dgeo.timeseries import SpatialTimeseries


@pytest.mark.parametrize("compress", [True, False])
def test_add_epochs_samples_elevations_and_timestamps(tmp_path, compress):
    filename = tmp_path / "timeseries.zip"
    timeseries = SpatialTimeseries(filename, compress=compress)
    timeseries.corepoints = Epoch(
        np.array([[0.0, 0.0, 100.0], [10.0, 0.0, -100.0]])
    )

    # Add epochs out of chronological order. The core-point Z values are
    # intentionally far from the sampled elevations to check that only XY is
    # used for the location lookup.
    timeseries.add_epochs(
        Epoch(
            np.array([[0.2, 0.0, 5.0], [9.8, 0.0, 8.0]]),
            timestamp=datetime.datetime(2024, 2, 1),
        )
    )
    timeseries.add_epochs(
        Epoch(
            np.array([[0.1, 0.0, 3.0], [10.1, 0.0, 6.0]]),
            timestamp=datetime.datetime(2024, 1, 1),
        )
    )

    np.testing.assert_allclose(timeseries.elevations, [[3.0, 5.0], [6.0, 8.0]])
    assert timeseries.timestamps == [
        datetime.datetime(2024, 1, 1),
        datetime.datetime(2024, 2, 1),
    ]

    restored = SpatialTimeseries(filename)
    np.testing.assert_allclose(restored.elevations, timeseries.elevations)
    assert restored.timestamps == timeseries.timestamps
    np.testing.assert_allclose(restored.corepoints.cloud, timeseries.corepoints.cloud)


def test_add_epoch_without_timestamp_raises(tmp_path):
    timeseries = SpatialTimeseries(tmp_path / "timeseries.zip")
    timeseries.corepoints = Epoch(np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(Py4DGeoError, match="timestamp"):
        timeseries.add_epochs(Epoch(np.array([[0.0, 0.0, 1.0]])))


def test_empty_epoch_produces_nan_elevations(tmp_path):
    timeseries = SpatialTimeseries(tmp_path / "timeseries.zip")
    timeseries.corepoints = Epoch(np.array([[0.0, 0.0, 0.0]]))
    timeseries.add_epochs(
        Epoch(np.empty((0, 3)), timestamp=datetime.datetime(2024, 1, 1))
    )

    assert np.isnan(timeseries.elevations[0, 0])
