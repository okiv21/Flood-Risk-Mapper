"""Package dated hotspot water-context rasters as local PMTiles v3 archives."""

import argparse
import io
import json
import tempfile
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from PIL import Image
from pmtiles.reader import Reader, MmapSource
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from rasterio.warp import reproject, transform_bounds
from floodrisk.deploy.tiled import file_hash

COLORS = {
    0: (235, 234, 226, 255),
    1: (25, 89, 146, 255),
    2: (26, 167, 137, 255),
    3: (230, 159, 0, 255),
}
LEGEND = {
    "0": "predicted dry",
    "1": "water, historical occurrence >=90%",
    "2": "water, historical occurrence <90%",
    "3": "water, occurrence unavailable",
    "255": "no radar data",
}


def export(root, observations, output, changes=False):
    output.mkdir(parents=True, exist_ok=True)
    palette = np.zeros((256, 4), dtype="uint8")
    for code, color in COLORS.items():
        palette[code] = color
    index = []
    for item in observations:
        name = item["site"]
        raster = (
            root
            / name
            / ("change_context.tif" if changes else "reference/water_context.tif")
        )
        if file_hash(raster) != item["change_sha256" if changes else "context_sha256"]:
            raise ValueError(f"Context raster changed at {name}")
        target = output / (name.lower().replace(" ", "_") + ".pmtiles")
        with (
            rasterio.open(raster) as src,
            tempfile.NamedTemporaryFile(dir=output) as temp,
        ):
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            header = {
                k: round(v * 1e7)
                for k, v in zip(
                    ["min_lon_e7", "min_lat_e7", "max_lon_e7", "max_lat_e7"], bounds
                )
            }
            header.update(tile_type=TileType.PNG, tile_compression=Compression.NONE)
            writer = Writer(temp)
            tiles = []
            for tile in sorted(
                mercantile.tiles(*bounds, zooms=range(8, 15)),
                key=lambda t: zxy_to_tileid(t.z, t.x, t.y),
            ):
                dest = np.full((256, 256), 255, dtype="uint8")
                reproject(
                    rasterio.band(src, 1),
                    dest,
                    dst_transform=rasterio.transform.from_bounds(
                        *mercantile.xy_bounds(tile), 256, 256
                    ),
                    dst_crs="EPSG:3857",
                    dst_nodata=255,
                )
                if (dest == 255).all():
                    continue
                png = io.BytesIO()
                Image.fromarray(palette[dest]).save(png, format="PNG")
                writer.write_tile(zxy_to_tileid(tile.z, tile.x, tile.y), png.getvalue())
                tiles.append(tile)
            writer.finalize(
                header,
                dict(
                    site=name,
                    observed_utc=item["observed_utc"],
                    legend=(
                        {
                            **LEGEND,
                            "0": "no predicted dry-to-water change",
                            "1": "new water, high historical occurrence",
                            "2": "new water, lower historical occurrence",
                            "3": "new water, historical occurrence unavailable",
                        }
                        if changes
                        else LEGEND
                    ),
                    **({"earlier_source": item["earlier_source"]} if changes else {}),
                    attribution="Copernicus Sentinel-1; EC JRC/Google",
                    scope="Paired dry-to-water change"
                    if changes
                    else "Experimental water observation",
                ),
            )
            temp.flush()
            reader = Reader(MmapSource(temp))
            if reader.header()["addressed_tiles_count"] != len(tiles):
                raise ValueError("Archive count mismatch")
            for tile in tiles:
                with Image.open(io.BytesIO(reader.get(tile.z, tile.x, tile.y))) as png:
                    png.verify()
            import shutil

            staging = target.with_suffix(".pmtiles.tmp")
            shutil.copyfile(temp.name, staging)
            staging.replace(target)
        index.append(
            dict(
                site=name,
                observed_utc=item["observed_utc"],
                file=target.name,
                tiles=len(tiles),
                sha256=file_hash(target),
                bounds=bounds,
            )
        )
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "observations", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--changes", action="store_true")
    args = parser.parse_args()
    export(
        args.root, json.loads(args.observations.read_text()), args.output, args.changes
    )
