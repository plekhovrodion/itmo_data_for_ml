"""DataCollectionAgent: Kaggle (₽/м²) + выгрузка объявлений etagi.com из HTML."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml

SCHEMA = [
    "price_per_m2",
    "total_price_rub",
    "area_m2",
    "rooms",
    "city_or_region",
    "address_text",
    "geo_lat",
    "geo_lon",
    "source",
    "listing_url",
    "text",
    "audio",
    "image",
    "label",
    "collected_at",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_etagi_var_data(html: str) -> dict[str, Any]:
    marker = "var data="
    idx = html.find(marker)
    if idx < 0:
        raise ValueError("В HTML нет встроенного var data= (страница не выдачи или антибот).")
    start = idx + len(marker)
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(html[start:])
    return obj


def _etagi_page_url(base: str, page: int) -> str:
    base = base.rstrip("/") + "/"
    if page <= 1:
        return base
    return f"{base}page/{page}/"


class DataCollectionAgent:
    def __init__(self, config: str | Path = "config.yaml") -> None:
        self.config_path = Path(config)
        self.cfg = _load_config(self.config_path)

    def load_kaggle_sample(self) -> pd.DataFrame:
        import kagglehub

        k = self.cfg["kaggle"]
        out_dir = Path(self.cfg["output"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_path = kagglehub.dataset_download(k["dataset"])
        root = Path(ds_path)
        csv_files = list(root.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"В {root} нет CSV после загрузки Kaggle.")
        path = csv_files[0]
        df = pd.read_csv(path, sep=k.get("sep", ";"), nrows=int(k["max_rows"]))

        df = df[df["area"].astype(float) > 0].copy()
        df["price_per_m2"] = df["price"].astype(float) / df["area"].astype(float)
        df["total_price_rub"] = df["price"].astype(float)
        df["area_m2"] = df["area"].astype(float)
        df["rooms"] = df["rooms"]
        df["city_or_region"] = df["id_region"].astype(str)
        df["address_text"] = ""
        df["geo_lat"] = df["geo_lat"]
        df["geo_lon"] = df["geo_lon"]
        df["source"] = f"kaggle:{k['dataset']}"
        df["listing_url"] = ""
        df["text"] = df.apply(
            lambda r: f"Регион {r['id_region']}, {r['rooms']} комн., {r['area_m2']:.1f} м²",
            axis=1,
        )
        df["audio"] = pd.NA
        df["image"] = pd.NA
        df["label"] = df["price_per_m2"].round(2)
        df["collected_at"] = _utc_now()

        cols = [c for c in SCHEMA if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        return df[cols + rest]

    def scrape_etagi(self) -> pd.DataFrame:
        e = self.cfg["etagi"]
        base = e["base_url"].rstrip("/") + "/"
        ua = e["user_agent"]
        timeout = float(e["request_timeout_sec"])
        pause = float(e["pause_sec"])
        max_rows = int(e["max_rows"])

        rows: list[dict[str, Any]] = []
        page = 1
        session = requests.Session()
        session.headers.update({"User-Agent": ua, "Accept-Language": "ru-RU,ru;q=0.9"})
        parsed = urlparse(base)
        site_origin = f"{parsed.scheme}://{parsed.netloc}"

        while len(rows) < max_rows:
            url = _etagi_page_url(base, page)
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            data = _parse_etagi_var_data(r.text)
            flats = data.get("lists", {}).get("flats") or []
            if not flats:
                break
            for f in flats:
                if len(rows) >= max_rows:
                    break
                price_m2 = float(str(f.get("price_m2", "0")).replace(" ", ""))
                price = float(str(f.get("price", "0")).replace(" ", ""))
                sq = float(f.get("square") or 0)
                meta = f.get("meta") or {}
                street = meta.get("street") or ""
                city = meta.get("city") or ""
                addr = f"{city}, {street}".strip(", ")
                oid = f.get("object_id")
                listing_url = f"{site_origin}/realty/{oid}/" if oid else ""

                rows.append(
                    {
                        "price_per_m2": price_m2,
                        "total_price_rub": price,
                        "area_m2": sq,
                        "rooms": f.get("rooms"),
                        "city_or_region": city,
                        "address_text": addr,
                        "geo_lat": f.get("la"),
                        "geo_lon": f.get("lo"),
                        "source": "scrape:spb.etagi.com/realty/flats",
                        "listing_url": listing_url,
                        "text": f"{addr}; {sq} м²; {f.get('metro_station', '')}",
                        "audio": pd.NA,
                        "image": f.get("main_photo"),
                        "label": round(price_m2, 2),
                        "collected_at": _utc_now(),
                    }
                )
            page += 1
            time.sleep(pause)

        return pd.DataFrame(rows)

    def merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=SCHEMA)
        out = pd.concat(frames, ignore_index=True)
        for c in SCHEMA:
            if c not in out.columns:
                out[c] = pd.NA
        extra = [c for c in out.columns if c not in SCHEMA]
        return out[SCHEMA + extra]

    def run(self) -> dict[str, Path]:
        out = self.cfg["output"]
        out_dir = Path(out["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        df_k = self.load_kaggle_sample()
        p_k = out_dir / out["kaggle_csv"]
        df_k.to_csv(p_k, index=False, encoding="utf-8")

        df_e = self.scrape_etagi()
        p_e = out_dir / out["etagi_csv"]
        df_e.to_csv(p_e, index=False, encoding="utf-8")

        merged = self.merge([df_k, df_e])
        p_m = out_dir / out["merged_csv"]
        merged.to_csv(p_m, index=False, encoding="utf-8")

        return {"kaggle": p_k, "etagi": p_e, "merged": p_m}


if __name__ == "__main__":
    agent = DataCollectionAgent()
    paths = agent.run()
    for k, v in paths.items():
        print(f"{k}: {v}")
