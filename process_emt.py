import io
import os
import sys
import json
import shutil
import hashlib
import zipfile
import unicodedata
import pandas as pd
import requests
import polyline

GTFS_URL = "https://opendata.vlci.valencia.es/dataset/ab058cf8-ad3e-4d9c-ac89-0c6367ecf351/resource/c81b69e6-c082-44dc-acc6-66fc417b4e66/download/google_transit.zip"
OUTPUT_DIR = "data"
HASH_FILE = "gtfs_sha256.txt"

def get_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def time_to_minutes(time_str: str) -> int:
    """Convierte 'HH:MM:SS' a minutos del día operativo (soporta >= 24h)."""
    if pd.isna(time_str):
        return -1
    parts = str(time_str).strip().split(":")
    if len(parts) >= 2:
        return int(parts[0]) * 60 + int(parts[1])
    return -1

def clean_text(val: str) -> str:
    """Normaliza texto Unicode (elimina \xa0 y espacios redundantes)."""
    if pd.isna(val):
        return ""
    normalized = unicodedata.normalize("NFKC", str(val))
    return " ".join(normalized.split())

def main():
    print("Descargando GTFS de EMT Valencia...")
    headers = {"User-Agent": "Mozilla/5.0 (EMT GTFS Sync Agent)"}
    resp = requests.get(GTFS_URL, headers=headers, timeout=60)
    resp.raise_for_status()
    raw_zip = resp.content
    current_hash = get_hash(raw_zip)

    # 1. Control de cambios por hash SHA256
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            last_hash = f.read().strip()
        if current_hash == last_hash:
            print("El feed es idéntico al último procesado. Deteniendo ejecución.")
            sys.exit(0)

    print("Actualización detectada. Purgando directorio anterior...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(f"{OUTPUT_DIR}/departures", exist_ok=True)

    zf = zipfile.ZipFile(io.BytesIO(raw_zip))

    # 2. Carga con las cabeceras exactas del feed
    routes = pd.read_csv(zf.open("routes.txt"), dtype={"route_id": str, "route_short_name": str})
    trips = pd.read_csv(zf.open("trips.txt"), dtype={
        "route_id": str, 
        "service_id": str, 
        "trip_id": str, 
        "trip_headsign": str, 
        "shape_id": str
    })
    stops = pd.read_csv(zf.open("stops.txt"), dtype={"stop_id": str})
    stop_times = pd.read_csv(zf.open("stop_times.txt"), dtype={"trip_id": str, "stop_id": str})
    
    calendar = pd.read_csv(zf.open("calendar.txt"), dtype={"service_id": str}) if "calendar.txt" in zf.namelist() else pd.DataFrame()
    calendar_dates = pd.read_csv(zf.open("calendar_dates.txt"), dtype={"service_id": str, "date": str}) if "calendar_dates.txt" in zf.namelist() else pd.DataFrame()

    # Mapeo route_id -> route_short_name (Línea: "4", "C1", etc.)
    route_map = dict(zip(routes["route_id"], routes["route_short_name"]))
    trips["line"] = trips["route_id"].map(route_map)

    # 3. stops.json
    print("Procesando paradas y líneas vinculadas...")
    trip_stops = stop_times[["trip_id", "stop_id"]].drop_duplicates()
    trip_lines = trip_stops.merge(trips[["trip_id", "line"]], on="trip_id")
    lines_per_stop = (
        trip_lines.dropna(subset=["line"])
        .groupby("stop_id")["line"]
        .unique()
        .apply(lambda x: sorted(list(set(str(l) for l in x))))
        .to_dict()
    )

    stops_list = []
    for _, row in stops.iterrows():
        sid = str(row["stop_id"])
        stops_list.append({
            "id": sid,
            "name": clean_text(row["stop_name"]),
            "lat": round(float(row["stop_lat"]), 5),
            "lon": round(float(row["stop_lon"]), 5),
            "lines": lines_per_stop.get(sid, [])
        })

    with open(f"{OUTPUT_DIR}/stops.json", "w", encoding="utf-8") as f:
        json.dump(stops_list, f, ensure_ascii=False, separators=(",", ":"))

    # 4. shapes.json consolidado
    if "shapes.txt" in zf.namelist():
        print("Codificando trazados en shapes.json...")
        shapes_df = pd.read_csv(zf.open("shapes.txt"), dtype={"shape_id": str})
        shapes_df.sort_values(by=["shape_id", "shape_pt_sequence"], inplace=True)

        encoded_shapes = {}
        for shape_id, group in shapes_df.groupby("shape_id"):
            coords = list(zip(group["shape_pt_lat"], group["shape_pt_lon"]))
            encoded_shapes[shape_id] = polyline.encode(coords)

        # Mapeo directo por línea -> shape_id -> datos
        trip_shapes = trips.dropna(subset=["shape_id", "line"])[["line", "shape_id", "trip_headsign"]].drop_duplicates(subset=["shape_id"])
        
        all_shapes = {}
        for line, group in trip_shapes.groupby("line"):
            line_str = str(line)
            all_shapes[line_str] = {}
            for _, r in group.iterrows():
                shp_id = r["shape_id"]
                if shp_id in encoded_shapes:
                    all_shapes[line_str][shp_id] = {
                        "headsign": clean_text(r["trip_headsign"]),
                        "poly": encoded_shapes[shp_id]
                    }
        
        with open(f"{OUTPUT_DIR}/shapes.json", "w", encoding="utf-8") as f:
            json.dump(all_shapes, f, ensure_ascii=False, separators=(",", ":"))

    # 5. Calendarios: Días de semana (1=Lunes .. 7=Domingo) y excepciones
    print("Mapeando vigencia y días de servicio...")
    service_info = {}
    day_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    
    if not calendar.empty:
        for _, row in calendar.iterrows():
            active_days = [idx + 1 for idx, day in enumerate(day_cols) if int(row[day]) == 1]
            service_info[str(row["service_id"])] = {
                "days": active_days,
                "dates": []
            }

    if not calendar_dates.empty:
        for _, row in calendar_dates.iterrows():
            sid = str(row["service_id"])
            d_str = str(row["date"])
            ex_type = int(row["exception_type"])
            
            if sid not in service_info:
                service_info[sid] = {"days": [], "dates": []}
                
            if ex_type == 1:
                service_info[sid]["dates"].append(d_str)

    # 6. Salidas por parada: departures/{stop_id}.json
    print("Generando salidas programadas por parada...")
    stop_times["m"] = stop_times["departure_time"].apply(time_to_minutes)
    valid_times = stop_times[stop_times["m"] >= 0]

    merged_trips = valid_times.merge(
        trips[["trip_id", "service_id", "trip_headsign", "line"]], 
        on="trip_id"
    )

    for stop_id, group in merged_trips.groupby("stop_id"):
        schedules = []
        service_groups = group.groupby(["line", "trip_headsign", "service_id"])
        
        for (line, headsign, s_id), rows in service_groups:
            srv = service_info.get(s_id, {"days": [], "dates": []})
            mins = sorted(list(set(rows["m"].tolist())))
            
            entry = {
                "line": str(line),
                "dest": clean_text(headsign),
                "days": srv["days"],
                "times": mins
            }
            if srv["dates"]:
                entry["dates"] = srv["dates"]
                
            schedules.append(entry)

        with open(f"{OUTPUT_DIR}/departures/{stop_id}.json", "w", encoding="utf-8") as f:
            json.dump({"id": str(stop_id), "schedules": schedules}, f, ensure_ascii=False, separators=(",", ":"))

    # 7. Guardar hash de la versión procesada
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        f.write(current_hash)

    print("Procesamiento completado con éxito.")

if __name__ == "__main__":
    main()
