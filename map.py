# app/map.py
from flask import Blueprint, render_template, request, jsonify, abort
from .models import Album, Result            # adjust import paths to your project
from .db import db                           # your SQLAlchemy instance

bp = Blueprint("map", __name__, url_prefix="/map")

# ---------- Page ----------------------------------------------------------------
@bp.route("")
def map_view():
    place_ids = request.args.get("places", "")
    album_id  = request.args.get("album")

    if album_id:
        album = Album.query.get_or_404(album_id)
        place_ids = []
        for ref in album.activities:
            res_id, idx = ref.split("-")
            res = Result.query.get(int(res_id))
            act = res.data["activities"][int(idx)]
            place_ids.append(act["place_name_normalized"])   # adjust key name
        place_ids = ",".join(place_ids)

    return render_template("map.html",
                           place_ids=place_ids.split(",") if place_ids else [])

# ---------- Bulk-info API --------------------------------------------------------
@bp.post("/api/bulk")
def bulk_places():
    ids = request.json.get("ids", [])
    out = []
    for pid in ids:
        meta = db.get_external_meta(pid)          # ↖ replace with your helper
        if not meta: continue
        out.append({
            "id"    : pid,
            "name"  : meta.get("name"),
            "lat"   : meta["lat"],
            "lng"   : meta["lng"],
            "rating_global": meta.get("rating"),
            "image_url": meta.get("hero_image"),
        })
    return jsonify(out)
