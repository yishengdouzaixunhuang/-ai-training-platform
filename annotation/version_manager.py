"""Annotation version manager - auto-snapshot, list, restore, diff."""
import os, json, shutil, time, glob
from datetime import datetime

MAX_VERSIONS = 100  # Keep at most 100 versions per image


def _versions_dir(project_dir):
    return os.path.join(project_dir, ".versions")


def _version_base(image_path):
    """Given image path, return base name for version files (without timestamp)."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return base


def _list_versions_for_image(project_dir, image_path):
    """Return sorted list of (version_path, timestamp_str) for an image, newest first."""
    vdir = _versions_dir(project_dir)
    base = _version_base(image_path)
    pattern = os.path.join(vdir, f"{base}_v*.json")
    files = glob.glob(pattern)
    versions = []
    for f in files:
        # Extract timestamp from filename: name_v20260610_123456.json
        name = os.path.basename(f)
        try:
            ts_str = name.rsplit("_v", 1)[1].replace(".json", "")
            ts = datetime.strptime(ts_str[:15], "%Y%m%d_%H%M%S")
            versions.append((f, ts))
        except (ValueError, IndexError):
            continue
    versions.sort(key=lambda x: x[1], reverse=True)
    return versions


def save_version(project_dir, image_path):
    """Save a snapshot of the current JSON as a new version.
    Returns the version path or None if no JSON exists."""
    from annotation.labelme_io import get_json_path
    json_path = get_json_path(image_path)
    if not os.path.exists(json_path):
        return None

    vdir = _versions_dir(project_dir)
    os.makedirs(vdir, exist_ok=True)

    base = _version_base(image_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    version_path = os.path.join(vdir, f"{base}_v{ts}.json")

    # Copy current JSON to version
    shutil.copy2(json_path, version_path)

    # Prune old versions
    _prune_old_versions(project_dir, image_path)

    return version_path


def _prune_old_versions(project_dir, image_path):
    """Remove oldest versions if exceeding MAX_VERSIONS."""
    versions = _list_versions_for_image(project_dir, image_path)
    if len(versions) > MAX_VERSIONS:
        for vpath, _ in versions[MAX_VERSIONS:]:
            try:
                os.remove(vpath)
            except OSError:
                pass


def list_versions(project_dir, image_path):
    """Return list of (version_path, ts_str, size_kb) for display, newest first."""
    versions = _list_versions_for_image(project_dir, image_path)
    result = []
    for vpath, ts in versions:
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        size_kb = os.path.getsize(vpath) / 1024.0
        result.append((vpath, ts_str, size_kb))
    return result


def restore_version(project_dir, image_path, version_path):
    """Restore a version: copy version file to current JSON path."""
    from annotation.labelme_io import get_json_path
    json_path = get_json_path(image_path)

    # First save current as a version (so restore is undoable)
    save_version(project_dir, image_path)

    # Copy version back
    shutil.copy2(version_path, json_path)
    return json_path


def get_version_diff(version_path_a, version_path_b):
    """Compare two version JSONs and return a diff summary.
    Returns dict with: shapes_added, shapes_removed, shapes_modified, total_a, total_b"""
    try:
        with open(version_path_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)
        with open(version_path_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)
    except Exception:
        return None

    shapes_a = data_a.get("shapes", [])
    shapes_b = data_b.get("shapes", [])

    # Simple diff: compare by label + point count
    def shape_key(s):
        pts = s.get("points", [])
        return (s.get("label", ""), len(pts), str(pts[:2]))

    keys_a = {shape_key(s) for s in shapes_a}
    keys_b = {shape_key(s) for s in shapes_b}

    added = len(keys_b - keys_a)
    removed = len(keys_a - keys_b)
    modified = min(len(shapes_a), len(shapes_b)) - len(keys_a & keys_b)

    # Count per label
    def count_by_label(shapes):
        counts = {}
        for s in shapes:
            lbl = s.get("label", "unknown")
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    return {
        "shapes_added": added,
        "shapes_removed": removed,
        "shapes_modified": max(0, modified),
        "total_a": len(shapes_a),
        "total_b": len(shapes_b),
        "labels_a": count_by_label(shapes_a),
        "labels_b": count_by_label(shapes_b),
    }


def auto_save_version_before_save(project_dir, image_path):
    """Hook to call before save_labelme_json. Creates a version snapshot."""
    from annotation.labelme_io import get_json_path
    json_path = get_json_path(image_path)
    if os.path.exists(json_path):
        save_version(project_dir, image_path)
