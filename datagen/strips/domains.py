"""Six task domains: 4 training + 2 held-out test domains.

Shared multi-agent mechanics (MoveTo / PickUp / PlaceDown / Handover /
CoPickUp / CoMoveTo / CoPlaceDown / Recharge / ClearPath) with per-domain
vocabulary (location & item names) so that cross-domain transfer is real.
Test domains additionally define UNIQUE action schemas never seen in training.
"""
from __future__ import annotations

from .core import ActionSchema, Domain

ROBOTS = ["alpha", "beta", "gamma", "delta"]

# ---------------------------------------------------------------- schemas ---

def shared_schemas() -> list:
    return [
        ActionSchema(
            "MoveTo", (("r", "robot"), ("l1", "location"), ("l2", "location")),
            pre=(("at", "r", "l1"), ("connected", "l1", "l2"), ("can_reach", "r", "l2"),
                 ("mobile", "r")),
            add=(("at", "r", "l2"),), delete=(("at", "r", "l1"),),
        ),
        ActionSchema(
            "PickUp", (("r", "robot"), ("i", "item"), ("l", "location")),
            pre=(("at", "r", "l"), ("item_at", "i", "l"), ("free", "r"), ("light", "i")),
            add=(("carrying", "r", "i"),), delete=(("item_at", "i", "l"), ("free", "r")),
        ),
        ActionSchema(
            "PlaceDown", (("r", "robot"), ("i", "item"), ("l", "location")),
            pre=(("at", "r", "l"), ("carrying", "r", "i")),
            add=(("item_at", "i", "l"), ("free", "r")), delete=(("carrying", "r", "i")),
        ),
        ActionSchema(
            "Handover", (("g", "robot"), ("t", "robot"), ("i", "item"), ("l", "location")),
            pre=(("at", "g", "l"), ("at", "t", "l"), ("carrying", "g", "i"), ("free", "t")),
            add=(("carrying", "t", "i"),), delete=(("carrying", "g", "i"), ("free", "t")),
            agents=("g", "t"),
        ),
        ActionSchema(
            "CoPickUp", (("r1", "robot"), ("r2", "robot"), ("i", "item"), ("l", "location")),
            pre=(("at", "r1", "l"), ("at", "r2", "l"), ("item_at", "i", "l"),
                 ("free", "r1"), ("free", "r2"), ("heavy", "i")),
            add=(("co_carrying", "r1", "r2", "i"),),
            delete=(("item_at", "i", "l"), ("free", "r1"), ("free", "r2"),
                    ("mobile", "r1"), ("mobile", "r2")),
            agents=("r1", "r2"),
        ),
        ActionSchema(
            "CoMoveTo", (("r1", "robot"), ("r2", "robot"), ("i", "item"),
                         ("l1", "location"), ("l2", "location")),
            pre=(("co_carrying", "r1", "r2", "i"), ("at", "r1", "l1"), ("at", "r2", "l1"),
                 ("connected", "l1", "l2"), ("can_reach", "r1", "l2"), ("can_reach", "r2", "l2")),
            add=(("at", "r1", "l2"), ("at", "r2", "l2")),
            delete=(("at", "r1", "l1"), ("at", "r2", "l1")),
            agents=("r1", "r2"),
        ),
        ActionSchema(
            "CoPlaceDown", (("r1", "robot"), ("r2", "robot"), ("i", "item"), ("l", "location")),
            pre=(("co_carrying", "r1", "r2", "i"), ("at", "r1", "l"), ("at", "r2", "l")),
            add=(("item_at", "i", "l"), ("free", "r1"), ("free", "r2"),
                 ("mobile", "r1"), ("mobile", "r2")),
            delete=(("co_carrying", "r1", "r2", "i"),),
            agents=("r1", "r2"),
        ),
        ActionSchema(
            "Recharge", (("r", "robot"), ("l", "location")),
            pre=(("at", "r", "l"), ("charge_station_at", "l")),
            add=(("charged", "r"),), delete=(),
        ),
        ActionSchema(
            "ClearPath", (("r", "robot"), ("l1", "location"), ("l2", "location")),
            pre=(("at", "r", "l1"), ("connected", "l1", "l2")),
            add=(("clear", "l1", "l2"),), delete=(),
        ),
    ]


def library_extra_schemas() -> list:
    return [
        ActionSchema(
            "CatalogBook", (("r", "robot"), ("i", "item"), ("l", "location")),
            pre=(("at", "r", "l"), ("item_at", "i", "l")),
            add=(("cataloged", "i"),), delete=(),
        ),
    ]


def greenhouse_extra_schemas() -> list:
    return [
        ActionSchema(
            "WaterPlants", (("r", "robot"), ("l", "location")),
            pre=(("at", "r", "l"),),
            add=(("watered", "l"),), delete=(),
        ),
        ActionSchema(
            "InspectPlants", (("r", "robot"), ("l", "location")),
            pre=(("at", "r", "l"),),
            add=(("inspected", "l"),), delete=(),
        ),
    ]


def kitchen_extra_schemas() -> list:
    """Primitive reserved for the independent external-domain split."""
    return [
        ActionSchema(
            "SanitizeCounter", (("r", "robot"), ("l", "location")),
            pre=(("at", "r", "l"),),
            add=(("sanitized", "l"),), delete=(),
        ),
    ]


# ---------------------------------------------------------------- domains ---

def _edges(pairs: list) -> list:
    return [("connected", a, b) for a, b in pairs] + [("connected", b, a) for a, b in pairs]


DOMAIN_SPECS = {
    # ------------------------------------------------------- training ----
    "warehouse": dict(
        split="train",
        locations=["shelf_a", "shelf_b", "shelf_c", "packing_station", "loading_dock", "charge_bay"],
        edges=_edges([("shelf_a", "packing_station"), ("shelf_b", "packing_station"),
                      ("shelf_c", "shelf_b"), ("packing_station", "loading_dock"),
                      ("charge_bay", "packing_station"), ("shelf_a", "shelf_c")]),
        charge_stations=["charge_bay"],
        items=["crate_red", "crate_blue", "box_small", "pallet_heavy"],
        heavy_items=["pallet_heavy"],
        delivery_locations=["packing_station", "loading_dock"],
        extra_schemas=[],
        scene_zh="仓储物流中心",
    ),
    "hospital": dict(
        split="train",
        locations=["pharmacy", "ward_east", "ward_west", "lab", "nurses_station", "charge_nook"],
        edges=_edges([("pharmacy", "nurses_station"), ("nurses_station", "ward_east"),
                      ("nurses_station", "ward_west"), ("lab", "nurses_station"),
                      ("charge_nook", "lab"), ("ward_east", "ward_west")]),
        charge_stations=["charge_nook"],
        items=["medicine_kit", "blood_sample", "linens_pack", "iv_stand_heavy"],
        heavy_items=["iv_stand_heavy"],
        delivery_locations=["ward_east", "ward_west", "lab"],
        extra_schemas=[],
        scene_zh="医院住院部",
    ),
    "search_rescue": dict(
        split="train",
        locations=["base_camp", "zone_north", "zone_south", "rubble_pile", "medical_tent", "charge_post"],
        edges=_edges([("base_camp", "zone_north"), ("base_camp", "zone_south"),
                      ("zone_north", "rubble_pile"), ("zone_south", "rubble_pile"),
                      ("medical_tent", "base_camp"), ("charge_post", "medical_tent")]),
        charge_stations=["charge_post"],
        items=["medkit", "water_pack", "thermal_blanket", "debris_slab_heavy"],
        heavy_items=["debris_slab_heavy"],
        delivery_locations=["medical_tent", "base_camp"],
        extra_schemas=[],
        scene_zh="地震灾后搜救现场",
    ),
    "office": dict(
        split="train",
        locations=["mail_room", "lobby", "meeting_room", "kitchen", "server_room", "charge_corner"],
        edges=_edges([("mail_room", "lobby"), ("lobby", "meeting_room"), ("lobby", "kitchen"),
                      ("server_room", "lobby"), ("charge_corner", "server_room"),
                      ("meeting_room", "kitchen")]),
        charge_stations=["charge_corner"],
        items=["document_folder", "parcel_box", "coffee_tray", "server_rack_heavy"],
        heavy_items=["server_rack_heavy"],
        delivery_locations=["meeting_room", "server_room", "kitchen"],
        extra_schemas=[],
        scene_zh="办公楼",
    ),
    # ---------------------------------------------------------- test -----
    "library": dict(
        split="test",
        locations=["front_desk", "reading_hall", "archive_room", "restoration_lab", "return_bin", "charge_pod"],
        edges=_edges([("return_bin", "front_desk"), ("front_desk", "reading_hall"),
                      ("reading_hall", "archive_room"), ("archive_room", "restoration_lab"),
                      ("charge_pod", "restoration_lab"), ("front_desk", "archive_room")]),
        charge_stations=["charge_pod"],
        items=["novel_stack", "rare_manuscript", "encyclopedia_set", "oak_bookcase_heavy"],
        heavy_items=["oak_bookcase_heavy"],
        delivery_locations=["reading_hall", "archive_room", "restoration_lab"],
        extra_schemas=library_extra_schemas(),
        scene_zh="图书馆",
    ),
    "greenhouse": dict(
        split="test",
        locations=["seedling_row_a", "seedling_row_b", "irrigation_station", "storage_shed",
                   "compost_area", "charge_spot"],
        edges=_edges([("seedling_row_a", "irrigation_station"), ("seedling_row_b", "irrigation_station"),
                      ("storage_shed", "irrigation_station"), ("compost_area", "storage_shed"),
                      ("charge_spot", "compost_area"), ("seedling_row_a", "seedling_row_b")]),
        charge_stations=["charge_spot"],
        items=["seedling_tray", "fertilizer_bag", "tool_crate", "water_tank_heavy"],
        heavy_items=["water_tank_heavy"],
        delivery_locations=["seedling_row_a", "seedling_row_b", "compost_area"],
        extra_schemas=greenhouse_extra_schemas(),
        scene_zh="农业温室",
    ),
    # ------------------------------------------------ independent external --
    # This domain is never included by build_dataset.py's train/val/test
    # quotas.  Its vocabulary and SanitizeCounter action are held out for a
    # post-hoc external-validity check.
    "kitchen": dict(
        split="external",
        locations=["prep_island", "cold_storage", "dining_room", "wash_station",
                   "pantry_shelf", "service_passage", "charge_rack"],
        edges=_edges([("prep_island", "cold_storage"), ("prep_island", "wash_station"),
                      ("cold_storage", "pantry_shelf"), ("wash_station", "dining_room"),
                      ("dining_room", "service_passage"), ("service_passage", "charge_rack")]),
        charge_stations=["charge_rack"],
        items=["produce_bin", "sauce_pot", "dish_stack", "oven_unit_heavy"],
        heavy_items=["oven_unit_heavy"],
        delivery_locations=["dining_room", "wash_station", "prep_island"],
        extra_schemas=kitchen_extra_schemas(),
        scene_zh="a professional teaching kitchen",
    ),
}


def build_domain(name: str, robots: list = None, extra_static: list = None,
                 items: list = None, edges: list = None) -> Domain:
    spec = DOMAIN_SPECS[name]
    robots = robots or ROBOTS[:2]
    items = items or spec["items"]
    static = list(edges if edges is not None else spec["edges"])
    static += [("charge_station_at", l) for l in spec["charge_stations"]]
    static += list(extra_static or [])
    types = {"robot": robots, "location": spec["locations"], "item": items}
    schemas = shared_schemas() + spec["extra_schemas"]
    return Domain(name, types, schemas, static)


def random_graph(locations: list, rng, extra_edges: int = 2) -> list:
    """Random connected graph over locations: random spanning tree + a few
    extra edges. Per-task topology kills map-memorization: the layout is only
    available from the NL input."""
    locs = list(locations)
    rng.shuffle(locs)
    pairs = []
    for i in range(1, len(locs)):
        j = rng.randint(0, i - 1)  # random parent among previous -> random tree
        pairs.append((locs[j], locs[i]))
    existing = {frozenset(p) for p in pairs}
    cand = [(a, b) for i, a in enumerate(locs) for b in locs[i + 1:]
            if frozenset((a, b)) not in existing]
    rng.shuffle(cand)
    pairs += cand[:extra_edges]
    return _edges(pairs)


def item_facts(spec_name: str) -> list:
    """('light', i) / ('heavy', i) static-ish dynamic facts for init state."""
    spec = DOMAIN_SPECS[spec_name]
    facts = []
    for i in spec["items"]:
        facts.append(("heavy" if i in spec["heavy_items"] else "light", i))
    return facts
