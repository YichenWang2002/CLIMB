"""Natural (English) display names for entities, per domain.

The NL input mentions entities as 'natural name (id)' on first use, e.g.
'shelf A (shelf_a)'; the BT XML uses the raw id. The mapping is consistent,
which is what the model must learn to ground NL into XML attributes.
"""

ROBOT_EN = {
    "alpha": "Alpha",
    "beta": "Beta",
    "gamma": "Gamma",
    "delta": "Delta",
}

LOC_EN = {
    "warehouse": {
        "shelf_a": "shelf A", "shelf_b": "shelf B", "shelf_c": "shelf C",
        "packing_station": "the packing station", "loading_dock": "the loading dock",
        "charge_bay": "the charging bay",
    },
    "hospital": {
        "pharmacy": "the pharmacy", "ward_east": "the east ward", "ward_west": "the west ward",
        "lab": "the laboratory", "nurses_station": "the nurses' station",
        "charge_nook": "the charging nook",
    },
    "search_rescue": {
        "base_camp": "the base camp", "zone_north": "the north zone", "zone_south": "the south zone",
        "rubble_pile": "the rubble pile", "medical_tent": "the medical tent",
        "charge_post": "the charging post",
    },
    "office": {
        "mail_room": "the mail room", "lobby": "the lobby", "meeting_room": "the meeting room",
        "kitchen": "the kitchen", "server_room": "the server room",
        "charge_corner": "the charging corner",
    },
    "library": {
        "front_desk": "the front desk", "reading_hall": "the reading hall",
        "archive_room": "the archive room", "restoration_lab": "the restoration lab",
        "return_bin": "the book return bin", "charge_pod": "the charging pod",
    },
    "greenhouse": {
        "seedling_row_a": "seedling row A", "seedling_row_b": "seedling row B",
        "irrigation_station": "the irrigation station", "storage_shed": "the storage shed",
        "compost_area": "the compost area", "charge_spot": "the charging spot",
    },
    "kitchen": {
        "prep_island": "the preparation island", "cold_storage": "cold storage",
        "dining_room": "the dining room", "wash_station": "the wash station",
        "pantry_shelf": "the pantry shelf", "service_passage": "the service passage",
        "charge_rack": "the charging rack",
    },
}

ITEM_EN = {
    "warehouse": {
        "crate_red": "the red crate", "crate_blue": "the blue crate",
        "box_small": "the small box", "pallet_heavy": "the heavy pallet",
    },
    "hospital": {
        "medicine_kit": "the medicine kit", "blood_sample": "the blood sample",
        "linens_pack": "the linens pack", "iv_stand_heavy": "the heavy IV stand",
    },
    "search_rescue": {
        "medkit": "the medkit", "water_pack": "the water pack",
        "thermal_blanket": "the thermal blanket", "debris_slab_heavy": "the heavy debris slab",
    },
    "office": {
        "document_folder": "the document folder", "parcel_box": "the parcel box",
        "coffee_tray": "the coffee tray", "server_rack_heavy": "the heavy server rack",
    },
    "library": {
        "novel_stack": "the stack of novels", "rare_manuscript": "the rare manuscript",
        "encyclopedia_set": "the encyclopedia set", "oak_bookcase_heavy": "the heavy oak bookcase",
    },
    "greenhouse": {
        "seedling_tray": "the seedling tray", "fertilizer_bag": "the fertilizer bag",
        "tool_crate": "the tool crate", "water_tank_heavy": "the heavy water tank",
    },
    "kitchen": {
        "produce_bin": "the produce bin", "sauce_pot": "the sauce pot",
        "dish_stack": "the stack of dishes", "oven_unit_heavy": "the heavy oven unit",
    },
}

SCENE_EN = {
    "warehouse": "a logistics warehouse",
    "hospital": "a hospital inpatient wing",
    "search_rescue": "a post-earthquake search-and-rescue site",
    "office": "an office building",
    "library": "a public library",
    "greenhouse": "an agricultural greenhouse",
    "kitchen": "a professional teaching kitchen",
}

# action descriptions for test-domain unique actions (mentioned inline in NL).
# Each description spells out, in prose, how the skill is invoked in the tree
# (which of the robot / item / location arguments it takes) so the model can
# ground a never-seen primitive into the learned XML conventions — this is the
# compositional-generalization test, not a guessing game.
UNIQUE_ACTION_EN = {
    "CatalogBook": "a skill named CatalogBook, invoked on a robot with an item and "
                   "the location where that item sits, which catalogs the item there",
    "WaterPlants": "a skill named WaterPlants, invoked on a robot with a location, "
                   "which waters the plants at that location",
    "InspectPlants": "a skill named InspectPlants, invoked on a robot with a location, "
                     "which inspects the plants at that location",
    "SanitizeCounter": "a skill named SanitizeCounter, invoked on a robot with a "
                       "location, which sanitizes the counter at that location",
}

# Primitive-rename augmentation pool (datagen/rename_skills.py): standard
# simple single-robot action -> candidate (new leaf tag name, English signature
# description) pairs. Descriptions mirror the UNIQUE_ACTION_EN style above so
# the model practices the meta-skill "read an unseen skill signature in prose,
# ground it into a same-named XML leaf" without leaking the real test-domain
# primitives. Only simple single-robot leaves are renamed; coordination nodes
# (Handover*/Co*/SignalReady/WaitReady) and control nodes are never touched.
RENAME_POOL = {
    "PickUp": [
        ("FetchItem", "a skill named FetchItem, invoked on a robot with an item and "
                      "the location where that item sits, which picks up the item there"),
        ("GrabItem", "a skill named GrabItem, invoked on a robot with an item and "
                     "the location where that item sits, which grabs the item there"),
        ("RetrieveItem", "a skill named RetrieveItem, invoked on a robot with an item and "
                         "the location where that item sits, which retrieves the item there"),
        ("CollectItem", "a skill named CollectItem, invoked on a robot with an item and "
                        "the location where that item sits, which collects the item there"),
        ("AcquireItem", "a skill named AcquireItem, invoked on a robot with an item and "
                        "the location where that item sits, which acquires the item there"),
    ],
    "PlaceDown": [
        ("PlaceItem", "a skill named PlaceItem, invoked on a robot with an item and "
                      "a location, which sets the item down there"),
        ("DepositItem", "a skill named DepositItem, invoked on a robot with an item and "
                        "a location, which deposits the item there"),
        ("DeliverItem", "a skill named DeliverItem, invoked on a robot with an item and "
                        "a location, which delivers the item there"),
        ("SetItemDown", "a skill named SetItemDown, invoked on a robot with an item and "
                        "a location, which sets the item down there"),
        ("UnloadItem", "a skill named UnloadItem, invoked on a robot with an item and "
                       "a location, which unloads the item there"),
    ],
    "MoveTo": [
        ("DriveTo", "a skill named DriveTo, invoked on a robot with the location it "
                    "moves from and the location it moves to, which drives the robot "
                    "along that connection"),
        ("NavigateTo", "a skill named NavigateTo, invoked on a robot with the location "
                       "it moves from and the location it moves to, which navigates the "
                       "robot along that connection"),
        ("TravelTo", "a skill named TravelTo, invoked on a robot with the location it "
                     "moves from and the location it moves to, which moves the robot "
                     "along that connection"),
        ("GoTo", "a skill named GoTo, invoked on a robot with the location it moves "
                 "from and the location it moves to, which moves the robot along that "
                 "connection"),
    ],
    "Recharge": [
        ("PowerUp", "a skill named PowerUp, invoked on a robot with a charging-station "
                    "location, which recharges the robot's battery there"),
        ("ChargeUp", "a skill named ChargeUp, invoked on a robot with a charging-station "
                     "location, which recharges the robot's battery there"),
        ("DockAndCharge", "a skill named DockAndCharge, invoked on a robot with a "
                          "charging-station location, which recharges the robot's battery there"),
        ("RechargeBattery", "a skill named RechargeBattery, invoked on a robot with a "
                            "charging-station location, which recharges the robot's battery there"),
    ],
    "ClearPath": [
        ("ClearDebris", "a skill named ClearDebris, invoked on a robot with the two ends "
                        "of a blocked passage, which clears the debris between them"),
        ("UnblockPath", "a skill named UnblockPath, invoked on a robot with the two ends "
                        "of a blocked passage, which unblocks the passage between them"),
        ("RemoveDebris", "a skill named RemoveDebris, invoked on a robot with the two ends "
                         "of a blocked passage, which removes the debris between them"),
        ("OpenPath", "a skill named OpenPath, invoked on a robot with the two ends of a "
                     "blocked passage, which clears the passage between them"),
    ],
}


def loc_en(domain: str, loc: str) -> str:
    return LOC_EN[domain].get(loc, loc)


def item_en(domain: str, item: str) -> str:
    return ITEM_EN[domain].get(item, item)


def robot_en(robot: str) -> str:
    return ROBOT_EN.get(robot, robot)
