"""
======================================================
  Blender Script: Place Characters into Chemical Scene
  Repo: SaumyaShreya1/blender-assets
======================================================

HOW TO RUN:
-----------
Option A — Terminal (recommended):
    blender "chemical scene.blend" --python place_characters_in_chemical_scene.py

Option B — Inside Blender:
    1. Open "chemical scene.blend" in Blender
    2. Go to Scripting tab
    3. Open this file and click "Run Script"

BEFORE RUNNING:
---------------
- Make sure ALL these files are in the SAME folder as this script:
    * chemical scene.blend          (your environment)
    * poland soldier clear s.blend
    * privage military contractor.blend
    * sity soldier for scetchfab.blend
    * soldiers.blend

NOTE: Your scene file is .unity — if you only have the Unity version,
scroll down to the bottom of this file for Unity instructions.
======================================================
"""

import bpy
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Folder where all your .blend files live
# Change this path if your files are in a different location
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# Characters to import with their positions in the chemical scene
# Format: (filename, object_name_in_file, (X, Y, Z) position, (X, Y, Z) rotation degrees)
CHARACTERS = [
    (
        "poland soldier clear s.blend",
        None,           # None = auto-detect first mesh/armature
        (-3.0, 0.0, 0.0),
        (0, 0, 0),
    ),
    (
        "privage military contractor.blend",
        None,
        (0.0, 0.0, 0.0),
        (0, 0, 0),
    ),
    (
        "sity soldier for scetchfab.blend",
        None,
        (3.0, 0.0, 0.0),
        (0, 0, 45),
    ),
    (
        "soldiers.blend",
        None,
        (6.0, 0.0, 0.0),
        (0, 0, -45),
    ),
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clear_scene():
    """Remove default objects (cube, light, camera) if present."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("[✓] Scene cleared")


def get_objects_from_blend(blend_path):
    """Return list of object names inside a .blend file."""
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
        return list(data_from.objects)


def import_character(blend_path, position, rotation_deg):
    """
    Append all objects from a .blend file and move them to `position`.
    Returns the list of newly added objects.
    """
    blend_path = os.path.join(ASSETS_DIR, blend_path)

    if not os.path.exists(blend_path):
        print(f"[✗] File not found: {blend_path}")
        return []

    available_objects = get_objects_from_blend(blend_path)
    if not available_objects:
        print(f"[✗] No objects found in: {blend_path}")
        return []

    print(f"[→] Importing from: {os.path.basename(blend_path)}")
    print(f"    Objects found: {available_objects}")

    # Remember existing objects before import
    before = set(bpy.data.objects.keys())

    # Append all objects from the file
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    # Link newly imported objects to the current scene
    new_objects = []
    for obj in bpy.data.objects:
        if obj.name not in before:
            bpy.context.collection.objects.link(obj)
            new_objects.append(obj)

    if not new_objects:
        print(f"[✗] No new objects were linked from: {blend_path}")
        return []

    # Select only the new objects and apply transform
    bpy.ops.object.select_all(action='DESELECT')
    root_objects = []

    for obj in new_objects:
        if obj.parent is None:  # only move root objects
            obj.select_set(True)
            root_objects.append(obj)

    # Move to target position
    import math
    for obj in root_objects:
        obj.location = position
        obj.rotation_euler = (
            math.radians(rotation_deg[0]),
            math.radians(rotation_deg[1]),
            math.radians(rotation_deg[2]),
        )

    print(f"    [✓] Placed {len(new_objects)} objects at {position}")
    return new_objects


def scale_character(objects, scale=1.0):
    """Apply uniform scale to a list of objects."""
    for obj in objects:
        if obj.parent is None:
            obj.scale = (scale, scale, scale)


def save_scene(output_path):
    """Save the final merged scene."""
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    print(f"\n[✓] Scene saved to: {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  Placing Characters into Chemical Scene")
    print("="*55)

    # Step 1: Check scene file
    scene_blend = os.path.join(ASSETS_DIR, "chemical scene.blend")
    if not os.path.exists(scene_blend):
        print("\n[!] 'chemical scene.blend' not found in:", ASSETS_DIR)
        print("    Your scene is a .unity file.")
        print("    Please export it from Unity as FBX first,")
        print("    then re-run this script.")
        print("\n    See UNITY INSTRUCTIONS at bottom of this script.\n")
        # Continue anyway — will build scene from scratch with characters only
        clear_scene()
    else:
        print("[✓] Chemical scene found — opening...")
        bpy.ops.wm.open_mainfile(filepath=scene_blend)

    # Step 2: Import each character
    all_imported = []
    for blend_file, obj_name, position, rotation in CHARACTERS:
        imported = import_character(blend_file, position, rotation)
        all_imported.extend(imported)
        # Optional: scale characters to match scene (adjust 1.0 as needed)
        scale_character(imported, scale=1.0)

    print(f"\n[✓] Total objects imported: {len(all_imported)}")

    # Step 3: Save merged scene
    output_path = os.path.join(ASSETS_DIR, "chemical_scene_with_characters.blend")
    save_scene(output_path)

    print("\n" + "="*55)
    print("  DONE! Open this file in Blender:")
    print(f"  {output_path}")
    print("="*55 + "\n")


main()


"""
======================================================
  UNITY INSTRUCTIONS (if you want to use Unity instead)
======================================================

Since your scene file is "chemical scene.unity", here's
how to bring the characters into Unity:

STEP 1 — Export characters from Blender (do this for each .blend):
    1. Open the .blend file in Blender
    2. File → Export → FBX (.fbx)
    3. Settings:
        - Apply Transform: ✅ ON
        - Armature: ✅ ON (if character has bones)
        - Mesh: ✅ ON
    4. Save each as:
        - poland_soldier.fbx
        - military_contractor.fbx
        - city_soldier.fbx
        - soldiers.fbx

STEP 2 — Import into Unity:
    1. Open Unity project containing "chemical scene.unity"
    2. Drag all .fbx files into Assets/Characters/ folder
    3. Open "chemical scene.unity" scene

STEP 3 — Place characters in scene:
    1. Drag each character from Assets into the Scene Hierarchy
    2. Use Transform to position them in the chemical lab
    Suggested positions (X, Y, Z):
        - Poland Soldier:       (-3, 0, 0)
        - Military Contractor:  ( 0, 0, 0)
        - City Soldier:         ( 3, 0, 0)
        - Soldiers:             ( 6, 0, 0)

STEP 4 — OR use this Unity C# script to auto-place them:
    Attach to an empty GameObject in the scene:

    ---------------------------------------------------
    using UnityEngine;

    public class PlaceCharacters : MonoBehaviour {
        public GameObject polandSoldier;
        public GameObject militaryContractor;
        public GameObject citySoldier;
        public GameObject soldiers;

        void Start() {
            Vector3[] positions = {
                new Vector3(-3, 0, 0),
                new Vector3( 0, 0, 0),
                new Vector3( 3, 0, 0),
                new Vector3( 6, 0, 0),
            };
            GameObject[] chars = {
                polandSoldier, militaryContractor,
                citySoldier, soldiers
            };
            for (int i = 0; i < chars.Length; i++) {
                if (chars[i] != null)
                    Instantiate(chars[i], positions[i],
                                Quaternion.identity);
            }
        }
    }
    ---------------------------------------------------

======================================================
"""
