"""Write workflows/phrase2stickman.json, the same graph stickman_i2v.py runs, in the format ComfyUI's UI loads
(drag the file onto the canvas, or Workflow > Open). Needs ComfyUI running, to read the node definitions.

    python export_workflow.py [--compile]

The UI version saves with ComfyUI's core Create Video + Save Video nodes, so it needs no extra node pack beyond comfyui-w2m.
Edit the mime in the green "mime" text box (the prompt node) and press Run.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # ComfyUI's embedded Python doesn't put the script folder on sys.path
import comfy_client as cc  # noqa: E402
import stickman_i2v as si  # noqa: E402

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
TITLES = {"1": "High-noise expert (NVFP4)", "2": "Low-noise expert (NVFP4)", "9": "Prompt: style + mime",
          "10": "Negative (unused at CFG 1)", "r1": "Reference stickman", "r2": "Reference to 480x832",
          "11": "Start image (encode cached)", "12": "Sampler: steps 0-2 (high noise)", "13": "Sampler: steps 2-4 (low noise)",
          "50": "Tiny VAE (fast decode)", "82": "Lanczos up to 480x832", "83": "Light sharpen",
          "62": "Compile high-noise blocks", "63": "Compile low-noise blocks"}
NOTE = ("Phrase2Stickman. Edit the mime sentence at the end of the Prompt node, then Run.\n"
        "Needs: the comfyui-w2m node, and the models listed in the README (python setup.py installs them).\n"
        "Sampling at 384x672 and scaling up to 480x832 is deliberate: 320 breaks gestures, 480+ is slower and no better.")


def to_api(compile_blocks):
    wf = si.build_workflow(si.MIMES["thumbs_up"], compile_blocks=compile_blocks)
    # core Create Video + Save Video instead of VHS_VideoCombine, so no extra node pack is needed
    wf.pop("15")
    wf["15"] = {"class_type": "CreateVideo", "inputs": {"images": ["83", 0], "fps": si.FPS}}
    wf["16"] = {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": "stickman/out", "format": "mp4", "codec": "h264"}}
    return wf


def is_widget(spec):
    t = spec[0]
    return isinstance(t, list) or t in WIDGET_TYPES


def convert(wf, oi):
    # depth = longest path from a source, for left-to-right layout
    depth = {}

    def d(k):
        if k not in depth:
            ups = [v[0] for v in wf[k]["inputs"].values() if isinstance(v, list)]
            depth[k] = 1 + max((d(u) for u in ups), default=-1)
        return depth[k]
    for k in wf:
        d(k)
    ids = {k: i + 1 for i, k in enumerate(sorted(wf, key=lambda k: (depth[k], k)))}
    rows, nodes, links, out_links = {}, [], [], {}
    for order, k in enumerate(sorted(wf, key=lambda k: ids[k])):
        n = wf[k]
        info = oi[n["class_type"]]
        spec_in = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        inputs, widgets = [], []
        for name in info["input_order"].get("required", []) + info["input_order"].get("optional", []):
            spec = spec_in[name]
            val = n["inputs"].get(name)
            if isinstance(val, list):  # a link
                lid = len(links) + 1
                links.append([lid, ids[val[0]], val[1], ids[k], len(inputs), oi[wf[val[0]]["class_type"]]["output"][val[1]]])
                out_links.setdefault((val[0], val[1]), []).append(lid)
                inputs.append({"name": name, "type": spec[0] if isinstance(spec[0], str) else "COMBO", "link": lid})
            elif is_widget(spec):
                widgets.append(val if val is not None else (spec[1].get("default") if len(spec) > 1 else None))
                if name in ("seed", "noise_seed"):
                    widgets.append("fixed")  # control_after_generate
                if n["class_type"] == "LoadImage":
                    widgets.append("image")  # the upload button widget
        col = depth[k]
        r = rows.get(col, 0)
        text = n["class_type"] == "CLIPTextEncode"
        w, h = (420, 260) if text else (330, 60 + 28 * max(len(widgets), len(inputs)))
        if n["class_type"] == "LoadImage":
            h = 380
        rows[col] = r + h + 40
        nodes.append({"id": ids[k], "type": n["class_type"], "pos": [60 + col * 400, 60 + r], "size": [w, h], "flags": {},
                      "order": order, "mode": 0, "inputs": inputs, "outputs": [], "properties": {"Node name for S&R": n["class_type"]},
                      "widgets_values": widgets, **({"title": TITLES[k]} if k in TITLES else {})})
        if k == "9":
            nodes[-1]["color"], nodes[-1]["bgcolor"] = "#232", "#353"
        for i, (oname, otype) in enumerate(zip(info["output_name"], info["output"])):
            nodes[-1]["outputs"].append({"name": oname, "type": otype, "links": out_links.get((k, i)), "slot_index": i})
    # output link lists were filled after the producing node was emitted when the consumer came later; patch them now
    by_id = {n["id"]: n for n in nodes}
    for (k, slot), lids in out_links.items():
        by_id[ids[k]]["outputs"][slot]["links"] = lids
    nodes.append({"id": len(nodes) + 1, "type": "Note", "pos": [60, -240], "size": [760, 150], "flags": {}, "order": len(nodes),
                  "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [NOTE], "title": "Read me"})
    return {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compile", action="store_true", help="include the W2MCompileWan nodes")
    ap.add_argument("--server", default=cc.SERVER)
    a = ap.parse_args()
    cc.ensure_comfy(a.server)
    try:
        oi = cc._get(a.server, "/object_info", timeout=120)
    finally:
        cc.stop_comfy()  # only if we started it
    ui = convert(to_api(a.compile), oi)
    dst = os.path.join(HERE, "workflows", "phrase2stickman" + ("_compile" if a.compile else "") + ".json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(ui, open(dst, "w", encoding="utf-8"), indent=1)
    print(f"[workflow] {os.path.relpath(dst, HERE)}: {len(ui['nodes'])} nodes, {len(ui['links'])} links")
