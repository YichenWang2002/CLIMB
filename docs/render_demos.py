import json, subprocess, xml.etree.ElementTree as ET
recs = [json.loads(l) for l in open('data/test.jsonl') if l.strip()]

# Demo records shown on the project page -- pinned to the exact missions whose
# natural-language texts appear in docs/index.html. Selection is by record
# index: the released dataset carries no tier labels.
DEMOS = [('demo_relay',    364),  # 3-agent relay, transient handover fault
         ('demo_heavy',      5),  # 2-robot cooperative transport, blocked passage
         ('demo_recovery', 360)]  # relay + CatalogBook service, blocked corridor

NL = chr(92) + 'n'

def emit(e, pfx, parent, lines, ctr):
    ctr[0] += 1
    nid = f"{pfx}{ctr[0]}"
    vals = [v for k, v in e.attrib.items()
            if k not in ('ID', 'BTCPP_format', 'main_tree_to_execute',
                         'num_attempts', 'success_threshold', 'failure_threshold')]
    lbl = e.tag
    if vals:
        lbl += NL + "(" + ",".join(vals) + ")"
    if e.get('success_threshold'):
        lbl += NL + "S>=" + e.get('success_threshold')
    shape = 'ellipse' if e.tag in ('IsAtLocation', 'IsItemAt', 'IsCarrying',
                                   'SignalReady', 'WaitReady') else 'box'
    color = '#6baed6' if e.tag in ('Sequence', 'Fallback', 'Parallel') else (
        '#a1d99b' if shape == 'ellipse' else '#fdd0a2')
    lines.append(f'  {nid}[label="{lbl}", shape={shape}, style="rounded,filled", '
                 f'fillcolor="{color}", fontname=Helvetica, fontsize=10];')
    if parent:
        lines.append(f'  {parent} -> {nid};')
    for c in e:
        emit(c, pfx, nid, lines, ctr)

for key, i in DEMOS:
    r = recs[i]
    xml = r['output']
    root = ET.fromstring(xml[xml.index('<root'):])
    lines = ['digraph G { rankdir=TB; splines=ortho; nodesep=0.3; ranksep=0.4; fontname=Helvetica;']
    ctr = [0]
    for bt in root.findall('BehaviorTree'):
        bid = bt.get('ID')
        lines.append(f'  subgraph cluster_{bid} {{ label="{bid}";')
        emit(bt[0], f"c{bid}_", None, lines, ctr)
        lines.append('  }')
    lines.append('}')
    open(f'docs/static/images/{key}.dot', 'w').write("\n".join(lines))
    subprocess.run(['dot', '-Tpng', '-Gdpi=150', '-o',
                    f'docs/static/images/{key}.png', f'docs/static/images/{key}.dot'], check=True)
    print(key, 'row', i, r['meta']['scenario'],
          'faults', [f['type'] for f in r['meta']['faults']])
    open(f'docs/static/images/{key}.txt', 'w').write(r['input'])
