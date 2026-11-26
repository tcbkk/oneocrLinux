#!/usr/bin/env python3
"""Summarise every dumped ONNX blob: inputs, outputs, opsets, op types, custom domains."""
import glob
import os
import sys

import onnx


def main(pattern):
    for p in sorted(glob.glob(pattern)):
        name = os.path.basename(p)
        try:
            m = onnx.load(p, load_external_data=True)
        except Exception as e:
            print("%-28s PARSE FAIL %s" % (name, e))
            continue
        g = m.graph
        def shape_of(v):
            try:
                return [d.dim_value if d.HasField("dim_value") else d.dim_param
                        for d in v.type.tensor_type.shape.dim]
            except Exception:
                return "?"
        ins = ["%s%s" % (i.name, shape_of(i)) for i in g.input]
        outs = ["%s%s" % (o.name, shape_of(o)) for o in g.output]
        ops = sorted({n.op_type for n in g.node})
        domains = sorted({n.domain for n in g.node if n.domain not in ("", "ai.onnx")})
        print("%-28s opset=%s prov=%s" % (name, [(o.domain, o.version) for o in m.opset_import],
                                          m.producer_name))
        print("    in : %s" % ins)
        print("    out: %s" % outs)
        print("    ops(%d): %s" % (len(ops), ",".join(ops)))
        if domains:
            print("    CUSTOM DOMAINS: %s" % domains)
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model_*.bin")
