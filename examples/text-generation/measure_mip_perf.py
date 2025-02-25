

import json

sample_json = {
    "method": "HOOKS",
    "mode": "QUANTIZE",
    "observer": "maxabs",
    "scale_method": "maxabs_pow2",
    "dump_stats_path": "./hqt_output/measure",
    "allowlist": {"types": [], "names":  []},
    "blocklist": {"types": [], "names":  []}
}

def main():
    sample_json["blocklist"]["names"] = "v_proj"
    print(sample_json)


if __name__ == "__main__":
    main()

