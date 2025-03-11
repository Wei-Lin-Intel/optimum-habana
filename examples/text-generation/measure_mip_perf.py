import os
import json
import subprocess
import argparse


sample_json = {
    "method": "HOOKS",
    "mode": "QUANTIZE",
    "observer": "maxabs",
    "scale_method": "maxabs_pow2",
    "dump_stats_path": "./hqt_output/measure",
    "allowlist": {"types": [], "names":  []},
    "blocklist": {"types": [], "names":  []}
}
# model.layers.22.self_attn.q_proj
# 'model.layers.30.self_attn.q_proj', 'model.layers.30.self_attn.k_proj', 'model.layers.30.self_attn.o_proj', 'model.layers.30.self_attn.matmul_qk', 'model.layers.30.self_attn.matmul_av', 'model.layers.30.self_attn.k_cache', 'model.layers.30.self_attn.v_cache', 'model.layers.30.mlp.gate_proj', 'model.layers.30.mlp.up_proj', 'model.layers.30.mlp.down_proj'

layers_to_check = {
    "q_proj": [ "self_attn.q_proj"],
    "o_proj": [ "self_attn.o_proj"],
    "v_proj": ["self_attn.v_proj", "self_attn.v_cache"],
    "k_proj": ["self_attn.k_proj", "self_attn.k_cache"],
    "matmul_qk": ["self_attn.matmul_qk"], 
    "matmul_av": ["self_attn.matmul_av"],
    "gate_proj": ["mlp.gate_proj"],
    "up_proj": ["mlp.up_proj"],
    "down_proj": ["mlp.down_proj"]
}

# always_quantize = ["cache"]
def run_cmd(name, i, env, args, config_filename=None):
    cmd = [
        "python", "run_generation.py",
        "--model_name_or_path", "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/",
        "--n_iterations", "5",
        "--input_embeds",
        "--use_hpu_graphs",
        "--batch_size", "10",
        "--output_dir", args.out_dir,
        "--out_name", f"block_{name}_{i}",
        "--simulate_dyn_prompt", "1024",
        "--max_new_tokens", "1"
    ]
        # "--use_kv_cache",
    
    print(f"\n=== Running experiment for : {name}_{i} ===")
    if config_filename:
        print("QUANT_CONFIG=", env["QUANT_CONFIG"])
    print("Environment Variables:")
    print("Command:", " ".join(cmd))
    
    # Run the command
    subprocess.run(cmd, env=env)


def run_per_layer(env, args):
    for i in range(32):
        for name, layers_to_quantize in layers_to_check.items():
            layers_to_quantize_config = [f"model.layers.{i}.{layer}" for layer in layers_to_quantize]
            # Create a copy of sample_json for this layer
            config_copy = dict(sample_json)
            
            # Insert layer name into blocklist["names"]
            config_copy["blocklist"]["names"] = layers_to_quantize_config 
            
            # Create a unique file name for the config
            config_filename = f"./additive_exp_configs/block_{name}_{i}.json"
            
            # Save updated JSON
            with open(config_filename, "w") as f:
                json.dump(config_copy, f, indent=4)
            
            # Prepare environment variables
            # env["BLOCKS_STATS_SAVE_PATH"] = f"./test_blocks_{layer}.txt"
            # env["MEASURE_BLOCKS"] = "True"
            env["QUANT_CONFIG"] = config_filename
            run_cmd(name, i, env, args, config_filename)


def run_all_layers(env, args):
    for name, layers_to_quantize in layers_to_check.items():
        layers_to_quantize_config = []
        for i in range(32):
            layers_to_quantize_config += [f"model.layers.{i}.{layer}" for layer in layers_to_quantize]
        print("layers_to_quantize", layers_to_quantize_config)
        # Create a copy of sample_json for this layer
        config_copy = dict(sample_json)
        
        # Insert layer name into blocklist["names"]
        config_copy["blocklist"]["names"] = layers_to_quantize 
        
        # Create a unique file name for the config
        config_filename = f"./additive_exp_configs/block_{name}_{i}.json"
        
        # Save updated JSON
        with open(config_filename, "w") as f:
            json.dump(config_copy, f, indent=4)
        
        # Prepare environment variables
        # env["BLOCKS_STATS_SAVE_PATH"] = f"./test_blocks_{layer}.txt"
        # env["MEASURE_BLOCKS"] = "True"
        env["QUANT_CONFIG"] = config_filename
        run_cmd(name, i, env, args, config_filename)


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("--per_layer", action="store_true")
    p.add_argument("--all_layers", action="store_true")
    p.add_argument("--out_dir", type=str)
    p.add_argument("--configs_dir", type=str, default="./exp_configs/")
    
    args = p.parse_args()
    if not args.per_layer and not args.all_layers:
        print("No option chosen")
        quit()

    env = os.environ.copy()
    # config_filename = f"./additive_exp_configs_only_prefill/all_fp8.json"
    # config_copy = dict(sample_json)
    # with open(config_filename, "w") as f:
    #     json.dump(config_copy, f, indent=4)
    #
    # env["QUANT_CONFIG"] = config_filename
    # run_cmd("all_fp8", 0, env, args)
    # if args.per_layer:
    #     run_per_layer(env, args)
    # elif args.all_layers:
    #     run_all_layers(env, args)


    # Test lm-head
    config_copy_lm_head = dict(sample_json)
    lm_head_config_filename = os.path.join(args.configs_dir, "block_lm_head.json")
    config_copy_lm_head["blocklist"]["names"] = ['lm_head']
    with open(lm_head_config_filename, "w") as f:
        json.dump(config_copy_lm_head, f, indent=4)

    env["QUANT_CONFIG"] = lm_head_config_filename 
    run_cmd("lm_head", 0, env, args)



    
            

if __name__ == "__main__":
    main()

