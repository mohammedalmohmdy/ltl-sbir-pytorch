
import json, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_json', required=True, help='Path to runs/.../eval.json')
    ap.add_argument('--out_prefix', default='plots', help='Output filename prefix')
    args = ap.parse_args()

    with open(args.eval_json, 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})
    cmc = data.get('cmc@1..K') or data.get('cmc', [])

    # Print metrics to console
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # CMC curve
    if cmc:
        plt.figure()
        plt.plot(range(1, len(cmc)+1), cmc, marker='o')
        plt.xlabel('Rank')
        plt.ylabel('CMC')
        plt.title('Cumulative Match Characteristic')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_cmc.png", dpi=150)
        print("Saved:", f"{args.out_prefix}_cmc.png")

if __name__ == "__main__":
    main()
