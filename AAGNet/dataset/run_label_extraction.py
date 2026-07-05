# -*- coding: utf-8 -*-
import argparse
import glob
import os
from multiprocessing import Pool

from extract_label_from_MFCADPP import generate_graph


def _worker(args):
    shape_dir, graph_dir, shape_name = args
    try:
        generate_graph(shape_dir, graph_dir, shape_name)
        return shape_name, True
    except Exception as e:
        return shape_name, str(e)


def initializer():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    shape_paths = glob.glob(os.path.join(args.step_path, "*.st*p"))
    shape_names = [os.path.splitext(os.path.basename(p))[0] for p in shape_paths]

    tasks = [(args.step_path, args.output, name) for name in shape_names]

    failures = []
    pool = Pool(processes=args.num_workers, initializer=initializer)
    try:
        from tqdm import tqdm
        for name, ok in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)):
            if ok is not True:
                failures.append((name, ok))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    print(f"Processed {len(tasks)} files, {len(failures)} failures.")
    if failures:
        with open(os.path.join(args.output, "failed_extractions.txt"), "w") as f:
            for name, err in failures:
                f.write(f"{name}\t{err}\n")


if __name__ == "__main__":
    main()
