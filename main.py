import argparse
from dataclasses import asdict
from datetime import time

from torch.utils.data import DataLoader

from core.wavenet import WaveNetCategorical
from util.metrics import save_random_postcue_plots
from util.util import *
from util.datasets import RandomWaveNetSegments
from util.Cfg import Cfg
from util.wavenet import dilations_1s_context, receptive_field
from core.training import train_model, eval_streaming_latency_comp, eval_postcue_window


def main():
    print("RUNNING FILE:", Path(__file__).resolve())

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_mode", choices=["pooled", "split"], default="pooled")

    # split-mode paths (theta-band pre-split)
    parser.add_argument("--valid_train_path", type=str, default="datasets/valid-cue-epochs-theta-training")
    parser.add_argument("--double_train_path", type=str, default="datasets/double-cue-epochs-theta-training")
    parser.add_argument("--valid_test_path", type=str, default="datasets/valid-cue-epochs-theta-test")
    parser.add_argument("--double_test_path", type=str, default="datasets/double-cue-epochs-theta-test")

    # data
    parser.add_argument("--valid_path", type=str, default="datasets/valid-cue-epochs")
    parser.add_argument("--double_path", type=str, default="datasets/double-cue-epochs")
    parser.add_argument("--channel_idx", type=int, default=2)

    # split
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.2)

    # run
    parser.add_argument("--out_dir", type=str, default="trained_wavenet")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")

    # eval settings
    parser.add_argument("--decode", type=str, default="expected", choices=["expected", "argmax"])
    parser.add_argument("--horizon_ms", type=float, default=50.0)
    parser.add_argument("--context_s", type=float, default=1.0)
    parser.add_argument("--eval_max_epochs", type=int, default=None, help="Max #epochs to evaluate (val/test). If None, evaluate all.")
    # streaming benchmark
    parser.add_argument("--stream_post_cue_s", type=float, default=3.1)
    parser.add_argument("--stream_step_ms", type=float, default=20.0)
    parser.add_argument("--no_stream_eval", action="store_true")

    # train overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--filters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    args = parser.parse_args()

    if args.eval_only:
        os.environ["WANDB_DISABLED"] = "true"

    out_dir = Path(args.out_dir)
    logger = setup_logger(out_dir)
    set_seed(0)

    cfg = Cfg()
    cfg.context_s = float(args.context_s)
    cfg.horizon_s = float(args.horizon_ms) / 1000.0

    # quantization range depends on existing dataset split:
    # - pooled => raw (±8e-5)
    # - split  => theta (±1.5e-5)
    if args.dataset_mode == "split":
        cfg.amp_min, cfg.amp_max = -1.5e-05, 1.5e-05
    else:
        cfg.amp_min, cfg.amp_max = -8.0e-05, 8.0e-05

    logger.info(f"USING amp_min/amp_max: {cfg.amp_min:.3e} / {cfg.amp_max:.3e} | n_bins={cfg.n_bins}")


    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.filters is not None:
        cfg.n_filters = int(args.filters)
    if args.lr is not None:
        cfg.lr = float(args.lr)

    dils = dilations_1s_context()
    cfg.receptive_field = int(receptive_field(cfg.kernel_size, dils))
    logger.info(
        f"Architecture: k={cfg.kernel_size}, "
        f"layers={len(dils)}, "
        f"RF={cfg.receptive_field} samples "
        f"({cfg.receptive_field/cfg.sfreq:.3f}s)"
    )


    logger.info("LOAD DATA...")
    t0 = time.perf_counter()

    if args.dataset_mode == "pooled":
        valid = load_json_epochs(resolve_epoch_path(args.valid_path))
        double = load_json_epochs(resolve_epoch_path(args.double_path))

        cfg.sfreq = float(valid.get("sfreq", double.get("sfreq", cfg.sfreq)))

        x_all = np.concatenate([valid["data"], double["data"]], axis=0)
        if isinstance(x_all, list):
            x_all = np.array(x_all)

        N, C, T = x_all.shape
        ch = int(args.channel_idx)
        if ch < 0 or ch >= C:
            raise ValueError(f"channel_idx={ch} out of range for C={C}")

        epochs_1d = [x_all[i, ch, :].astype(np.float64) for i in range(N)]
        logger.info(f"sfreq={cfg.sfreq} | n_epochs={N} | T={T} | load_dt={time.perf_counter()-t0:.2f}s")

        train_ids, val_ids, test_ids = split_epochs(
            N, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed
        )
        tr_list = [epochs_1d[int(i)] for i in train_ids]
        va_list_full = [epochs_1d[int(i)] for i in val_ids]
        te_list_full = [epochs_1d[int(i)] for i in test_ids]

        logger.info(f"splits (uLAR-style): train={len(tr_list)} val={len(va_list_full)} test={len(te_list_full)}")

    else:
        valid_tr = load_json_epochs(resolve_epoch_path(args.valid_train_path))
        double_tr = load_json_epochs(resolve_epoch_path(args.double_train_path))
        valid_te = load_json_epochs(resolve_epoch_path(args.valid_test_path))
        double_te = load_json_epochs(resolve_epoch_path(args.double_test_path))

        cfg.sfreq = float(valid_tr.get("sfreq", double_tr.get("sfreq", cfg.sfreq)))

        x_tr = np.concatenate([valid_tr["data"], double_tr["data"]], axis=0)
        x_te = np.concatenate([valid_te["data"], double_te["data"]], axis=0)
        if isinstance(x_tr, list): x_tr = np.array(x_tr)
        if isinstance(x_te, list): x_te = np.array(x_te)

        Ntr, C, T = x_tr.shape
        Nte = x_te.shape[0]

        ch = int(args.channel_idx)
        if ch < 0 or ch >= C:
            raise ValueError(f"channel_idx={ch} out of range for C={C}")

        epochs_tr_1d = [x_tr[i, ch, :].astype(np.float64) for i in range(Ntr)]
        epochs_te_1d = [x_te[i, ch, :].astype(np.float64) for i in range(Nte)]

        logger.info(
            f"sfreq={cfg.sfreq} | train_epochs={Ntr} | test_epochs={Nte} | T={T} | load_dt={time.perf_counter()-t0:.2f}s"
        )

        train_ids, val_ids, _ = split_epochs(
            Ntr, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed
        )
        tr_list = [epochs_tr_1d[int(i)] for i in train_ids]
        va_list_full = [epochs_tr_1d[int(i)] for i in val_ids]
        te_list_full = epochs_te_1d
        logger.info(
            f"splits (on TRAIN pool): train={len(tr_list)} val={len(va_list_full)} | explicit test={len(te_list_full)}"
        )

    max_eval = int(args.eval_max_epochs) if args.eval_max_epochs is not None else None
    va_list = va_list_full[:max_eval] if max_eval is not None else va_list_full
    te_list = te_list_full[:max_eval] if max_eval is not None else te_list_full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    run_name = f"eeg-wavenet-ch{ch}-h{args.horizon_ms:.0f}ms"
    wb = wandb_init(asdict(cfg), run_name=run_name)

    model = WaveNetCategorical(n_bins=cfg.n_bins, n_filters=cfg.n_filters, kernel_size=cfg.kernel_size)

    if args.ckpt is not None:
        bundle = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(bundle, dict) and "model" in bundle:
            model.load_state_dict(bundle["model"])
        else:
            model.load_state_dict(bundle)
        logger.info(f"Loaded checkpoint: {args.ckpt}")

        if torch.cuda.is_available():
            logger.info(f"[GPU CHECK] cuda mem allocated MB={torch.cuda.memory_allocated()/1024**2:.1f}")


    model.to(device)

    if not args.eval_only:
        ckpt_dir = out_dir / "checkpoints"

        train_ds = RandomWaveNetSegments(
            epochs_1d=tr_list,
            seq_len=cfg.seq_len,
            n_samples=cfg.train_samples_per_epoch,
            amp_min=cfg.amp_min,
            amp_max=cfg.amp_max,
            n_bins=cfg.n_bins,
            rng=np.random.default_rng(1),
        )

        ys = []
        for i in range(10):
            _, y0 = train_ds[i]
            ys.append(y0)
        ycat = torch.cat(ys)
        logger.info(
            f"DEBUG train_ds bins: unique={int(torch.unique(ycat).numel())} "
            f"minbin={int(ycat.min())} maxbin={int(ycat.max())}"
        )


        val_ds_tmp = RandomWaveNetSegments(
            epochs_1d=va_list_full,
            seq_len=cfg.seq_len,
            n_samples=cfg.val_samples_fixed,
            amp_min=cfg.amp_min,
            amp_max=cfg.amp_max,
            n_bins=cfg.n_bins,
            rng=np.random.default_rng(2),
        )
        fixed_pairs = list(val_ds_tmp.pairs)
        val_ds = RandomWaveNetSegments(
            epochs_1d=va_list_full,
            seq_len=cfg.seq_len,
            n_samples=len(fixed_pairs),
            amp_min=cfg.amp_min,
            amp_max=cfg.amp_max,
            n_bins=cfg.n_bins,
            rng=np.random.default_rng(2),
            fixed_pairs=fixed_pairs,
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)


        rf = cfg.receptive_field
        counts = torch.zeros(cfg.n_bins, dtype=torch.long)
        total = 0
        for _, yb in val_loader:
            y_v = yb[:, rf:].reshape(-1)  # ignora i primi rf come fai in training
            counts += torch.bincount(y_v.cpu(), minlength=cfg.n_bins)
            total += int(y_v.numel())
        p = counts.float() / max(total, 1)
        unigram_ce = (-(p.clamp_min(1e-12).log()) * counts.float()).sum() / max(total, 1)
        logger.info(f"BASELINE val_unigram_ce={unigram_ce.item():.6e} (lower is better)")


        logger.info(f"Checkpoints: {ckpt_dir}")
        logger.info(f"TRAIN for up to {cfg.epochs} epochs...")

        model = train_model(
            model=model,
            train_ds=train_ds,
            val_loader=val_loader,
            rf=cfg.receptive_field,
            cfg=cfg,
            device=device,
            logger=logger,
            wb_run=wb,
            ckpt_dir=ckpt_dir,
        )

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, out_dir / "final.pt")
        logger.info(f"Saved: {out_dir/'final.pt'}")

    model.eval()

    logger.info(
        f"EVAL: {args.horizon_ms:.0f}ms post-cue window using {cfg.context_s:.1f}s pre-cue context..."
    )
    m_val = eval_postcue_window(model, va_list, cfg, device, decode=args.decode)
    m_test = eval_postcue_window(model, te_list, cfg, device, decode=args.decode)
    logger.info(f"VAL {args.horizon_ms:.0f}ms-postcue:  {m_val}")
    logger.info(f"TEST {args.horizon_ms:.0f}ms-postcue: {m_test}")

    save_random_postcue_plots(
        model=model,
        epochs_1d=te_list,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        n_plots=10,
        split_name="test",
        seed=0,
    )
    logger.info(f"Saved 10 random post-cue plots to: {out_dir}")

    (out_dir / f"val_metrics_{int(round(args.horizon_ms))}ms_postcue.json").write_text(json.dumps(m_val, indent=2))
    (out_dir / f"test_metrics_{int(round(args.horizon_ms))}ms_postcue.json").write_text(json.dumps(m_test, indent=2))

    if wb is not None:
        wb.log({f"val_postcue/{k}": float(v) for k, v in m_val.items() if k != "N"})
        wb.log({f"test_postcue/{k}": float(v) for k, v in m_test.items() if k != "N"})

    if not args.no_stream_eval:
        logger.info(
            f"EVAL: STREAMING latency-comp (predict x(t+{args.horizon_ms:.0f}ms) for many t) | post_cue_s={args.stream_post_cue_s} | step_ms={args.stream_step_ms}"
        )
        m_val_s = eval_streaming_latency_comp(
            model,
            va_list,
            cfg,
            device,
            decode=args.decode,
            post_cue_s=float(args.stream_post_cue_s),
            step_ms=float(args.stream_step_ms),
            max_epochs=max_eval,
        )
        m_test_s = eval_streaming_latency_comp(
            model,
            te_list,
            cfg,
            device,
            decode=args.decode,
            post_cue_s=float(args.stream_post_cue_s),
            step_ms=float(args.stream_step_ms),
            max_epochs=max_eval,
        )
        logger.info(f"VAL streaming {args.horizon_ms:.0f}ms-ahead:  {m_val_s}")
        logger.info(f"TEST streaming {args.horizon_ms:.0f}ms-ahead: {m_test_s}")

        (out_dir / f"val_stream_{int(round(args.horizon_ms))}ms_step{int(round(args.stream_step_ms))}ms.json").write_text(
            json.dumps(m_val_s, indent=2)
        )
        (out_dir / f"test_stream_{int(round(args.horizon_ms))}ms_step{int(round(args.stream_step_ms))}ms.json").write_text(
            json.dumps(m_test_s, indent=2)
        )

        if wb is not None:
            wb.log({f"val_stream/{k}": float(v) for k, v in m_val_s.items() if k != "N"})
            wb.log({f"test_stream/{k}": float(v) for k, v in m_test_s.items() if k != "N"})

    if wb is not None:
        wb.finish()

    logger.info("DONE")


if __name__ == "__main__":
    main()
