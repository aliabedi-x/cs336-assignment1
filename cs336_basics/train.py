import cs336_basics.model as model
from pathlib import Path
import argparse
import math
import numpy as np
import torch
import random

# Hyperparameters: vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
# Optimizer Hyperparameters: max_lr, min_lr, beta1, beta2, eps, weight_decay, warmpup_iters, cosine_cycle_iters, max_grad_norm
# Training Configuration: batch_size, max_iters, device, train_path, val_path, checkpoint_path, interval_path, eval_interval, eval_iters, seed

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Adding argparse to be able to adjust Hyperparmeters via terminal
def params():
    parser = argparse.ArgumentParser() 
    # Hyperp
    parser.add_argument("--vocab-size",type=int,default=10000)
    parser.add_argument("--context-length",type=int,default=128)
    parser.add_argument("--d-model",type=int,default=128)
    parser.add_argument("--num-layers",type=int,default=2)
    parser.add_argument("--num-heads",type=int,default=4)
    parser.add_argument("--d-ff",type=int,default=384)
    parser.add_argument("--rope-theta",type=float,default=10000.0)

    # Opt Hyperp
    parser.add_argument("--max-lr",type=float,default=0.001)
    parser.add_argument("--min-lr",type=float,default=0.0001)
    parser.add_argument("--beta1",type=float,default=0.9)
    parser.add_argument("--beta2",type=float,default=0.999)
    parser.add_argument("--eps",type=float,default=1e-8)
    parser.add_argument("--weight-decay",type=float,default=0.01)
    parser.add_argument("--warmup-iters",type=int,default=10)
    parser.add_argument("--cosine-cycle-iters",type=int,default=100)
    parser.add_argument("--max-grad-norm",type=float,default=1.0)

    # Train config 
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--max-iters",type=int,default=100)
    parser.add_argument("--device",type=str,default="mps")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--log-interval",type=int,default=10)
    parser.add_argument("--eval-interval",type=int,default=20)
    parser.add_argument("--eval-iters",type=int,default=5)
    parser.add_argument("--checkpoint-interval",type=int,default=50)
    parser.add_argument("--train-path",type=str,required=True)
    parser.add_argument("--val-path",type=str,required=True)
    parser.add_argument("--checkpoint-path",type=str,required=True)
    parser.add_argument("--resume-from",type=str,required=False)
    args = parser.parse_args()
    return args

def validate_config(args):
    if args.vocab_size <= 0:
        raise ValueError("vocab_size must be positive (> 0)")
    if args.context_length <= 0:
        raise ValueError("context_length must be positive (> 0)")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive (> 0)")
    if args.d_model <= 0:
        raise ValueError("d_model must be positive (> 0)")
    if args.num_heads <= 0:
        raise ValueError("num_heads must be positive (> 0)")
    if args.num_layers <= 0:
        raise ValueError("num_layers must be positive (> 0)")
    if args.d_ff <= 0:
        raise ValueError("d_ff must be positive (> 0)")

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = args.d_model // args.num_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even (head_dim % 2 == 0)")

    if not (0 <= args.min_lr <= args.max_lr) or args.max_lr <= 0:
        raise ValueError("Learning rates must satisfy 0 <= min_lr <= max_lr with max_lr > 0")

    if not (0 <= args.beta1 < 1) or not (0 <= args.beta2 < 1):
        raise ValueError("beta1 and beta2 must be in [0, 1)")

    if args.eps <= 0:
        raise ValueError("eps must be positive (> 0)")

    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative (>= 0)")

    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive (> 0)")

    if not (0 <= args.warmup_iters < args.cosine_cycle_iters):
        raise ValueError("warmup_iters and cosine_cycle_iters must satisfy 0 <= warmup_iters < cosine_cycle_iters")

    if args.max_iters <= 0:
        raise ValueError("max_iters must be positive (> 0)")

    if args.rope_theta <= 0:
        raise ValueError("rope_theta must be positive (> 0)")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive (> 0)")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive (> 0)")
    if args.eval_iters <= 0:
        raise ValueError("eval_iters must be positive (> 0)")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive (> 0)")
    if not (0 <= args.seed < 2**32):
        raise ValueError("seed must satisfy 0 <= seed < 2**32")

    for name in ("rope_theta", "max_lr", "min_lr", "beta1", "beta2", "eps", "weight_decay", "max_grad_norm"):
        val = getattr(args, name)
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite (not NaN or Inf)")

def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested, but MPS is not available.")
        return torch.device("mps")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but CUDA is not available.")
        return torch.device("cuda")
    else:
        raise ValueError(f"Invalid device name: '{device_name}'. Expected one of 'cpu', 'mps', or 'cuda'.")

def load_dataset(path: str, context_length: int) -> np.memmap:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

    if file_size % 2 != 0:
        raise ValueError(
            f"File size ({file_size} bytes) is not divisible by 2 (uint16 itemsize)."
        )

    tokens = np.memmap(file_path, dtype=np.uint16, mode="r")

    # برای ساخت یک نمونه ورودی x به طول context_length و هدف y شیفت‌یافته، حداقل context_length + 1 توکن نیاز است
    if tokens.size < context_length + 1:
        raise ValueError(
            f"Not enough tokens ({tokens.size}) for context_length={context_length}."
        )

    return tokens
   
def train_step(transformer,optimizer,train_data, iteration, args,device) -> float: 
    # 1. Calculate learning rate
    new_lr = model.learing_rate_schedule(
        t=iteration,
        a_max=args.max_lr,
        a_min=args.min_lr,
        T_w=args.warmup_iters,
        T_c=args.cosine_cycle_iters,
    )

    # 2. Update optimizer learning rate
    for group in optimizer.param_groups:
        group["lr"] = new_lr

    # 3. Sample input and target tensors
    x, y = model.data_loading(
        x=train_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )

    # 4. Forward pass
    logits = transformer(x)

    # 5. Calculate cross-entropy loss
    loss = model.cross_entropy(logits, y)

    # 6. Zero gradients
    optimizer.zero_grad()

    # 7. Backward pass
    loss.backward()

    # 8. Gradient clipping
    model.gradient_clipping(transformer.parameters(), args.max_grad_norm)

    # 9. Optimizer step
    optimizer.step()

    # 10. Return loss as Python float
    return float(loss.item())

def evaluate(transformer,val_data,args,device) -> float:
    transformer.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = model.data_loading(
                x=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )

            logits = transformer(x)
            loss = model.cross_entropy(logits, y)
            total_loss += float(loss.item())

    transformer.train()
    mean_loss = total_loss / args.eval_iters
    return float(mean_loss)




def main():
    args = params()
    validate_config(args)

    device = resolve_device(args.device)
    set_seed(args.seed)

    train = load_dataset(args.train_path, args.context_length)
    valid = load_dataset(args.val_path, args.context_length)

    transformer = model.TransformerLm(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device=device,
    )
    adamw = model.AdamW(
        transformer.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    total = sum(p.numel() for p in transformer.parameters())
    print(f"Total params: {total}")

    start_iteration = 0
    if args.resume_from:
        start_iteration = model.load_checkpoint(args.resume_from, transformer, adamw)

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    transformer.train()
    for iteration in range(start_iteration, args.max_iters):
        loss = train_step(transformer, adamw, train, iteration, args, device)
        step = iteration + 1

        if step % args.log_interval == 0:
            print(f"[Train] iter {step}: {loss}")

        if step % args.eval_interval == 0:
            evalt = evaluate(transformer, valid, args, device)
            print(f"[Val] iter {step}: {evalt}")

        if step % args.checkpoint_interval == 0 or step == args.max_iters:
            path = Path(args.checkpoint_path)
            ckpt_path = path.with_name(f"{path.stem}_step_{step}.pt")
            model.save_checkpoint(transformer, adamw, step, ckpt_path)
    

if __name__ == "__main__":
    main()


