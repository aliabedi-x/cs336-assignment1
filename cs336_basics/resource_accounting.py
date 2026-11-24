
vocab_size = 50257
context_length = 1024
num_layers = 48
d = 1600
num_heads = 25
dff = 4288
batch_size = 1
T = context_length


attention_params = 4 * (d ** 2)
rmsnorm_block_params = 2 * d          
final_rmsnorm_params = d              
swiglu_params = 3 * d * dff

block_params = attention_params + rmsnorm_block_params + swiglu_params
lm_head_params = vocab_size * d
token_emb_params = vocab_size * d

total_params = (block_params * num_layers) + final_rmsnorm_params + lm_head_params + token_emb_params
memory_bytes = total_params * 4
memory_gb = memory_bytes / 1e9

print("--- Part A: Parameters & Memory GPT-2 XL  ---")
print(f"Params per block: {block_params:,}")
print(f"Total params:     {total_params:,}")
print(f"Memory (bytes):   {memory_bytes:,}")
print(f"Memory (~GB):     {memory_gb:.2f} GB")


d_head = d // num_heads

f_projections = batch_size * 4 * 2 * T * (d ** 2)
f_qk = batch_size * num_heads * 2 * T * T * d_head
f_av = batch_size * 2 * (T ** 2) * d
f_swiglu = batch_size * 3 * 2 * T * d * dff

block_flops = f_projections + f_qk + f_av + f_swiglu
all_blocks_flops = block_flops * num_layers
f_lm_head = batch_size * 2 * T * d * vocab_size
total_flops = all_blocks_flops + f_lm_head

print("\n--- Part B: FLOPs GPT-2 XL ---")
print(f"F_projections:  {f_projections:,}")
print(f"F_QK:           {f_qk:,}")
print(f"F_AV:           {f_av:,}")
print(f"F_SwiGLU:       {f_swiglu:,}")
print(f"Block total:    {block_flops:,}")
print(f"All 48 blocks:  {all_blocks_flops:,}")
print(f"F_LMHead:       {f_lm_head:,}")
print(f"Total FLOPs:    {total_flops:,}")


print("--- Part D: Different model size analysis ---")
print("--- SMALL ---")

vocab_size = 50257
context_length = 1024
num_layers = 12
d = 768
num_heads = 12
dff = 4288
batch_size = 1
T = context_length
attention_params = 4 * (d ** 2)
rmsnorm_block_params = 2 * d          
final_rmsnorm_params = d              
swiglu_params = 3 * d * dff

d_head = d // num_heads

f_projections = batch_size * 4 * 2 * T * (d ** 2)
f_qk = batch_size * num_heads * 2 * T * T * d_head
f_av = batch_size * 2 * (T ** 2) * d
f_swiglu = batch_size * 3 * 2 * T * d * dff

block_flops = f_projections + f_qk + f_av + f_swiglu
all_blocks_flops = block_flops * num_layers
f_lm_head = batch_size * 2 * T * d * vocab_size
total_flops = all_blocks_flops + f_lm_head

print(f"F_projections:  {f_projections:,}")
print(f"F_QK:           {f_qk:,}")
print(f"F_AV:           {f_av:,}")
print(f"F_SwiGLU:       {f_swiglu:,}")
print(f"Block total:    {block_flops:,}")
print(f"All 48 blocks:  {all_blocks_flops:,}")
print(f"F_LMHead:       {f_lm_head:,}")
print(f"Total FLOPs:    {total_flops:,}")

print("--- MEDIUM ---")
vocab_size = 50257
context_length = 1024
num_layers = 24
d = 1024
num_heads = 16
dff = 4288
batch_size = 1
T = context_length

d_head = d // num_heads

f_projections = batch_size * 4 * 2 * T * (d ** 2)
f_qk = batch_size * num_heads * 2 * T * T * d_head
f_av = batch_size * 2 * (T ** 2) * d
f_swiglu = batch_size * 3 * 2 * T * d * dff

block_flops = f_projections + f_qk + f_av + f_swiglu
all_blocks_flops = block_flops * num_layers
f_lm_head = batch_size * 2 * T * d * vocab_size
total_flops = all_blocks_flops + f_lm_head

print(f"F_projections:  {f_projections:,}")
print(f"F_QK:           {f_qk:,}")
print(f"F_AV:           {f_av:,}")
print(f"F_SwiGLU:       {f_swiglu:,}")
print(f"Block total:    {block_flops:,}")
print(f"All 48 blocks:  {all_blocks_flops:,}")
print(f"F_LMHead:       {f_lm_head:,}")
print(f"Total FLOPs:    {total_flops:,}")

print("--- LARGE ---")
vocab_size = 50257
context_length = 1024
num_layers = 36
d = 1280
num_heads = 20
dff = 4288
batch_size = 1
T = context_length

d_head = d // num_heads

f_projections = batch_size * 4 * 2 * T * (d ** 2)
f_qk = batch_size * num_heads * 2 * T * T * d_head
f_av = batch_size * 2 * (T ** 2) * d
f_swiglu = batch_size * 3 * 2 * T * d * dff

block_flops = f_projections + f_qk + f_av + f_swiglu
all_blocks_flops = block_flops * num_layers
f_lm_head = batch_size * 2 * T * d * vocab_size
total_flops = all_blocks_flops + f_lm_head

print(f"F_projections:  {f_projections:,}")
print(f"F_QK:           {f_qk:,}")
print(f"F_AV:           {f_av:,}")
print(f"F_SwiGLU:       {f_swiglu:,}")
print(f"Block total:    {block_flops:,}")
print(f"All 48 blocks:  {all_blocks_flops:,}")
print(f"F_LMHead:       {f_lm_head:,}")
print(f"Total FLOPs:    {total_flops:,}")



print("--- Part E: Parameters & Memory GPT-2 XL, Increased context_length  ---")
vocab_size = 50257
context_length = 16384
num_layers = 48
d = 1600
num_heads = 25
dff = 4288
batch_size = 1
T = context_length

d_head = d // num_heads

f_projections = batch_size * 4 * 2 * T * (d ** 2)
f_qk = batch_size * num_heads * 2 * T * T * d_head
f_av = batch_size * 2 * (T ** 2) * d
f_swiglu = batch_size * 3 * 2 * T * d * dff

block_flops = f_projections + f_qk + f_av + f_swiglu
all_blocks_flops = block_flops * num_layers
f_lm_head = batch_size * 2 * T * d * vocab_size
total_flops = all_blocks_flops + f_lm_head

print("\n--- Part B: FLOPs GPT-2 XL ---")
print(f"F_projections:  {f_projections:,}")
print(f"F_QK:           {f_qk:,}")
print(f"F_AV:           {f_av:,}")
print(f"F_SwiGLU:       {f_swiglu:,}")
print(f"Block total:    {block_flops:,}")
print(f"All 48 blocks:  {all_blocks_flops:,}")
print(f"F_LMHead:       {f_lm_head:,}")
print(f"Total FLOPs:    {total_flops:,}")
