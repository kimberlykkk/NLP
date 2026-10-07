# 任务三实验记录（手写 LoRA + SFT + DPO，Qwen2.5-0.5B）

> 本机：RTX 5070 Ti Laptop（12GB）+ torch 2.11.0+cu128 + transformers 5.2.0
> 数据：MOSS-003 SFT 子集 1 万条（取 8000，ModelScope `mxnaxvex/moss-003-sft-data-10w`）
>       DPO 偏好数据 Human-Like-DPO（ModelScope `okwinds/Human-Like-DPO-Dataset`，21768 对）
> 基座：`models/Qwen2.5-0.5B`（ModelScope，988MB）

## 最终状态（自检三项全过）

| 测试 | 结果 | 关键数值 |
|---|---|---|
| `lora_param_count` | ✅ | 可训练 540,672 / 494,573,440 = **0.109%**（阈值 <5%） |
| `loss_masking` | ✅ | mask 比例 **0.422**（区间 0.2-0.9，user/marker 全 -100） |
| `sft_vs_base` | ✅ | `ckpt/sft/` 非空（8.80M 参数 LoRA） |

## 实现要点

- **手写 LoRA**（`src/lora.py`）：A `(r,in)` kaiming 初始化、B `(out,r)` 零初始化、
  scaling=alpha/r；inject 后**整个基座冻结**只有 A/B 可训练（自检按
  `["q_proj","v_proj"], r=8, alpha=16` 校验占比）；`merge_lora` 折回原权重。
  训练用全 7 个目标层（q/k/v/o/gate/up/down），r=16, alpha=32。
- **Qwen chat 模板**（`src/chat.py`）：`<|im_start|>{role}\n{content}<|im_end|>\n`；
  loss masking 用 tokenizer offsets 精确判定"保留区间"。
- **SFT**：8000 条 MOSS，batch 8 / max_len 384 / 1 epoch，loss 1.32 → **1.196**，
  1000 步约 15 分钟。
- **DPO**：policy = base+SFT LoRA；ref = base+SFT 冻结；β=0.1；
  logp 按 response token 均值（错位 1 的 next-token 口径）。
  终版配置：1500 对 / batch 4 / max_len 256 / lr 1e-5 / 1 epoch，
  loss 0.69 → **0.23**，margin -0.11 → **5.53**，win_rate 0.36 → **0.89**（375 步，3 分钟）。

## 消融/现象观察（写进报告）

1. **模型规模×LoRA**：8.80M 可训练参数（0.5B 的 1.75%）。LoRA 全线性层 vs 只 q/v：
   前者效果更好（任务三 README 建议实验）。
2. **DPO 学习率是第一敏感旋钮**：lr=1e-4 时 1 epoch margin 冲到 80、win_rate 1.0、
   生成坍塌成整段"！！！"（Human-Like 数据全是感叹号/表情风格，模型学到
   "全感叹号" 这一个输出）——鲜明地展示了 DPO 过训/模式坍塌；lr=1e-5 后
   margin 收在 5.5、win_rate 0.89，输出恢复正常。
3. **DPO 数据分布的局限**：Human-Like-DPO 是英文口语化闲聊，对中文指令任务的
   影响是"语调"层面的（相比 SFT 更结构化地起势），属于正常现象；
   想显著改变中文行为应换中文偏好数据（任务 README 建议探索方向）。
4. base vs SFT vs DPO 的可见差异：base 只会续写废话；SFT 开始"会对话"；
   DPO 在句式组织上略优于 SFT（见 results/compare.md）。

## 踩坑记录（写报告/复盘很有价值）

1. **LoRA 注入要冻结整个基座**，不是只冻结被包裹的 Linear——否则可训练占比 95%，
   `lora_param_count` 直接不通过。
2. **LoRA 参数必须在原权重所在 device/dtype 上创建**：模型 `.to('cuda')` 之后才注入，
   会把 A/B 留在 CPU 报 mat2 device 错误。
3. **SFT loss 的 next-token 错位**（最大坑）：`logits[t]` 要与 `labels[t+1]` 对齐。
   一开始写成对齐（预测"当前 token"）→ 模型学"看到自己就复读"，
   loss 假性收敛到 0.0002，生成却永远是换行死循环。改成错位后
   loss 1.32→1.20，模型立刻学会了真实指令跟随。
4. **assistant 头部（`<|im_start|>assistant\n`）也要作为训练目标**：否则小模型
   学不会"用户说完→生成 assistant 头部"的转换，生成退化为复读 base 先行习惯。
5. **DPO 的 logp 用 next-token 口径**（log_probs[:, :-1] 对 ids[:, 1:] 聚集）。
6. bf16 直接过 CE（别 .float() 转 151936 类 logits：单 batch 转 fp32 就 3.7GB OOM）。
