"""交互式文白互译 REPL — 加载基座 + LoRA adapter"""
from mlx_lm import load, generate

MODEL = "./models/Qwen2.5-1.5B-Instruct"
ADAPTER = "adapters"
SYSTEM = "你是一个精通文言文的翻译助手，能够在白话文与文言文之间互相转换。"

def main():
    print(f"加载 {MODEL} + LoRA({ADAPTER}) ...")
    model, tokenizer = load(MODEL, adapter_path=ADAPTER)
    print("就绪。输入 'q' 退出。前缀示例: '请将以下白话文翻译为文言文：...'\n")
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"q", "quit", "exit"}:
            break
        if not user:
            continue
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        print(f"模型> {out}\n")

if __name__ == "__main__":
    main()
