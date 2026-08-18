import time
import torch
from model.model import VCC  # Import VCC model

# For GFLOPs calculation
try:
    from thop import profile
    thop_available = True
except ImportError:
    thop_available = False
    print("警告: 'thop' 库未找到。GFLOPs 将不会被计算。请运行 'pip install thop' 来安装它。")


if __name__ == '__main__':
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # --- Model Instantiation ---
    print("\n正在实例化 VCC 模型...")
    # Instantiate VCC with current model architecture
    model = VCC(
        in_chans=3,
        out_chans=1,
        backbone_name='convnext_tiny',
        load_pretrained=False,  # Set to True if you want to load pretrained weights
        temporal_backbone_name='convnext_atto.d2_in1k',
        temporal_load_pretrained=True,  # Set to False for clean profiling
        save_wave_images=False,
        debug_high_freq=False
    ).to(device)
    model.eval()  # 重要！避免验证阶段因训练模式导致的计算不稳定
    print("VCC 模型实例化完成，并设置为评估模式。")

    # --- 1. Parameter Count ---
    print("\n--- 模型参数量 ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    print(f"模型可训练参数量: {params_m:.2f} M")

    # --- 2. FLOPs/GFLOPs Calculation ---
    print("\n--- 计算复杂度 (FLOPs) ---")
    if thop_available:
        # GFLOPs input resolution (e.g., 384x384, 2 frames, 3 channels)
        h_gflops, w_gflops = 384, 384
        model_num_frames_gflops = 2  # VCC requires 2 frames
        model_num_channels_gflops = 3
        # VCC input format: [2, C, H, W]
        dummy_input_gflops = torch.randn(model_num_frames_gflops, model_num_channels_gflops, h_gflops, w_gflops).to(device)
        
        print(f"正在计算 FLOPs (输入分辨率: {h_gflops}x{w_gflops}, {model_num_frames_gflops}帧)...")
        try:
            macs, params_thop = profile(model, inputs=(dummy_input_gflops,), verbose=False)
            flops = macs * 2  # 1 MAC ≈ 2 FLOPs
            gflops = flops / 1e9  # 转换为 GFLOPs
            mflops = flops / 1e6  # 转换为 MFLOPs
            
            print(f"\n计算复杂度:")
            print(f"  MACs (乘加运算):     {macs / 1e9:.2f} G")
            print(f"  FLOPs (浮点运算):    {flops:.0f}")
            print(f"  MFLOPs:              {mflops:.2f} M")
            print(f"  GFLOPs:              {gflops:.2f} G")
            print(f"\n说明:")
            print(f"  - 1 MAC = 1 乘法 + 1 加法 = 2 FLOPs")
            print(f"  - 1 GFLOPs = 10^9 FLOPs")
            print(f"  - 输入尺寸: [{model_num_frames_gflops}, {model_num_channels_gflops}, {h_gflops}, {w_gflops}]")
        except Exception as e:
            print(f"计算 FLOPs 失败: {e}")
            print("  请检查模型结构是否与 'thop' 兼容，或尝试更新 'thop' 库。")
    else:
        print("FLOPs 计算已跳过，因为 'thop' 库不可用。")
        print("请运行: pip install thop")

    # --- 3. Inference Time Measurement ---
    print("\n--- 推理时间测量 ---")
    # Inference input resolution (e.g., 1024x1024, 2 frames, 3 channels)
    h_inf, w_inf = 1024, 1024
    model_num_frames_inf = 2  # VCC requires 2 frames
    model_num_channels_inf = 3
    num_samples = 100  # Number of samples for inference timing
    
    print(f"正在生成 {num_samples} 个测试样本 (分辨率: {h_inf}x{w_inf}, {model_num_frames_inf}帧)...")
    # VCC input format: [2, C, H, W]
    inputs = [torch.randn(model_num_frames_inf, model_num_channels_inf, h_inf, w_inf).to(device) for _ in range(num_samples)]
    print("测试样本生成完毕。")

    # --- Warm-up 阶段 ---
    warmup_runs = 20 # Increased warm-up runs
    print(f"正在进行预热 ({warmup_runs} 次运行)...")
    with torch.no_grad():
        for i in range(min(warmup_runs, num_samples)): # Ensure warmup doesn't exceed available samples
            _ = model(inputs[i])
    print("预热完成。")

    # --- 正式计时阶段 ---
    print(f"正在使用 {num_samples} 个样本进行正式计时...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter() # Use perf_counter for more precision

    with torch.no_grad():  # 禁用梯度计算
        for i in range(num_samples):
            _ = model(inputs[i])

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    print("计时完成。")
    
    # --- 统计结果 ---
    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_samples) * 1000  # 单样本平均延迟（毫秒）
    avg_fps = num_samples / total_time if total_time > 0 else 0
    time_for_100_samples = (total_time / num_samples) * 100 if num_samples > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"{'最终结果汇总':^60}")
    print(f"{'='*60}")
    
    print(f"\n【模型信息】")
    print(f"  模型名称:            VCC (Video Crowd Counting)")
    print(f"  输入格式:            [T, C, H, W] = [{model_num_frames_inf}, {model_num_channels_inf}, {h_inf}, {w_inf}]")
    print(f"  可训练参数量:        {params_m:.2f} M")
    
    if thop_available:
        print(f"\n【计算复杂度】(输入: {h_gflops}x{w_gflops})")
        print(f"  FLOPs:               {flops:.0f}")
        print(f"  GFLOPs:              {gflops:.2f} G")
        print(f"  MACs:                {macs / 1e9:.2f} G")
    
    print(f"\n【推理速度】(输入: {h_inf}x{w_inf}, {num_samples}个样本)")
    print(f"  总耗时:              {total_time:.4f} 秒")
    print(f"  平均延迟:            {avg_latency_ms:.2f} 毫秒/样本")
    print(f"  吞吐量:              {avg_fps:.2f} FPS")
    print(f"  100次推理等效时间:   {time_for_100_samples:.4f} 秒")
    
    print(f"\n【论文表格格式】")
    print(f"  Params (M):          {params_m:.2f}")
    if thop_available:
        print(f"  GFLOPs:              {gflops:.2f}")
    print(f"  Inference Time (s):  {time_for_100_samples:.4f} (100 samples)")
    
    print(f"\n{'='*60}")
    
    if not thop_available:
        print("\n⚠️  提示: 要计算 GFLOPs，请安装 'thop' 库:")
        print("    pip install thop")
