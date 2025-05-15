import torch
import torch.cuda
import torch.autograd.profiler as torch_profiler
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# Change this import to use tiny_yolov3_str architecture
from lava.lib.dl.slayer.object_detection.models.tiny_yolov3_str import Network
from lava.lib.dl import slayer

class CUDAProfiler:
    """CUDA memory and performance profiler for SDNN YOLO models."""
    
    def __init__(self, 
                 network,
                 device: torch.device,
                 batch_size: int = 1,
                 img_size: tuple = (224, 224),
                 time_steps: int = 1):
        """Initialize the profiler with network and parameters."""
        self.network = network
        self.device = device
        self.batch_size = batch_size
        self.img_size = img_size
        self.time_steps = time_steps
        
        # Generate dummy input tensor
        self.input = torch.randn(
            batch_size, 3, img_size[0], img_size[1], time_steps,
            device=self.device
        )
        
        # Stats storage
        self.layer_time: Dict[str, float] = {}
        self.layer_memory: Dict[str, float] = {}
        self.overall_stats: Dict[str, float] = {}
    
    def profile_forward_pass(self, warmup: int = 3, runs: int = 10, 
                            detailed: bool = True, save_trace: bool = False):
        """Profile the network's forward pass with PyTorch profiler."""
        # Warmup runs
        print(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            with torch.no_grad():
                self.network(self.input)
        
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()
        
        # Profile with torch profiler
        print(f"Profiling network with batch size {self.batch_size}...")
        
        with torch_profiler.profile(use_device='cuda') as prof:
            for _ in range(runs):
                with torch.no_grad():
                    output, counts = self.network(self.input)
        
        # Get overall statistics
        memory_stats = torch.cuda.memory_stats(self.device)
        
        self.overall_stats = {
            'peak_allocated_memory_MB': torch.cuda.max_memory_allocated(self.device) / 1024**2,
            'peak_reserved_memory_MB': torch.cuda.max_memory_reserved(self.device) / 1024**2,
            'current_allocated_MB': memory_stats["allocated_bytes.all.current"] / 1024**2,
            'current_reserved_MB': memory_stats["reserved_bytes.all.current"] / 1024**2
        }
        
        # Print overall statistics
        print("\n===== OVERALL MEMORY STATISTICS =====")
        print(f"Peak allocated memory: {self.overall_stats['peak_allocated_memory_MB']:.2f} MB")
        print(f"Peak reserved memory:  {self.overall_stats['peak_reserved_memory_MB']:.2f} MB")
        print(f"Current allocated:     {self.overall_stats['current_allocated_MB']:.2f} MB")
        print(f"Current reserved:      {self.overall_stats['current_reserved_MB']:.2f} MB")
        
        # Print detailed profiling results
        print("\n===== DETAILED CUDA OPERATION STATISTICS =====")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20))
        
        if detailed:
            # Extract per-block statistics
            self._analyze_blocks(prof)
        
        if save_trace:
            prof.export_chrome_trace("yolo_trace.json")
            print("\nChrome trace file saved to: yolo_trace.json")
            print("View it at chrome://tracing/")
        
        return self.overall_stats
    
    def profile_layer_by_layer(self):
        """Profile each layer individually to identify memory usage per layer."""
        print("\n===== PROFILING NETWORK LAYER BY LAYER =====")
        
        self.network.eval()
        
        # Create storage for tracking memory and performance
        memory_trackers = []
        
        # Modified for tiny_yolov3_str architecture
        block_groups = {
            'input_blocks': self.network.input_blocks,
            'backend_blocks': self.network.backend_blocks,
            'head1_backend': self.network.head1_backend,
            'head1_blocks': self.network.head1_blocks,
            'head2_backend': self.network.head2_backend,
            'head2_blocks': self.network.head2_blocks
        }
        
        # Collect all blocks for profiling
        all_blocks = []
        block_names = []
        
        for group_name, blocks in block_groups.items():
            for i, block in enumerate(blocks):
                all_blocks.append(block)
                block_names.append(f"{group_name}_{i}")
                memory_trackers.append({
                    'name': f"{group_name}_{i}",
                    'before': 0,
                    'after': 0,
                    'diff': 0
                })
        
        # Hook for memory tracking
        def make_hook(idx):
            def hook(module, inp, output):
                torch.cuda.synchronize()
                memory_trackers[idx]['after'] = torch.cuda.memory_allocated()
                memory_trackers[idx]['diff'] = memory_trackers[idx]['after'] - memory_trackers[idx]['before']
                print(f"Layer {memory_trackers[idx]['name']} - "
                      f"Memory: {memory_trackers[idx]['diff'] / 1024**2:.2f} MB")
            return hook
        
        # Register hooks
        handles = []
        for i, block in enumerate(all_blocks):
            handles.append(block.register_forward_hook(make_hook(i)))
        
        # Input tensor
        with torch.no_grad():
            x = self.input
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            
            # Process through each layer individually
            # For tiny_yolov3_str, we'll follow the forward pass structure
            
            # Input blocks
            for i, block in enumerate(self.network.input_blocks):
                idx = block_names.index(f"input_blocks_{i}")
                torch.cuda.synchronize()
                memory_trackers[idx]['before'] = torch.cuda.memory_allocated()
                
                # Time the execution
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer input_blocks_{i} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                # Track event counts
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer input_blocks_{i} - "
                          f"Event rate: {event_rate:.4f}")
            
            # Backend blocks
            for i, block in enumerate(self.network.backend_blocks):
                idx = block_names.index(f"backend_blocks_{i}")
                torch.cuda.synchronize()
                memory_trackers[idx]['before'] = torch.cuda.memory_allocated()
                
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer backend_blocks_{i} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer backend_blocks_{i} - "
                          f"Event rate: {event_rate:.4f}")
            
            # Store backend output for head2
            backend_out = x
            
            # Head1 backend
            for i, block in enumerate(self.network.head1_backend):
                idx = block_names.index(f"head1_backend_{i}")
                torch.cuda.synchronize()
                memory_trackers[idx]['before'] = torch.cuda.memory_allocated()
                
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer head1_backend_{i} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer head1_backend_{i} - "
                          f"Event rate: {event_rate:.4f}")
            
            # Store head1_backend output
            h1_backend_out = x
            
            # Head1 blocks
            for i, block in enumerate(self.network.head1_blocks):
                idx = block_names.index(f"head1_blocks_{i}")
                torch.cuda.synchronize()
                memory_trackers[idx]['before'] = torch.cuda.memory_allocated()
                
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer head1_blocks_{i} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer head1_blocks_{i} - "
                          f"Event rate: {event_rate:.4f}")
            
            # Head2 backend starting from h1_backend_out
            x = h1_backend_out
            for i, block in enumerate(self.network.head2_backend):
                idx = block_names.index(f"head2_backend_{i}")
                torch.cuda.synchronize()
                memory_trackers[idx]['before'] = torch.cuda.memory_allocated()
                
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer head2_backend_{i} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer head2_backend_{i} - "
                          f"Event rate: {event_rate:.4f}")
            
            # Concatenate for head2 input as in the model's forward method
            x = torch.concat([x, backend_out], dim=1)
            
            # Head2 blocks
            for i, block in enumerate(self.network.head2_blocks):
                idx = block_names.index(f"head2_blocks_{i}")
                torch.cuda.synchronize()
                memory_trackers[idx]['before'] = torch.cuda.memory_allocated()
                
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer head2_blocks_{i} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer head2_blocks_{i} - "
                          f"Event rate: {event_rate:.4f}")
        
        # Remove all hooks
        for handle in handles:
            handle.remove()
        
        return memory_trackers
    
    def profile_with_memory_tracking(self):
        """Profile network with precise memory tracking at each step."""
        self.network.eval()
        memory_usage = []
        
        def track_memory():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        
        # Initial memory usage
        base_mem = track_memory()
        memory_usage.append(("Initial", base_mem))
        
        with torch.no_grad():
            # Forward through blocks with memory tracking
            x = self.input
            memory_usage.append(("Input", track_memory()))
            
            # Input blocks
            for i, block in enumerate(self.network.input_blocks):
                x = block(x)
                memory_usage.append((f"Input Block {i}", track_memory()))
            
            # Backend blocks
            for i, block in enumerate(self.network.backend_blocks):
                x = block(x)
                memory_usage.append((f"Backend Block {i}", track_memory()))
            
            # Store backend output
            backend_out = x
            
            # Head1 backend
            for i, block in enumerate(self.network.head1_backend):
                x = block(x)
                memory_usage.append((f"Head1 Backend {i}", track_memory()))
            
            # Store h1_backend
            h1_backend = x
            
            # Head1 blocks
            for i, block in enumerate(self.network.head1_blocks):
                x = block(x)
                memory_usage.append((f"Head1 Block {i}", track_memory()))
            
            head1 = x
            
            # Head2 backend (from h1_backend)
            x = h1_backend
            for i, block in enumerate(self.network.head2_backend):
                x = block(x)
                memory_usage.append((f"Head2 Backend {i}", track_memory()))
            
            # Concat with backend output
            x = torch.concat([x, backend_out], dim=1)
            memory_usage.append(("Concat", track_memory()))
            
            # Head2 blocks
            for i, block in enumerate(self.network.head2_blocks):
                x = block(x)
                memory_usage.append((f"Head2 Block {i}", track_memory()))
            
            head2 = x
            
            # Final processing
            head1_raw = self.network.yolo_raw(head1)
            head2_raw = self.network.yolo_raw(head2)
            memory_usage.append(("YOLO Raw", track_memory()))
            
            if not self.network.training:
                output1 = self.network.yolo(head1_raw, self.network.anchors[0])
                output2 = self.network.yolo(head2_raw, self.network.anchors[1])
                output = torch.concat([output1, output2], dim=1)
                memory_usage.append(("YOLO Final", track_memory()))
        
        print("\n===== MEMORY USAGE THROUGHOUT NETWORK =====")
        print(f"{'Layer':20s} | {'Absolute (MB)':15s} | {'Relative (MB)':15s}")
        print("-" * 55)
        
        for i, (name, mem) in enumerate(memory_usage):
            if i == 0:
                rel_mem = 0
            else:
                rel_mem = mem - memory_usage[i-1][1]
            print(f"{name:20s} | {mem:15.2f} | {rel_mem:15.2f}")
            
        return memory_usage
    
    def _analyze_blocks(self, prof):
        """Analyze profiling results for each network block."""
        print("\n===== BLOCK-LEVEL ANALYSIS =====")
        
        # This needs to be modified for the tiny_yolov3_str architecture
        # Group blocks by their module type for analysis
        block_groups = {
            'input_blocks': self.network.input_blocks,
            'backend_blocks': self.network.backend_blocks,
            'head1_backend': self.network.head1_backend,
            'head1_blocks': self.network.head1_blocks,
            'head2_backend': self.network.head2_backend,
            'head2_blocks': self.network.head2_blocks
        }
        
        # Print available operation keys for debugging
        print("Available operation keys in profiler output:")
        stats = prof.key_averages()
        all_keys = [stat.key for stat in stats]
        
        # Print a few sample keys to understand the format
        for i, key in enumerate(all_keys[:5]):  # Print first 5 keys
            print(f"Key {i}: {key}")
        
        # Extract block-related operations with more flexible matching
        block_stats = {}
        
        # Initialize stats containers for each block
        for group_name, blocks in block_groups.items():
            for i in range(len(blocks)):
                block_name = f"{group_name}_{i}"
                block_stats[block_name] = {
                    'cuda_time': 0, 
                    'cpu_time': 0,
                    'cuda_memory': 0,
                    'calls': 0,
                    'operations': []
                }
        
        # Try to match operations to blocks
        for stat in stats:
            for group_name, blocks in block_groups.items():
                for i in range(len(blocks)):
                    # Try different pattern formats that might match block operations
                    patterns = [
                        f"{group_name}.{i}",
                        f"{group_name}[{i}]",
                        f"{group_name}_{i}",
                    ]
                    
                    matched = False
                    for pattern in patterns:
                        if pattern and pattern in stat.key:
                            block_name = f"{group_name}_{i}"
                            block_stats[block_name]['cuda_time'] += stat.cuda_time_total
                            block_stats[block_name]['cpu_time'] += stat.cpu_time_total
                            if hasattr(stat, 'cuda_memory_usage') and stat.cuda_memory_usage > 0:
                                block_stats[block_name]['cuda_memory'] += stat.cuda_memory_usage
                            block_stats[block_name]['calls'] += 1
                            block_stats[block_name]['operations'].append(stat.key)
                            matched = True
                            break
        
        # Print block statistics
        print(f"\n{'Block':20s} | {'CUDA Time (ms)':15s} | {'CPU Time (ms)':15s} | {'Memory (MB)':15s} | {'Operation Count':15s}")
        print("-" * 90)
        
        for block_name, stats in block_stats.items():
            cuda_time_ms = stats['cuda_time'] / 1000  # μs to ms
            cpu_time_ms = stats['cpu_time'] / 1000    # μs to ms
            memory_mb = stats['cuda_memory'] / (1024**2) if stats['cuda_memory'] > 0 else 0
            op_count = len(stats['operations'])
            
            print(f"{block_name:20s} | {cuda_time_ms:15.2f} | {cpu_time_ms:15.2f} | {memory_mb:15.2f} | {op_count:15d}")
            
        # Fall back to manual layer-by-layer analysis if no operations were captured
        total_ops = sum(len(stats['operations']) for stats in block_stats.values())
        if total_ops == 0:
            print("\nNo block operations found in profiler output.")
            print("The layer-by-layer profiling will provide more detailed information.")

    def visualize_memory_usage(self, memory_data, filename='memory_usage.png'):
        """Visualize memory usage across network layers."""
        layers = [x[0] for x in memory_data]
        absolute = [x[1] for x in memory_data]
        
        # Calculate relative memory for each step
        relative = [0]
        for i in range(1, len(absolute)):
            relative.append(absolute[i] - absolute[i-1])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Absolute memory plot
        ax1.plot(absolute, 'o-', linewidth=2, markersize=8)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=60, ha='right')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('Absolute Memory Usage')
        ax1.grid(True)
        
        # Relative memory plot
        bars = ax2.bar(range(len(relative)), relative)
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=60, ha='right')
        ax2.set_ylabel('Memory Change (MB)')
        ax2.set_title('Relative Memory Usage (Change at Each Layer)')
        
        # Color bars based on whether memory increased or decreased
        for i, bar in enumerate(bars):
            if relative[i] >= 0:
                bar.set_color('tab:blue')
            else:
                bar.set_color('tab:red')
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Memory usage visualization saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CUDA Memory and Performance Profiler for YOLO SDNN')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for profiling')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], help='Image dimensions (height width)')
    parser.add_argument('--time_steps', type=int, default=1, help='Number of time steps')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for averaging')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--save_trace', action='store_true', help='Save Chrome trace file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-layer analysis')
    parser.add_argument('--layer_by_layer', action='store_true', help='Profile each layer individually')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes for the model')
    parser.add_argument('--threshold', type=float, default=0.1, help='Neuron threshold')
    parser.add_argument('--tau_grad', type=float, default=0.1, help='Surrogate gradient time constant')
    parser.add_argument('--scale_grad', type=float, default=0.1, help='Surrogate gradient scaling')
    parser.add_argument('--clamp_max', type=float, default=5.0, help='Clamping maximum value')
    args = parser.parse_args()
    
    # Initialize CUDA device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("WARNING: CUDA not available, running on CPU!")
    
    # Create network - using tiny_yolov3_str architecture
    network = Network(
        num_classes=args.num_classes,
        threshold=args.threshold,
        tau_grad=args.tau_grad,
        scale_grad=args.scale_grad,
        clamp_max=args.clamp_max
    ).to(device)
    
    network.eval()  # Set to evaluation mode
    
    # Print basic network info
    print("\n===== NETWORK INFORMATION =====")
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Network architecture: {network.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Count total blocks
    block_count = (len(network.input_blocks) + len(network.backend_blocks) + 
                  len(network.head1_backend) + len(network.head1_blocks) +
                  len(network.head2_backend) + len(network.head2_blocks))
    print(f"Total blocks: {block_count}")
    
    # Initialize profiler
    profiler = CUDAProfiler(
        network=network,
        device=device,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        time_steps=args.time_steps
    )
    
    # Run detailed profiling
    overall_stats = profiler.profile_forward_pass(
        warmup=args.warmup, 
        runs=args.runs,
        detailed=args.detailed,
        save_trace=args.save_trace
    )
    
    # Optionally profile layer by layer
    if args.layer_by_layer:
        memory_trackers = profiler.profile_layer_by_layer()
    
    # Detailed memory tracking through network execution
    memory_data = profiler.profile_with_memory_tracking()
    
    # Visualize memory usage
    profiler.visualize_memory_usage(memory_data)
    
    print("\nProfiling complete!")