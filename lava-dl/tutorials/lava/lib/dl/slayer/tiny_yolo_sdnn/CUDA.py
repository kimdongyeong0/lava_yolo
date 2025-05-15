import torch
import torch.cuda
import torch.autograd.profiler as torch_profiler  # Renamed here
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List
import time

# Import Network from yolo_kp
from lava.lib.dl.slayer.object_detection.models.yolo_kp import Network
from lava.lib.dl import slayer

class CUDAProfiler:
    """CUDA memory and performance profiler for SDNN YOLO models."""
    
    def __init__(self, 
                 network,
                 device: torch.device,
                 batch_size: int = 1,
                 img_size: tuple = (224, 224),
                 time_steps: int = 1):
        """Initialize the profiler with network and parameters.
        
        Parameters
        ----------
        network : torch.nn.Module
            The network to profile
        device : torch.device
            CUDA device to use
        batch_size : int
            Batch size for profiling
        img_size : tuple
            Image dimensions (height, width)
        time_steps : int
            Number of time steps for temporal data
        """
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
        
        # Profile with torch profiler (compatible with older PyTorch versions)
        print(f"Profiling network with batch size {self.batch_size}...")
        
        with torch_profiler.profile(use_device='cuda') as prof:
            for _ in range(runs):  # Run multiple times for more stable measurements
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
            prof.export_chrome_trace("yolo_kp_trace.json")
            print("\nChrome trace file saved to: yolo_kp_trace.json")
            print("View it at chrome://tracing/")
        
        return self.overall_stats
    
    def profile_layer_by_layer(self):
        """Profile each layer individually to identify memory usage per layer."""
        print("\n===== PROFILING NETWORK LAYER BY LAYER =====")
        
        self.network.eval()
        layers_to_profile = ['blocks', 'heads']
        
        # Create separate hooks for each block
        memory_trackers = []
        
        # Step through blocks module list
        if hasattr(self.network, 'blocks'):
            for i, block in enumerate(self.network.blocks):
                memory_trackers.append({
                    'name': f'block_{i}',
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
        for i, block in enumerate(self.network.blocks):
            handles.append(block.register_forward_hook(make_hook(i)))
        
        # Input tensor
        with torch.no_grad():
            x = self.input
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            
            # Process through each layer individually
            for i, block in enumerate(self.network.blocks):
                torch.cuda.synchronize()
                memory_trackers[i]['before'] = torch.cuda.memory_allocated()
                
                # Time the execution
                start = time.time()
                x = block(x)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"Layer {memory_trackers[i]['name']} - "
                      f"Time: {elapsed*1000:.2f} ms")
                
                # Track event counts
                if hasattr(x, 'shape'):
                    event_rate = slayer.utils.event_rate(x)
                    print(f"Layer {memory_trackers[i]['name']} - "
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
            
            for i, block in enumerate(self.network.blocks):
                x = block(x)
                memory_usage.append((f"Block {i}", track_memory()))
                
            # Process detection head
            y = self.network.heads[0](x)
            memory_usage.append(("Head", track_memory()))
            
            if not self.network.training:
                head_output = self.network.yolo_raw(y)
                memory_usage.append(("YOLO Raw", track_memory()))
                
                output = self.network.yolo(head_output, self.network.anchors[0])
                memory_usage.append(("YOLO Final", track_memory()))
        
        print("\n===== MEMORY USAGE THROUGHOUT NETWORK =====")
        print(f"{'Layer':15s} | {'Absolute (MB)':15s} | {'Relative (MB)':15s}")
        print("-" * 50)
        
        for i, (name, mem) in enumerate(memory_usage):
            if i == 0:
                rel_mem = 0
            else:
                rel_mem = mem - memory_usage[i-1][1]
            print(f"{name:15s} | {mem:15.2f} | {rel_mem:15.2f}")
            
        return memory_usage
    
    def _analyze_blocks(self, prof):
        """Analyze profiling results for each network block."""
        print("\n===== BLOCK-LEVEL ANALYSIS =====")
        
        # Print available operation keys for debugging
        print("Available operation keys in profiler output:")
        stats = prof.key_averages()
        all_keys = [stat.key for stat in stats]
        
        # Print a few sample keys to understand the format
        for i, key in enumerate(all_keys[:5]):  # Print first 5 keys
            print(f"Key {i}: {key}")
        
        # Extract block-related operations with more flexible matching
        block_stats = {}
        for i in range(len(self.network.blocks)):
            block_name = f"Block {i}"
            block_stats[block_name] = {
                'cuda_time': 0, 
                'cpu_time': 0,
                'cuda_memory': 0,
                'calls': 0,
                'operations': []
            }
        
        # More flexible pattern matching
        for stat in stats:
            for i in range(len(self.network.blocks)):
                # Try different pattern formats that might match block operations
                patterns = [
                    f"blocks.{i}",
                    f"blocks[{i}]",
                    f"block_{i}",
                ]
                
                matched = False
                for pattern in patterns:
                    if pattern and pattern in stat.key:
                        block_name = f"Block {i}"
                        block_stats[block_name]['cuda_time'] += stat.cuda_time_total
                        block_stats[block_name]['cpu_time'] += stat.cpu_time_total
                        if hasattr(stat, 'cuda_memory_usage') and stat.cuda_memory_usage > 0:
                            block_stats[block_name]['cuda_memory'] += stat.cuda_memory_usage
                        block_stats[block_name]['calls'] += 1
                        block_stats[block_name]['operations'].append(stat.key)
                        matched = True
                        break
        
        # Print block statistics
        print(f"\n{'Block':10s} | {'CUDA Time (ms)':15s} | {'CPU Time (ms)':15s} | {'Memory (MB)':15s} | {'Operation Count':15s}")
        print("-" * 85)
        
        for block_name, stats in block_stats.items():
            cuda_time_ms = stats['cuda_time'] / 1000  # μs to ms
            cpu_time_ms = stats['cpu_time'] / 1000    # μs to ms
            memory_mb = stats['cuda_memory'] / (1024**2) if stats['cuda_memory'] > 0 else 0
            op_count = len(stats['operations'])
            
            print(f"{block_name:10s} | {cuda_time_ms:15.2f} | {cpu_time_ms:15.2f} | {memory_mb:15.2f} | {op_count:15d}")
            
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
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Absolute memory plot
        ax1.plot(absolute, 'o-', linewidth=2, markersize=8)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title('Absolute Memory Usage')
        ax1.grid(True)
        
        # Relative memory plot
        bars = ax2.bar(range(len(relative)), relative)
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha='right')
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
    
    # Create network
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
    print(f"Number of blocks: {len(network.blocks)}")
    
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