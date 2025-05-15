#!/usr/bin/env python3
import subprocess
import sys
import time

# 순서대로 실행할 명령어 리스트
commands = [
    ["python3", "train_sdnn.py", "--model", "yolo_kp", "--num_classes", "4", "--epoch", "300", "--b", "8", "--exp", "b8"],
    ["python3", "train_sdnn.py", "--model", "yolo_kp", "--num_classes", "4", "--epoch", "300", "--b", "6", "--exp", "b6"],
    ["python3", "train_sdnn.py", "--model", "yolo_kp", "--num_classes", "4", "--epoch", "300", "--b", "4", "--exp", "b4"],
    ["python3", "train_sdnn.py", "--model", "yolo_kp", "--num_classes", "4", "--epoch", "300", "--b", "2", "--exp", "b2"],
]

POLL_INTERVAL = 60  # GPU idle 대기 간격 (초)

def get_gpu_pids():
    """nvidia-smi 로 현재 GPU 상의 compute 앱(PID)을 조회합니다."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        return []  # nvidia-smi 실패 시 빈 리스트로 간주
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return [int(pid) for pid in lines]

def wait_for_gpu_idle():
    """GPU에 실행 중인 프로세스가 없을 때까지 대기합니다."""
    while True:
        pids = get_gpu_pids()
        if not pids:
            return
        print(f"⏳ GPU busy, PIDs running: {pids}. {POLL_INTERVAL}s 후 재확인...")
        time.sleep(POLL_INTERVAL)

def run_with_oom_fallback(cmd):
    """
    주어진 cmd를 실행하고, 'CUDA out of memory' 감지 시 False 반환,
    아니면 True 반환합니다.
    """
    print(f"\n▶️ 실행 중: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    saw_oom = False

    for line in proc.stdout:
        print(line, end="")
        if "CUDA out of memory" in line:
            saw_oom = True

    proc.wait()
    return not saw_oom

def main():
    for cmd in commands:
        # GPU가 완전히 idle 상태일 때까지 대기
        wait_for_gpu_idle()

        success = run_with_oom_fallback(cmd)
        if success:
            print(f"\n✅ 배치 사이즈 {cmd[-1]} 로 성공적으로 완료되었습니다.")
            sys.exit(0)
        else:
            print("⚠️  OOM 감지, 다음 배치 사이즈로 재시도합니다.")

    print("\n❌ 모든 시도에서 OOM이 발생했습니다. 스크립트를 종료합니다.")
    sys.exit(1)

if __name__ == "__main__":
    main()