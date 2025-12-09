from typing import Dict

from hta.trace_analysis import TraceAnalysis


def analyze_run(
    breakdown: str,
    trace_dir: str,
    trace_files: Dict[int, str] | None = None,
):
    analyzer = TraceAnalysis(trace_dir=trace_dir, trace_files=trace_files)

    if breakdown == "temporal":
        return analyzer.get_temporal_breakdown()
    elif breakdown == "idle":
        return analyzer.get_idle_time_breakdown()
    elif breakdown == "communication_overlap":
        return analyzer.get_comm_comp_overlap()
    elif breakdown == "kernel_launch_stats":
        return analyzer.get_cuda_kernel_launch_stats()
    else:
        raise NotImplementedError()
