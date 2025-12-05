from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="/home/henry/vidarr/log/tinyvit")

# temporal_breakdown_df = analyzer.get_temporal_breakdown()

idle_time_df = analyzer.get_idle_time_breakdown()
